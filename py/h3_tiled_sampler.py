"""
H3 Tiled Sampler

H3 视频模型专属分块采样节点, 采用 LTX 2.3 式的 2D (H×W) 分块采样原理.

设计目标 (参考 LTX 2.3 tiling 思路, 结合 H3 双模态特点):
  - 数据全程在显存上计算, 绝不掉到 CPU 内存 (半卡死/缓慢生成的根本原因).
  - 按可用显存自适应计算分块数/块大小, 用户无需猜 n_tiles, 设错也不会爆显存.
  - 复用全局噪声 + 全局文本条件 (guider.raw_conds), 仅在 latent 上切块,
    不逐块改条件 → 最大限度保住提示词内容, 减少漂移.
  - 每块算完即释放, 循环内不做 torch.cuda.empty_cache() (会强制同步变慢),
    只在结束统一 soft_empty_cache.
  - 支持 H×W 二维分块 + 可分离余弦窗口 (1D×1D 外积) 融合, 消除接缝.

与 LTX 2.3 的关键差异:
  - H3 是 (video, audio) 双模态 tuple, 音频 passthrough, 不参与采样.
  - 时长由 VAE 编码决定, 入口只做最小帧数保护, 绝不截断.
  - 不假设 model 有 process_latent_out, 用 hasattr 探测.

适用场景: H3 768p/2K 上采样精修时 token 数过多导致 attention 性能下降,
         通过空间分块让每块在 H3 DiT 训练分布的 token 数内工作.
不适用: 重去噪 (从纯噪声起步). 空间分块会破坏全局一致性.
"""

import math

import torch
import comfy.utils
import comfy.model_management as model_management
from comfy.nested_tensor import NestedTensor
import latent_preview


# H3 视频 VAE 训练约束
H3_VIDEO_FRAMES = 17       # 输入视频帧数硬约束
H3_LATENT_CHANNELS = 24    # H3 video latent 通道数

# 每个 (帧 × latent 平面像素) 的 DiT 激活峰值字节估算 (保守启发式).
# 仅用于 auto 模式估算分块数; 想要精确控制请切 manual 模式.
_BYTES_PER_PLANAR_TOKEN = 4096


# ─────────────────────────────────────────────────────────────────────────────
# H3 视频/音频提取与重建
# ─────────────────────────────────────────────────────────────────────────────

def _h3_extract(samples, debug=False):
    """
    H3 sampler 输出格式:
      1. NestedTensor (PyTorch built-in): 包含 video (5D) + audio
      2. 单 5D tensor: 仅视频
      3. tuple (video, audio): 视频在前, 音频在后
      4. list [video, audio]: 同 tuple
    返回 (video_5d, audio_or_None, format_info).
    """
    type_name = type(samples).__name__

    # --- NestedTensor (comfy.nested_tensor.NestedTensor, NOT torch.Tensor subclass) ---
    if hasattr(samples, "is_nested") and samples.is_nested:
        try:
            parts = list(samples.unbind())
            video = None
            audio = None
            for p in parts:
                if isinstance(p, torch.Tensor):
                    if p.dim() == 5 and video is None:
                        video = p
                    elif video is not None and audio is None:
                        audio = p
            if video is not None:
                if debug:
                    print(f"  \u00b7 [H3 extract] NestedTensor video={tuple(video.shape)} "
                          f"audio={tuple(audio.shape) if audio is not None else None}")
                return video, audio, {"type": "nested_tensor"}
        except Exception as e:
            if debug:
                print(f"  \u00b7 [H3 extract] NestedTensor unbind failed: {e}")

    # --- 普通 tensor ---
    if isinstance(samples, torch.Tensor):
        if debug:
            print(f"  \u00b7 [H3 extract] plain tensor {tuple(samples.shape)}")
        return samples, None, {"type": "tensor"}

    # --- tuple / list ---
    if isinstance(samples, (tuple, list)):
        video = None
        audio = None
        for i, item in enumerate(samples):
            if isinstance(item, torch.Tensor):
                if item.dim() == 5 and video is None:
                    video = item
                elif video is not None and audio is None:
                    audio = item
        if video is not None:
            fmt = "tuple" if isinstance(samples, tuple) else "list"
            if debug:
                print(f"  \u00b7 [H3 extract] {fmt} video={tuple(video.shape)} "
                      f"audio={tuple(audio.shape) if audio is not None else None}")
            return video, audio, {"type": fmt}
        raise TypeError(
            f"H3 extract: {type_name} 中未找到 5D video tensor. "
            f"items: {[type(it).__name__ for it in samples]}"
        )

    pub_attrs = [a for a in dir(samples) if not a.startswith("_")][:25]
    raise TypeError(
        f"H3 extract: 不支持的格式 '{type_name}'. "
        f"期望 5D tensor / NestedTensor / (tensor, tensor) tuple. "
        f"可用属性: {pub_attrs}"
    )


def _h3_reconstruct(video, audio, format_info, debug=False):
    """严格保持输入格式. 不引入新结构."""
    fmt = format_info.get("type", "tensor")
    if fmt == "nested_tensor":
        parts = [video] + ([audio] if audio is not None else [])
        return NestedTensor(parts)
    if fmt == "tensor":
        return video
    if fmt == "tuple":
        return (video, audio) if audio is not None else (video,)
    if fmt == "list":
        return [video, audio] if audio is not None else [video]
    # fallback
    return (video, audio) if audio is not None else video


# ─────────────────────────────────────────────────────────────────────────────
# H3 帧数调整
# ─────────────────────────────────────────────────────────────────────────────

def _adjust_frame_count(latent_5d, target_frames, mode, debug=False):
    """
    调整 video latent 的 T 维 (最小帧数保护).

    只保证 T 维不低于目标长度, T 已达标时原样返回, 绝不截断 —— 时长由上游 VAE
    编码决定 (如 5s 视频 latent T≈37 会完整保留).
    """
    B, C, T, H, W = latent_5d.shape
    target_T = round((target_frames - 3) / 4) + 1

    if T >= target_T:
        return latent_5d

    if mode == "error":
        raise ValueError(
            f"H3: latent 时间维 T={T}, 期望至少 T={target_T} "
            f"(对应 {target_frames} 帧). 当前 mode=error, 请调整输入或换模式."
        )

    if mode == "replicate_last":
        pad_n = target_T - T
        last = latent_5d[:, :, -1:, :, :].expand(-1, -1, pad_n, -1, -1)
        out = torch.cat([latent_5d, last], dim=2)
        if debug:
            print(f"  \u00b7 [frame] replicate_last: T {T} -> {target_T} (+{pad_n})")
    elif mode == "zero":
        pad_n = target_T - T
        zeros = torch.zeros(
            B, C, pad_n, H, W,
            dtype=latent_5d.dtype, device=latent_5d.device
        )
        out = torch.cat([latent_5d, zeros], dim=2)
        if debug:
            print(f"  \u00b7 [frame] zero: T {T} -> {target_T} (+{pad_n})")
    else:
        raise ValueError(f"H3: 未知 pad 模式 '{mode}'")

    return out.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# 分块数学 (与模型无关)
# ─────────────────────────────────────────────────────────────────────────────

def _tile_spans(size, n_tiles, overlap):
    """
    沿一条轴生成等长均匀重叠切片 [(start, end), ...].

    每块长度完全一致, 无缝隙完整覆盖 [0, size), 相邻块重叠由实际切片位置决定.
    首尾块只在一侧有 overlap (边缘不补).
    """
    if n_tiles <= 1 or size <= n_tiles:
        if size <= 0:
            raise ValueError(f"tile_spans: 轴长必须为正, 实际 {size}")
        return [(0, size)]

    tile_len = min(size, max(int(math.ceil((size + (n_tiles - 1) * overlap) / n_tiles)),
                             overlap + 1))
    if tile_len >= size:
        return [(0, size)]

    starts = torch.linspace(0, size - tile_len, n_tiles).round().long().tolist()
    spans = []
    for s in starts:
        spans.append((int(s), int(s + tile_len)))
    return spans


def _auto_split(height, width, frames, overlap, max_tiles, vram_budget_frac, debug=False):
    """
    按可用显存自适应估算 H 轴和 W 轴各分多少块 (auto 模式).

    估算模型: 单块去噪的 DiT 激活峰值 ≈ frames × (块 H×W) × 每 token 字节.
    用可用显存的 vram_budget_frac 作为单块噪声预留给算, 反推出目标块平面面积,
    再按 H/W 比例拆成 n_h × n_v, 并 clamp 到 [1, max_tiles].

    这是保守启发式, 想精确控制请切 manual 模式逐个设定 h_tiles / v_tiles.
    """
    device = model_management.get_torch_device()
    free = model_management.get_free_memory(device)
    budget = max(free, 1) * max(vram_budget_frac, 0.05)

    # 目标单块平面 token 数
    planar_budget = budget / (max(frames, 1) * _BYTES_PER_PLANAR_TOKEN)
    if planar_budget <= 0 or planar_budget >= height * width:
        return 1, 1, planar_budget, budget

    total = height * width
    n_est = max(1, int(math.ceil(total / planar_budget)))
    n_h = int(math.ceil(math.sqrt(n_est * height / width)))
    n_v = int(math.ceil(math.sqrt(n_est * width / height)))
    n_h = max(1, min(n_h, max_tiles))
    n_v = max(1, min(n_v, max_tiles))
    if debug:
        print(f"  \u00b7 [auto] free={free/1e9:.2f}GB budget={budget/1e9:.2f}GB "
              f"planar={planar_budget:.0f} -> {n_h}x{n_v} tiles")
    return n_h, n_v, planar_budget, budget


def _make_window_1d(length, ov_left, ov_right, dtype, device):
    """
    1D cosine 窗口: 中间全 1, 两侧 overlap 区做 (1+cos)/2 渐变.
    ov_left/ov_right: 该侧重叠 token 数 (无邻居则 0).
    """
    w = torch.ones(length, dtype=dtype, device=device)
    if ov_left > 0:
        n = min(ov_left, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos(t * 3.14159265)
            w[:n] = torch.minimum(w[:n], fade)
    if ov_right > 0:
        n = min(ov_right, length // 2 + 1)
        if n > 0:
            t = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[:-1]
            fade = 0.5 - 0.5 * torch.cos((1 - t) * 3.14159265)
            w[-n:] = torch.minimum(w[-n:], fade)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# 主节点
# ─────────────────────────────────────────────────────────────────────────────

class H3TiledSampler:
    """
    H3 视频模型分块采样节点 (LTX 2.3 式 2D 分块).

    沿 H 和 W 独立切块, 每块独立采样, 可分离余弦窗口融合.
    全程显存计算, 按可用显存自适应分块, 复用全局噪声与全局条件.
    音频 passthrough, 不参与采样.

    使用方法:
      1. 接入 H3 对应的 noise / guider / sampler / sigmas / latent
      2. tile_mode=auto: 按显存自适应; tile_mode=manual: 手动设 h_tiles/v_tiles
      3. 首次使用开 debug=True 验证分块布局
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise": ("NOISE", {
                    "tooltip": "H3 噪声生成器."
                }),
                "guider": ("GUIDER", {
                    "tooltip": "H3 CFG/STG guider."
                }),
                "sampler": ("SAMPLER", {
                    "tooltip": "采样算法."
                }),
                "sigmas": ("SIGMAS", {
                    "tooltip": "H3 噪声调度."
                }),
                "latent_image": ("LATENT", {
                    "tooltip": "H3 video latent. 期望 5D [B,24,T,H/16,W/16]."
                }),
            },
            "optional": {
                "bypass_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True: 单次采样, 完全等同 SamplerCustomAdvanced. "
                               "用于对比和调试."
                }),
                "tile_mode": (["auto", "manual"], {
                    "default": "auto",
                    "tooltip": "auto: 按可用显存自适应分块; manual: 手动设 h_tiles/v_tiles."
                }),
                "h_tiles": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "manual 模式下的横向 (W 轴) 分块数. auto 模式为上限."
                }),
                "v_tiles": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "manual 模式下的纵向 (H 轴) 分块数. auto 模式为上限."
                }),
                "tile_overlap": ("INT", {
                    "default": 8, "min": 0, "max": 32, "step": 1,
                    "tooltip": "相邻块在 latent 域的重叠 token 数 (两个轴通用)."
                }),
                "max_size_for_no_tile": ("INT", {
                    "default": 24, "min": 8, "max": 256, "step": 1,
                    "tooltip": "目标轴大小 <= 此值时自动 bypass."
                }),
                "vram_budget_frac": ("FLOAT", {
                    "default": 0.30, "min": 0.05, "max": 0.90, "step": 0.01,
                    "tooltip": "auto 模式下, 单块去噪可使用的可用显存比例 (保守取值). "
                               "越高块越大越快但越易爆显存."
                }),
                "target_frames": ("INT", {
                    "default": 17, "min": 1, "max": 512, "step": 1,
                    "tooltip": "最小帧数保护 (非截断目标). 输入 latent 时长达标时完整保留; "
                               "不足时按 frame_padding_mode 补齐."
                }),
                "frame_padding_mode": (["replicate_last", "zero", "error"], {
                    "default": "replicate_last",
                    "tooltip": "帧数不足时的填充方式."
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "打印每个 tile 的 shape / value range."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample_tiled"
    CATEGORY = "10S Nodes/Sampling"
    DESCRIPTION = (
        "H3 视频模型专属 2D 分块采样 (LTX 2.3 式). 沿 H×W 分块, 每块独立采样后 "
        "可分离余弦窗口融合. 按可用显存自适应分块, 全程显存计算, 复用全局噪声与条件, "
        "最大限度保住提示词. 音频 passthrough."
    )

    def sample_tiled(self, noise, guider, sampler, sigmas, latent_image,
                     bypass_tiling=False,
                     tile_mode="auto", h_tiles=2, v_tiles=2, tile_overlap=8,
                     max_size_for_no_tile=24, vram_budget_frac=0.30,
                     target_frames=17, frame_padding_mode="replicate_last",
                     debug=False):

        latent = latent_image.copy()
        raw_samples = latent["samples"]

        if debug:
            print(f"\u2192 [H3] TiledSampler: input type="
                  f"{type(raw_samples).__name__} bypass={bypass_tiling}")

        # 1. 提取 video (5D) 和 audio
        try:
            video_tensor, audio_tensor, fmt_info = _h3_extract(raw_samples, debug)
        except TypeError as e:
            print(f"\u2192 [H3] TiledSampler: extract failed: {e}")
            raise

        if video_tensor.dim() != 5:
            raise ValueError(
                f"H3: video latent 必须是 5D [B,C,T,H,W], "
                f"实际 {video_tensor.dim()}D shape={tuple(video_tensor.shape)}"
            )

        if video_tensor.shape[1] != H3_LATENT_CHANNELS:
            print(f"\u2192 [H3] \u26a0  video latent 通道数 "
                  f"{video_tensor.shape[1]} != 预期 {H3_LATENT_CHANNELS}. "
                  f"继续采样但结果可能异常.")

        # 2. 帧数调整
        video_tensor = _adjust_frame_count(
            video_tensor, target_frames, frame_padding_mode, debug
        )

        B, C, F, H, W = video_tensor.shape
        latent["samples"] = video_tensor  # 暂时只放 video, 重建时再放回 audio

        # 3. bypass 路径 (进入采样全程显存)
        if bypass_tiling:
            if debug:
                print(f"  \u00b7 bypass: 单次采样 (shape={tuple(video_tensor.shape)})")
            return self._single_pass(
                noise, guider, sampler, sigmas, latent,
                video_tensor, audio_tensor, fmt_info, debug
            )

        # 4. auto-bypass: 两个轴都小或显存预算足够整幅
        do_tile_h = H > max_size_for_no_tile
        do_tile_w = W > max_size_for_no_tile
        if not do_tile_h and not do_tile_w:
            if debug:
                print(f"  \u00b7 auto-bypass (H,W) <= max_size_for_no_tile")
            return self._single_pass(
                noise, guider, sampler, sigmas, latent,
                video_tensor, audio_tensor, fmt_info, debug
            )

        # 5. 计算分块布局
        if tile_mode == "manual":
            n_h = max(1, min(h_tiles, 8)) if do_tile_h else 1
            n_v = max(1, min(v_tiles, 8)) if do_tile_w else 1
            planar_budget, budget = None, None
        else:
            n_h, n_v, planar_budget, budget = _auto_split(
                H, W, F, tile_overlap, 8, vram_budget_frac, debug
            )

        if n_h == 1 and n_v == 1:
            if debug:
                print(f"  \u00b7 auto-bypass (单块即可容纳)")
            return self._single_pass(
                noise, guider, sampler, sigmas, latent,
                video_tensor, audio_tensor, fmt_info, debug
            )

        h_spans = _tile_spans(H, n_h, tile_overlap)
        w_spans = _tile_spans(W, n_v, tile_overlap)
        if debug:
            print(f"  \u00b7 layout: axis=HxW {H}x{W} tiles={n_h}x{n_v} "
                  f"overlap={tile_overlap} h_spans={h_spans} w_spans={w_spans}")

        # 6. 全程显存: 把 video/noise/累加器都留在 GPU, 结束才回中间设备
        device = model_management.get_torch_device()
        dtype = video_tensor.dtype

        video_tensor = video_tensor.to(device=device)
        full_noise = noise.generate_noise({"samples": video_tensor}).to(device=device)
        if debug:
            print(f"  \u00b7 noise shape={tuple(full_noise.shape)} on {device}")

        # 7. 累加器 (fp32 防累积误差) + 权重掩码
        output = torch.zeros(B, C, F, H, W, dtype=torch.float32, device=device)
        weights = torch.zeros(B, C, F, H, W, dtype=torch.float32, device=device)
        denoised_output = torch.zeros_like(output)
        denoised_present = False

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # 清理 minimax_payload 中的 layout 缓存和 cond_video_latents
        # (解决两阶段高清化时分辨率变化导致的形状不匹配)
        H3TiledSampler._clean_minimax_layout(guider, debug)

        for h_idx, (hs, he) in enumerate(h_spans):
            for w_idx, (ws, we) in enumerate(w_spans):
                tile_h = he - hs
                tile_w = we - ws

                tile_latent = video_tensor[:, :, :, hs:he, ws:we].contiguous()
                tile_noise = full_noise[:, :, :, hs:he, ws:we].contiguous()

                if debug:
                    print(f"  \u00b7 tile[{h_idx},{w_idx}] "
                          f"h=[{hs}:{he}) w=[{ws}:{we}) "
                          f"shape={tuple(tile_latent.shape)}")

                # 全局条件不变 (guider.raw_conds), 只在 latent 上切块 -> 保住提示词.
                # 但每块用自己位置的噪声, 并复用全局噪声内容, 保证边界连续性.
                x0_output = {}
                callback = latent_preview.prepare_callback(
                    guider.model_patcher, sigmas.shape[-1] - 1, x0_output
                )

                tile_samples = guider.sample(
                    tile_noise, tile_latent, sampler, sigmas,
                    denoise_mask=None,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=noise.seed,
                ).to(device=device)

                if debug and isinstance(tile_samples, torch.Tensor):
                    print(f"    sampled: shape={tuple(tile_samples.shape)} "
                          f"range=[{tile_samples.min().item():.3f},"
                          f"{tile_samples.max().item():.3f}]")

                # 8. 可分离 2D 余弦窗口 (H 轴 ± W 轴外积), 消除接缝
                #     渐变长度固定用设定的 tile_overlap (避免每块额外算实际重叠).
                win_h = _make_window_1d(
                    tile_h,
                    tile_overlap if h_idx > 0 else 0,
                    tile_overlap if h_idx < len(h_spans) - 1 else 0,
                    torch.float32, device,
                )
                win_w = _make_window_1d(
                    tile_w,
                    tile_overlap if w_idx > 0 else 0,
                    tile_overlap if w_idx < len(w_spans) - 1 else 0,
                    torch.float32, device,
                )
                window = win_h.view(1, 1, 1, -1, 1) * win_w.view(1, 1, 1, 1, -1)

                tf = tile_samples.float()
                output[:, :, :, hs:he, ws:we] += tf * window
                weights[:, :, :, hs:he, ws:we] += window

                # 9. denoised output (x0 预测)
                try:
                    if hasattr(guider.model_patcher.model, "process_latent_out") \
                            and x0_output.get("x0") is not None:
                        model = guider.model_patcher.model
                        x0_proc = model.process_latent_out(x0_output["x0"])
                        if isinstance(x0_proc, torch.Tensor) \
                                and x0_proc.shape == tile_samples.shape:
                            denoised_present = True
                            denoised_output[:, :, :, hs:he, ws:we] += \
                                x0_proc.float().to(device=device) * window
                except Exception as e:
                    if debug:
                        print(f"    \u26a0  x0 处理失败: {type(e).__name__}: {e}")

                # 每块算完即释放, 不在此处 empty_cache (会强制同步变慢)
                del tile_samples, tf, window, win_h, win_w, tile_latent, tile_noise

        del full_noise

        # 10. 权重归一化
        wmin = weights.min().item()
        wmax = weights.max().item()
        if debug:
            print(f"  \u00b7 final weights: min={wmin:.4f} max={wmax:.4f}")
        if wmin < 1e-3:
            print(f"\u2192 [H3] \u26a0  weight min={wmin:.4f} 太小, "
                  f"建议增大 tile_overlap.")
        if wmax > 1.05:
            print(f"\u2192 [H3] \u26a0  weight max={wmax:.4f} > 1.05, "
                  f"cosine 渐变异常.")

        output = output / weights.clamp(min=1e-8)
        if denoised_present:
            denoised_output = denoised_output / weights.clamp(min=1e-8)
        del weights

        # 11. 回到中间设备 (仅在结束时统一清理显存, 不打断循环)
        intermediate_device = model_management.intermediate_device()
        output = output.to(dtype=dtype, device=intermediate_device)
        if denoised_present:
            denoised_output_final = denoised_output.to(
                dtype=dtype, device=intermediate_device
            )
            del denoised_output
        else:
            denoised_output_final = output
        model_management.soft_empty_cache()

        if debug:
            print(f"\u2192 [H3] final output: shape={tuple(output.shape)} "
                  f"dtype={output.dtype}")

        # 12. 重建 (恢复输入格式, 放回 audio)
        reconstructed = _h3_reconstruct(output, audio_tensor, fmt_info, debug)
        denoised_reconstructed = _h3_reconstruct(
            denoised_output_final, audio_tensor, fmt_info, debug
        )

        out_dict = latent.copy()
        out_dict["samples"] = reconstructed
        out_denoised_dict = latent.copy()
        out_denoised_dict["samples"] = denoised_reconstructed

        return (out_dict, out_denoised_dict)

    @staticmethod
    def _clean_minimax_layout(guider, debug=False):
        """清理 minimax_payload 中的 layout 缓存和 cond_video_latents.

        解决两阶段高清化流程中, 第二遍采样时 latent 分辨率变化导致:
          - cond_video_latents 分辨率不匹配
          - layout 缓存了旧分辨率的 PackedLayout
        """
        if hasattr(guider, 'original_conds'):
            for cond_key, cond_list in guider.original_conds.items():
                for cond in cond_list:
                    if isinstance(cond, dict) and 'minimax_refs' in cond:
                        if debug:
                            print(f"  · [H3] 清理 minimax_refs [{cond_key}] "
                                  f"({len(cond['minimax_refs'])} 个 ref)")
                        del cond['minimax_refs']

        if hasattr(guider, 'model_patcher') and hasattr(guider.model_patcher, 'model'):
            model = guider.model_patcher.model
            if hasattr(model, '_cached_extra_conds'):
                cached = model._cached_extra_conds
                if isinstance(cached, dict):
                    for k, v in cached.items():
                        if hasattr(v, 'cond') and isinstance(v.cond, dict):
                            if 'layout' in v.cond:
                                if debug:
                                    print(f"  · [H3] 清理已缓存的 layout")
                                del v.cond['layout']
                            if 'cond_video_latents' in v.cond:
                                if debug:
                                    print(f"  · [H3] 清理已缓存的 cond_video_latents")
                                del v.cond['cond_video_latents']

    @staticmethod
    def _single_pass(noise, guider, sampler, sigmas, latent_dict,
                     video_tensor, audio_tensor, fmt_info, debug=False):
        """单次采样 (bypass 或自动跳过分块时使用).

        ⚠ 必须传入 NestedTensor 格式 (video+audio), 否则 EasyCache / unpack_latents
        等中间件会把 plain 5D tensor 当成 list 处理, 导致 IndexError.
        """
        H3TiledSampler._clean_minimax_layout(guider, debug)

        latent_for_sample = _h3_reconstruct(video_tensor, audio_tensor, fmt_info, debug)
        latent_dict["samples"] = latent_for_sample

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = guider.sample(
            noise.generate_noise(latent_dict),
            latent_for_sample,
            sampler,
            sigmas,
            denoise_mask=None,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        samples = samples.to(model_management.intermediate_device())

        out = latent_dict.copy()
        out["samples"] = samples

        out_denoised = out.copy()
        try:
            model = guider.model_patcher.model
            if hasattr(model, "process_latent_out") and "x0" in x0_output and x0_output["x0"] is not None:
                x0_proc = model.process_latent_out(x0_output["x0"])
                if isinstance(x0_proc, torch.Tensor) and x0_proc.shape == samples.shape:
                    out_denoised["samples"] = x0_proc
        except Exception:
            pass

        return (out, out_denoised)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "H3TiledSampler": H3TiledSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TiledSampler": "H3 Tiled Sampler",
}
