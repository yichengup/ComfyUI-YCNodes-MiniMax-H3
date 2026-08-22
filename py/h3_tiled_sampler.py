"""
H3 Tiled Sampler
作者：亦诚 改造成H3节点，原始节点来自https://github.com/TenStrip/10S-Comfy-nodes
H3 视频模型专属分块采样节点.
空间分块 (H/W 轴) + cosine 融合, 严格保持 H3 视频/音频独立模态.

与 LTX Tiled Sampler 的关键差异:
  - 视频+音频是 (tensor, tensor) tuple, 不是 LTX NestedTensor wrapper
  - 没有 combined flat tensor, 因此不需要 _unflatten_ltx_combined
  - 时长由 VAE 编码决定, 入口只做最小帧数保护, 绝不截断
  - 不假设 model 有 process_latent_out, 用 hasattr 探测
  - 音频仅 passthrough, 不做 tile_carrying

适用场景: H3 768p/2K 上采样精修时 token 数过多导致 attention 性能下降,
        通过空间分块让每块在 H3 DiT 训练分布的 token 数内工作.

不适用: 重去噪 (从纯噪声起步). 空间分块会破坏全局一致性.
"""

import torch
import comfy.sample
import comfy.utils
import comfy.model_management
from comfy.nested_tensor import NestedTensor
import latent_preview


# H3 视频 VAE 训练约束
H3_VIDEO_FRAMES = 17       # 输入视频帧数硬约束
H3_LATENT_CHANNELS = 24    # H3 video latent 通道数
H3_LATENT_TIME = 5         # 17 帧 → 5 latent frame (vae_ratio_t=4 + token_drop=3)


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

    只保证 T 维不低于目标长度 (target_frames 对应的最小 latent 帧数).
    T 已达标时原样返回, 绝不截断 —— 采样器不改视频时长,
    时长由上游 VAE 编码决定 (如 5s 视频 latent T≈37 会完整保留).
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
# 分块数学 (与模型无关, 复用 LTX sampler 的成熟实现)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_tile_starts(total, n_tiles, overlap):
    """
    计算每个 tile 在 axis 上的 start 位置, 保证相邻 tile 有 overlap 区域.

    算法:
      - 每个 tile 的有效覆盖区域 (不含 overlap) = total / n_tiles (均匀分配)
      - 实际 tile 大小 = 有效 + 两侧 overlap (边缘 tile 只加一侧)
      - start 位置 = i * stride - overlap (边缘 clamp 到 0)
      - 返回 (starts, tile_size) 其中 tile_size 是含 overlap 的完整 tile 大小
    """
    if n_tiles <= 1:
        return [0], total

    stride = total / n_tiles  # 每个 tile 的有效步长 (float)

    starts = []
    for i in range(n_tiles):
        start = int(i * stride - overlap)
        start = max(0, start)
        starts.append(start)

    # 去重 (相邻 start 可能因 clamp 到 0 而重合)
    dedup = []
    for s in starts:
        if not dedup or s > dedup[-1]:
            dedup.append(s)
    starts = dedup

    # tile_size: 最大 tile 的覆盖范围 (含 overlap)
    # 中间 tile: stride + 2*overlap, 边缘 tile: stride + overlap
    tile_size = int(stride) + 2 * overlap + 1  # +1 向上取整安全余量

    return starts, tile_size


def _make_window_1d(length, ov_left, ov_right, dtype, device):
    """
    1D cosine 窗口: 中间全 1, 两侧 overlap 区做 (1+cos)/2 渐变.
    ov_left: 左侧 overlap token 数
    ov_right: 右侧 overlap token 数
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
    H3 视频模型分块采样节点.

    空间分块沿 H 或 W 切, 每块独立采样, cosine 窗口融合.
    严格保持输入 latent 格式 (单 tensor / tuple / list).
    音频 passthrough, 不参与采样.

    使用方法:
      1. 接入 H3 对应的 noise / guider / sampler / sigmas / latent
      2. 调节 tile_axis / n_tiles / tile_overlap
      3. 首次使用开 debug=True 验证
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
                "tile_axis": (["auto", "H", "W"], {
                    "default": "auto",
                    "tooltip": "沿哪条空间轴切. auto 取较长轴."
                }),
                "n_tiles": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "分块数. 1 等同 bypass."
                }),
                "tile_overlap": ("INT", {
                    "default": 8, "min": 0, "max": 32, "step": 1,
                    "tooltip": "相邻块在 latent 域的重叠 token 数."
                }),
                "max_size_for_no_tile": ("INT", {
                    "default": 24, "min": 8, "max": 256, "step": 1,
                    "tooltip": "目标轴大小 <= 此值时自动 bypass."
                }),
                "target_frames": ("INT", {
                    "default": 17, "min": 1, "max": 512, "step": 1,
                    "tooltip": "最小帧数保护 (非截断目标). 输入 latent 时长达标时完整保留; "
                               "不足时按 frame_padding_mode 补齐. 5s 视频 latent T≈37 不会被砍."
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
    CATEGORY = "YCNodes-MiniMax-H3/Sampling"
    DESCRIPTION = (
        "H3 视频模型专属分块采样. 沿 H/W 空间分块, 每块独立采样后 "
        "cosine 融合. 完整保留输入 latent 时长 (绝不截断), 不足时补齐. "
        "音频 passthrough."
    )

    def sample_tiled(self, noise, guider, sampler, sigmas, latent_image,
                     bypass_tiling=False,
                     tile_axis="auto", n_tiles=2, tile_overlap=8,
                     max_size_for_no_tile=24, target_frames=17,
                     frame_padding_mode="replicate_last",
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

        # 3. bypass 路径
        if bypass_tiling:
            if debug:
                print(f"  \u00b7 bypass: 单次采样 (shape={tuple(video_tensor.shape)})")
            return self._single_pass(
                noise, guider, sampler, sigmas, latent,
                video_tensor, audio_tensor, fmt_info, debug
            )

        # 4. 选择 tile 轴
        if tile_axis == "auto":
            tile_axis = "H" if H >= W else "W"
        axis_size = H if tile_axis == "H" else W

        if axis_size <= max_size_for_no_tile or n_tiles <= 1:
            if debug:
                reason = ("axis_size \u2264 max" if axis_size <= max_size_for_no_tile
                          else f"n_tiles={n_tiles}")
                print(f"  \u00b7 auto-bypass ({reason})")
            return self._single_pass(
                noise, guider, sampler, sigmas, latent,
                video_tensor, audio_tensor, fmt_info, debug
            )

        # 5. 计算 tile 区间
        starts, tile_size = _compute_tile_starts(axis_size, n_tiles, tile_overlap)
        if debug:
            print(f"  \u00b7 axis={tile_axis} size={axis_size} "
                  f"n_tiles={n_tiles} overlap={tile_overlap} "
                  f"starts={starts} tile_size={tile_size}")

        # 6. 准备 device/dtype
        device = comfy.model_management.get_torch_device()
        dtype = video_tensor.dtype

        video_tensor = video_tensor.to(device=device)
        full_noise = noise.generate_noise({"samples": video_tensor}).to(device=device)

        if debug:
            print(f"  \u00b7 noise shape={tuple(full_noise.shape)} on {device}")

        # 7. 分块采样 + cosine 融合
        output = torch.zeros_like(video_tensor, dtype=torch.float32, device=device)
        weights_shape = (1, 1, 1,
                         H if tile_axis == "H" else 1,
                         W if tile_axis == "W" else 1)
        weights = torch.zeros(weights_shape, dtype=torch.float32, device=device)
        denoised_output = torch.zeros_like(output)
        denoised_present = False

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # 清理 minimax_payload 中的 layout 缓存和 cond_video_latents
        # (解决两阶段高清化时分辨率变化导致的形状不匹配)
        H3TiledSampler._clean_minimax_layout(guider, debug)

        for tile_idx, ax_start in enumerate(starts):
            if tile_axis == "H":
                ax_end = min(ax_start + tile_size, H)
                tile_latent = video_tensor[:, :, :, ax_start:ax_end, :].contiguous()
                tile_noise = full_noise[:, :, :, ax_start:ax_end, :].contiguous()
            else:
                ax_end = min(ax_start + tile_size, W)
                tile_latent = video_tensor[:, :, :, :, ax_start:ax_end].contiguous()
                tile_noise = full_noise[:, :, :, :, ax_start:ax_end].contiguous()

            actual_size = tile_latent.shape[3 if tile_axis == "H" else 4]

            if debug:
                print(f"  \u00b7 tile {tile_idx+1}/{len(starts)}: "
                      f"range=[{ax_start},{ax_end}) "
                      f"shape={tuple(tile_latent.shape)}")

            # 准备 x0 捕获
            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output
            )

            # 采样
            tile_samples = guider.sample(
                tile_noise, tile_latent, sampler, sigmas,
                denoise_mask=None,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise.seed,
            )

            tile_samples = tile_samples.to(device=device)

            if debug and isinstance(tile_samples, torch.Tensor):
                v_min = tile_samples.min().item()
                v_max = tile_samples.max().item()
                v_mean = tile_samples.mean().item()
                print(f"    sampled: shape={tuple(tile_samples.shape)} "
                      f"range=[{v_min:.3f},{v_max:.3f}] mean={v_mean:.3f}")

            # 计算 overlap 和窗口
            has_prev = tile_idx > 0
            has_next = tile_idx < len(starts) - 1
            ov_left = 0
            ov_right = 0
            if has_prev:
                prev_end = starts[tile_idx - 1] + tile_size
                ov_left = max(0, min(prev_end, ax_end) - ax_start)
            if has_next:
                next_start = starts[tile_idx + 1]
                ov_right = max(0, ax_end - max(ax_start, next_start))

            window_1d = _make_window_1d(
                actual_size, ov_left, ov_right, torch.float32, device
            )
            if tile_axis == "H":
                window = window_1d.view(1, 1, 1, -1, 1)
                output[:, :, :, ax_start:ax_end, :] += tile_samples.float() * window
                weights[:, :, :, ax_start:ax_end, :] += window
            else:
                window = window_1d.view(1, 1, 1, 1, -1)
                output[:, :, :, :, ax_start:ax_end] += tile_samples.float() * window
                weights[:, :, :, :, ax_start:ax_end] += window

            # x0 预测 (denoised output)
            try:
                # 探测 model 是否有 process_latent_out
                model = guider.model_patcher.model
                if hasattr(model, "process_latent_out") and "x0" in x0_output and x0_output["x0"] is not None:
                    x0_proc = model.process_latent_out(x0_output["x0"])
                    if isinstance(x0_proc, torch.Tensor) and x0_proc.shape == tile_samples.shape:
                        denoised_present = True
                        x0_proc = x0_proc.to(device=device)
                        if tile_axis == "H":
                            denoised_output[:, :, :, ax_start:ax_end, :] += \
                                x0_proc.float() * window
                        else:
                            denoised_output[:, :, :, :, ax_start:ax_end] += \
                                x0_proc.float() * window
            except Exception as e:
                if debug:
                    print(f"    \u26a0  x0 处理失败: {type(e).__name__}: {e}")

            if debug:
                print(f"    fades: left={ov_left} right={ov_right} "
                      f"weight_acc min={weights.min().item():.3f}")

            # 内存清理
            del tile_samples, tile_latent, tile_noise, window, window_1d
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del full_noise

        # 8. 权重归一化
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

        # 9. 回到中间设备
        intermediate_device = comfy.model_management.intermediate_device()
        output = output.to(dtype=dtype, device=intermediate_device)
        if denoised_present:
            denoised_output_final = denoised_output.to(
                dtype=dtype, device=intermediate_device
            )
            del denoised_output
        else:
            denoised_output_final = output
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if debug:
            print(f"\u2192 [H3] final output: shape={tuple(output.shape)} "
                  f"dtype={output.dtype}")

        # 10. 重建 (恢复输入格式, 放回 audio)
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
          - cond_video_latents 分辨率不匹配 (cond_video_rows 形状 vs img_update 形状)
          - layout 缓存了旧分辨率的 PackedLayout

        策略:
          1. 清理 guider.original_conds 中的 minimax_refs, 防止
             model.extra_conds() 把旧分辨率的 cond_video_latents 注入 payload
          2. 清理 model 上可能缓存的旧 layout (通过 extra_conds 闭包)
        """
        # 1. 清理 original_conds 中的 minimax_refs (主要修复)
        if hasattr(guider, 'original_conds'):
            for cond_key, cond_list in guider.original_conds.items():
                for cond in cond_list:
                    if isinstance(cond, dict) and 'minimax_refs' in cond:
                        if debug:
                            print(f"  · [H3] 清理 minimax_refs [{cond_key}] "
                                  f"({len(cond['minimax_refs'])} 个 ref, 防止分辨率不匹配)")
                        del cond['minimax_refs']

        # 2. 清理已缓存的 layout (辅助: 如果 model 的 extra_conds 已缓存了旧 layout)
        if hasattr(guider, 'model_patcher') and hasattr(guider.model_patcher, 'model'):
            model = guider.model_patcher.model
            if hasattr(model, 'diffusion_model'):
                dm = model.diffusion_model
                # H3 的 _forward 会把 layout 回写到 payload dict.
                # 如果 payload dict 被 CONDConstant 复用, 则需要清理.
                # 通过 extra_conds 的闭包无法直接访问 payload dict,
                # 但这里尝试通过 model.extra_conds 清理.
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
        # 清理 minimax_payload 中的 layout 缓存和 cond_video_latents
        # (解决两阶段高清化时分辨率变化导致的形状不匹配)
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
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent_dict.copy()
        out["samples"] = samples

        # denoised output
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
