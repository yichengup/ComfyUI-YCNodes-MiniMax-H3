"""
H3 Prompt Relay —— MiniMax H3 时间分段提示词控制

H3 使用打包自注意力（packed self-attention，text + cond + audio + video 在
同一序列中走同一个 self-attention），不存在独立 cross-attention。因此通过对
自注意力矩阵中 video query → text key 的路径施加时间惩罚 mask 来实现
"不同时间段只关注对应 prompt" 的效果。

原理：
  1. 接收官方节点（MiniMaxH3ImageToVideo 等）输出的 CONDITIONING
  2. 按 | 分隔的多段 prompt 分别 tokenize，记录每段 token 在文本中的起止位置
  3. 自动检测 CONDITIONING 中是否有 vision token（首尾帧），自动偏移 token 区间
  4. 每段 prompt 分配一个时间区间（帧数），计算中点、窗口半径、sigma 参数
  5. 在 H3 的 DiTBlock 自注意力中，对 video token 查询 text token 的路径
     施加高斯惩罚：距离段中点越远惩罚越大，段内基本无惩罚
  6. 其他注意力路径（text↔text、video↔video、audio↔*）不受影响

用法：
  CLIP → [MiniMaxH3ImageToVideo] → CONDITIONING ─┐
  CLIP ───────────────────────────────────────────┤
  latent ───────────────────────────────────────→ [H3PromptRelay] → MODEL → [采样器]
                                                  ↑                   CONDITIONING
                                                  └─── (透传) ───────────┘

  local_prompts 直接把官方 prompt 原文复制过来，在需要分段的位置加 | 即可。
  例如官方 prompt 是 "猫在草地上追逐蝴蝶"，想要 3 段：
  local_prompts = "猫在草地上 | 追逐蝴蝶 | 蝴蝶停落"

节点输出 patched MODEL + 透传的 CONDITIONING
"""

import logging
import math
import types

import torch

import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention

_LOG = logging.getLogger("h3_prompt_relay")

# H3 帧率映射常量（与 comfy/ldm/minimax/model.py 一致）
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


# ═════════════════════════════════════════════════════════════════════════════
# 像素帧 ↔ 潜在帧 转换
# ═════════════════════════════════════════════════════════════════════════════

def _pixel_frames_from_latent_t(latent_t):
    """给定 latent 帧数，返回总共覆盖的像素帧数."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _distribute_pixel_segments(pixel_lengths, latent_t):
    """将像素帧段长度按比例分配到 latent 帧空间."""
    total_pixel = _pixel_frames_from_latent_t(latent_t)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    latent_lengths = []
    for pl in pixel_lengths:
        ll = max(1, round(pl * latent_t / total_pixel))
        latent_lengths.append(ll)

    diff = latent_t - sum(latent_lengths)
    idx = 0
    while diff > 0:
        latent_lengths[idx % len(latent_lengths)] += 1
        diff -= 1
        idx += 1
    while diff < 0:
        if latent_lengths[idx % len(latent_lengths)] > 1:
            latent_lengths[idx % len(latent_lengths)] -= 1
            diff += 1
        idx += 1

    return latent_lengths


# ═════════════════════════════════════════════════════════════════════════════
# Token 映射
# ═════════════════════════════════════════════════════════════════════════════

def _get_raw_tokenizer(clip):
    """从 ComfyUI CLIP 对象提取底层 HuggingFace tokenizer."""
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer
    raise RuntimeError(
        "H3PromptRelay: 无法从 CLIP 对象提取 tokenizer. "
        f"可用属性: {[a for a in dir(tokenizer_wrapper) if not a.startswith('_')]}"
    )


def _map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    """增量 tokenize，返回 (合并后 prompt, 每段 token 起止区间)."""
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0

    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt

    for i, plp in enumerate(prefixed_locals):
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"H3PromptRelay: 段 {i} prompt 未产生额外 token: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges


# ═════════════════════════════════════════════════════════════════════════════
# 段元数据构建
# ═════════════════════════════════════════════════════════════════════════════

def _build_segments(token_ranges, segment_lengths_latent, epsilon):
    """为每段 prompt 构建时间惩罚参数."""
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    q_token_idx = []
    frame_cursor = 0

    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths_latent):
        if L <= 0:
            continue
        midpoint = (2 * frame_cursor + L) / 2.0
        base_window = max(L / 2.0 - 2.0, 0.0)
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": base_window,
            "sigma": sigma,
            "strength": 1.0,
        })
        frame_cursor += L

    return q_token_idx


# ═════════════════════════════════════════════════════════════════════════════
# 每帧空间 token 数
# ═════════════════════════════════════════════════════════════════════════════

def _frame_rows(latent_h, latent_w):
    """每帧的空间 token 数 = (H/patch) * (W/patch)，patch = 2."""
    return (latent_h // 2) * (latent_w // 2)


# ═════════════════════════════════════════════════════════════════════════════
# 自注意力 mask 函数（动态布局）
# ═════════════════════════════════════════════════════════════════════════════

def _build_h3_mask_fn(q_token_idx, text_len, frame_rows, latent_t):
    """
    构建 H3 打包自注意力的时间惩罚 mask 函数。

    H3 打包序列中 video 始终在末尾，因此 video_start 和 video_end 可以
    从实际序列长度 Lq 动态推导，无需预计算布局。这确保适配带/不带 keyframe
    的所有场景。

    text_len 是 CONDITIONING 的总 token 数（包括 vision token），
    token_ranges 已经过偏移调整，与 CONDITIONING 中的位置对齐。

    q_token_idx 中每段 token 区间已自动跳过 global_prompt 的 token 列，
    因此 global_prompt 的 token 列始终为 0（不惩罚），全程对所有帧可见。

    返回的 mask_fn 签名为 (Lq, Lk, dtype, device, transformer_options) -> mask or None.
    mask 形状 [Lq, Lk]，被加到注意力分数上（负值 = 惩罚）。
    """

    def mask_fn(Lq, Lk, dtype, device, transformer_options):
        if Lq != Lk:
            return None

        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            return None

        if Lq <= text_len:
            return None

        video_end = Lq
        video_start = Lq - latent_t * frame_rows

        if video_start < 0:
            return None

        n_video = video_end - video_start
        if n_video <= 0:
            return None

        video_local = torch.arange(n_video, device=device)
        frame_indices = video_local.float() / frame_rows

        penalty = torch.zeros(n_video, text_len, dtype=torch.float32, device=device)

        for seg in q_token_idx:
            local = seg["local_token_idx"].to(device=device)
            d = (frame_indices[:, None] - seg["midpoint"]).abs()
            cost = seg["strength"] * (torch.relu(d - seg["window"]) ** 2) / (2.0 * seg["sigma"] ** 2)
            penalty[:, local] = cost.to(torch.float32)

        mask = torch.zeros(Lq, Lk, dtype=dtype, device=device)
        mask[video_start:video_end, :text_len] = -penalty.to(dtype)

        return mask

    return mask_fn


# ═════════════════════════════════════════════════════════════════════════════
# 模型 patch
# ═════════════════════════════════════════════════════════════════════════════

def _patched_attn_forward(self, x, rope_freqs=None, transformer_options={}):
    """
    H3 Attention.forward 的 patch 版本。

    与原版唯一区别：从 transformer_options 读取 h3_prompt_relay_mask_fn，
    调用它生成时间惩罚 mask，传给 optimized_attention。
    """
    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)

    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    mask_fn = transformer_options.get("h3_prompt_relay_mask_fn")
    mask = None
    if mask_fn is not None:
        mask = mask_fn(s, s, q.dtype, q.device, transformer_options)

    out = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True,
                              transformer_options=transformer_options)
    return self.out_proj(out.squeeze(0))


def _apply_h3_patches(model_clone):
    """对 H3 模型的每个 DiTBlock.attn.forward 打 patch."""
    diff_model = model_clone.get_model_object("diffusion_model")

    if not hasattr(diff_model, "blocks"):
        raise RuntimeError("H3PromptRelay: 模型没有 blocks 属性，不是 H3 模型")

    for i, block in enumerate(diff_model.blocks):
        attn = block.attn
        key = f"diffusion_model.blocks.{i}.attn.forward"
        model_clone.add_object_patch(
            key,
            types.MethodType(_patched_attn_forward, attn)
        )

    _LOG.info("H3PromptRelay: 已 patch %d 个 DiTBlock 的 Attention 层", len(diff_model.blocks))


# ═════════════════════════════════════════════════════════════════════════════
# 节点
# ═════════════════════════════════════════════════════════════════════════════

class H3PromptRelay:
    """
    H3 时间分段提示词控制节点。

    用法：
      官方节点 [MiniMaxH3ImageToVideo] → conditioning 接本节点 conditioning 输入
      CLIP 同时接入官方节点和本节点
      local_prompts = 官方 prompt 原文，在分段处加 | 分隔符
      本节点输出 MODEL 接采样器，CONDITIONING 透传官方节点的 conditioning
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "H3 模型 (MiniMax H3)."
                }),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "来自官方节点 (MiniMaxH3ImageToVideo) 的 conditioning."
                }),
                "clip": ("CLIP", {
                    "tooltip": "H3 CLIP 模型 (Qwen3-VL). 与官方节点使用同一个 CLIP."
                }),
                "latent": ("LATENT", {
                    "tooltip": "H3 视频 latent. 5D [B,24,T,H,W] 或 NestedTensor."
                }),
                "global_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "全局风格/主题基调（可选，300字以内，全程对所有帧可见，不分段）",
                    "tooltip": "可选的全局提示词。用于定义视频的整体风格、画质、色调等基础设定。\n\n这段文本的 token 全程对所有帧可见，不会被时间分段 mask 惩罚。\n留空则只用 local_prompts 的分段内容。"
                }),
                "local_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "官方 prompt 原文，在分段处加 | 分隔符",
                    "tooltip": "时间分段提示词。\n\n直接将官方节点的 prompt 原文复制过来，在想分段的位置加上 | 分隔符即可。\n\n例如官方 prompt 是「史诗奇幻战斗视频，巨龙在海中怒吼，战士腾空迎战」，想分 3 段：\n  「史诗奇幻战斗视频，巨龙在海中怒吼 | 战士腾空迎战 | 风暴平息后日出」\n\n每段内容对应一个时间段，段数需与 segment_lengths 一致。"
                }),
                "segment_lengths": ("STRING", {
                    "default": "",
                    "placeholder": "例如: 49, 49, 49 或留空自动均分",
                    "tooltip": "逗号分隔的像素帧数，每段长度。留空则自动均分。"
                }),
                "epsilon": ("FLOAT", {
                    "default": 0.001, "min": 1e-6, "max": 0.99, "step": 0.0001,
                    "tooltip": "惩罚衰减参数。越小边界越锐利（默认 0.001）。增大可柔化段间过渡。"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING",)
    RETURN_NAMES = ("model", "positive",)
    FUNCTION = "apply"
    CATEGORY = "YCNodes-MiniMax-H3/conditioning"
    DESCRIPTION = (
        "H3 时间分段提示词控制。\n"
        "接收官方节点 (MiniMaxH3ImageToVideo) 的 conditioning，\n"
        "在 H3 打包自注意力中注入时间惩罚 mask，\n"
        "透传 conditioning 给采样器。\n\n"
        "global_prompt（可选）：风格/主题基调，全程对所有帧可见。\n"
        "local_prompts：官方 prompt 原文，在分段处加 | 分隔符。"
    )

    def apply(self, model, conditioning, clip, latent, global_prompt, local_prompts,
              segment_lengths, epsilon):
        # ── 1. 解析本地 prompt ──
        locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
        if not locals_list:
            raise ValueError("H3PromptRelay: 至少需要一段本地 prompt（用 | 分隔）")

        # ── 2. 从 conditioning 提取实际的总 token 数 ──
        if not isinstance(conditioning, list) or len(conditioning) == 0:
            raise ValueError("H3PromptRelay: conditioning 格式无效")
        cond_text_len = int(conditioning[0][0].shape[1])  # [1, L, dim]

        _LOG.info("H3PromptRelay: conditioning text_len=%d", cond_text_len)

        # ── 3. Tokenize 纯文本 prompt 获取 token 区间 ──
        raw_tokenizer = _get_raw_tokenizer(clip)
        global_base = global_prompt.strip() if global_prompt else ""
        full_prompt, token_ranges = _map_token_indices(
            raw_tokenizer, global_base, locals_list
        )

        my_text_len = token_ranges[-1][1] if token_ranges else 0
        shift = cond_text_len - my_text_len

        _LOG.info("H3PromptRelay: 全局提示 tokens=%d, 分段 tokens=%d, conditioning tokens=%d, shift=%d",
                  len(raw_tokenizer(global_base)["input_ids"]) - (1 if getattr(raw_tokenizer, "add_eos", False) else 0),
                  my_text_len, cond_text_len, shift)

        if shift < 0:
            raise ValueError(
                "H3PromptRelay: conditioning 的 token 数 (%d) 小于本地 prompt 总 token 数 (%d)。"
                "请减少 local_prompts 总字数，或减少分段数。" % (cond_text_len, my_text_len)
            )

        # 偏移 token 区间以对齐 conditioning 中的实际位置
        if shift > 0:
            _LOG.info("H3PromptRelay: 检测到 vision token (首尾帧), 偏移所有 token 区间 +%d", shift)
            token_ranges = [(s + shift, e + shift) for s, e in token_ranges]

        for i, (s, e) in enumerate(token_ranges):
            _LOG.info("H3PromptRelay: 段 %d 调整后 tokens=[%d:%d] (%d tokens)", i, s, e, e - s)

        # ── 4. 从 latent 提取视频维度 ──
        samples = latent["samples"]
        video = None
        if hasattr(samples, "is_nested") and samples.is_nested:
            for p in samples.unbind():
                if isinstance(p, torch.Tensor) and p.dim() == 5:
                    video = p
                    break
        elif isinstance(samples, (tuple, list)):
            for p in samples:
                if isinstance(p, torch.Tensor) and p.dim() == 5:
                    video = p
                    break
        elif isinstance(samples, torch.Tensor) and samples.dim() == 5:
            video = samples

        if video is None:
            raise ValueError("H3PromptRelay: 无法从 latent 提取 5D video tensor")

        latent_t = video.shape[2]
        latent_h = video.shape[3]
        latent_w = video.shape[4]

        _LOG.info("H3PromptRelay: latent_t=%d latent_h=%d latent_w=%d",
                  latent_t, latent_h, latent_w)

        # ── 5. 分配段长度 ──
        if segment_lengths.strip():
            pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
            n_seg = len(locals_list)
            if len(pixel_lengths) < n_seg:
                avg = sum(pixel_lengths) / max(len(pixel_lengths), 1)
                _LOG.warning(
                    "H3PromptRelay: segment_lengths (%d) 少于 local_prompts (%d) 段，"
                    "自动补齐剩余段为 %.0f 像素帧",
                    len(pixel_lengths), n_seg, avg
                )
                pixel_lengths += [int(avg)] * (n_seg - len(pixel_lengths))
            elif len(pixel_lengths) > n_seg:
                _LOG.warning(
                    "H3PromptRelay: segment_lengths (%d) 多于 local_prompts (%d) 段，"
                    "自动截断多余段",
                    len(pixel_lengths), n_seg
                )
                pixel_lengths = pixel_lengths[:n_seg]
        else:
            total_pixel = _pixel_frames_from_latent_t(latent_t)
            step = max(1, total_pixel // len(locals_list))
            pixel_lengths = [step] * len(locals_list)

        latent_lengths = _distribute_pixel_segments(pixel_lengths, latent_t)
        _LOG.info("H3PromptRelay: pixel_lengths=%s  latent_lengths=%s",
                  pixel_lengths, latent_lengths)

        # ── 6. 构建段元数据 & mask 函数 ──
        frame_r = _frame_rows(latent_h, latent_w)
        q_token_idx = _build_segments(token_ranges, latent_lengths, epsilon)
        mask_fn = _build_h3_mask_fn(q_token_idx, cond_text_len, frame_r, latent_t)

        # ── 7. 打 patch ──
        model_clone = model.clone()
        _apply_h3_patches(model_clone)

        to = model_clone.model_options.setdefault("transformer_options", {})
        to["h3_prompt_relay_mask_fn"] = mask_fn

        # ── 8. 透传 conditioning ──
        return (model_clone, conditioning,)


NODE_CLASS_MAPPINGS = {
    "H3PromptRelay": H3PromptRelay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptRelay": "H3 Prompt Relay",
}