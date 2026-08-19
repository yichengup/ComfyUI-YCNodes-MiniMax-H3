"""
H3 Distance Attention Patcher (v3 - Patch Size & Frame Count Fixed)

针对 MiniMax H3 视频模型全景场景下肢体和面部容易被背景同化或扯碎的瓶颈。
本节点采用时空高斯感受野遮罩（Gaussian Receptive Field Masking），强行约束模型在中前期去噪时的局部注意力。

[v3 修复]:
  1. 修复 original_shape 没有除以 patch_size [1,2,2] 导致网格坐标 H/W 扩大 2 倍，
     与 token 数不匹配后触发 1D 兜底，产生"满屏水泡/网格圆圈"的画面割裂 Bug。
  2. 新增 num_frames 参数，支持不同帧数视频，不再硬编码 T=5。
  3. 兜底逻辑从纯因数分解改为基于 num_frames + 原始分辨率反推，更可靠。
"""

import torch
import torch.nn.functional as F
import math

class H3DistanceAttentionPatcher:
    """
    通过高斯距离惩罚强制限制自注意力感受野，阻断大面积背景对微小手部/脸部的特征同化与扯碎。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "输入的 MiniMax H3 视频模型。"}),
                "receptive_field_scale": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 100.0, "step": 0.5,
                    "tooltip": "感受野尺度因子。数值越小，注意力越局限于局部（抗干扰越强）；数值越大越接近全局注意力。"
                }),
                "temporal_weight": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "时间轴相对于空间轴的距离权重。"
                }),
                "start_at_sigma": ("FLOAT", {
                    "default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "开始应用距离限制的 Sigma 阈值。建议 3.0 ~ 5.0（避开前几步的宏观构图期）。"
                }),
                "end_at_sigma": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "结束限制的 Sigma 阈值（默认 0.0 直至结束）。"
                }),
                "num_frames": ("INT", {
                    "default": 17, "min": 1, "max": 256, "step": 1,
                    "tooltip": "视频总帧数。H3 VAE 时间下采样 4x，潜空间帧数 = ceil(num_frames/4)。默认 17 帧 → 5 潜帧。"
                }),
                "original_width": ("INT", {
                    "default": 864, "min": 128, "max": 2048, "step": 16,
                    "tooltip": "生成视频的宽度。用于兜底解算（original_shape 可用时优先使用）。"
                }),
                "original_height": ("INT", {
                    "default": 480, "min": 128, "max": 2048, "step": 16,
                    "tooltip": "生成视频的高度。"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_attention"
    CATEGORY = "MiniMax-H3/注意力"
    DESCRIPTION = "通过高斯距离惩罚强制限制自注意力感受野，阻断大面积背景对微小手部/脸部的特征同化与扯碎。"

    def patch_attention(self, model, receptive_field_scale, temporal_weight, start_at_sigma, end_at_sigma, num_frames, original_width, original_height):
        patched_model = model.clone()

        # H3 架构常量
        VAE_TEMPORAL_DOWNSAMPLE = 4   # VAE 时间下采样倍数
        VAE_SPATIAL_DOWNSAMPLE = 16   # VAE 空间下采样倍数
        PATCH_SIZE_T = 1              # patch_size[0]
        PATCH_SIZE_H = 2              # patch_size[1]
        PATCH_SIZE_W = 2              # patch_size[2]

        # 由帧数推导潜空间帧数（向上取整）
        latent_frames = math.ceil(num_frames / VAE_TEMPORAL_DOWNSAMPLE)

        # 兜底：用分辨率推导潜空间 patch 网格尺寸
        latent_h = original_height // VAE_SPATIAL_DOWNSAMPLE
        latent_w = original_width // VAE_SPATIAL_DOWNSAMPLE
        fallback_patch_h = latent_h // PATCH_SIZE_H
        fallback_patch_w = latent_w // PATCH_SIZE_W

        def patched_attn1(q, k, v, extra_options):
            n_heads = extra_options.get("n_heads", 1)
            B, L, D = q.shape
            d_head = D // n_heads

            q_heads = q.view(B, L, n_heads, d_head).transpose(1, 2)
            k_heads = k.view(B, k.shape[1], n_heads, d_head).transpose(1, 2)
            v_heads = v.view(B, v.shape[1], n_heads, d_head).transpose(1, 2)

            current_sigma = extra_options.get("sigmas", [999.0])[0]

            if start_at_sigma >= current_sigma >= end_at_sigma:
                T, H_patch, W_patch = None, None, None

                # 【优先】从 ComfyUI 核心上下文抓取 original_shape
                transformer_options = extra_options.get("transformer_options", {})
                orig_shape = transformer_options.get("original_shape", None)
                if orig_shape is None:
                    orig_shape = extra_options.get("original_shape", None)

                if orig_shape is not None and len(orig_shape) == 5:
                    # orig_shape = (B, C, T_latent, H_latent, W_latent)
                    T = orig_shape[2]
                    # 关键修复：潜空间维度除以 patch_size 得到 token 网格尺寸
                    H_patch = orig_shape[3] // PATCH_SIZE_H
                    W_patch = orig_shape[4] // PATCH_SIZE_W

                # 验证：网格 token 数是否匹配
                if T is not None and H_patch is not None and W_patch is not None:
                    video_tokens = T * H_patch * W_patch
                    if video_tokens > L:
                        # 网格算出来比实际 token 多，不可信，走兜底
                        T, H_patch, W_patch = None, None, None

                # 【兜底】用 num_frames 和分辨率反推
                if T is None or H_patch is None or W_patch is None:
                    # 尝试从 L 和已知的帧数、空间分辨率反推
                    # L = video_tokens + extra_tokens，video_tokens ≈ latent_frames * fallback_patch_h * fallback_patch_w
                    video_tokens_est = latent_frames * fallback_patch_h * fallback_patch_w
                    if video_tokens_est <= L:
                        T = latent_frames
                        H_patch = fallback_patch_h
                        W_patch = fallback_patch_w
                    else:
                        # 最后兜底：因数分解
                        T = latent_frames
                        L_spat = L // T
                        found = False
                        for test_h in range(int(math.sqrt(L_spat)), 0, -1):
                            if L_spat % test_h == 0:
                                H_patch = test_h
                                W_patch = L_spat // test_h
                                if original_width > original_height and H_patch > W_patch:
                                    H_patch, W_patch = W_patch, H_patch
                                found = True
                                break
                        if not found:
                            H_patch = 1
                            W_patch = L_spat

                # 仅取视频 token 部分构造距离遮罩
                video_tokens = T * H_patch * W_patch
                video_tokens = min(video_tokens, L)

                # 构造 3D 时空坐标网格
                grid_t, grid_y, grid_x = torch.meshgrid(
                    torch.arange(T, dtype=torch.float32, device=q.device),
                    torch.arange(H_patch, dtype=torch.float32, device=q.device),
                    torch.arange(W_patch, dtype=torch.float32, device=q.device),
                    indexing="ij"
                )

                coords = torch.stack([
                    grid_t.flatten() * temporal_weight,
                    grid_y.flatten(),
                    grid_x.flatten()
                ], dim=-1)  # (video_tokens, 3)

                # 计算两两欧氏距离
                coords_sq = torch.sum(coords ** 2, dim=-1, keepdim=True)
                dist_matrix = coords_sq + coords_sq.T - 2 * torch.matmul(coords, coords.T)
                dist_matrix = torch.clamp(dist_matrix, min=0.0)
                dist_matrix = torch.sqrt(dist_matrix)  # (video_tokens, video_tokens)

                # 高斯距离惩罚遮罩
                mask_full = - (dist_matrix / receptive_field_scale).to(dtype=q.dtype, device=q.device)

                # 如果 L > video_tokens（有 extra tokens），扩展 mask
                if L > video_tokens:
                    # 填充：extra tokens 不受距离限制（填 0 = 无衰减）
                    extra_len = L - video_tokens
                    mask_full = F.pad(mask_full, (0, extra_len, 0, extra_len), value=0.0)

                out_heads = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, attn_mask=mask_full)
            else:
                out_heads = F.scaled_dot_product_attention(q_heads, k_heads, v_heads)

            out = out_heads.transpose(1, 2).contiguous().view(B, L, D)
            return out

        # 全量覆盖注册机制（H3 50 层 DiT）
        for i in range(24):
            patched_model.set_model_attn1_replace(patched_attn1, "input", i)
            patched_model.set_model_attn1_replace(patched_attn1, "output", i)
        patched_model.set_model_attn1_replace(patched_attn1, "middle", 0)

        for i in range(48):
            patched_model.set_model_attn1_replace(patched_attn1, "double_block", i)
            patched_model.set_model_attn1_replace(patched_attn1, "single_block", i)

        return (patched_model,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "H3DistanceAttentionPatcher": H3DistanceAttentionPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3DistanceAttentionPatcher": "H3 Distance Attention Patcher",
}