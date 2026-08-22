"""
MiniMaxH3ImageToVideoTail 节点 —— 二采（后段采样）版 Image-to-Video 条件节点。
作者：亦诚
与官方 MiniMaxH3ImageToVideo 的关系：
    · 官方节点总是输出一个"空 AV latent"（纯噪声起点），无法用于二采续跑。
    · 本节点新增可选 video_latent 输入：
        传入时 → latent 输出直接透传该 latent（保留一采的噪声态/中间态），
                  并把它当作"实际视频帧数"重新计算关键帧锚点；
        未传入 → 行为与官方节点完全一致（空 latent + keyframe 条件）。

二采高清用法：
    一采输出 LATENT ─► video_latent
    首帧/末帧        ─► first_frame / last_frame（可选）
    width/height     必须设为"二采目标分辨率"（= 二采 latent 的像素分辨率），
                      这样关键帧 latent（vae.encode 后 1/16）才能与二采 latent 对齐。
    apply_keyframes  =disable 时跳过关键帧注入（适合 latent 已含首末帧信息、
                      且不想处理 keyframe 分辨率匹配的场景）。

实现完全自包含：复制官方节点的辅助函数，不 import 官方私有 API。
"""

import math

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy_api.latest import io

# 与 ComfyUI nodes.MAX_RESOLUTION 一致（16384），自包含避免模块解析依赖
MAX_RESOLUTION = 16384
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24
AUDIO_LATENT_FPS = 40


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数（与官方 comfy_extras/nodes_minimax_h3.py 逻辑一致，自包含）
# ─────────────────────────────────────────────────────────────────────────────

def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def _resize(image, width, height, crop):
    """image [B, H, W, C] -> [B, height, width, 3]"""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _latent_video_t(latent):
    """从 latent 提取视频时间维 T。支持 NestedTensor / tuple / 5D tensor。"""
    samples = latent["samples"]
    if hasattr(samples, "is_nested") and samples.is_nested:
        for p in samples.unbind():
            if isinstance(p, torch.Tensor) and p.dim() == 5:
                return p.shape[2]
    if isinstance(samples, (tuple, list)):
        for p in samples:
            if isinstance(p, torch.Tensor) and p.dim() == 5:
                return p.shape[2]
    if isinstance(samples, torch.Tensor) and samples.dim() == 5:
        return samples.shape[2]
    return None


def _latent_frame_count(latent, fallback_length=124):
    """从 latent 时间维 T 反推实际视频帧数（17k+5 网格的反函数）。"""
    t = _latent_video_t(latent)
    if t is None:
        # 无法解析时回退到 length 对齐值，保证 keyframe 锚点不会崩
        return align_frame_count(max(5, fallback_length))
    if t <= 2:
        return 5
    return ((t - 2) // 5) * 17 + 5


def _latent_spatial_dims(latent):
    """从 latent 提取视频空间维度 (H, W 像素). 返回 (pixel_h, pixel_w)."""
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
    if video is not None:
        return video.shape[3] * 16, video.shape[4] * 16
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 节点
# ─────────────────────────────────────────────────────────────────────────────

class MiniMaxH3ImageToVideoTail(io.ComfyNode):
    """二采版 Image-to-Video：可透传已有视频 latent 并重算关键帧锚点。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageToVideoTail",
            display_name="MiniMax H3 Image to Video (Tail)",
            category="YCNodes-MiniMax-H3/conditioning",
            description=(
                "二采版 Image-to-Video 条件节点。video_latent 传入时透传该 "
                "latent（而非生成空 latent），并按实际帧数重算关键帧锚点，"
                "用于后段采样/高清精修。width/height 须与二采 latent 分辨率一致。"
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32,
                             tooltip="二采目标分辨率宽（须等于二采 latent 像素宽）"),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32,
                             tooltip="二采目标分辨率高（须等于二采 latent 像素高）"),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="帧数（24fps，对齐 17k+5 网格）。未传入 video_latent 时决定空 latent；传入时作为帧数反推的兜底"),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Latent.Input("video_latent", optional=True,
                                tooltip="二采输入：已有视频 latent（一采输出）。传入后替代空 latent，并按它的实际帧数重算关键帧锚点"),
                io.Combo.Input("apply_keyframes", options=["enable", "disable"], default="enable",
                               tooltip="disable 时跳过关键帧注入（latent 已含首末帧信息时的简洁模式）"),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                first_frame=None, last_frame=None, video_latent=None,
                apply_keyframes="enable") -> io.NodeOutput:
        # 1. 决定 latent 与帧数; 二采时自动对齐 keyframe 尺寸
        if video_latent is not None:
            latent = video_latent
            frame_count = _latent_frame_count(latent, fallback_length=length)
            # 自动从 latent 提取像素尺寸, 覆盖用户输入的 width/height
            latent_h, latent_w = _latent_spatial_dims(latent)
            if latent_h is not None and latent_w is not None:
                if width != latent_w or height != latent_h:
                    print(f"[MiniMaxH3ImageToVideoTail] 二采: 自动对齐 keyframe 尺寸 "
                          f"{width}x{height} → {latent_w}x{latent_h} (与 video_latent 一致)")
                width, height = latent_w, latent_h
        else:
            latent, frame_count = _empty_av_latent(width, height, length)

        # 2. 收集首/末帧（仅当启用关键帧且提供了帧时）
        images = []
        keyframes = []
        if apply_keyframes == "enable":
            if first_frame is not None:
                img = _resize(first_frame[:1], width, height, "disabled")
                images.append(img)
                keyframes.append({"resolved_frame_index": 0, "image": img})
            if last_frame is not None:
                img = _resize(last_frame[:1], width, height, "center")
                images.append(img)
                # 关键：锚点重算到二采 latent 的实际帧数
                keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        # 3. 文本/图像 tokenize（Qwen 视觉 token 需要首/末帧图）
        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)

        # 4. 关键帧条件注入
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })

        return io.NodeOutput(cond, latent)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ImageToVideoTail": MiniMaxH3ImageToVideoTail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ImageToVideoTail": "MiniMax H3 Image to Video (Tail)",
}
