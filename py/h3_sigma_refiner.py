"""
H3 Sigma Refiner
--作者 亦诚
H3 低噪细节精修器 (解决高速运动边缘像素颗粒与闪烁)
对低噪点区间（低 Sigma 阶段）进行局部“微雕”加步。
"""

import math
import torch


class H3SigmaRefiner:
    """
    对低噪点区间（低 Sigma 阶段）进行局部“微雕”加步，消除高速运动边缘的马赛克与像素紊乱。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", {
                    "tooltip": "输入的原始噪声序列。"
                }),
                "extra_steps": ("INT", {
                    "default": 1, "min": 0, "max": 15, "step": 1,
                    "tooltip": "在低噪点区间额外增加的细节平滑步数。"
                }),
                "start_at_sigma": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 20.0, "step": 0.01,
                    "tooltip": "启动细节加步的 Sigma 阈值。H3 推荐设在 2.0 ~ 3.5 之间。"
                }),
                "end_at_sigma": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "结束细化的 Sigma 边界（默认 0.0）。"
                }),
                "spacing": (["cosine", "linear", "exponential"], {
                    "default": "cosine",
                    "tooltip": "插值分布曲线。cosine（余弦）在趋近于 0 时分布更密，消噪效果最丝滑。"
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "refine_sigmas"
    CATEGORY = "YCNodes-MiniMax-H3/scheduler"
    DESCRIPTION = "针对 H3 视频模型高动态场景优化。在低噪点区（低 Sigma）局部加密采样步长，消解运动物体的边缘像素颗粒。"

    def refine_sigmas(self, sigmas, extra_steps, start_at_sigma, end_at_sigma, spacing):
        if extra_steps <= 0:
            return (sigmas,)

        # 确保在 CPU 侧处理数值
        sigmas_cpu = sigmas.detach().cpu()

        # 寻找进入 start_at_sigma 阈值的第一个索引位置
        idx = -1
        for i, s in enumerate(sigmas_cpu):
            if s <= start_at_sigma:
                idx = i
                break

        # 如果未达到阈值，或仅剩最后一步 (0.0)，无需处理
        if idx == -1 or idx >= len(sigmas_cpu) - 1:
            return (sigmas,)

        # 提取头部未修改序列与待精修的起始点
        unmodified_head = sigmas_cpu[:idx]
        A = sigmas_cpu[idx].item()
        B = max(end_at_sigma, sigmas_cpu[-1].item())

        original_tail_len = len(sigmas_cpu) - idx
        new_tail_len = original_tail_len + extra_steps

        # 1D 插值因子映射 [0.0, 1.0]
        t = torch.linspace(0.0, 1.0, steps=new_tail_len)

        if spacing == "cosine":
            factor = (1.0 - torch.cos(t * math.pi)) / 2.0
        elif spacing == "exponential":
            alpha = 3.0
            factor = (torch.exp(t * alpha) - 1.0) / (math.exp(alpha) - 1.0)
        else:  # linear
            factor = t

        # 根据对应曲线计算出新的高密度平滑尾端
        new_tail = A + (B - A) * factor

        # 如果原始序列最末端是 0.0 且 B > 0.0，需要保留最终的绝对收敛点 0.0
        if sigmas_cpu[-1].item() == 0.0 and B > 0.0:
            new_tail = torch.cat([new_tail, torch.tensor([0.0])])

        # 拼接头部和精置尾部
        new_sigmas = torch.cat([unmodified_head, new_tail])

        return (new_sigmas.to(device=sigmas.device, dtype=sigmas.dtype),)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "H3SigmaRefiner": H3SigmaRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SigmaRefiner": "H3 Sigma Refiner",
}


