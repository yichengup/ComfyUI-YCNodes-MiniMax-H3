"""
H3 Dynamic CFG Scheduler (动态CFG调度)

针对 MiniMax H3 视频模型，根据去噪阶段动态调整 CFG 引导强度。
原理：高 sigma（早期构图）用低 CFG 保大形，低 sigma（后期细节）用高 CFG 提细节。

与固定 CFG 的区别：
  H3 flow matching 默认 CFG=1.0，动态范围很小，微调即可。
  动态 CFG 0.9→1.1：早期松（构图灵活），后期紧（细节锐利）。
"""

import torch
from typing import Callable


class H3DynamicCFGScheduler:
    """
    动态 CFG 调度：sigma 越大 CFG 越低（保构图），sigma 越小 CFG 越高（提细节）。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "输入的 MiniMax H3 视频模型。"}),
                "cfg_low": ("FLOAT", {
                    "default": 0.9, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "高 sigma（早期构图）时的 CFG。建议 0.8~0.95，过低构图会散。"
                }),
                "cfg_high": ("FLOAT", {
                    "default": 1.1, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "低 sigma（后期细节）时的 CFG。建议 1.05~1.2，过高会过饱和。"
                }),
                "start_at_sigma": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "开始动态调度的 Sigma 阈值。之上用采样器原始 CFG。"
                }),
                "end_at_sigma": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "结束动态调度的 Sigma 阈值。"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "YCNodes-MiniMax-H3/scheduler"
    DESCRIPTION = "根据去噪阶段动态调整 CFG：早期低 CFG 保构图，后期高 CFG 提细节。"

    def patch(self, model, cfg_low, cfg_high, start_at_sigma, end_at_sigma):
        patched_model = model.clone()
        sigma_range = start_at_sigma - end_at_sigma

        def sampler_cfg_function(args: dict) -> torch.Tensor:
            sigma = args["sigma"].max().detach().cpu().item()
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            x = args["input"]
            cond_scale = args["cond_scale"]

            # 在目标区间内线性插值 CFG
            if start_at_sigma >= sigma >= end_at_sigma and sigma_range > 0:
                t = (start_at_sigma - sigma) / sigma_range
                dynamic_scale = cfg_low + (cfg_high - cfg_low) * t
            else:
                dynamic_scale = cond_scale

            # CFG: denoised = uncond + (cond - uncond) * scale
            # 返回噪声预测: noise = x - denoised
            denoised = uncond_pred + (cond_pred - uncond_pred) * dynamic_scale
            return x - denoised

        patched_model.set_model_sampler_cfg_function(sampler_cfg_function)
        return (patched_model,)


NODE_CLASS_MAPPINGS = {
    "H3DynamicCFGScheduler": H3DynamicCFGScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3DynamicCFGScheduler": "H3 Dynamic CFG Scheduler",
}