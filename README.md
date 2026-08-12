# ComfyUI-YCNodes-MiniMax-H3

专为 MiniMax H3 视频模型打造的 ComfyUI 节点包。当前包含 **H3 Sigma Refiner**——针对 H3 高动态场景的低噪声细节精修器。

## 功能

### H3 Sigma Refiner

解决 H3 视频中**高速运动物体边缘的像素颗粒与闪烁**问题。

原理：对低 Sigma（低噪点）区间的噪声调度进行**局部"微雕"加步**——保留原始调度的高噪头部不动，从阈值点起把尾部重采样成一条更长、更平滑的曲线，让模型在细节收尾阶段多走几步，从而消解运动边缘的马赛克与像素紊乱。

## 节点参数 （按默认值设置就好，效果就稳）

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `sigmas` | SIGMAS | - | - | 输入的原始噪声序列 |
| `extra_steps` | INT | 1 | 0 ~ 15 | 在低噪点区间额外增加的细节平滑步数 |
| `start_at_sigma` | FLOAT | 0.7 | 0.0 ~ 20.0 | 启动细节加步的 Sigma 阈值（H3 推荐 2.0 ~ 3.5） |
| `end_at_sigma` | FLOAT | 0.0 | 0.0 ~ 5.0 | 结束细化的 Sigma 边界（默认 0.0 = 一路收敛到末点） |
| `spacing` | COMBO | cosine | cosine / linear / exponential | 尾部插值分布曲线 |

### spacing 曲线说明

- **cosine**（默认）：在趋近于 0 时分布更密，消噪效果最丝滑，适合细节收尾。
- **linear**：均匀分布，行为最可预测。
- **exponential**：能量前移，尾部大步走向末点（对低噪微雕通常不如 cosine）。

## 接线位置

```
BasicScheduler → (sigmas) → H3 Sigma Refiner → (sigmas) → SamplerCustomAdvanced
```

把本节点插在**调度器（BasicScheduler 等）与采样器（SamplerCustomAdvanced）之间**，替换掉原本直连的 sigmas 连线即可。

## 安装

1. 将整个 `ComfyUI-YCNodes-MiniMax-H3` 目录放入 `ComfyUI/custom_nodes/` 下。
2. 重启 ComfyUI。
3. 在节点面板搜索 `H3 Sigma Refiner`（分类 `YCNodes-MiniMax-H3/Sampling`）。

无任何第三方依赖，仅需 PyTorch。

## 目录结构

```
ComfyUI-YCNodes-MiniMax-H3/
├── __init__.py          # 插件入口，自动加载 py/ 下所有节点
└── py/
    └── h3_sigma_refiner.py   # H3 Sigma Refiner 节点
```

## 使用建议
按这个值设置，就是默认值

<img width="444" height="253" alt="image" src="https://github.com/user-attachments/assets/6bbb04a6-d5bb-4dbc-85c9-5885eb9d43fb" />
