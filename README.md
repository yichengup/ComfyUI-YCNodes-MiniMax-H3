# ComfyUI-YCNodes-MiniMax-H3

专为 MiniMax H3 视频模型打造的 ComfyUI 节点包，包含 4 个节点，覆盖时间分段提示词控制、注意力感受野约束、动态 CFG 调度和低噪细节精修。

无第三方依赖，仅需 PyTorch。

---

## 节点总览

| 节点 | 分类 | 功能 |
|------|------|------|
| H3 Prompt Relay | conditioning | 时间分段提示词控制，不同时间段只关注对应 prompt |
| H3 Distance Attention Patcher | 注意力 | 时空高斯感受野遮罩，防止背景同化局部细节 |
| H3 Dynamic CFG Scheduler | scheduler | 根据去噪阶段动态调整 CFG 引导强度 |
| H3 Sigma Refiner | scheduler | 低噪区间局部加步，消除运动边缘像素颗粒 |

---

## 1. H3 Prompt Relay（时间分段提示词控制）

**原理：** H3 使用打包自注意力（text + cond + audio + video 在同一序列），不存在独立 cross-attention。本节点通过对自注意力矩阵中 video query -> text key 路径施加时间惩罚 mask，实现不同时间段只关注对应 prompt 的效果。

**用法：** 将官方 prompt 原文复制到 `local_prompts`，在分段处加 `|` 分隔符。`global_prompt`（可选）填入全局风格基调，全程对所有帧可见。支持官方图生视频节点（`MiniMaxH3ImageToVideo`）及多参版本（`MiniMaxH3ImageToVideoMultiParams`）。

```
CLIP -> [MiniMaxH3ImageToVideo / MultiParams] -> CONDITIONING ─┐
CLIP ──────────────────────────────────────────────────────────┤
latent ──────────────────────────────────────────────────────> [H3PromptRelay] -> MODEL -> [采样器]
                                                                ↑                   CONDITIONING
                                                                └─── (透传) ─────────┘
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | MODEL | - | H3 模型 |
| `conditioning` | CONDITIONING | - | 官方节点输出的 conditioning |
| `clip` | CLIP | - | H3 CLIP 模型 |
| `latent` | LATENT | - | H3 视频 latent |
| `global_prompt` | STRING | 空 | 全局风格/主题基调，全程对所有帧可见（可选） |
| `local_prompts` | STRING | 空 | 分段提示词，用 `\|` 分隔 |
| `segment_lengths` | STRING | 空 | 逗号分隔的像素帧数，留空自动均分 |
| `epsilon` | FLOAT | 0.001 | 惩罚衰减参数，越小边界越锐利 |

---

## 2. H3 Distance Attention Patcher（距离注意力约束）

**原理：** 针对全景场景下肢体和面部容易被背景同化或扯碎的问题，通过时空高斯感受野遮罩强行约束模型在中前期去噪时的局部注意力，阻断大面积背景对微小细节的特征同化。

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `model` | MODEL | - | - | H3 模型 |
| `receptive_field_scale` | FLOAT | 1.0 | 1.0 ~ 100.0 | 感受野尺度，越小越局部 |
| `temporal_weight` | FLOAT | 3.0 | 0.1 ~ 10.0 | 时间轴相对空间轴的距离权重 |
| `start_at_sigma` | FLOAT | 2.5 | 0.0 ~ 20.0 | 开始约束的 Sigma 阈值 |
| `end_at_sigma` | FLOAT | 0.0 | 0.0 ~ 5.0 | 结束约束的 Sigma 阈值 |
| `num_frames` | INT | 17 | 1 ~ 256 | 视频总帧数 |
| `original_width` | INT | 864 | 128 ~ 2048 | 视频宽度 |
| `original_height` | INT | 480 | 128 ~ 2048 | 视频高度 |

---

## 3. H3 Dynamic CFG Scheduler（动态 CFG 调度）

**原理：** 根据去噪阶段动态调整 CFG 引导强度。高 sigma（早期构图）用低 CFG 保大形，低 sigma（后期细节）用高 CFG 提细节。H3 flow matching 默认 CFG=1.0，动态范围很小，微调即可。

**接线：** 插在 MODEL 和采样器之间。

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `model` | MODEL | - | - | H3 模型 |
| `cfg_low` | FLOAT | 0.9 | 0.5 ~ 2.0 | 高 sigma 时的 CFG（早期构图） |
| `cfg_high` | FLOAT | 1.1 | 0.5 ~ 2.0 | 低 sigma 时的 CFG（后期细节） |
| `start_at_sigma` | FLOAT | 3.0 | 0.0 ~ 20.0 | 开始动态调度的 Sigma 阈值 |
| `end_at_sigma` | FLOAT | 0.0 | 0.0 ~ 5.0 | 结束动态调度的 Sigma 阈值 |

---

## 4. H3 Sigma Refiner（低噪细节精修）

**原理：** 对低 Sigma 区间进行局部加步——保留原始调度的高噪头部不动，从阈值点起把尾部重采样成更长、更平滑的曲线，让模型在细节收尾阶段多走几步，消除高速运动边缘的马赛克与像素紊乱。

**接线：** 插在调度器和采样器之间。

```
BasicScheduler -> (sigmas) -> H3 Sigma Refiner -> (sigmas) -> SamplerCustomAdvanced
```

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `sigmas` | SIGMAS | - | - | 原始噪声序列 |
| `extra_steps` | INT | 1 | 0 ~ 15 | 低噪区间额外增加的步数 |
| `start_at_sigma` | FLOAT | 0.7 | 0.0 ~ 20.0 | 启动加步的 Sigma 阈值 |
| `end_at_sigma` | FLOAT | 0.0 | 0.0 ~ 5.0 | 结束细化的 Sigma 边界 |
| `spacing` | COMBO | cosine | cosine / linear / exponential | 尾部插值分布曲线 |

**spacing 曲线：**
- **cosine**（默认）：趋近 0 时分布更密，消噪最丝滑。
- **linear**：均匀分布。
- **exponential**：能量前移，尾部大步走向末点。

---

## 安装

1. 将 `ComfyUI-YCNodes-MiniMax-H3` 目录放入 `ComfyUI/custom_nodes/` 下。
2. 重启 ComfyUI。
3. 节点面板搜索 `H3` 即可找到全部节点。

## 推荐工作流接线

```
[CLIP] ───────────────────────────────────────────────┐
[图像] -> [MiniMaxH3ImageToVideo / MultiParams] -> COND ──┤
                                                      ├─> [H3PromptRelay] -> MODEL ─┐
[CLIP] ───────────────────────────────────────────────┘                            │
                                                                                   ├─> [H3DynamicCFGScheduler] -> MODEL -> [采样器]
[BasicScheduler] -> [H3SigmaRefiner] -> SIGMAS ────────────────────────────────────┘
```
