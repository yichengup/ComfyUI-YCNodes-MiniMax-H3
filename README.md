# ComfyUI-YCNodes-MiniMax-H3
### 推荐只接我的节点：调度器-我的节点-采样器

专为 MiniMax H3 视频模型打造的 ComfyUI 节点包。包含两个核心节点：
- **H3 Sigma Refiner**：低噪尾部细节精修器
- **H3 Dynamic CFG Scheduler**：动态 CFG 调度器

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

---

### H3 Dynamic CFG Scheduler

在 SigmaRefiner 涂抹颗粒后，**补回细节锐度**。H3 flow matching 默认 CFG=1.0，本节点按 sigma 线性拉高 CFG，让后期细节阶段模型更"听话"。

原理：高 sigma（早期构图）用低 CFG 保大形灵活度，低 sigma（后期细节）用高 CFG 补回锐度。

## 节点参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `model` | MODEL | - | - | 输入的 H3 视频模型 |
| `cfg_low` | FLOAT | 0.9 | 0.5 ~ 2.0 | 高 sigma（早期构图）时的 CFG。建议 0.8~0.95 |
| `cfg_high` | FLOAT | 1.1 | 0.5 ~ 2.0 | 低 sigma（后期细节）时的 CFG。建议 1.05~1.2 |
| `start_at_sigma` | FLOAT | 3.0 | 0.0 ~ 20.0 | 开始动态调度的 Sigma 阈值 |
| `end_at_sigma` | FLOAT | 0.0 | 0.0 ~ 5.0 | 结束动态调度的 Sigma 阈值 |

### 推荐用法

**单节点（最简）**：单独使用 H3 Sigma Refiner 解决颗粒闪烁问题。

**组合链路（推荐）**：
```
基本调度器 → H3SigmaRefiner (0.7→0.0 cosine 涂抹) → H3DynamicCFGScheduler (0.2→0.0 CFG 1.0→1.15 补锐度) → 采样器
```

**配置参考**：
| 节点 | 关键参数 |
|------|---------|
| H3SigmaRefiner | start=0.7, end=0.0, spacing=cosine, extra_steps=1~2 |
| H3DynamicCFGScheduler | cfg_low=1.0, cfg_high=1.15, start=0.2, end=0.0 |

**采样器 CFG 必须填 1.0**（H3 默认），区间外用这个值。

## 接线位置

```
BasicScheduler → (sigmas) → H3 Sigma Refiner → (sigmas) → SamplerCustomAdvanced
                                                                       ↑
                                                          H3 Dynamic CFG Scheduler
                                                          （MODEL 链上，采样器前）
```

- **H3 Sigma Refiner** 插在**调度器与采样器之间**的 sigmas 线上
- **H3 Dynamic CFG Scheduler** 插在**MODEL 链**上，任意位置（在采样器前即可）

## 安装

1. 将整个 `ComfyUI-YCNodes-MiniMax-H3` 目录放入 `ComfyUI/custom_nodes/` 下。
2. 重启 ComfyUI。
3. 在节点面板搜索 `H3 Sigma Refiner` / `H3 Dynamic CFG Scheduler`（分类 `YCNodes-MiniMax-H3/Sampling`）。

无任何第三方依赖，仅需 PyTorch。

## 目录结构

```
ComfyUI-YCNodes-MiniMax-H3/
├── __init__.py                  # 插件入口，自动加载 py/ 下所有节点
└── py/
    ├── h3_sigma_refiner.py      # H3 Sigma Refiner 节点
    └── h3_dynamic_cfg_scheduler.py  # H3 Dynamic CFG Scheduler 节点
```

## 使用建议
按这个值设置，就是默认值

<img width="444" height="253" alt="image" src="https://github.com/user-attachments/assets/6bbb04a6-d5bb-4dbc-85c9-5885eb9d43fb" />
