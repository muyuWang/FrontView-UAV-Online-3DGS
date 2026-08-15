# TGBR 计算路由强化与 PanoAir 验证

## 1. 修改动机

原 TGBR 只在优化器更新前将未晋升 Gaussian 的 SH3 梯度清零，并在更新后把对应
系数约束为零。所有 Gaussian 仍分配完整 SH3 参数和 Adam 状态，gsplat 也仍对所有
可见 Gaussian 执行 SH3 forward/backward。因此原实现证明的是外观容量分配，而不是
计算型 LOD。

本次修改把 TGBR 扩展为计算路由：正常在线优化步中，已晋升 Gaussian 计算 SH3，
未晋升 Gaussian 只计算 SH2；TGBR 证据步临时对未晋升行开启 SH3 probe，以保留原有
反事实梯度和晋升逻辑。

## 2. 实现

核心实现位于 `utils_new/heterogeneous_sh_rasterizer.py`。每次渲染仍只有：

1. 一次 packed `fully_fused_projection`；
2. 一次全局 tile intersection 和深度排序；
3. 一次 `rasterize_to_pixels` alpha blending。

变化只发生在投影后、混合前的颜色计算：

```text
base rows   = visible rows with appearance_sh_degree < 3
target rows = visible rows with appearance_sh_degree >= 3

normal step: base rows -> SH2, target rows -> SH3
probe step:  all visible rows -> SH3
```

没有把 SH2 和 SH3 分成两张图渲染，因此不会破坏全局深度顺序或产生两次 alpha
blending。在线优化只在非 supplemental 关键帧的最后一步开启 probe。

为降低 TGBR 自身状态，21 维方向 EMA 可使用带固定尺度的 FP16。当前推荐设置使用
`scale=4096` 后再写入 FP16，避免原始微小梯度下溢；固定正尺度不改变理论排序目标。
前 10 次证据更新保持原 dense 路径，避免初始化阶段的极小 backward 数值差异被在线
几何和后续出生决策放大。

PLY 不保存 `appearance_sh_degree`。`render.py` 从 PLY 重载时会关闭计算路由并按完整
SH 阶渲染；未激活 SH3 系数在训练中已保持为严格零，因此该行为与在线最终外观一致。

## 3. 推荐资源配置

配置：
`configs/frontview_uav/panoair_tgbr_compute_routed_cap60_full.yaml`

```yaml
StreamingAppearanceLOD:
  enabled: true
  birth_degree: 2
  target_degree: 3
  max_target_fraction: 0.60
  selection_mode: gradient_agreement
  compute_routing: true
  compute_routing_warmup_evidence_updates: 10
  gradient_ema_dtype: float16
  gradient_ema_scale: 4096.0
```

该配置是资源/质量折中，不替换 PSNR 优先的 dense TGBR-75 默认配置。

## 4. 完整 PanoAir 结果

协议：完整 2230 帧、seed 43、四卡 matched 启动；重建阶段
`Results.skip_eval=true`，保存 PLY 后统一用 `render.py --skip_lpips` 计算 PSNR/SSIM
并生成 `render_vs_gt.mp4`。本表的指标协议与历史内置 `eval_gaussians` 数值不可混用。

结果清单：
`Logs_frontview_uav/benchmarks/tgbr_compute_matched_full_seed43_v136_20260806/manifest.json`

| 方法 | PSNR | SSIM | 在线时间 | GS | SH3 GS |
|---|---:|---:|---:|---:|---:|
| static SH3 | 26.481985 | 0.792935 | 3655.65 s | 680364 | 全部 |
| dense TGBR-75 | **26.541345** | 0.792501 | 3646.31 s | 672981 | 494031 |
| routed TGBR-60 | 26.471929 | 0.792469 | 3656.74 s | 671797 | 395773 |
| routed shuffled-60 | 26.431877 | 0.791507 | 3652.38 s | 672666 | 396718 |

同一配置的第二次 routed TGBR-60 完整运行达到
`26.519311 dB / 0.792165 SSIM`，说明当前在线轨迹存在约 0.05 dB 的运行波动。对应
清单位于：
`Logs_frontview_uav/benchmarks/tgbr_compute_frontier_full_seed43_v137_20260807/manifest.json`。

在严格同批、近似等 SH3 容量对照中，TGBR-60 相对 shuffled-60 为：

```text
PSNR: +0.040052 dB
SSIM: +0.000962
```

这比只比较 static SH3 更直接地支持“时间梯度方向一致性决定高阶 band 位置”的动机。

## 5. 资源结果

routed TGBR-60 的完整运行统计为：

```text
SH3 band 可见行跳过比例（计算路由激活期间）: 35.54%
SH 基函数项减少比例（计算路由激活期间）: 15.55%
21 维方向 EMA: 53.91 MB -> 26.91 MB，减少 50%
单次完整运行的 rasterization 调用: 每次 render 仍为 1
```

首组 matched 完整实验中四路在线时间为 3646--3657 s，差异小于 0.3%，因此当前证据
支持“没有速度退化”，不支持稳定 wall-time 加速。峰值显存仅从约 9599.7 MB 下降到
9584.6 MB，因为完整 SH3 参数和 Adam 状态仍为 dense，且自定义 packed 路由需要临时
索引张量。不能把 50% 的 TGBR 证据状态压缩写成 50% 总显存压缩。

## 6. 结论与边界

本次修改让 TGBR 从纯 mask 容量路由变成了真实跳过未激活 SH3 forward/backward 的
计算路由，并保留一次投影和一次混合。TGBR 相对 shuffled 的完整序列增益支持其选择
信号；资源收益也可由执行统计直接验证。

仅凭上述 SH 计算路由实验，当前不能声称：

- routed TGBR 的 PSNR 超过 dense TGBR-75；
- 总显存显著下降；
- wall time 已稳定加速；
- FP16 EMA 与 FP32 在所有场景严格等价。

若只沿 SH 参数方向追求显著总显存下降，仍需把 inactive SH3 参数及 Adam moment
改为真正分组或稀疏存储；上述计算路由没有伪称已经完成这一层重构。下一节改为约束
历史观测驻留，解决的是另一项更主要的峰值来源。

## 7. 有界因果重放驻留

### 7.1 动机

TGBR 的在线决策只需要当前重放批次和已经累计的逐 Gaussian 梯度证据。把全部历史
关键帧 RGB、图像金字塔和稀疏深度永久留在 GPU，不会增加 TGBR 的证据充分性，却会
让观测显存随关键帧数 `K` 线性增长。该开销在完整 PanoAir 上比 SH 参数本身更大，
也是前述计算路由未能显著降低总峰值显存的主要原因。

因此当前 TGBR 增加 **bounded causal replay residency**：

1. 历史关键帧观测保存在 CPU；
2. 因果重放调度器选出本次更新所需的至多 `B` 个视图；
3. 只把这 `B` 个视图的 RGB 金字塔、稀疏深度和 point IDs 暂存到 GPU；
4. 完成同一组 Gaussian loss、反向传播和 TGBR 证据更新；
5. 更新结束后立即把该批观测迁回 CPU。

该路径不删除视图，不减少优化 step，不改变 loss，也不改变 Gaussian 出生预算。它只
改变历史观测的驻留位置。GPU 观测显存由

```text
O(K * H * W)  ->  O(B * H * W),  B << K
```

且 Gaussian 参数、Adam 状态和 TGBR 证据状态保持不变。实现开关位于：

```yaml
StreamingAppearanceLOD:
  bounded_replay_residency_enabled: true
```

启用后，`SceneMapper` 会强制历史关键帧按需驻留，不再依赖调用者手工修改
`Mapper.pin_kf_gpu`。运行结果额外保存 `tgbr_replay_residency`，包括原始与有效驻留
策略、单次最大暂存视图数、暂存观测字节数和最终 GPU 常驻历史帧数。

### 7.2 完整 PanoAir 结果

固定 seed 43、完整 2230 帧、GPU 4 串行计时，评测跳过 LPIPS。基线与 EBRR 使用相同
TGBR 配置、重放视图、优化步数和 GS 预算：

| 方法 | PSNR | SSIM | 在线时间 | 峰值 allocated | 峰值 reserved | GS |
|---|---:|---:|---:|---:|---:|---:|
| TGBR 常驻历史帧 | 26.494330 | 0.792062 | 566.03 s | 10.038 GB | 11.096 GB | 670316 |
| TGBR + bounded residency | **26.551353** | **0.792523** | 716.64 s | **2.755 GB** | **3.410 GB** | 669956 |

相对变化：

```text
peak allocated: -72.56%
peak reserved:  -69.27%
PSNR:           +0.05702 dB
SSIM:           +0.00046
在线时间:       +26.61%
吞吐率:          3.11 fps
```

结果目录：

```text
基线: Logs_frontview_uav/benchmarks/tgbr_directional_views_a_full_sequential_seed43_v151_20260807
EBRR: Logs_frontview_uav/benchmarks/tgbr_bounded_residency_full_seed43_v162_20260807
```

显式开关的 200 帧回归还记录到：历史关键帧 `34`，最终 GPU 常驻历史帧 `0`，单次最大
暂存视图 `6`，最大暂存观测约 `78.56 MB`。PSNR/SSIM 为
`28.848314/0.804002`。结果目录为：

```text
Logs_frontview_uav/benchmarks/tgbr_ebrr_wiring_200_seed43_20260807
```

### 7.3 结论边界

该机制满足“总峰值显存下降至少 9% 且质量不下降”的目标，实际 allocated 降幅为
72.56%。代价是 CPU-GPU 观测传输使在线时间增加 26.61%，但 3.11 fps 仍高于当前
1 fps 的在线要求。

EBRR 应作为 TGBR 的资源执行设计，而不是单独宣称为新的外观模型。TGBR 的核心算法
贡献仍是反事实 SH3 梯度的跨时间方向一致性与固定预算晋升；EBRR 的作用是让这套因果
证据路由在长序列上具有显著、可测量的 GPU 显存收益。CPU 仍保存全部历史观测，因此
它提供的是有界 GPU residency，不是全系统严格有界内存保证。
