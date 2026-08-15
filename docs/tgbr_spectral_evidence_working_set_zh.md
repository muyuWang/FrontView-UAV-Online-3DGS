# TGBR 谱证据工作集

## 1. 问题

原 TGBR 根据在线梯度证据，把 Gaussian 的外观容量分成 SH2 和 SH3。它能够决定哪些
Gaussian 需要高阶视角相关表示，但完整历史关键帧的 RGB 金字塔、稀疏深度和 point
IDs 仍会常驻 GPU。关键帧数记为 `N_t` 时，这部分显存为 `O(N_tHW)`，长序列会持续
增长。完整 PanoAir 有 599 个关键帧，该历史观测已经成为约 10 GB 峰值显存的主要
来源，远大于 SH 参数本身。

这里不再给 TGBR 叠加独立的 EBRR。TGBR 被扩展为联合管理两类资源：

1. 每个 Gaussian 的活跃角向基维度；
2. GPU 上同时常驻的历史角向证据视图。

对应开关是：

```yaml
StreamingAppearanceLOD:
  spectral_residency_enabled: true
  spectral_residency_basis_budget: 5472.0
  spectral_residency_max_views: 512
  bounded_replay_residency_enabled: false
```

## 2. 联合谱预算

设最近一次 TGBR 证据选择后实际晋升到 SH3 的 Gaussian 比例为 `r_t`。代码读取的是
`current_target_fraction`，该值由实际 `appearance_sh_degree` 统计得到，不是配置中的
容量上限。SH2 和 SH3 分别有 9 和 16 个基函数，因此平均活跃基维度为

```text
D_t = (1 - r_t) * 9 + r_t * 16 = 9 + 7r_t.
```

TGBR 用同一个 `r_t` 决定 GPU 证据工作集上限：

```text
K_t = min(K_max, floor(B / D_t)).
```

所以任意时刻都有

```text
D_t * K_t <= B.
```

当更多 Gaussian 晋升到 SH3 时，每个视图对应的角向表示负载增大，允许常驻的历史
视图数随之收缩。推荐配置 `B=5472, K_max=512` 在 `r_t=0` 时最多常驻 512 帧；当
`r_t=0.75` 时 `D_t=14.25`，最终常驻约 384 帧。这个策略只使用当前时刻已经产生的
TGBR 证据，因而是因果的。

核心公式实现于 `utils_new/streaming_appearance_lod.py` 的
`spectral_residency_limit()`。驻留执行和统计位于
`utils_new/scene_mapper.py` 的 `_spectral_resident_camera_ids()`、
`_stage_camera_for_optimization()` 和 `_enforce_spectral_residency()`。

## 3. 数据流

每次在线更新仍执行原来的关键帧选择、优化步数和损失：

```text
KFGraph 选择当前优化视图
        |
        v
视图已在谱工作集? -- yes --> 直接读取 GPU 观测
        |
        no
        v
临时搬入 RGB / sparse depth / point IDs
        |
        v
执行完全相同的 render、loss、backward 和 TGBR 证据更新
        |
        v
更新后保留最近 K_t 个关键帧，其余观测迁回 CPU
```

该路径没有删除关键帧、没有减少 replay 视图、没有跳过优化 step，也不改变 Gaussian
出生、几何、opacity 或 loss 权重。`KFGraph` 选中的非驻留帧仍会按需搬回 GPU 并参加
同一次优化。

## 4. 性质

### 4.1 显存上界

令单帧观测显存上界为 `M_obs`，地图参数、优化器和一次优化的临时显存为 `M_map` 与
`M_tmp`。全部常驻时：

```text
M_peak <= M_map + N_t * M_obs + M_tmp.
```

谱证据工作集开启后：

```text
M_peak <= M_map + K_max * M_obs + M_tmp.
```

因此历史观测的 GPU 显存从随序列长度线性增长变为由配置常数约束。

### 4.2 优化目标不变

设第 `t` 步 KFGraph 选择的视图集合为 `S_t`，搬运算子为 `T`。CPU 到 GPU 的张量搬运
不改变张量数值，所以 `T(I_j)=I_j`。新旧路径在同一个 `S_t` 上计算相同的
`L(theta_t; S_t)`。在精确算术下，其 loss 和梯度相同；实际 CUDA 运行仍可能因在线
系统的非确定性产生小幅轨迹波动，因此实验只把指标视为“未退化”，不把随机正增益
解释为质量贡献。

## 5. PanoAir 结果

最终协议为完整 2230 帧、seed 43、GPU 4 单卡顺序运行、CPU 无其他重建任务；重建后
用 `render.py --skip_lpips` 计算 PSNR/SSIM。共进行三组同卡配对，顺序分别为
baseline->K384、K384->baseline、baseline->K384。结构化汇总位于：

```text
Logs_frontview_uav/benchmarks/
  tgbr_spectral_residency_k384_repeats_20260810/three_pair_summary.json
```

| 配对 | allocated | reserved | 在线时间 | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| repeat 1 | -26.43% | -24.88% | +3.98% | +0.14595 dB | +0.000682 |
| repeat 2 | -26.55% | -24.85% | +0.92% | -0.05284 dB | -0.000318 |
| repeat 3 | -26.40% | -23.21% | +5.84% | +0.04944 dB | -0.000008 |
| mean | **-26.46%** | **-24.31%** | **+3.58%** | **+0.04751 dB** | **+0.000119** |
| sample std | 0.08 pp | 0.96 pp | 2.48 pp | 0.09941 dB | 0.000512 |

三组 allocated 显存降幅都超过 26.4%，显存结论稳定。平均在线时间代价低于 5%，但
单组最大为 5.84%，因此不能声称每次运行都低于 5%。PSNR/SSIM 的均值没有退化，第二
组存在 `-0.05284 dB/-0.000318` 的小幅波动；结合在线系统本身的运行方差，当前结论
应写成“平均质量保持”，不能写成谱驻留提高了重建质量。

原始清单位于：

```text
Logs_frontview_uav/benchmarks/
  tgbr_spectral_residency_k384_pair_20260810/manifest.json
  tgbr_spectral_residency_k384_repeats_20260810/manifest.json
```

资源前沿位于
`Logs_frontview_uav/benchmarks/tgbr_spectral_residency_frontier_20260810/manifest.json`。
最终工作集 48/96/192/384 帧时，allocated 显存分别降低约
67.26%/61.45%/49.99%/26.36%；K384 的缓存命中率为 91.35%，因此比激进小缓存更接近
原始在线速度。

## 6. 论文表述边界

可以声称：

- TGBR 从单纯的 SH2/SH3 容量分配扩展为角向容量与证据驻留的联合预算；
- 工作集满足明确的 `D_t K_t <= B` 约束；
- 不改变关键帧集合、优化步数和目标函数；
- 三组完整 PanoAir 同卡实验将 peak allocated 显存平均降低 26.46%，平均在线时间代价
  为 3.58%，PSNR/SSIM 均值保持。

不能单独声称：

- CPU/GPU 张量搬运本身具有 novelty；
- K384 的正 PSNR 增量由谱驻留直接产生；
- 当前单场景结果已经证明跨数据集泛化；
- 固定 16 帧或零缓存无法获得相近的显存降幅。它们确实能降低显存，但传输开销更大，
  且不提供由当前 TGBR 活跃基维度决定的联合资源约束。
