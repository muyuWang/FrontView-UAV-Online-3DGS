# Front-view UAV 在线重建阶段结论（2026-07-23）

## 1. 结论

当前应保留的有效方法是 **Perspective-Balanced Survivor Densification
(PBSD)**，不是 q_g、P/M/S/A、PCRO、CRWD/CDWD 或 ACDT。

PBSD 在在线阶段生效，不依赖序列结束后的 appearance replay。它保留
ORB-SLAM3 VI 位姿、持久世界点、MODP 的即时 commit、HashBlock 和原优化
步数，只改变 DepthCov 候选的预算顺序与 survivor 分配：

1. 在高残差未覆盖区域采样 `2x` 候选池。
2. 先执行原有 DepthCov 不确定性过滤。
3. 对过滤后的 survivor 分配固定接纳预算，避免筛选前限额造成欠填。
4. 按预测 metric depth 分为 `<20 m`、`20-50 m`、`>=50 m`。
5. UAV 配额为 `0.25/0.45/0.30`；某层不足时从所有剩余 survivor 回填。
6. 通过 HashBlock 后在当前帧立即 commit，不增加优化步数或候选等待状态。

相对 MODP，真正的方法差异是 **uncertainty-filter-first、fixed-budget、
metric-depth survivor allocation**。ORB-VI 和持久世界点是共同输入基础，不应
与 PBSD novelty 混为一个贡献。

## 2. q_g 与 P/M/S/A 结论

同一 PanoAir 200 帧、ORB-VI/persistent-world、seed 42 协议：

| 方法 | PSNR (dB) | GS | 在线时间 (s) | 结论 |
|---|---:|---:|---:|---|
| 当前代码 MODP | 28.0907 | 48,945 | 118.3（并发） | 基础对照 |
| hard q_g | 24.1612 | 2,447 | 91.9 | 接纳密度坍缩，否决 |
| P/M/S/A | 28.5120 | 77,185 | 444.0 | 质量提高但约 3.75x 慢 |
| PM-only | 28.2496 | 50,425 | 309.5 | 弱收益且低效 |
| MODP 3200 点 | 28.7273 | 73,723 | 26.1 | 容量对照超过 P/M/S/A |

因此 q_g/P/M/S/A 不作为主方法。P/M/S/A 的增益主要来自更多小 GS，而不是
晋升逻辑本身；hard q_g 则错误地把输入点质量问题转化为大量拒绝。

## 3. PBSD 200 帧结果

以下是同一 GPU 顺序运行的主要结果。seed 42 存在约 `0.01-0.03 dB` 的进程
非确定性，结论使用 paired 差值而不是混合不同运行的绝对值。

### Seed 42

| 方法 | PSNR | SSIM | LPIPS | GS | 在线时间 (s) |
|---|---:|---:|---:|---:|---:|
| 高容量 MODP | 28.8192 | 0.80338 | 0.28886 | 76,878 | 26.66 |
| 2x pool-uniform | 28.9557 | 0.80443 | 0.28975 | 76,604 | 25.69 |
| PBSD-equal | 28.9775 | 0.80614 | 0.29288 | 76,181 | 26.98 |
| PBSD-UAV | **29.0751** | **0.80759** | 0.29437 | 76,285 | 24.82 |
| shuffled-depth-band | 28.8393 | 0.80378 | **0.28871** | 76,385 | 27.59 |

### Seed 43

| 方法 | PSNR | SSIM | LPIPS | GS | 在线时间 (s) |
|---|---:|---:|---:|---:|---:|
| 高容量 MODP | 28.8678 | 0.80512 | **0.27665** | 81,923 | 27.17 |
| 2x pool-uniform | 28.8212 | 0.80474 | 0.27610 | 82,007 | 27.07 |
| PBSD-equal | 29.0245 | 0.80848 | 0.27730 | 82,208 | 26.02 |
| PBSD-UAV | **29.0601** | **0.80938** | 0.28254 | 82,184 | 25.85 |
| shuffled-depth-band | 28.8238 | 0.80483 | 0.27682 | 82,302 | 26.33 |

PBSD-UAV 相对高容量 MODP 的 PSNR 提升分别为 `+0.2559` 和 `+0.1923 dB`。
相对 shuffled-depth-band 分别为 `+0.2358` 和 `+0.2364 dB`。shuffled 控制
保持 survivor 池、层配额、GS 数和优化器不变，仅打乱 depth label，因此两次
近乎相同的下降说明真实透视深度位置是 load-bearing variable，而不只是 2x 池
或容量增加。

旧的 200 帧稳定最高结果为 `28.1121 dB`；当前有效 PBSD-UAV 达到
`29.0751 dB`。更高的 `29.1527 dB` 来自被否决的 ACDT shuffled 控制，不能
作为方法结果。

## 4. 完整 PanoAir 结果

完整 ORB-VI/persistent-world 序列共有 2230 帧，seed 42：

| 方法 | PSNR | SSIM | LPIPS | GS | 在线时间 (s) |
|---|---:|---:|---:|---:|---:|
| 高容量 MODP（extra=3400） | 26.2057 | 0.79493 | **0.29422** | 641,792 | 411.09 |
| PBSD-UAV（extra=3200） | **26.4303** | **0.79691** | 0.29885 | **600,072** | **389.34** |
| 差值 | **+0.2246** | **+0.00199** | +0.00463 | -6.50% | -5.29% |

该对照给 MODP 更多 GS，PBSD 仍在 PSNR/SSIM、数量和时间上占优。PBSD 的
完整序列 PSNR 也高于此前约 `26.3558 dB` 的 Evidence LOD 结果，但 LPIPS
回退，因此不能宣称所有指标全面提升。

每 10 帧抽样、直接忽略缓存从最终 PLY 渲染时，PBSD 相对高容量 MODP 为
`+0.1922 dB PSNR / +0.00180 SSIM / +0.00437 LPIPS`。可见 far GS 的加权
比例从 `18.61%` 提高到 `22.74%`。RGB、depth 和 primitive 视频未观察到
新增地面 floater。

完整结果目录：

- MODP: `Logs_frontview_uav/PanoAir-frontview-orbvi-matched-full/2026-07-23-02-29-32_frontview_matched_full_pbsd_full_screen_seed42_v1_seed42_gpu2`
- PBSD: `Logs_frontview_uav/PanoAir-frontview-orbvi-pbsd-uav-full/2026-07-23-02-29-32_frontview_pbsd_uav_full_pbsd_full_screen_seed42_v1_seed42_gpu3`

## 5. 被否决的新方向

### PCRO

因果视差射线梯度预条件在单卡 paired seed 42/43 上分别比 equal-count 低
`0.0930/0.0788 dB`，否决。

### CRWD/CDWD

因果重投影/新显露评分分别低于 shuffled-score 或 oversampled-uniform，说明
正确分数不是 load-bearing variable，否决。

### ACDT / WorldGauge

原 WorldGauge 设计存在自条件泄漏：同一稀疏点既条件化 DepthCov，又被当作
校正真值。修正版将 500 个稀疏锚点拆成 450 个条件点和 50 个 held-out
校准点，并在同一次 DepthCov 前向中查询校准点与候选。

严格单卡 seed 42：

| 方法 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| split-only | 29.0411 | 0.80747 | 0.29764 |
| ACDT true binding | 29.0649 | **0.80896** | **0.29417** |
| shuffled residual location | **29.1527** | 0.80887 | 0.29553 |

真实绑定改善了感知指标，但 shuffled 控制的 PSNR 更高 `0.0879 dB`，因此
“世界点空间绑定”不能解释 PSNR 增益，按预注册 falsification 标准否决。
代码保留为默认关闭的可复现实验路径，不并入 PBSD。

## 6. 创新性边界

115 篇去重文献和 7 篇近邻全文/回退审计的结论为 **Level 3 - Medium
Overlap**。最接近的工作包括 SplatMAP、G2-Mapping、GeoGS-SLAM 和 Revising
Densification。

不能声称的新内容：

- online monocular Gaussian mapping；
- adaptive/depth-aware/uncertainty-aware densification；
- 固定 densification 容量或 growth cap；
- UAV 使用 ORB-VI 位姿与持久世界点本身。

当前可辩护的 narrow delta：

> SplatMAP 根据动态 DROID 深度/位姿可靠性更新、删除和新增 Gaussians，而
> PBSD 固定外部 VI 世界，在 DepthCov 过滤之后对 metric-depth survivor 分配
> 不变的即时出生预算；在完整 PanoAir 上以更少 GS 和更低在线时间取得
> `+0.2246 dB PSNR / +0.00199 SSIM`，但 LPIPS 回退 `0.00463`。

完整 novelty 报告：`ideaspark_run/pbsd-scoop/report.md`。

## 7. 当前限制与下一步门槛

PBSD 已经是可证伪的方法改动，不是增加优化时间的工程堆叠，但仍不足以直接
形成强论文结论：

1. `20/50 m` 边界与 `0.25/0.45/0.30` 配额目前是 PanoAir 设定，需在 Road
   和 LVBA 上验证或改为由相机/场景尺度因果确定的自适应边界。
2. LPIPS 在 200 帧与完整序列均回退，PBSD-UAV 是 distortion-oriented
   变体；PBSD-equal 更平衡但 PSNR 较低。
3. 完整序列目前只有 seed 42，需要至少补一个完整 seed 或跨场景重复。
4. 论文主表必须同时报告 pool-uniform、shuffled-depth-band、高容量 MODP、
   GS 数、在线时间和 far/near 分区指标。
5. 在感知指标未改善前，claim 应限定为“透视预算纠偏提高 PSNR/SSIM 与资源
   效率”，不能写成“全面提高重建质量”。

## 8. 验证状态

- 全量测试：`164 passed, 1 warning`
- `py_compile`：通过
- `git diff --check`：通过
- 未执行 commit、tag、reset 或 checkout
- 原有用户 dirty 文件与历史日志均未覆盖或删除
