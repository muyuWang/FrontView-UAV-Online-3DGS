# Temporal Gradient Band Routing (TGBR)

## 1. 当前结论

TGBR 是当前 hash-free front-view UAV 管线中的在线外观容量分配模块。它不直接改变
Gaussian 的出生候选、位置、尺度、旋转或优化步数，也不增加一次渲染；它只决定
哪些 Gaussian 可以从 SH2 单调晋升到 SH3。由于外观梯度会影响 opacity 优化和后续
pruning，不同分配仍可能间接产生不同的最终 GS 存活数。

seed 43 的完整 PanoAir 对照中，TGBR-75 达到 26.92350 dB，是当前单次完整序列
最高 PSNR。seed 44 中 TGBR 达到 26.83500 dB，分别比 SH2、静态 SH3 和
shuffled-75 高 0.08844、0.13865 和 0.00309 dB。两个 seed 中 TGBR 都取得最高
PSNR，但对 shuffled 的优势从 0.04483 dB 缩小到 0.00309 dB，且 seed 44 的
SSIM/LPIPS 略差于 shuffled。因此当前证据支持“弱但同向的两 seed PSNR 信号”，
不支持稳定幅度、全指标支配或跨场景普适提升。

## 2. 完整数据流及与 MODP 的边界

当前正式配置为
`configs/frontview_uav/panoair_orbvi_tsc_tgbr75_sh3_full.yaml`。其解析后的有效路径是：

1. 使用 ORB-SLAM3 VI/fused 位姿和 canonical persistent world points，而不是原始
   COLMAP 稀疏输入。
2. 稀疏点和 DepthCov 候选仍由 MODP 的 proposal 主干产生；DepthCov 使用 PBSD 的
   depth-stratified 固定预算采样。
3. 深度小于 80 m 的候选及稀疏点由 TSC 连续尺度覆盖判断空间责任；尺度半径系数为
   0.3，颜色兼容阈值为 0.15。
4. 深度不小于 80 m 的未跟踪 DepthCov 候选沿用 FPR-80，在当前帧的投影责任内竞争。
5. `HashBlock.use_hash=false`。完整实验中的 `hash_query_rows` 和 `hash_set_rows` 都为零。
6. 通过出生筛选的 Gaussian 立即进入永久地图；没有 candidate bank 或延迟 commit。
7. 所有 Gaussian 以 SH2 出生，TGBR 在每次晋升时按当前 GS 数设置 75% 的 SH3
   容量上限。

因此，当前方法已经脱离 MODP 的 HashBlock，但不能声称完全脱离 MODP：候选生成、
DepthCov、关键帧优化和立即 commit 仍来自 MODP。当前新增的地图组织是
`PBSD + TSC + FPR-80`，TGBR 是其上的在线外观 LOD。

## 3. TGBR 的计算

对 Gaussian `i`，令尚未激活的 SH3 band 为 `B3`。SH3 有 7 个基函数和 3 个颜色
通道，因此每个 Gaussian 的 band 梯度是 21 维向量：

```text
g_i^t = d L_t / d c_i,B3
```

SH3 系数在出生时严格为零，但仍处于可微的 SH3 渲染图中。因此一次标准反向传播就能
给出“如果允许该 band 学习，当前损失希望它如何变化”的反事实梯度，不需要额外
forward/backward。

在每个非 supplemental 关键帧优化的最后一步，从 packed projection 中取本批次可见的
Gaussian，并更新梯度向量 EMA：

```text
m_i^t = beta * m_i^(t-1) + (1 - beta) * g_i^t
beta = 0.9
```

按该 Gaussian 的有效观测次数 `n_i` 消除冷启动偏差：

```text
m_hat_i = m_i / (1 - beta^n_i)
q_i = ||m_hat_i||_2^2
```

`q_i` 保留梯度方向信息。如果不同关键帧对同一 SH3 系数提出相反更新，向量 EMA 会
相互抵消；只有跨时刻保持一致的残差方向才获得高分。这是 TGBR 与失败的 CGBR
(`EMA[||g||^2]`) 的核心差异。

每 10 次证据更新执行一次晋升：

1. 仅考虑至少观测 2 次、尚未达到 SH3 且 `q_i > 0` 的 Gaussian。
2. 当前总容量为 `floor(0.75 * N)`，已晋升 Gaussian 占用该容量。由于晋升不可逆，
   后续 opacity pruning 理论上可能改变最终 SH3 比例；因此 75% 是晋升时上限，不是
   pruning 后仍严格保持的全程不变量。
3. 对剩余容量按 `q_i` 取 top-k，并执行不可逆的 SH2 -> SH3 晋升。
4. 晋升发生在梯度屏蔽之前，因此新晋升项可以使用本次已计算的梯度立即更新。
5. 其余 SH3 梯度随后清零；Adam step 后再次将未激活系数约束为严格零。

21 维状态只在 gradient 模式首次使用时懒创建。以 674k Gaussian 计，向量 EMA 的
float32 主状态约 56.6 MB；默认关闭时不产生这部分开销。

## 4. 因果对照

`gradient_shuffled` 对照执行同样的反向传播、可见性收集、向量/标量 EMA、更新时刻和
容量计算，但不使用向量 EMA 的排序，而是在满足最小观测数且标量梯度能量为正的集合中
随机选择。实际实验的 SH3/GS 数量高度匹配，但两个 eligible 集合并非代码意义上的逐项
恒等，因此它是等容量位置随机对照，不应写成严格 matched-eligibility 对照。它排除了
“仅增加 SH3 参数量”和“仅执行梯度统计”的主要解释。

CGBR 使用标量 `EMA[||g_i^t||^2]`。在 seed 43 的 200 帧实验中：

| 方法 | PSNR | LPIPS | GS |
|---|---:|---:|---:|
| CGBR-75 | 29.16903 | 0.26796 | 83,281 |
| shuffled-75 | 29.23227 | 0.26681 | 83,504 |

CGBR 比 shuffled 低 0.06324 dB。该负结果说明瞬时梯度能量主要选择冲突或噪声残差，
不能作为有效证据。TGBR 改为 `||EMA[g]||^2` 后，在相同 200 帧协议中比 shuffled 高
0.09518 dB。

## 5. 已完成结果

### 5.1 PanoAir 200 帧，seed 43

结果：
`Logs_frontview_uav/benchmarks/tsc_tgbr75_matched_200_seed43_v130_2026-07-30/manifest.json`

| 方法 | PSNR | SSIM | LPIPS | GS | online time |
|---|---:|---:|---:|---:|---:|
| SH2 | 29.14496 | 0.809478 | 0.269784 | 83,172 | 716.37 s |
| 静态 SH3 | 29.23924 | 0.810143 | 0.267360 | 83,491 | 723.67 s |
| shuffled-75 | 29.17074 | 0.809911 | 0.268794 | 83,378 | 678.89 s |
| TGBR-75 | **29.26592** | 0.809962 | 0.269314 | 83,409 | 670.50 s |

### 5.2 PanoAir 完整序列，seed 43

结果：
`Logs_frontview_uav/benchmarks/tsc_tgbr75_matched_full_seed43_v131_2026-07-30/manifest.json`

| 方法 | PSNR | SSIM | LPIPS | GS | online time |
|---|---:|---:|---:|---:|---:|
| SH2 | 26.75531 | 0.801989 | 0.277246 | 660,373 | 7,755.68 s |
| 静态 SH3 | 26.88800 | 0.804690 | **0.269406** | 683,883 | 7,967.50 s |
| shuffled-75 | 26.87867 | 0.804300 | 0.272582 | 674,217 | 7,432.27 s |
| TGBR-75 | **26.92350** | **0.804747** | 0.270219 | 674,201 | 7,966.53 s |

TGBR 对 SH2 同时改善 PSNR、SSIM 和 LPIPS；对静态 SH3 提高 PSNR/SSIM，但 LPIPS
差 0.00081，不能宣称支配静态 SH3 的全部指标。TGBR 与 shuffled 只差 16 个 GS，
且 PSNR 高 0.04483 dB、LPIPS 低 0.00236。

四路并发共享 CPU、磁盘和数据读取，因此绝对 wall time 不可与历史单卡运行直接比较。
同批对照中，TGBR 与静态 SH3 的在线时间基本一致，相对 SH2 增加约 2.72%。

### 5.3 PanoAir 完整序列，seed 44

结果：
`Logs_frontview_uav/benchmarks/tsc_tgbr75_matched_full_seed44_v132_2026-07-29/manifest.json`

| 方法 | PSNR | SSIM | LPIPS | GS | online time |
|---|---:|---:|---:|---:|---:|
| SH2 | 26.74656 | 0.803685 | 0.275184 | 706,988 | 7,117.54 s |
| 静态 SH3 | 26.69635 | 0.804728 | 0.271782 | 735,165 | 7,339.43 s |
| shuffled-75 | 26.83191 | **0.805464** | **0.270492** | 726,265 | 8,091.01 s |
| TGBR-75 | **26.83500** | 0.805362 | 0.270821 | 721,440 | 7,253.45 s |

TGBR 相对 shuffled 的总体 PSNR 高 0.00309 dB，逐帧 PSNR 差的中位数为
0.00883 dB，1% trimmed mean 为 0.01158 dB，胜帧比例为 51.93%。因此正向结果
不是仅由少数正异常帧造成，但效应已经接近实验波动量级。与此同时，TGBR 的 SSIM
低 0.000102，LPIPS 高 0.000328，不能把 seed 44 表述为全指标改善。

seed 44 中 TGBR/shuffled 的最终 GS 数分别为 721,440/726,265，SH3 数分别为
533,765/537,733，SH3 比例分别为 73.986%/74.041%。两者容量比例高度接近，但最终
GS 相差 4,825（0.66%），不再像 seed 43 那样只差 16 个 GS。原因是 75% 约束发生在
晋升时，后续 opacity pruning 和外观梯度对地图存活产生反馈。该对照应称为近似等容量，
不能称为严格等 GS 数或严格等 SH3 数。

### 5.4 两个完整 seed 汇总

| 方法 | mean PSNR | mean SSIM | mean LPIPS | mean GS | mean online time |
|---|---:|---:|---:|---:|---:|
| SH2 | 26.75094 | 0.802837 | 0.276215 | 683,680.5 | 7,436.61 s |
| 静态 SH3 | 26.79218 | 0.804709 | 0.270594 | 709,524.0 | 7,653.47 s |
| shuffled-75 | 26.85529 | 0.804882 | 0.271537 | 700,241.0 | 7,761.64 s |
| TGBR-75 | **26.87925** | **0.805054** | **0.270520** | 697,820.5 | 7,609.99 s |

两 seed 平均下，TGBR 相对 SH2 的 PSNR/SSIM/LPIPS 差为
`+0.12831 / +0.002217 / -0.005695`，相对静态 SH3 为
`+0.08707 / +0.000345 / -0.000074`，相对 shuffled 为
`+0.02396 / +0.000172 / -0.001017`。TGBR 对 shuffled 的单 seed PSNR 增益范围为
`[+0.00309, +0.04483] dB`，说明均值为正但方差不可忽略。

八个完整运行均返回 0、处理 2,230 帧和 599 个关键帧，并且
`hash_query_rows == hash_set_rows == 0`；TSC 后端均为 `scipy_kdtree`。两 seed
平均在线时间中，TGBR 比 SH2 慢约 2.31%，比静态 SH3 快约 0.59%。由于每批四路并发
且 seed 44 同时受到其他作业的 CPU/I/O 竞争，这些时间只证明仍处于同一效率量级，不能
证明稳定加速。

## 6. Prior-art 审计

### 6.1 最接近工作

1. Papantonakis et al., *Reducing the Memory Footprint of 3D Gaussian Splatting*,
   PACM CGIT 2024, arXiv:2406.17074。该方法在训练中点使用全部输入视角，按
   transmittance 加权的颜色方差和截断高阶 band 后的颜色差，决定每个 Gaussian 保留
   SH0--SH3 中的哪一级；随后继续离线优化。目标是压缩和渲染加速。
2. Liu et al., *Adaptive Spherical Harmonics Degree Allocation for 3D Gaussian
   Splatting Compression*, ETAI 2026, DOI:10.1109/ETAI68332.2026.11485220。
   可检索摘要表明它联合渲染重要性、几何观测充分性和多视角颜色变化，为每个 Gaussian
   分配 SH0--SH3。全文未开放，因此不能排除实现细节上的进一步重合。
3. Huang et al., *StructGS: Adaptive Spherical Harmonics and Rendering Enhancements
   for Superior 3D Gaussian Splatting*, IEEE TMM 2025, arXiv:2503.06462。它按初始
   opacity 和点间距离初始化不同阶 SH；论文明确说明调整仍局限在初始化阶段。
4. Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*,
   SIGGRAPH 2023, arXiv:2308.04079。它使用 view-space 位置梯度做几何 densification，
   不是未激活外观 band 的容量分配。
5. Wu et al., *Monocular Online Reconstruction with Enhanced Detail Preservation*,
   SIGGRAPH 2025。MODP 定义了当前在线候选和优化主干，但没有 per-Gaussian SH LOD。

### 6.2 审计结论

保守结论为 **Level 3 - Medium Overlap**。已有工作已经覆盖“为每个 Gaussian 分配
不同 SH 阶数”这一大类机制，因此不能把 adaptive SH 本身作为 novelty。当前仍可辩护的
delta 是：

> 不同于使用完整训练视角的颜色变化进行事后 SH 裁剪，TGBR 在单遍在线 UAV 建图中
> 直接复用未激活 SH band 的反事实梯度，并以跨关键帧梯度方向一致性在固定容量下执行
> 因果、单调的 band 晋升，无需未来帧或额外渲染。

这一 delta 比 Evidence LOD 更强，因为 TGBR 不等待序列结束，也不是完整轨迹 replay；
但它仍是 appearance allocation，不能替代对整体 hash-free map organization 的论证。

## 7. 可证伪条件和下一步

以下任一结果都会削弱或否定当前方法：

1. 后续 seed 中 TGBR 对近似等容量 shuffled 不再保持正 PSNR，或多 seed 置信区间
   无法排除零增益，说明当前方向一致性信号不足以形成稳定收益。seed 44 仅以
   0.00309 dB 弱通过该条件，不能视为强复现。
2. Road 或 LVBA 场景中 TGBR 相对 SH2/静态 SH3持续退化，说明它只适配 PanoAir。
3. 等 SH3 数量或等显存对照解释全部增益，说明方法只是容量变化。
4. 去掉时间方向一致性后结果不变，说明 `||EMA[g]||^2` 不是有效机制。
5. 单卡运行时在线时间显著超出静态 SH3，说明当前并发计时掩盖了成本。

论文级最小验证集应包含：至少两个 PanoAir seed、Road 和两个 LVBA 场景；每个场景都
报告 SH2、静态 SH3、TGBR-75 和 shuffled-75，并同时报告 PSNR、SSIM、LPIPS、GS 数、
SH3 数、在线时间及 Hash/TSC 统计。PanoAir 两 seed 已完成，但跨场景证据仍缺失；当前
不应提交或打标签，也不应把 TGBR 描述为已经验证的普适改进。
