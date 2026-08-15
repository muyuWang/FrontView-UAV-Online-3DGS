# PanoAir seq1 位姿与初始化点云实验报告（2026-07-21）

## 实验问题

本轮只改变 MODP 的输入预处理，不接入在线 tracker。所有正式对照使用同一 mapper、`true_qg`、seed 42、200 帧和 100 步后优化。原始数据没有删除；`start110` 和 `start130` 均为派生数据，图像使用符号链接。

需要区分两个问题：

1. 切掉静止前缀是否改善 MODP 初始化和运行时间；
2. 换用更可靠的稀疏轨迹是否改善草坪几何并减少漂浮。

`start110` 的原始 COLMAP 分支复用了全序列 COLMAP 位姿和点云，因此它只测试第一个问题。DISK-LightGlue 点和 fused 点是在 source 110 之后重新匹配、建轨和三角化的，因此它们也测试第二个问题，但位姿仍来自原有 COLMAP 世界。

## 输入分支

| 分支 | source 范围 | 初始化几何 |
|---|---:|---|
| colmap_start0 | 0-199 | 全序列 COLMAP 的 canonical 点和位姿 |
| colmap_start110 | 110-309 | 同一个全序列 COLMAP 世界，切掉前 110 帧 |
| colmap_start130 | 130-329 | 同一个全序列 COLMAP 世界，再切 20 帧 |
| learned4096_start110 | 110-309 | COLMAP 位姿 + DISK-LightGlue 多视图轨迹三角化 |
| fused_start110 | 110-309 | 过滤 COLMAP 轨迹 + 去重后的 learned 轨迹 |

learned 分支获得 16,148 个点，轨迹长度中位数 7，重投影误差中位数 0.77 px，三角化角中位数 5.87 度，位置标准差中位数 0.083 m，但每帧可见点中位数仅 513。fused 分支有 25,188 个 canonical 点，其中 10,224 个过滤后的 COLMAP 点和 14,964 个 learned 点。

## 200 帧结果

以下完整指标只在各分支自己的 200 帧上计算，因此 start0 的完整 PSNR 包含大量容易拟合的静止帧，不能直接当作公平质量排名。

| 分支 | PSNR | SSIM | 覆盖率 | 近侧 PSNR | GS | 重建时间 | 秒/帧 |
|---|---:|---:|---:|---:|---:|---:|---:|
| colmap_start0 | 16.187 | 0.522 | 86.11% | 18.663 | 47,771 | 237.9 s | 1.190 |
| colmap_start110 | 14.625 | 0.486 | 87.04% | 20.352 | 54,008 | 178.8 s | 0.894 |
| colmap_start130 | 14.723 | 0.483 | 87.54% | 20.584 | 54,693 | 179.5 s | 0.897 |
| fused_start110 | 12.490 | 0.298 | 57.83% | 14.168 | 32,540 | 146.6 s | 0.733 |
| learned4096_start110 | 11.197 | 0.095 | 29.74% | 12.085 | 16,833 | 122.2 s | 0.611 |

对齐到相同 source 130-309 后，`colmap_start110` 和 `colmap_start130` 的 PSNR 分别为 14.410 和 14.378，覆盖率分别为 87.24% 和 87.09%。继续删除 110-129 没有收益。

在相同 source 120-160 上，`colmap_start0` 与 `colmap_start110` 的 PSNR 分别为 14.711 和 14.205，但近侧 PSNR 分别为 18.857 和 19.066。删除前缀使 200 帧重建时间降低 24.9%，近侧指标略有改善，但没有改善草坪的深度结构。

## 几何诊断

使用 source 150 的图像下半区域选择可见绿色 GS，再以 RANSAC 拟合草坪平面。该指标会混入树木等绿色物体，只作为跨分支相对诊断，不能解释为真实地面误差。

| 分支 | 草坪平面内点率 | 全部候选到平面中位数 | 全部候选到平面 P95 |
|---|---:|---:|---:|
| colmap_start0 | 22.9% | 0.205 m | 20.48 m |
| colmap_start110 | 16.7% | 0.352 m | 17.34 m |
| colmap_start130 | 16.5% | 0.251 m | 16.91 m |
| fused_start110 | 41.4% | 0.039 m | 0.348 m |
| learned4096_start110 | 52.2% | 0.028 m | 0.196 m |

RGB、深度和 opacity 视频与该结果一致：原始 COLMAP 分支覆盖完整，但草坪和上方植被由大量深度离散的 GS 构成；learned/fused 的地面明显更平整，但建筑和植被出现大面积空洞。这不是单纯增加优化步数能够修复的现象，而是几何纯度和覆盖率之间的输入矛盾。

固定 RTK 位姿的 DISK-LightGlue 三角化也已经证伪：相同 80 帧下，现有 calibrated RTK 产生 6,697 条候选轨迹但接受 0 点，另一种 `world_camera` 约定也只接受 54 点；同一 matcher 在 COLMAP 位姿下接受 1,284 点。当前 RTK 外参、方向约定或时间同步与虚拟 pinhole 图像不具备像素级一致性，不能直接替代 COLMAP 位姿。

## 结论

1. 前 110 帧静止会浪费在线计算，但不是漂浮草坪的主要原因。删除它们是合理的数据清理，主要收益是速度。
2. 继续删除到 130 帧没有收益，正式输入应保留 source 110 起点。
3. 原始 COLMAP 的低重投影误差不等于可靠深度。低视差、草地重复纹理和错误匹配仍可形成重投影误差较低但深度错误的点。
4. learned 轨迹证明可以得到明显更干净的地面几何，但 513 点/帧不足以支撑 MODP 场景覆盖。
5. 当前 fused 策略过滤过强：它把几何错误和建立外观覆盖所需的弱轨迹一起删除，所以漂浮减少但 PSNR、SSIM 和覆盖率同时下降。
6. 现阶段不应把 learned-only 或当前 fused 分支提升为 500 帧正式设置；它们没有通过质量门槛。在已删除静止前缀的候选中，当前质量最好的输入仍是 `colmap_start110`，但它没有解决根因。

## 下一步

下一版应采用“置信度分层而不是硬删除”：

1. 从 source 110 重新运行一次 SfM，使用位移/视差驱动的关键帧选择，而不是让静止帧和固定时间步长主导建图；用 DISK-LightGlue 或 SIFT + LightGlue 建轨并做全局 BA。
2. 高置信轨迹作为永久几何锚点；中置信 COLMAP 点保留覆盖，但使用低初始 opacity、有限位置 trust radius 和延迟晋升；只删除确定的 cheirality/极端不确定点。
3. 在草坪等弱纹理区域用高置信稀疏锚点拟合局部平面，对中置信点施加平面距离先验，不引入密集深度。
4. 保留每个图像网格的最低轨迹预算，避免 learned-only 的建筑/边缘覆盖塌缩。
5. 下一轮通过门槛应同时满足：覆盖率至少 80%，草坪 proxy P95 小于 0.5 m，相同 source 帧 PSNR 相对 raw COLMAP 下降不超过 0.3 dB，在线重建接近或优于 0.9 s/帧。

## 产物

- 主实验：`Logs_panoair_preprocess/benchmarks/preprocess200_causal_v2/manifest.json`
- start0 控制：`Logs_panoair_preprocess/benchmarks/preprocess200_start0_control_v1/manifest.json`
- 每个 run 下：`videos_full/`、`takeoff_source_120_160/`、`validation_metrics.json`、`worldtest_evaluation.json`
