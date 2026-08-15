# Front-view UAV PBSD 跨场景验证（2026-07-23）

## 1. 实验问题

验证 PanoAir 上有效的 Perspective-Balanced Survivor Densification（PBSD）能否在不按场景调参的条件下泛化到：

- HorizonGS `road_street1`（用户请求中的 `Rad` 按该场景解释）
- LVBA `Red_Sculpture`
- LVBA `HKU_Cultural_Center_01`

本轮固定使用 PanoAir 参数：候选池扩大 2 倍，深度分层为 `<20m / 20-50m / >=50m`，survivor 配额为 `0.25 / 0.45 / 0.30`。每个场景的 baseline/PBSD 除实验名和 `FrontViewSampling` 外，解析后的配置完全一致；seed 均为 42。

## 2. 完整序列结果

| 场景 | 方法 | PSNR | SSIM | LPIPS | GS 数 | 在线时间 (s) |
|---|---|---:|---:|---:|---:|---:|
| road_street1 | baseline | 22.52107 | 0.662765 | 0.370433 | 1,317,627 | 370.59 |
| road_street1 | PBSD | 22.56049 | 0.664341 | 0.368495 | 1,330,830 | 369.38 |
| Red Sculpture | baseline | 25.83172 | 0.817941 | 0.152158 | 3,514,779 | 1938.09 |
| Red Sculpture | PBSD | 25.85167 | 0.818092 | 0.151993 | 3,512,146 | 1899.80 |
| HKU Cultural Center | baseline | 21.99395 | 0.644560 | 0.374064 | 8,731,608 | 4179.59 |
| HKU Cultural Center | PBSD | 21.98938 | 0.644281 | 0.377103 | 8,798,909 | 4252.43 |

PBSD 相对 baseline：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS | Δ在线时间 |
|---|---:|---:|---:|---:|---:|
| road_street1 | +0.03941 dB | +0.001576 | -0.001938 | +1.00% | -0.33% |
| Red Sculpture | +0.01995 dB | +0.000151 | -0.000165 | -0.07% | -1.98% |
| HKU Cultural Center | -0.00457 dB | -0.000279 | +0.003039 | +0.77% | +1.74% |

## 3. 分区诊断

### road_street1

40 帧等步长诊断中，PBSD 的 far PSNR/SSIM 分别提高 `+0.2493 dB/+0.02231`，far-edge SSIM 提高 `+0.01778`；近景 PSNR 提高 `+0.0452 dB`，近景 SSIM 仅回退 `-0.00040`。可见 far GS 加权占比从 `44.19%` 增至 `45.13%`。结果符合“把有限 survivor 预算移向透视下欠表达区域”的机制。

### Red Sculpture

102 帧等步长诊断中，far PSNR/SSIM 提高 `+0.5770 dB/+0.01175`，far-edge PSNR/SSIM 提高 `+0.2981 dB/+0.00994`。但近景 PSNR/SSIM 为 `-0.0353 dB/-0.00210`，far LPIPS 回退 `+0.000619`，说明存在轻微近远景权衡。全序列三项指标仍全部优于 baseline。

### HKU Cultural Center

PBSD 候选池的三个深度段计数为：

```text
<20m:    2,248,266
20-50m:     41,568
>=50m:           61
```

候选几乎全部落在 `<20m`，固定米制配额无法形成真正的远景重分配。99 帧诊断中，可见 far GS 加权占比仅为 baseline/PBSD 的 `0.44%/0.45%`。PBSD 还产生了极少量投影半径异常大的 far outlier：平均 far radius 约 `1002 px`，单帧最大观察值 `8296 px`；RGB/primitive 抽帧未看到新的整屏 floater，但全序列 LPIPS 明确回退，因此该异常不能忽略。

## 4. 结论

1. PBSD 在 Road 和 Red 上同时改善完整序列 PSNR、SSIM、LPIPS，且速度和 GS 数保持同一量级；这比只在 PanoAir 上有效更有说服力。
2. HKU 是明确负结果，所以当前固定 `20/50m` PBSD 不能作为跨场景默认方法，也不能宣称已通过普适性验证。
3. 失败不是“配额强度还没调好”，而是固定米制深度在不同 SLAM 尺度/场景深度分布下不具备可比语义。继续逐场景调阈值会削弱方法性和可证伪性。
4. 下一版应把分层变量从绝对米制深度改为在线可校准的透视量，例如候选深度分位数、相对当前可见地图深度，或投影 footprint；并规定当中远层支持不足时严格退化为 baseline survivor 选择。该 fallback 必须由预先定义的支持度条件触发，而不是按场景人工开关。

## 5. 结果位置

- 机器可读汇总：`Logs_frontview_uav_crossscene/crossscene_final_summary.json`
- 原始并行 manifest：`Logs_frontview_uav_crossscene/benchmarks/pbsd_crossscene_full_seed42_2026-07-23_v2/manifest.json`
- Road 视频：各 run 的 `videos_validation_stride4/`
- Red 视频：各 run 的 `videos_validation_stride10/`
- HKU 视频：各 run 的 `videos_validation_stride20/`

本轮没有删除或覆盖旧日志，没有修改用户已有 HKU 配置，没有提交或打标签。
