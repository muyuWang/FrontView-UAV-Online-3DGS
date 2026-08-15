# UAV 前视远场投影责任：实现与 PanoAir 验证

## 结论

当前可保留候选为 **Far-Field Projective Responsibility (FPR-80)**。它保留已验证有效的 UAV VI 位姿、canonical persistent world points、PBSD 候选采样和近场 HashBlock，只替换深度不小于 80m 的未跟踪 DepthCov 候选去重。

该方法不是完全 hashless，也不能宣称已经摆脱 MODP 主干。它的贡献是一个窄而可证伪的 range-selective responsibility：近场三维世界占用仍由 HashBlock 管理；远场不再写入世界哈希，而是在当前关键帧的 `(u, v, log-depth)` cell 内保留 residual 最大的候选。

## 数据流差异

PBSD：

1. persistent sparse world points 产生候选。
2. DepthCov 在高残差未覆盖区域补点。
3. PBSD 按深度预算选择 DepthCov 候选。
4. sparse 和全部 DepthCov 候选都查询、竞争并写入 HashBlock。
5. 通过后 immediate commit。

FPR-80：

1. 前三步与 PBSD 完全相同。
2. sparse 和 depth < 80m 的 DepthCov 仍执行原 HashBlock 路径。
3. depth >= 80m 的 DepthCov 绕过 world hash。
4. 远场候选按 12px image cell 和 1.10 depth ratio 的 log-depth bin 分组，每组保留 residual 最大行。
5. 远场行 immediate commit，但不写入 HashBlock。

实现入口：

- `utils_new/frontview_far_field.py`
- `utils_new/gaussian_models.py`
- `configs/frontview_uav/panoair_orbvi_pbsd_far_projective80_200.yaml`
- `configs/frontview_uav/panoair_orbvi_pbsd_far_projective80_full.yaml`

## 200 帧结果

seed43 paired：

| 方法 | PSNR | SSIM | LPIPS | GS |
|---|---:|---:|---:|---:|
| PBSD | 29.07846 | 0.80888 | 0.28205 | 81,983 |
| FPR-80 | 29.13756 | 0.80939 | 0.27964 | 82,128 |

seed42 paired 和 shuffled control：

| 方法 | PSNR | SSIM | LPIPS | GS |
|---|---:|---:|---:|---:|
| PBSD | 29.07216 | 0.80774 | 0.29530 | 76,256 |
| FPR-80 | 29.06892 | 0.80755 | 0.29338 | 76,229 |
| shuffled FPR-80 | 29.01770 | 0.80717 | 0.29521 | 76,164 |

seed42 中 FPR-80 的 PSNR 与 PBSD 基本持平，但比保持每帧 bypass 数量的 shuffled control 高 0.05122dB。seed43 中 FPR-80 同时改善三项指标。不能用单个 seed 宣称稳定的大幅提升。

## 完整 2230 帧结果

paired seed42：

| 方法 | PSNR | SSIM | LPIPS | GS | 在线时间 |
|---|---:|---:|---:|---:|---:|
| PBSD | 26.36194 | 0.79651 | 0.29977 | 600,177 | 498.04s |
| FPR-80 | 26.41480 | 0.79690 | 0.29934 | 599,308 | 504.57s |
| 变化 | +0.05286 | +0.00039 | -0.00043 | -0.15% | +1.31% |

完整运行中，1,245,706 个 host rows 里有 4,687 行绕过 HashBlock，2,045 个远场同 cell 重复被 projective responsibility 抑制。该结果不是增加 GS 数或增加优化步数得到的。

结果目录：

- `Logs_frontview_uav/benchmarks/pbsd_far_projective_full_paired_seed42_v31_2026-07-23`
- `Logs_frontview_uav/benchmarks/pbsd_far_projective_controls_seed42_v30_2026-07-23`
- `Logs_frontview_uav/benchmarks/pbsd_far_projective_seed43_v29_2026-07-23`

## 已否证的全 TLPB 路径

Track-Anchored Layered Projective Birth 在 200 帧窗口可达到 29.15923dB，并显著改善 LPIPS；真实 layer responsibility 也优于 shuffled control。但完整 2230 帧结果为 26.05166dB，低于 paired PBSD 的 26.30599dB，且 LPIPS 退化。active track ledger、multi-layer atlas、bounded overflow 和 near-hash hybrid 都没有修复完整序列退化。

因此 TLPB 只能作为负结果和后续研究原型，不能作为当前主方法。

## Novelty 边界

可辩护部分：

- 根据前视 UAV 的 metric range，仅把真正远场的 untracked DepthCov 从世界占用改为 ray-depth responsibility。
- host filter、commit recheck 和 occupancy write 三处使用同一责任身份。
- 用 per-frame count-preserving shuffled responsibility 控制检验真实远场位置。

不能单独声称创新的部分：

- residual-guided densification。
- image-space NMS 或 log-depth binning。
- near/far 分层本身。
- VI sparse points 与 Gaussian mapping。

当前 novelty 应定位为 **modest incremental, mechanism-validated**。后续要成为更强论文贡献，需要在 Road、Red Sculpture、HKU/LVBA 上验证，并把固定 80m 阈值改成无需场景调参、具有物理或统计定义的远场边界。

## 运行命令

200 帧：

```bash
CUDA_VISIBLE_DEVICES=0 /home/wmy/anaconda3/envs/worldvln/bin/python slam_new.py \
  --config configs/frontview_uav/panoair_orbvi_pbsd_far_projective80_200.yaml \
  --exp_name panoair_fpr80_200_seed42 --seed 42
```

完整序列：

```bash
CUDA_VISIBLE_DEVICES=0 /home/wmy/anaconda3/envs/worldvln/bin/python slam_new.py \
  --config configs/frontview_uav/panoair_orbvi_pbsd_far_projective80_full.yaml \
  --exp_name panoair_fpr80_full_seed42 --seed 42
```

PBSD 回滚快照仍为 commit `94c8362ae76df9b74c3914cf3ac2626f440a3ef4`。任何后续泛化失败都应回到该快照对应配置，而不是覆盖当前最好结果。
