# Canonical WorldTest-GS 200/500 帧实验报告

## 最终结论

1. **坐标故障是否修复：是。** hybrid 同一 track 的世界点离散度为 median 0.0891 m / p95 1.8607 m，COLMAP-canonical 与 RTK-canonical 都为 0 / 0 m。说明原始逐帧 RTK 反投影破坏了持久 world identity。
2. **漂浮地面是否消失：是，但 canonical 结果较稀疏。** 人工检查 frame 120-160 的 RGB、depth、opacity、primitive contact sheet；COLMAP/RTK canonical 未再出现随起飞抬起的连续绿色地面层。hybrid 当前视角 RGB 仍可很密，但侧视几何存在多层。
3. **`q_g` 是否优于 matched controls：在 hybrid mismatch 上是，在 canonical 数据上否。** hybrid true 的 future-invalid rate 为 7.25%，matched-delay/equal-count/shuffled 分别为 71.46%/70.60%/70.05%，且每帧 commit 数完全相同。canonical 上五种方法都是 0%，图像质量基本相同。
4. **当前方法 claim：收窄。** 保留“all-path held-out world-consistency birth gate 能在系统性 pose/depth mismatch 下优先选出可持久点”的机制证据；不声称它在已经一致的 canonical 输入上提升质量，也不声称当前 MVP 已完成严格 nuisance-marginalized Bayes factor。
5. **尚未解决的限制：** hybrid permanent coverage 只有约 5%，PSNR 约 9-11 dB，不是可用重建；COLMAP-canonical 也较稀疏；lawn 3 cm 绝对厚度没有达标；offline cache 非完全在线因果；完整 Schur nuisance covariance 和 Gaussian scene prior 尚未实现。

## 实验设置

- 日期：2026-07-20
- 数据：PanoAir `seq1` 前 200 帧
- seed：42
- 分辨率：960 x 640
- GPU：RTX 4090
- 优化：每帧 10 iterations，初始化 20，post refinement 100
- 第一轮按协议关闭 frequency detail、track/surface/flow detail、DepthCov permanent commit 和 archive
- 正式指标和 PLY 均关闭 shadow

## 坐标因果实验

| 输入 | 世界契约 | track dispersion median / p95 | frame 120-150 median / p95 | GS | 全序列 PSNR | coverage | online time |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid diagnostic | 无效 frame-local | 0.0891 / 1.8607 m | 0.0936 / 0.7776 m | 207,324 | 21.394 dB | 98.94% | 158.71 s |
| COLMAP canonical | 有效 persistent | 0 / 0 m | 0 / 0 m | 29,359 | 15.405 dB | 81.70% | 147.67 s |
| RTK canonical | 有效 persistent | 0 / 0 m | 0 / 0 m | 39,540 | 13.039 dB | 61.59% | 150.85 s |

全局 Sim(3) 的 hybrid trajectory RMSE 只有 0.0483 m，但严格 point-ID dispersion 仍达到 p95 1.86 m。这说明仅看 trajectory alignment 会漏掉“同一个点被逐帧写到不同 world 位置”的故障。

hybrid PSNR 高是因为 unsafe fast path 每帧写入大量 frame-local 点，不代表其三维几何正确。frame 130-140 和 140-150 的 PSNR 分别降至 18.33 和 14.10 dB，侧视图可见厚层；canonical 图虽稀疏，但不再有连续抬起的草地层。

审计日志：

- `Logs_worldtest_gs/audits/hybrid/world_frame_audit.json`
- `Logs_worldtest_gs/audits/colmap/world_frame_audit.json`
- `Logs_worldtest_gs/audits/rtk/world_frame_audit.json`

## 200 帧机制对照

### Hybrid stress input

所有方法逐帧精确匹配 493 个 committed groups，`bypass_count=0`。

| 方法 | GS | future-invalid | lawn ROI p95 | frame 80-120 PSNR | 120-130 | 130-140 | 140-150 | 150-200 | late coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| true `q_g` | 495 | **7.25%** | **8.54 m** | 11.192 | 11.135 | 10.538 | 9.253 | 9.682 | 5.00% |
| matched-delay | 497 | 71.46% | 18.48 m | 11.235 | 11.177 | 10.569 | 9.270 | 9.691 | 5.62% |
| equal-count random | 497 | 70.60% | 18.51 m | 11.222 | 11.164 | 10.563 | 9.264 | 9.687 | 5.41% |
| shuffled-`q_g` | 496 | 70.05% | 16.90 m | 11.226 | 11.167 | 10.561 | 9.269 | 9.689 | 5.54% |
| NPO-Lite | 497 | 62.50% | 12.31 m | 11.208 | 11.150 | 10.547 | 9.263 | 9.696 | 5.45% |

`q_g` 的收益不是 PSNR：所有 493 点地图都太稀疏，黑区占绝大部分。收益是等量、等时提交下 future-invalid 大幅下降，primitive/opacity 图从 controls 的放射状散点变为稳定的地面与建筑立面轮廓。固定 frame 120 底半幅 lawn ROI 的绝对 p95 仍远高于 3 cm，因此地面厚度成功条件没有达到，只能作为相对辅助证据。

### COLMAP-canonical input

所有方法逐帧精确匹配 23,958 个 committed groups，`bypass_count=0`，future-invalid 都为 0%。

| 方法 | GS | frame 80-120 PSNR | 120-130 | 130-140 | 140-150 | 150-200 | late coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| true `q_g` | 23,145 | 16.409 | 16.240 | 14.557 | 11.934 | 12.530 | 79.07% |
| matched-delay | 23,170 | 16.545 | 16.375 | 14.630 | 11.947 | 12.546 | 80.30% |
| equal-count random | 23,146 | 16.547 | 16.369 | 14.632 | 11.948 | 12.546 | 80.33% |
| shuffled-`q_g` | 23,160 | 16.547 | 16.376 | 14.638 | 11.947 | 12.545 | 80.29% |
| NPO-Lite | 23,149 | 16.547 | 16.373 | 14.630 | 11.945 | 12.544 | 80.31% |

在坐标已经一致时，score 排序不提供可测收益，true 甚至低约 0.1 dB。这否定了“`q_g` 普遍提升 canonical 重建质量”的宽泛说法，但不否定它在 pose/depth mismatch 下的选择能力。

## 运行时间

| 输入/方法 | internal online time | process wall time | 相对 coordinate fast path |
|---|---:|---:|---:|
| COLMAP coordinate fast path | 147.67 s | 未记录 | 1.00x |
| COLMAP true `q_g` | 214.12 s | 235.48 s | 1.45x |
| hybrid unsafe fast path | 158.71 s | 未记录 | 1.00x |
| hybrid true `q_g` | 163.08 s | 187.43 s | 1.03x |

离线 sparse-track cache 在 SLAM CUDA timer 启动前构建，因此 wall time 比 internal time 更完整。COLMAP 路径的 1.45x 尚未达到严格实时，但仍与 baseline 同一量级；controls 为了保持未选 cache group 并重新排序，耗时更高，不是部署路径。

## Frame 120-160 人工检查

关键图：

- canonical coordinate RGB：`Logs_worldtest_gs/PanoAir-seq1-worldtest-coordinate-colmap-200/2026-07-20-04-59-00_worldtest_colmap_coordinate200_causal_v1_seed42_gpu1/takeoff_120_160/contact_sheets/render_vs_gt_frames_120_160.jpg`
- hybrid true primitives：`Logs_worldtest_gs/PanoAir-seq1-worldtest-qg-hybrid-200/2026-07-20-06-39-48_worldtest_hybrid_true_qg_qg_true200_final_v1_seed42_gpu1/takeoff_120_160/contact_sheets/render_primitives.jpg`
- hybrid matched primitives：`Logs_worldtest_gs/PanoAir-seq1-worldtest-qg-hybrid-200-matched_delay/2026-07-20-06-49-39_worldtest_matched_delay_controls_hybrid200_final_v2_seed42_gpu4/takeoff_120_160/contact_sheets/render_primitives.jpg`
- hybrid true opacity：`Logs_worldtest_gs/PanoAir-seq1-worldtest-qg-hybrid-200/2026-07-20-06-39-48_worldtest_hybrid_true_qg_qg_true200_final_v1_seed42_gpu1/takeoff_120_160/contact_sheets/render_opacity.jpg`
- hybrid matched opacity：`Logs_worldtest_gs/PanoAir-seq1-worldtest-qg-hybrid-200-matched_delay/2026-07-20-06-49-39_worldtest_matched_delay_controls_hybrid200_final_v2_seed42_gpu4/takeoff_120_160/contact_sheets/render_opacity.jpg`

人工判断：

- canonical coordinate run：没有连续绿色平面随相机起飞抬升；缺点是远处建筑和草地覆盖稀疏。
- hybrid true `q_g`：没有 controls 中明显的放射状地面散点，保留下来的点形成较稳定的水平地面和竖直立面。
- matched/equal/shuffled：frame 120-160 中大量点以当前相机为中心向外发散，属于非持久 frame-local geometry。
- 所有 493 点 hybrid 方法 RGB 都接近空图，不能据此声称重建质量可用。

每个最终 run 都包含：

```text
videos_full/render_vs_gt.mp4
videos_full/render_depth.mp4
videos_full/render_opacity.mp4
videos_full/render_primitives.mp4
takeoff_120_160/*.mp4
takeoff_120_160/contact_sheets/*.jpg
validation_metrics.json
worldtest_evaluation.json
```

## 自动化测试

最终结果：`128 passed, 1 warning`。

覆盖：固定 Sim(3) 投影不变性、canonical identity、相容/不相容三视图、纯旋转/非有限 abstain、Laplace 与 200k Monte Carlo、所有 birth 路径证书检查、shadow alpha cap、control exact count，以及 cached offline control pool 回归。

当前 worktree 的 tracked `git diff --stat`（包含任务开始前已有的相关修改，不代表全部由本次新增）为：

```text
9 files changed, 3251 insertions(+), 147 deletions(-)
```

本任务新增的 `configs/worldtest_gs/`、`utils_new/worldtest_gs/`、审计/benchmark/后处理脚本、测试和两份文档仍是 untracked 文件，因此不会出现在上述 `git diff --stat` 中；`git status --short` 已保留它们，未清理用户原有 worktree。

## 复现命令

```bash
cd /home/wmy/workspace_vla/Online-3DGS-Monocular

/home/wmy/anaconda3/envs/worldvln/bin/python scripts/benchmark_worldtest_gs.py \
  --suite qg_true --frames 200 --gpu-ids 0,1 --seed 42 \
  --tag qg_true200_repro

# 分别使用两个 true run 的 worldtest_commit_schedule.json：
/home/wmy/anaconda3/envs/worldvln/bin/python scripts/benchmark_worldtest_gs.py \
  --suite controls --frames 200 --input colmap --schedule <COLMAP_SCHEDULE> \
  --gpu-ids 0,1,2,3 --seed 42 --tag controls_colmap200_repro

/home/wmy/anaconda3/envs/worldvln/bin/python scripts/benchmark_worldtest_gs.py \
  --suite controls --frames 200 --input hybrid --schedule <HYBRID_SCHEDULE> \
  --gpu-ids 4,5,6,7 --seed 42 --tag controls_hybrid200_repro
```

## Claim 判定

matched-delay、equal-count 与 shuffled-`q_g` 三个反事实对照没有证伪“在系统性 world-frame mismatch 下，held-out score 选择了更持久的点”，因为 true 的 invalid rate 从约 70% 降至 7.25%。但 canonical 对照证明该 score 不是普遍质量增强器，低 coverage 也证明仅靠 admission 不能完成稠密重建。

因此当前 claim 只能收窄为：**在明确 pose/depth world-identity 冲突的 online Gaussian birth 场景中，all-path held-out world-consistency admission 比等量、等等待和 score-shuffled birth 更少产生未来失效点。** 完整 nuisance-marginalized Bayesian 版本仍需实现后重新验证，才具备论文级 solid claim。

## 500 帧扩展实验

### 故障定位

首个 500 帧 run 在 frame 200 后停止增长，是因为 `orb_point_ids` 只有 0-199。`scripts/restore_colmap_point_ids.py` 使用 exact float32 canonical coordinate identity 恢复了 frame 200-499 的 300 个 sidecar；500 帧共核对 5,000,000 行，没有使用 nearest neighbor。修复后 frame 400-500 仍提交 11,709 个 certified roots，说明后段退化不再是 admission 停止。

v3 的可见 GS 从 frame 400 的 40,746 降到 frame 499 的 21,801，且无深度天空被黑背景占据。完全冻结 certified root means 的 40 帧消融使 PSNR 从 19.772 降到 19.046，因此硬冻结被否决。5 cm canonical birth trust region 保留了 40 帧质量，并把 500 帧 auxiliary nearest-canonical p95 从约 26.2 cm 压到 11.35 cm；该 nearest 指标包含 split children，只是辅助诊断，不冒充严格 identity。

700-step all-frame replay 让每帧至少均匀出现两次后才启用 hard mining。它将 raw permanent-only PSNR 从 16.839 提到 16.892 dB，证明后段主瓶颈不是简单的优化步数不足。200 帧 pose-prepass 对照也被否决：PSNR 18.423 -> 17.952，SSIM 0.676 -> 0.570，并额外增加 42.7 秒。

### 世界方向背景

最终增加一个与 permanent GS 分离的无限远背景 sidecar。它不进入 Parameter、Adam、HashBlock、certificate、PLY 或 GS count：

1. 只读取已经输入的训练帧上方 40%，按亮度和饱和度选择 depthless sky observation；
2. 用 COLMAP-canonical camera rotation 把像素射线转换到 world direction；
3. 在 256 x 512 environment grid 中，每帧每个 bin 最多投一票；
4. 只有至少 20 个不同帧支持的方向才能写入背景；
5. 渲染时仅按这些 world directions 和 `(1 - permanent opacity)` 合成稳健 median sky RGB。

这不是 source fast path，也不签发虚假的 Gaussian certificate。它解决的是 sparse/monocular depth 本来无法表示的无限远天空，不能替代建筑、草地和自行车的有限深度几何。

### 500 帧结果

| 版本 | GS | PSNR | SSIM | coverage | 400-500 PSNR | internal / wall |
|---|---:|---:|---:|---:|---:|---:|
| v2 identity fix | 51,265 | 15.013 | 0.503 | 90.75% | 12.523 | 494.14 / 535.64 s |
| v3 detail + 300 replay | 77,975 | 16.839 | 0.726 | 95.29% | 13.093 | 452.47 / 497.97 s |
| v6 5 cm trust + 700 replay, permanent only | 77,979 | 16.892 | 0.728 | 95.44% | 13.235 | 469.59 / 515.78 s |
| v6 + directional background | 77,979 | **21.261** | **0.736** | **98.71%** | **22.165** | reconstruction time unchanged |

最终分段 PSNR：0-200 为 20.623，200-300 为 21.456，300-400 为 21.436，400-500 为 22.165。future-invalid rate 为 0，certificate bypass 为 0；57,667 个 certified roots、8,192 个 split parents、24,576 个 inherited children，最终 77,979 GS。

最终视频与指标：

```text
Logs_worldtest_gs/PanoAir-seq1-worldtest-qg-colmap-500/
  2026-07-20-15-17-50_worldtest_colmap_true_qg_trust5cm_replay700_v6_seed42_gpu1/
    videos_directional_background/render_vs_gt.mp4
    validation_metrics_directional_background.json
    worldtest_evaluation_directional_background.json
    background_model.json
```

### 500 帧结论边界

- frame 120-160 raw depth/opacity/primitive 检查中仍未出现随起飞抬升的连续绿色地面层，canonical 坐标修复继续成立。
- 超过 20 dB 的正式结果是 `permanent GS + independently stored directional far field`，不是 PLY-only PSNR；报告必须同时保留 16.892 dB raw 数字。
- 环境背景修复了天空黑洞和后段 coverage，但 edge PSNR 基本不变。树冠边界仍有浅色斑点，草纹、窗格和近处有限深度物体仍偏糊；这些需要更好的 finite-depth carrier/appearance model，不能靠背景继续提高。
