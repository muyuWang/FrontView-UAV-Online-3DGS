# AeroCommit-MVP

## 目标与边界

AeroCommit-MVP 在 MODP 原有 proposal、MOHV、渲染和优化器之上，改变“新 Gaussian 何时成为永久地图参数”。它不使用未来帧，不使用 GT depth，不改变 3DGS rasterizer。当前实现的核心不是 P/M 两层重命名，而是一个真实的 ephemeral candidate/commit 边界：waiting candidate 不占 `Parameter`、gradient、Adam state、HashBlock occupancy，也不参与 coverage、render、prune 或普通 PLY。

三种模式：

- `baseline`: host proposal 立即 commit，关闭 AeroCommit 时保持原路径。
- `npo_gate`: host proposal 进入 CandidateBank，满足多视图证据后由 NPO-Lite 决定 commit。
- `aerocommit_mvp`: 增加 commit-time 逆深度修正、纹理保持的颜色校正、稳定残差 split、字节预算和 FP16 CPU archive。

## 每帧数据流

```text
current RGB + ORB sparse depth + causal pose
  -> MODP propose_new_gaussians (read-only HashBlock query)
  -> ORB / high-confidence DepthCov fast path
  -> remaining proposal grouping (image cell + log depth)
  -> projected-grid candidate association
  -> WAIT until >= 3 representative support views
  -> top-K batched NPO-Lite
  -> WAIT or COMMIT
  -> bounded inverse-depth refinement + texture-preserving color correction
  -> stable residual detail split
  -> commit_proposals (the only permanent mutation)
  -> normal MODP optimization / prune / render
  -> active-byte check and optional FP16 CPU archive
```

候选最多保留 reference、最大视差、最新和最低 residual 四个 support。候选存储在 CPU，只有 top-K 风险计算搬到 GPU。默认关闭的 `fuse_support_proposals` 和 `latest_consistent` snapshot 已做失败消融：粗 cell 内没有逐点对应，直接融合会产生错误遮挡，因此正式配置使用 reference snapshot。

## Proposal 与 commit

`GaussianModel.propose_new_gaussians()` 复用 MODP 的 sparse/DepthCov 采样、尺度初始化、颜色读取、HashBlock 查询和 conflict filter，但不调用 `setOccupy`，不扩展 Gaussian tensor。

`GaussianModel.commit_proposals()` 会再次检查 occupancy，然后才：

1. 调用 `HashBlock.setOccupy`；
2. 扩展 means/scales/quats/opacities/SH；
3. 扩展 Adam `exp_avg` 和 `exp_avg_sq`；
4. 将新 group 加入 active/valid group。

baseline wrapper 等价于 `proposal -> immediate commit`。

## 双通道 admission

最终配置使用两条因果路径：

- ORB sparse depth 直接 commit；
- DepthCov 的 log-depth 标准差转成 `confidence = 1 - std / valid_threshold`，`confidence >= 0.35` 直接 commit；
- 其余低置信 DepthCov proposal 必须累计三视图并经过 NPO-Lite。

这避免了“所有 residual 点都立即永久化”，也避免把已经高置信的深度无意义地等待三帧。20/40 帧 sweep 中，阈值 0.70 太保守；0.50 在 20 帧最好但 40 帧左侧容量不足；0.35 在 40/158 帧取得最稳定的质量-速度折中。

## NPO-Lite

第一版只门控 inverse depth `rho`。对每个 support patch 累积：

```text
H_rr = J_rho^T W J_rho
H_rx = J_rho^T W J_pose
H_xx = J_pose^T W J_pose

I_e = sum(H_rr - H_rx (H_xx + Sigma_pose^-1 + lambda I)^-1 H_xr)
risk = g_upper / max(I_e - curvature_margin, min_information)
```

`g_upper` 由 patch residual MAD、association instability 和 pose projected uncertainty 组成。当前 pose covariance 来源是 fixed diagonal prior，扰动顺序为 `[tx, ty, tz, rx, ry, rz]`，采用左乘 camera perturbation。日志和文档只称 empirical commitment risk，不称 formal certificate。

## Commit refinement 与 detail split

commit refinement 在固定历史 pose 下批量优化候选 inverse depth，最多修正 reference depth 的 20%。颜色融合只估计一个稳健的组级颜色偏移，并保留每个 proposal 的相对色差；旧实现把整个组覆盖为单一中值色，会直接抹掉纹理，已经修复。

detail split 只作用于已 commit、至少三视图支持且 residual/radius 达标的候选。一个 parent 由沿参考相机两个切向轴的四个 child 替代，child scale 为 0.55。侧向 candidate 在排序中有更高优先级。

## Budget 与 archive

`ActiveBudgetManager` 统计 parameter、gradient、Adam、candidate 和 archive bytes。archive 以可移除 group 为单位，保存 FP16 CPU means/scales/quats/opacities/SH 和 bbox；恢复时重建 GPU Parameter 并将 Adam moments 置零。完整 PLY 合并 active 与 archive，`render.py` 自动优先读取 `point_cloud_aerocommit_full.ply`。

当前限制：bootstrap 和 confidence fast-path 的大 host group 不能被拆分归档，因此 `max_active_trainable_gaussians=350000` 是 best-effort 而非严格上限。158 帧中实际 archive 4,326 GS，active 仍为 473,141；但 active parameters/Adam 和峰值显存都低于 immediate control。下一版应将 fast-path 按 metric chunk 建组，才能严格执行预算。

## 运行

```bash
export CUDA_HOME=/home/wmy/.local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/DepthCov-Modified:$PYTHONPATH"

# 原 MODP baseline
CUDA_VISIBLE_DEVICES=7 /home/wmy/anaconda3/envs/worldvln/bin/python \
  slam_new.py --config configs/aerocommit/baseline_road_street1.yaml \
  --seed 42 --exp_name baseline_full158

# NPO gate
CUDA_VISIBLE_DEVICES=7 /home/wmy/anaconda3/envs/worldvln/bin/python \
  slam_new.py --config configs/aerocommit/npo_gate_road_street1.yaml \
  --seed 42 --exp_name npo_gate_full158

# AeroCommit-MVP
CUDA_VISIBLE_DEVICES=7 /home/wmy/anaconda3/envs/worldvln/bin/python \
  slam_new.py --config configs/aerocommit/aerocommit_mvp_road_street1.yaml \
  --seed 42 --exp_name aerocommit_full158
```

渲染与评测：

```bash
/home/wmy/anaconda3/envs/worldvln/bin/python render.py \
  --run_dir RUN_DIR --device cuda:0 --skip_novel --skip_depth --skip_primitives

/home/wmy/anaconda3/envs/worldvln/bin/python scripts/evaluate_render_vs_gt.py \
  --run_dir RUN_DIR
```

## road_street1 实测

相同输入、pose、分辨率、200 次 post-refinement、seed 42。完整 158 帧含约 89 秒共同 cold gsplat JIT：

| 方法 | PSNR | SSIM | Edge PSNR | Side PSNR | 时间 | Active/Full GS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Immediate control | 20.276 | 0.6588 | 18.492 | 20.130 | 222.44 s | 514,647 |
| AeroCommit-MVP | 20.258 | 0.6537 | 18.496 | 20.095 | 230.44 s | 473,141 / 477,467 |

含 cold JIT runtime ratio 为 1.036；扣除共同 89 秒后估计为 1.060。AeroCommit PSNR 低 0.019 dB，edge PSNR 高 0.004 dB，active GS 少 8.1%。目标 PSNR 25 未达到；当前瓶颈仍是单目 depth/pose 和有限在线外观优化，而不是 coverage（两者均约 99.922%）。

失败消融必须保留：纯 NPO gate 20 帧为 17.176 dB；只加 ORB sparse fast-path 为 19.279 dB；粗 candidate 多视图 proposal 并集为 18.956 dB。它们说明 gate 必须与可信 depth fast-path、commit refinement 和 detail capacity 联合使用。
