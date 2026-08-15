# Codex 实施任务：Canonical WorldTest-GS

你现在位于仓库：

`/home/wmy/workspace_vla/Online-3DGS-Monocular`

请直接检查当前代码、数据转换产物和已有实验，然后实现、测试并运行下面的方法。不要只给方案或伪代码；要完成代码、配置、单元测试、200 帧实验、可视化检查和中文报告。当前 worktree 可能有用户修改，禁止回退、覆盖或清理与本任务无关的改动。

## 1. 任务目标

实现 **Canonical WorldTest-GS：单一世界坐标约束下的证据门控 Gaussian birth**。

本任务必须同时解决两个不同问题，但不能把二者混成一个贡献：

1. **确定性故障修复：** 修复 PanoAir seq1 中“COLMAP camera depth 被逐帧通过 RTK pose 放回世界坐标”导致的非持久几何，防止同一草地被写成多层并在起飞后表现为漂浮绿色 GS。
2. **研究机制：** 所有新 proposal 在成为永久 GS 前，都必须通过 shared-world 与 independent-view 的 World Identity Evidence `q_g` 检验。novelty 只主张这个 all-path、nuisance-marginalized、held-out predictive birth test；不要把坐标转换、弱视差 abstention、PMSA 或普通 delayed initialization 宣称为创新。

最终需要回答：漂浮是否由坐标不一致引起；修复坐标后是否消失；`q_g` 是否在 matched count/delay 条件下仍优于等待、少提交或随机提交。

## 2. 开始前必须阅读

- `docs/researchstudio_panoair_idea_diagnosis.md`
- `ideaspark_run/world-consistent-gaussian-admission/phase4/idea.std.zh.md`
- `ideaspark_run/world-consistent-gaussian-admission/phase4/idea.detail.en.md`
- `utils_new/aerocommit/manager.py`
- `utils_new/aerocommit/admission_policy.py`
- `utils_new/aerocommit/candidate_bank.py`
- `utils_new/aerocommit/npo_lite.py`
- `utils_new/aerocommit/dataset_geometry.py`
- `utils_new/gaussian_models.py`
- `utils_new/scene_mapper.py`
- 三个数据目录中的 `conversion_stats.json`：
  - `data/Online3DGS_PanoAir/seq1`
  - `data/Online3DGS_PanoAir/seq1_colmap_consistent`
  - `data/Online3DGS_PanoAir/seq1_rtk_triangulated_smoke200`

先检查当前 config inheritance。当前某些名为 `smoke200` 的配置可能已经继承到 `seq1_colmap_consistent`，不能根据文件名猜输入数据。

## 3. 不允许违反的世界坐标契约

新增一个明确的 `WorldFrameContract`，并让永久提交 API fail closed。

只允许以下两类 geometry mode：

### A. COLMAP canonical mode

- camera pose、sparse point、track identity 均来自同一个 COLMAP reconstruction。
- 若需要 RTK metric/world 朝向，只允许估计**一个全局固定 Sim(3)**，并将同一个 Sim(3) 同时作用于所有 COLMAP camera centers、camera orientations 和 3D points。
- 全局 Sim(3) 拟合误差很大时，地图仍保留在 COLMAP canonical world；RTK 只能作为诊断、弱 pose prior 或最终导出变换，不能逐帧替换 mapping pose。
- 不能为了贴合 RTK 而用随帧变化的变换分别移动已经具有同一 COLMAP point ID 的世界点。

### B. RTK canonical mode

- camera pose 使用 RTK pose。
- sparse geometry 必须由图像 track 在这些 RTK poses 下重新三角化，例如已有的 `seq1_rtk_triangulated_smoke200`。
- 禁止复用 COLMAP camera depth 后再通过 RTK pose 独立反投影。

原始 `seq1` hybrid mode 只能用于负向 stress test，默认不得产生永久 GS。若 `conversion_stats.json` 标记 `frame_local_reprojected`、pose/depth frame ID 不同、或持久性检查失败，则所有 permanent birth 必须 abstain 或程序在配置要求下明确报错，不能自动退回 fast path。

每个 proposal/group 至少携带：

`world_frame_id, geometry_mode, calibration_version, source_frame_id, track_id, pose_source, depth_source`

`commit_proposals` 或其唯一上层入口必须要求一个有效的 `AdmissionCertificate`。增加测试，证明任何 sparse、DepthCov、residual、track detail、surface detail 和 flow detail 路径都不能绕过该检查。

## 4. 先做坐标因果实验，不要直接调 admission

建立三个显式、不依赖隐式 inheritance 的 200 帧配置，并写入新的日志根目录 `Logs_worldtest_gs/`：

1. `hybrid_diagnostic_200`：原始 `seq1`，只用于复现和测量，不允许作为最终方法输入。
2. `colmap_canonical_200`：`seq1_colmap_consistent`，pose 与 sparse points 始终处于同一 canonical world。
3. `rtk_canonical_200`：`seq1_rtk_triangulated_smoke200`，RTK pose 与 RTK-pose triangulated points 配套。

先关闭 archive、frequency detail split、surface/flow/track detail 和 DepthCov permanent commit，使三组只比较坐标来源。固定 seed、帧、分辨率、优化步数和 commit budget。

增加 `scripts/audit_world_frame.py`，至少输出：

- 数据模式、pose/depth 来源和 contract 判定；
- 全局 Sim(3) trajectory alignment 的 RMSE/median/p90，但不能用这个指标代替点的持久性；
- 同一 track/point ID 在多帧得到的 world-point dispersion median/p95；
- 相邻帧可对应 sparse points 的 world distance median/p95；
- frame 120-150 分段统计；
- machine-readable JSON 和简洁 Markdown 表。

若数据没有保存 track ID，优先修改转换/读取流程保留原 COLMAP point ID；不能用最近邻距离冒充严格 identity，只能把最近邻结果标记为辅助指标。

**因果判定：** 如果 canonical 输入相对 hybrid 没有显著降低同轨迹 world dispersion，或者 canonical 模式仍在起飞时产生相同漂浮，则当前“pose-depth 坐标不一致是首因”的诊断不成立或不完整。此时先停下 admission 调参，定位投影 convention、camera-to-world/world-to-camera、尺度、外参或时间同步问题。

## 5. 实现 World Identity Evidence

建议新增 `utils_new/worldtest_gs/`，保持 AeroCommit 现有实现可作为 baseline，不做大范围无关重构。

### 5.1 ShadowGroup

所有 sparse、DepthCov 和 residual proposal 先形成临时 `ShadowGroup`。每条 observation 保存：

`frame_id, uv, inverse_depth, inverse_depth_variance, pose_id, pose_covariance, source_kind, track_confidence, rgb`

- sparse proposal 使用真实 COLMAP/ORB track ID；
- 无 track 的 DepthCov/residual proposal 使用 forward-backward optical flow 加现有 patch descriptor，cycle error 必须不大于 1.5 px；
- 每帧最多一条 observation；至少三个不同 keyframes 才能计算证据；
- 未提交 group 不能进入 `Parameter`、Adam、coverage、pruning、HashBlock、active budget 或 archive。

若已有 COLMAP track 已经带有至少三视图 observations，可离线/首次出现时计算同一个 `q_g` 并缓存 certificate。这是“缓存过的同一检验”，不是 sparse source fast path。

### 5.2 共享 nuisance

窗口共享而不是每个 proposal 独立拟合：

`T_i' = Exp((B_i c_w)^) T_i`

`rho_i' = exp(a_w) * rho_i + b_w`

- 至少四个 keyframes 时使用每四帧一个 knot 的 clamped cubic B-spline；不足四帧退化为常量修正；
- 固定第一帧 pose correction 消除 gauge；
- 用 pose/depth covariance 构造零均值 prior；
- 从窗口中全部有效 shadow tracks 用 variance-normalized Huber residual 和 Gauss-Newton 求 `theta_w=(c_w,a_w,b_w)` 的 MAP 与 covariance；
- 没有有效 tracks、未阻尼 information rank deficient、non-finite 时整个窗口 fail closed。

注意：nuisance 是为了吸收一致的低维校准误差，不允许每个 group 拥有独立 pose transform 来掩盖坐标不相容。

### 5.3 竞争模型和 `q_g`

对每个 group 轮流留出一个支持视图 `h`：

- `H_W`：其余视图共享一个世界点 `X_g`，并边缘化窗口共享 nuisance covariance；使用 Schur complement 得到 point covariance，再预测留出视图的 `(u,v,rho)`。
- `H_F`：留出视图拥有独立 latent point，不使用其他视图的信息；在与 `H_W` 完全相同的 proper scene prior 下积分。
- scene prior 只能由配置的 camera frusta、固定 near/far 和 camera envelope 构造，不能根据当前 reconstruction residual 调参。
- 固定 `kappa in {0.5, 1, 2}` 三个 prior scale，并取最保守结果。

计算：

`e_gh = log p(D_gh | H_W, D_g\h) - log p(D_gh | H_F)`

`q_g = min_{h,kappa} [e_gh - 1.645 * sqrt(v_gh)]`

只有 `q_g > log(19)` 才能签发 `AdmissionCertificate` 并创建永久 GS。少于三视图、未阻尼 information matrix 的 `lambda_min/lambda_max < 1e-6`、或任意 non-finite 情况均令 `q_g=-inf`。

不要把普通 reprojection threshold、DepthCov confidence、NPO risk 或 parallax threshold 政名为 `q_g`。`H_F` 不能先用被留出 observation 拟合后再给自己打分，否则 Bayes comparison 无效。为 Laplace predictive density 写一个小规模 Monte Carlo 数值对照测试，防止公式实现看似正常但校准错误。

### 5.4 可撤销影响

永久和 shadow splats 分开 rasterize。shadow 只在其 support window 中可见；每像素累计 shadow alpha 超过 0.1 时等比缩放到 0.1。分别保存 permanent render、shadow-assisted render 和 shadow mask。

正式 PSNR/SSIM、PLY 和 ground geometry 评测必须关闭 shadow。shadow 只解决在线当前视图暂时无覆盖的问题，不能被算作持久地图质量。

### 5.5 职责分离

- frequency score 只能在 commit 后决定 detail/split budget；
- archive 只能接收带有效 certificate 的 committed group；
- child splat 继承 parent certificate/provenance；
- 输出 active-only 与 committed-active-plus-archive 两份结果；
- 删除 `trusted_sparse_fast_path` 和 `trusted_depthcov_fast_path` 对永久状态的实际豁免。为兼容旧配置可以保留字段，但 WorldTest 模式下必须忽略并记录 warning。

## 6. 必须实现的对照和可证伪实验

先跑 40 帧单元/流程 smoke，再跑 200 帧。200 帧全部固定相同 seed、输入帧、分辨率、优化步数和 post-refinement。

### 坐标修复对照

- hybrid + current immediate/fast path；
- COLMAP canonical + 相同 admission；
- RTK canonical + 相同 admission。

这组只证明故障原因，不算 novelty。

### `q_g` 机制对照

在 COLMAP canonical 输入和 hybrid stress input 上分别运行：

- true `q_g` all-path gate；
- matched-delay：匹配 true `q_g` 每帧的等待时间分布；
- equal-count random：精确匹配每帧 commit 数；
- shuffled-`q_g`：只在同 age/source bucket 内打乱分数，并匹配每帧 commit 数；
- all-gated NPO-Lite；
- current fast path baseline。

第一轮关闭 frequency refinement 和 archive；只有几何结论成立后，再以完全相同 admission 结果重新启用二者，验证它们是正交模块。

### 核心指标

- frame 80-120、120-130、130-140、140-150、150-200 的 novel-view PSNR/SSIM；
- 可配置 lawn ROI 内 committed GS 的 RANSAC ground-plane point-to-plane median/p95，阈值默认 3 cm；
- 同一 track world dispersion 和跨帧重投影误差；
- future-view invalid commit rate：已提交 group 在至少两个后续可观测视图中的 normalized pixel/depth residual median > 3；
- 每帧 proposed/shadow/evaluated/committed/expired 数量；
- 按 source 统计 `q_g`、最差 held-out view、最差 prior scale、rank failure 原因；
- active/archive GS 数、真实 bytes、峰值显存、平均/峰值 keyframe latency；
- frame 120-160 的 render video、depth/opacity diagnostic 和 PLY 侧视图，人工确认是否还有绿色地面层悬空。

## 7. 成功条件与 kill switch

不能只用“看起来更好”判定成功。至少满足：

1. canonical world 相对 hybrid 显著降低同 track world dispersion，且 frame 130 附近不再突增；
2. COLMAP canonical 或 RTK canonical 的 200 帧结果不再出现明显悬浮草地层；
3. true `q_g` 在相同 commit count/delay 下，ground thickness、invalid commit rate 或 takeoff novel-view quality 明显优于 shuffled/equal-count/matched-delay；
4. consistent input 上不能通过大面积拒绝来换质量：同时报告 coverage 和 commit 数，PSNR 不得被选择性省略；
5. 所有永久 birth 均有 certificate，telemetry 中 bypass count 必须为 0。

以下任一结果都必须在报告中明确判定机制 claim 失败：

- canonical coordinate repair 已完全解决漂浮，而 true `q_g` 与 matched controls 等价；
- `q_g` 的收益只能由更少 GS 或更长等待解释；
- score 对 `kappa` 或数值 damping 极度敏感；
- hybrid 输入下 `q_g` 在起飞前后没有降低，仍大量提交错误地面；
- 为保证效果不得不恢复任何 source-based permanent fast path。

如果 `q_g` 被证伪，保留 canonical-world 工程修复，但不要继续声称 WorldTest-GS novelty。

## 8. 测试要求

至少增加以下自动化测试：

1. 一个固定 Sim(3) 同时变换 camera poses 和 points 后，投影保持一致；
2. 同一 canonical track 跨帧只对应一个 world identity；
3. pose/depth 人工不相容时 `q_g` 显著下降；
4. 一致三视图且视差充分时 `q_g` 通过；
5. 纯旋转/弱视差、rank deficient、non-finite 时 abstain；
6. Laplace predictive density 与小规模 Monte Carlo 近似一致；
7. sparse、DepthCov、residual 和 detail 路径不能无 certificate commit；
8. shadow 不注册参数、不进入 Adam/archive，且 per-pixel alpha cap 正确；
9. shuffled-score/equal-count control 精确匹配 commit count；
10. 现有相关测试无回归。

## 9. 实施顺序

严格按下面顺序推进，每一步失败先修复，不要直接开始长实验：

1. 审计 config inheritance、pose convention、数据 metadata 和当前 git diff；
2. 实现 world-frame audit 与三组 coordinate-only 40/200 帧对照；
3. 只有坐标诊断得到支持后，实现 `WorldFrameContract` 和 fail-closed commit certificate；
4. 实现 ShadowGroup、共享 nuisance 和 `q_g`，先通过 synthetic tests；
5. 接入所有 proposal/birth 路径并证明 bypass count=0；
6. 跑 40 帧 true/shuffled/matched controls；
7. 再跑 200 帧 takeoff 实验和视觉检查；
8. 结论成立后才允许跑完整 2368 帧，不要用 full run 代替机制验证。

运行 GPU 实验前先用 `nvidia-smi` 选择空闲 GPU，不要终止其他用户进程。不要覆盖已有 `Logs_progressive_uav_new`。

## 10. 最终交付

完成后提交以下内容：

- 实现代码、配置、测试和统一 benchmark 脚本；
- `docs/worldtest_gs_mvp.md`：数据流、坐标契约、公式、假设、复杂度和限制；
- `docs/worldtest_gs_experiment_report_zh.md`：命令、配置、日志路径、逐项结果表、frame 120-160 可视化、失败项和最终判断；
- machine-readable summary JSON；
- 可直接复现 40/200 帧实验的命令；
- `git diff --stat` 和测试结果。

最终回复必须先给出明确结论，格式如下：

1. **坐标故障是否修复：是/否，证据是什么。**
2. **漂浮地面是否消失：是/否，在哪些帧和视角检查。**
3. **`q_g` 是否优于 matched controls：是/否。**
4. **当前方法 claim：保留、收窄或放弃。**
5. **尚未解决的限制。**

不要因为代码完成或单元测试通过就宣称研究假设成立；研究结论只能来自上述因果对照和 falsification controls。
