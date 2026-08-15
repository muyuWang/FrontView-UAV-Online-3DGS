# PanoAir 在线单目 3DGS：当前方法、故障诊断与 ResearchStudio 审查问题

## 1. 审查目标

本文件记录当前 AeroCommit/Frequency-Responsibility 方法的真实实现、PanoAir seq1
故障证据和待检验的因果判断。需要 ResearchStudio 独立回答：

1. 当前关于绿色地面 GS 在起飞后漂浮的诊断是否正确，是否遗漏了更主要的原因。
2. 不再保留严格 `P -> M -> S -> A` 状态机后，原本的核心研究问题和 novelty 是否仍成立。
3. 在在线、单目、弱视差、位姿和深度均可能有系统误差的条件下，能否形成一个更 solid、
   novel、可实现且可证伪的方法。

## 2. 当前希望保留的研究问题

原始设计使用 `P -> M -> S -> A` 表示 proposal、mutable、stable 和 archive。
此前的 novelty 审查认为，完整状态机过于宽泛，最可辩护的科学问题应收缩为：

> 在单目弱视差和 pose/depth 不确定条件下，一个候选 Gaussian 何时获得不可逆的永久地图写入权？

因此，PMSA 可以只是系统组织方式；真正的机制应是 observability-conditioned 或
world-consistency-conditioned irreversible admission。

## 3. 当前代码的实际数据流

当前 seq1 运行并不是严格 PMSA，而是：

1. MODP 在线优化 active GS。
2. 从每帧 COLMAP sparse points 和高 RGB 残差像素生成 proposals。
3. DepthCov 从 sparse depth 条件化估计部分半稠密深度和 confidence。
4. sparse proposal 直接进入 trusted fast path；DepthCov confidence >= 0.35 也可直接进入。
5. 每帧 fast path 最多永久提交 1200 GS。
6. 只有其余 proposal 进入跨帧 candidate association 和 NPO-Lite 风险门。
7. frequency score 主要改变采样和损失权重，不实际控制大部分永久提交。
8. active map 超过约 350k 后，历史 group 以 FP16 移入 CPU archive；archive 不再训练、
   prune 或 reactivate，最终完整 PLY 又无条件拼回。

`frequency_responsibility.py` 中实现了 Shapley geometry responsibility，但在线 admission
流程没有调用 `geometry_responsibility_decision`。因此当前名称中的 responsibility 主要不是
实际的写入责任机制。

## 4. PanoAir 数据坐标问题

正式 seq1 数据组合了：

- RTK GT position/quaternion 生成的 `T_camera_world`；
- COLMAP reconstruction 产生的每帧 sparse camera depth。

GT pose 导出时，转换程序先用 COLMAP visual pose 选取可见世界点，再把点变到当前 COLMAP
相机坐标，最后使用该帧 RTK pose 独立放回目标世界坐标。该过程只有在 COLMAP 轨迹和 RTK
轨迹可由一个稳定全局 Sim(3) 解释、且 COLMAP 深度尺度跨帧一致时，才保持持久世界几何。

现有证据：

- 正式 RTK/COLMAP 混合导出的轨迹对齐 RMSE 为 3.236 m，中位数为 2.733 m。
- COLMAP pose/point 保持一致的 300 帧导出，对齐 RMSE 为 0.038 m，中位数为 0.036 m。
- 混合导出相邻帧 point cloud 没有完全相同的世界点；COLMAP 一致导出通常保留约
  7,500-9,875 个相同点（每帧 10,000 点）。
- 混合导出在 120-129 帧的相邻点云 median nearest-neighbor distance 约为 0.0038 m，
  130-149 帧上升到约 0.0584 m。

待审查假设：同一个 COLMAP 视觉表面被逐帧重新嵌入不同的 RTK 世界位置，导致同一草地
形成多层或移动表面。单帧投影可以正确，但跨帧不是同一个持久世界平面。

## 5. 40 帧正常、200 帧起飞后失败的证据

RTK 相机轨迹本身数值平滑：

- frame 130 相对 frame 0 位移约 0.0037 m；
- frame 150 位移约 0.259 m；
- frame 199 位移约 1.001 m；
- 前 200 帧最大相邻位移约 0.0204 m。

相机约在 frame 130 开始实际起飞。重建指标同时发生突变：

| Frame range | Mean PSNR | Mean SSIM |
| --- | ---: | ---: |
| 80-120 | 29.04 | 0.952 |
| 120-130 | 28.30 | 0.944 |
| 130-140 | 22.47 | 0.717 |
| 140-150 | 21.65 | 0.586 |

待审查解释：40 帧处于近乎固定视角，只证明 RGB 外观可以拟合，无法验证三维世界一致性。
起飞产生视差后，之前重合的错误草地层才表现为绿色 GS 悬浮和建筑竖向拖影。

## 6. Admission 机制没有阻止故障

200 帧运行统计：

- total committed GS: 268,246；
- ordinary fast-path GS: 234,000；
- DepthCov-confidence fast-path GS: 59,088；
- candidate commits: 145；
- risk evaluations: 752；
- archived GS at frame 200: 0。

起飞附近：

- frame 128 的 NPO risk median 为 5.216，risk threshold 为 0.2；
- frame 132 的 NPO risk median 为 0.641；
- 两帧仍各自通过 fast path 永久提交 1200 GS。

待审查判断：NPO 在少数 candidate 上已经检测到 transition 风险，但主要 fast path 绕过
风险门，因此核心机制不是 load-bearing。仅关闭 DepthCov 也不足够，因为 sparse fast path
仍会大量永久写入。

## 7. 其他因素的当前判断

### Pose optimization

前 200 帧在线 pose correction 平均约 1.38 mm，最大约 4.53 mm；旋转最大约 0.42 度。
它不足以吸收 COLMAP-depth/RTK-pose 的系统不相容。后优化 `opt_cam=false`。

### Geometry optimization

训练的 depth、normal、distortion loss 权重均为 0；后优化没有启用 `use_all_frames`，并从
step 360 开始冻结 means、rotation 和 scale。优化能力不足会放大问题，但更多近视角 RGB
优化本身不能唯一确定正确几何，也可能巩固错误半透明表面。

### Frequency sampling

草地具有高梯度和持续 residual，因此被高频采样优先选择。frequency 当前主要决定“在哪里
多加点”，不决定几何是否可信，也没有缩小高频 GS footprint，因而可能加速错误草地写入。

### Archive

200 帧失败发生时 archive 数量为 0，因此 archive 不是故障起点。完整 2368 帧运行最终有
345,570 active GS 和 1,738,694 archived GS；active-only PSNR 为 16.147，拼回 archive
后的完整 PLY PSNR 为 13.873。archive 是完整序列的严重放大因素，但不是起飞时的首因。

## 8. 需要 ResearchStudio 判断的策略方向

请不要默认以下策略正确，而应结合近期文献和 novelty collision 独立审查：

1. 放弃“PMSA 状态机本身是贡献”的表述，将核心收缩为 world-consistency-conditioned
   irreversible Gaussian admission。
2. 所有来源（包括 sparse depth）只能提供 prior，不能凭来源直接获得永久提交豁免。
3. 为保持在线效率，允许 proposal 先以低影响、可撤销状态参与 rendering，但永久 commit
   必须通过跨帧世界一致性、视差、pose nuisance 消除和 leave-one-view-out stability 检查。
4. 将 DepthCov confidence 与 pose/depth coordinate consistency 分开建模；内部 confidence
   不能替代对系统偏差的检测。
5. frequency 只在几何通过 admission 后分配 detail/budget，不再充当几何可信度。
6. archive 是正交的存储状态；完整导出需要 confidence/visibility/consistency gate。

## 9. 新方法的硬约束和 falsification 要求

新策略应满足：

- 在线单目 UAV 3DGS；单卡可实现；不依赖测试时 dense GT depth。
- 可以使用 RGB、相机 pose、稀疏 SfM/ORB 深度及其观测轨迹；若引入学习深度，必须明确
  尺度、偏差和不确定度如何与世界坐标联合校准。
- 机制必须控制绝大多数不可逆提交，而不是仅处理少数 candidate。
- 必须给出可以推翻自身的实验，不只给常规 ablation。
- 最低控制组包括：world-consistent COLMAP pose/points、当前 RTK/COLMAP hybrid、
  all-gated admission、matched-delay、shuffled-score、equal-count/random admission、
  DepthCov-off、active-only 与 archive-full。
- 核心指标至少包括 takeoff 前后 novel-view PSNR/SSIM、地面厚度或高度漂移、跨帧重投影
  一致性、错误永久提交率、内存和在线耗时。

ResearchStudio 最终应只提出一个中心机制清晰的方法，并明确：最接近的已有工作、真正新增
的机制、成立所需假设、失败条件、计算成本，以及哪个实验结果出现时应放弃该 idea。
