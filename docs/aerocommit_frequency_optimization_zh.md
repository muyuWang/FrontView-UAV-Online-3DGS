# AeroCommit 高频清晰度优化方案

## 结论先行

当前模糊不是覆盖不足。现有 158 帧评测中，Immediate 与 AeroCommit 的覆盖率都约为 99.92%，但 AeroCommit 渲染的 Laplacian 方差约为 116，GT 约为 821。问题是同一高频残差可能同时来自位姿误差、逆深度误差和外观表达不足；如果不区分责任，继续 split 或加入更多点只会把错误平均得更密。

不建议第一步就加入稠密深度。先修复 DepthCov 的非方形图像高宽颠倒，再运行 SH1、all-frame pose replay 和高频损失的匹配预算基线。与此同时，在 shadow 模式验证 P/G/A 责任归因。只有当几何责任 `phi_G` 在留一视图后仍为正且主导 P/A 时，才应让新深度影响永久几何。

## 已确认的实现问题

`DepthCovEstimator.query()` 和 `query_tensor()` 原先把 PyTorch 的 `size=(H, W)` 写成了 `(W, H)`。对本数据的 `1600 x 896` 图像，模型实际接收的是 `H x W = 1600 x 896`，回采样的 covariance 也同样颠倒。DepthCov confidence 因此不能被当作可信的快速路径依据。修复后必须重新生成结果，旧的 `confidence >= 0.35` 消融不能直接沿用。

## 两条并行路线

### 路线 A：先做清晰度工程基线

实验配置为 `configs/aerocommit/aerocommit_frequency_road_street1.yaml`，默认保持 200-step post-refinement，以便与当前结果匹配。它只增加：

- `sh_degree: 1`，允许一阶视角相关颜色，降低跨视图颜色平均；
- `opt_cam: true` 和 `use_all_frames: true`，让历史 RGB 共同修正位姿与 Gaussian；
- edge/detail loss，把有限优化预算集中到自行车、树叶、栏杆和窗框等高频区域。

仓库已有的旧 progressive 消融显示，单独打开 pose 或 SH1 只有约 0.2 到 0.3 dB 收益；all-frame replay 与 pose、SH1、detail loss 联合后才有明显收益。因此不能把“SH0 改 SH1”单独写成方法贡献。

### 20 帧匹配预算结果

修复 DepthCov 后，使用相同 seed 42、输入、pose、20 帧和 200-step post-refinement 运行了两组：

| 方法 | PSNR | SSIM | edge PSNR | render LapVar | coverage |
|---|---:|---:|---:|---:|---:|
| AeroCommit SH0、局部窗口 | 20.176 | 0.663 | 18.743 | 87.31 | 99.673% |
| SH1、all-frame pose、detail loss | 20.597 | 0.683 | 19.247 | 79.29 | 99.732% |

新配置提升了 `0.421 dB` PSNR、`0.504 dB` edge PSNR 和 `0.020` SSIM，但 render LapVar 下降约 9.2%，而 GT LapVar 约为 465。它减少了部分边缘错位，仍然没有恢复高频能量。该结果支持“先做责任诊断”，不支持“SH1/all-frame 已解决模糊”。

### 路线 B：AeroCommit-F shadow 责任诊断

对每个有三视图 ORB 支撑的 `16 x 16` 高频 patch，构造三类临时干预：

- `P`：在 ORB pose covariance 内修正局部位姿；
- `G`：在 DepthCov/ORB 允许区间内修正逆深度；
- `A`：固定深度，只增加颜色、opacity 和切平面 footprint 的外观容量。

八个 coalition 从同一冻结状态开始，在 checkerboard fit 像素上使用相同预算拟合，在另一半像素上评分。三玩家精确 Shapley 将 held-out 高频损失下降分为 `phi_P`、`phi_G` 和 `phi_A`。几何只有在以下条件同时满足时才有资格进入后续 live admission 实验：

1. 至少三个历史帧提供几何内点 ORB 支撑；
2. `phi_G > 0` 且严格大于 `phi_P` 与 `phi_A`；
3. 删除任意一个支撑视图后，`phi_G` 仍为正；
4. shuffled-`phi_G` 在保持 commit 数量和时刻不变时退回 matched-delay 基线。

当前代码只实现 mutation-free shadow arithmetic 与 defect-injection 测试，不会修改 live Parameter、Adam 或 commit。原因是 IdeaSpark novelty audit 为 `advance`，但独立 implementability audit 仍要求补齐 P/G/A 的具体优化器、损失和最终 commit 来源。未通过 known-source defect attribution 前，不应把它接进在线 mapper。

IdeaSpark 的完整中文版卡片位于 `ideaspark_run/aerocommit-frequency-faithful-geometry/phase4/idea.std.zh.md`，实现审计位于同目录的 `phase4_implementability.json`。

## 是否需要稠密深度

### 可以先不需要的情况

如果路线 A 明显提高 edge PSNR/Laplacian variance，且 shadow 诊断主要得到 `phi_P` 或 `phi_A`，稠密深度不是主解。此时应优先优化 pose replay、外观阶数、曝光建模和 footprint，而不是给每个像素强加深度。

### 确实需要的情况

如果修复后的 DepthCov 仍出现以下现象，则稠密深度有价值：

- `phi_G` 在薄结构和纹理区域稳定主导；
- ORB 过稀，当前逆深度区间无法约束叶片、车轮辐条和远处立面；
- 几何重投影误差随时间系统累积，而提高 SH 阶数不能消除双边缘。

## 基于现有数据引入深度的方案

数据已有 158 张 RGB、相机内参、ORB pose 和逐帧 ORB sparse depth，足以构造不依赖 GT depth 的深度先验：

1. 用冻结的单目模型生成每帧 dense inverse depth。优先比较 Depth Anything V2 Metric Outdoor 与 UniDepth V2，而不是只选一个模型。
2. 在每帧使用 ORB sparse depth 对预测 inverse depth 做带 Huber 损失的 affine 对齐 `rho_aligned = a * rho_mono + b`。少于足够 ORB 内点时不产生强先验。
3. 用模型间分歧、ORB 对齐残差、图像边缘和跨帧 track 重投影误差组成逐像素 uncertainty；动态物体、天空、反光和遮挡边界设为低可信。
4. 将对齐深度只作为 `G` 的区间或正则，而不是直接提交 Gaussian。建议使用截断不确定度加权逆深度损失和梯度损失，且只作用于 `phi_G` 通过的区域。
5. 稠密深度必须加入三组控制：raw monocular depth、ORB-aligned depth、uncertainty-gated aligned depth。若第三组不能稳定优于前两组，说明收益来自额外监督而不是责任门控。

这条方案的关键不是“引入更强的深度网络”，而是让外部深度保持 ORB 的场景尺度，并把它限制在已经被反事实诊断判定为几何责任的区域。这样深度可以提高可观测性，但不会接管整个地图。

## 最小验证顺序

1. 修复 DepthCov 后重跑 20 帧 Immediate/AeroCommit，确认 confidence 分布和 fast-path 数量是否改变。
2. 用相同 seed、pose、commit budget 和 200-step refinement 跑 SH0 与新配置的匹配对照。
3. 在合成 P/G/A defect 上要求 top-1 attribution accuracy 大于 90%，且 live state hash 不变。
4. 再做真实 patch shadow replay，并报告 `phi_P/phi_G/phi_A` 分布与 leave-one-view 通过率。
5. 只有第 3、4 步通过后，运行 `phi_G`、matched-delay 和 shuffled-score admission；最后才添加 ORB-aligned dense depth。
