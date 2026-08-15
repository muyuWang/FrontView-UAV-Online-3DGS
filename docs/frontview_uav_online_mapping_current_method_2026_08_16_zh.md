# 前视无人机自适应可观测性责任在线 3D Gaussian 建图

> 文档状态：当前最优代码方案的代码一致性说明
>
> 审计分支：`research/hashless-frontview-lod-7-24`
>
> 统一验证入口：`scripts/run_current_best_cross_scene_validation.py`
>
> 方法配置源：`configs/360dvo_coverage_recovery/mountains_uncertainty_bootstrap_repair_best.yaml`
>
> 讨论范围：只讨论 mapping。输入图像、内参、位姿和稀疏几何观测已经按时间顺序到达；tracking 和数据预处理不作为本文算法贡献。

## 1. 方法的一句话定义

当前方法是一套面向前视无人机视频的、无 HashBlock 的在线 Gaussian 建图系统。它不再用固定米制距离决定近场和远场，而是先在当前帧中自适应分配增密预算，再依据候选的几何可观测性，把 Gaussian 出生责任交给连续世界尺度覆盖 TSC 或投影责任 FPR；对有限深度证据不足但外观仍可靠的区域，则把几何责任与外观责任分离，并用有界的因果方向档案补充低视差外观。Gaussian 的高阶 SH 容量由在线梯度证据单调分配。

当前有效方法可概括为：

```text
Adaptive PBSD
  + observability-typed TSC/FPR
  + budget-derived projective responsibility
  + observability and residual conditioned footprint trust
  + finite-depth geometry/appearance dual responsibility
  + K96 causal directional archive with uncertainty bootstrap
  + TGBR-75
```

这里的核心不是把场景机械地切成两个距离层，而是让不同证据回答不同问题：

- 当前帧哪些深度域应获得出生预算，由当前候选的 log-depth 分布决定。
- 一个候选是否适合在世界坐标中承担重复抑制责任，由视差和深度精度决定。
- 一个 Gaussian 的有限深度是否应参与几何深度，由有限 SE(3) 投影和无限远 rotation-only 投影的竞争决定。
- 一个未被可靠几何解释的像素是否可由历史外观补充，由两个因果历史源的一致性决定。
- 一个 Gaussian 是否需要 SH3，由未激活高阶频带的持续梯度方向决定。

## 2. 任务设定

### 2.1 每帧输入

对时间 (t=0,1,\ldots,T-1)，映射器依次接收：

- RGB 图像 (I_t\in[0,1]^{H\times W\times3})；
- 相机内参 (K_t)；
- 世界到相机变换 (T_{cw,t})；
- 当前帧稀疏几何观测及其有效深度；
- 可选的稀疏点 ID 和世界坐标字段。

当前通用最优 launcher 明确设置：

```yaml
Model:
  frequency_sampling:
    preserve_sparse_track_geometry: false
```

因此，当前主 proposal 路径保留的是“稀疏来源”这一可靠性类别，而不是完整的 persistent-ID 身份契约：稀疏世界点先形成稀疏深度，再按当前相机反投影；`sparse_valid=true` 会保留，但 proposal track ID 通常为 `-1`。后文所有当前方法描述均按 `responsibility_basis=source` 展开，不能把它表述为 persistent-identity responsibility。

### 2.2 在线地图状态

时刻 (t) 的 Gaussian 地图为

\[
\mathcal G_t=\{G_i\}_{i=1}^{N_t},
\]

其中

\[
G_i=(\boldsymbol\mu_i,\mathbf s_i,\mathbf q_i,
\alpha_i,\mathbf c_i,d_i,q_i^{g},q_i^{a}).
\]

- (\boldsymbol\mu_i\)：世界坐标均值；
- (\mathbf s_i\)：三轴尺度；
- (\mathbf q_i\)：旋转四元数；
- (\alpha_i\)：opacity；
- (\mathbf c_i\)：最高到 SH3 的颜色系数；
- (d_i\in\{2,3\})：TGBR 分配的有效 SH 阶数；
- (q_i^g\)：几何深度责任置信度；
- (q_i^a\)：证书作用前的外观责任置信度。

除 Gaussian 参数外，系统维护：

- TSC 的有界增量 KD-tree 覆盖表；
- 每个 Gaussian 的出生来源、深度不确定性和 footprint 约束；
- TGBR 的可见次数和高阶梯度 EMA；
- 最多 96 个常规方向锚点和最多 48 个不确定性启动锚点；
- 覆盖恢复的新生组状态。

### 2.3 输出

严格的几何输出是 `point_cloud.ply`，包含 Gaussian 几何、opacity、SH 和两类深度责任置信度。方向档案单独保存为：

```text
frontview_directional_layer.pt
```

最终图像渲染使用 Gaussian 地图和该 sidecar。方向档案不增加 Gaussian，不修改 Gaussian 几何，也不反馈控制后续 Gaussian 出生。

### 2.4 在线边界

以下过程只使用当前帧和历史帧，且不知道序列总长度：

- 候选生成与不确定性过滤；
- Adaptive PBSD；
- TSC/FPR 路由与准入；
- footprint trust；
- 有限深度证书；
- 关键帧优化与覆盖恢复；
- TGBR 证据更新；
- 方向锚点采集与有界压缩。

当前实验协议在序列结束后还有固定 100 步 post refinement。论文中应分别报告“最后一帧完成时的 online stage”和“固定 100 步后的 post-refinement stage”，不能把后优化时间隐藏在在线 FPS 之外。

## 3. 前视 UAV 的核心困难和设计动机

### 3.1 透视导致出生预算失衡

前视 UAV 图像中，近处地面、道路边缘和树枝通常占据更多像素并产生更强残差。若直接在全部高残差像素中取固定数量，近场会长期占满 DepthCov 补点预算，山体、天空边界和道路远端缺乏表示。

固定的 20 m、50 m 或 80 m 分界也不稳健。相同的米制距离在不同相机焦距、飞行高度、场景尺度和轨迹阶段中对应完全不同的投影难度，并且在线系统不应依赖完整序列的尺度统计。

因此当前方案在每个关键帧上只依据当时的 DepthCov 幸存者，学习三个 log-depth 域，再做等机会预算分配。这是 Adaptive PBSD 的动机。

### 3.2 远场 XYZ 不稳定，但图像责任仍然稳定

对深度 (z) 很大的候选，较小的逆深度误差会转化为很大的世界坐标偏移。同一山体像素可以在连续帧中被反投影到不同 XYZ；若所有候选都使用固定世界哈希占用，可能出现：

- 错误深度落入不同体素，导致同一结构重复出生并形成多层漂浮；
- 粗体素过早判定占用，远景缺失无法补充；
- 米制边界改变时，责任规则突然跳变。

前视运动又提供了一个更自然的判断量：候选在历史视角中的视差能否分辨其 Gaussian footprint 和深度不确定性。当前方法据此把责任类型化：可度量的候选交给世界空间 TSC，低可观测且处于当前最远 log-depth 域的候选交给投影空间 FPR。

### 3.3 远景外观可观测，不等于有限深度可观测

天空、极远山体和低纹理区域可以具有稳定颜色，但有限深度往往不可辨识。把“有限深度不可信”直接等同于“该区域没有任何有效信息”，会让方向外观层失去所有权；反过来，让这些点以完整权重进入几何深度，又会制造近天空和漂浮深度。

因此当前方案为同一个 Gaussian 维护两种责任：

- 几何责任 (q^g)：决定该 Gaussian 对 metric depth 和几何判断贡献多少；
- 外观责任 (q^a)：保留其作为可见外观解释的能力。

有限深度证书只衰减 (q^g)，不衰减 (q^a)。这就是 geometry/appearance responsibility decoupling。

### 3.4 相邻历史帧平均会制造拖影

方向层曾使用多个历史源平均或跨段平滑。对山脊和画面边界，这会把两个不同投影位置同时写入当前帧，表现为山体重影、边缘闪烁和场景断层。

当前方向档案保留两个源做一致性认证，但只使用最匹配的第一个源提供颜色。档案按稀疏掉线 episode 管理，并用时间有序 Ward 合并保持时序局部性，避免跨很长时间段混合外观。

### 3.5 高阶外观容量不应平均分配

前视 UAV 的观察方向通常连续而狭窄。大量 Gaussian 使用 SH2 已足够，只有少数边缘、反光或方向性纹理需要 SH3。全部使用 SH3 会给证据不足的点分配高频自由度，全部使用 SH2 又可能欠拟合。

TGBR 用在线反向传播中未激活 SH3 频带的持续下降方向选择晋升对象，在最多 75% 的预算内把 SH3 分给真正有证据的 Gaussian。

## 4. 总体数据流

```text
RGB + K + W2C pose + sparse geometric observations
                         |
                         v
                frame / keyframe decision
                         |
            +------------+-------------+
            |                          |
       non-keyframe                  keyframe
            |                          |
   sparse-source proposals       optimize current map
            |                    for a fixed 10 steps
            |                          |
            |                 geometry_render residual
            |                          |
            |                  DepthCov candidate pool
            |                          |
            |               uncertainty survivor filter
            |                          |
            |                  Adaptive PBSD budget
            |                          |
            +------------+-------------+
                         |
                 merged proposal batch
                         |
          finite-depth observability measurements
                         |
        +----------------+------------------+
        |                                   |
 metric/source-resolved              unresolved farthest
        |                               log-depth regime
        v                                   v
 TSC continuous world cover       FPR image/log-depth cells
        |                                   |
        +----------------+------------------+
                         |
              footprint trust bound
                         |
            finite-depth certificate
                         |
              immediate map commit
                         |
         future unified rendering/optimization
                         |
       TGBR online SH2 -> SH3 capacity allocation
                         |
       causal directional archive only at output render
```

覆盖恢复位于 frame/keyframe decision 之后：当稀疏点不足导致普通关键帧检查拒绝一帧时，它根据时间间隔、位姿变化和未解释图像比例重新认证该帧；认证成功后仍进入同一套 DepthCov、Adaptive PBSD、TSC/FPR 和 commit 路径。

## 5. 一帧的真实执行顺序

### 5.1 初始化阶段

当前方法使用 4 个初始化帧，每次进行 10 步优化。地图不足以提供可靠 residual 时，补点从图像平面候选中产生；地图建立后，补点集中于高残差、低覆盖且不与稀疏来源冲突的区域。

### 5.2 普通关键帧

关键帧执行顺序是“先优化已有地图，再从剩余误差出生新 Gaussian”：

1. 选择当前活动窗口和全局窗口。
2. 对现有 Gaussian 地图运行 10 个优化步骤。
3. 用基础 Gaussian 图像 `geometry_render` 渲染当前帧。
4. 计算渲染与真值之间的结构残差。
5. 收集稀疏来源候选。
6. 在高残差、非稀疏覆盖区域构造最多 (2B) 个 DepthCov 输入像素，其中 (B=3200)。
7. DepthCov 输出深度和标准差，并先执行不确定性过滤。
8. Adaptive PBSD 从过滤后的幸存者中选最多 (B) 个。
9. 合并稀疏来源和 DepthCov 来源。
10. 计算视差、投影半径、log-depth 标准差和责任类型。
11. TSC 或 FPR 抑制重复出生。
12. footprint trust 设置新生尺度扩张上限。
13. 有限深度证书生成 (q^g) 和 (q^a)。
14. 幸存者立即加入永久 Gaussian 地图。
15. 更新 TGBR 证据和方向档案观测。

新生 Gaussian 不会回头改变本帧用于出生决策的 residual，因此方向档案的输出也不会形成自我强化的出生闭环。

### 5.3 非关键帧

非关键帧不运行完整 DepthCov 增密，只处理稀疏来源 proposal。若存在覆盖恢复新生组，则按配置每 5 帧执行 1 步轻量跟随优化。

### 5.4 序列结束

系统先保存 online-stage Gaussian 和方向 sidecar，再运行固定 100 步 post refinement，最后激活方向层并保存最终地图。当前 post refinement 不优化相机位姿。

## 6. 候选生成和初始化

### 6.1 候选来源

当前主路径有两类候选：

1. `sparse`：由当前稀疏几何观测形成，深度先验置信度为 1。
2. `depthcov*`：从高残差像素经 DepthCov 得到，带连续深度置信度。

DepthCov 候选必须先通过 `std_valid_threshold=0.06`。当前代码把幸存者置信度写为由深度标准差导出的后验量，并在后续用于 responsibility 和 footprint trust。

### 6.2 反投影

对像素 (mathbf u=[u,v,1]^\top) 和深度 (z)：

\[
\mathbf x_c=zK_t^{-1}\mathbf u,
\qquad
\mathbf X_w=T_{cw,t}^{-1}[\mathbf x_c^\top,1]^\top.
\]

当前 sparse-source 主路径同样按稀疏深度和相机位姿恢复 proposal 世界坐标。它不使用 persistent ID 做重复出生判断。

### 6.3 Gaussian 初始尺度

设平均焦距为

\[
\bar f=\frac{f_x+f_y}{2}.
\]

当前初始化使用随透视尺度增长的 log scale：

\[
\ell_0=\log\left(\frac{0.5z}{\bar f}\right)+0.5.
\]

颜色取曝光校正后的当前像素，初始 opacity 为 0.5。模型物理张量支持到 SH3，但有效外观容量以 SH2 出生。

## 7. Adaptive PBSD：当前帧自适应透视平衡增密

### 7.1 模块定义

**名称**：Adaptive Perspective-Balanced Survivor Densification

**位置**：DepthCov 不确定性过滤之后，TSC/FPR 路由之前。

**实现**：

- `utils_new/frontview_sampling.py::_lloyd_log_depth_regimes`
- `utils_new/frontview_sampling.py::adaptive_log_depth_indices`
- `utils_new/gaussian_models.py::propose_new_gaussians`

### 7.2 输入和输出

输入：

- 当前帧 DepthCov 幸存者深度 (z_i)；
- 置信度 (c_i)；
- residual (r_i)；
- 固定出生预算 (B)。

输出：最多 (B) 个 DepthCov 幸存者下标。稀疏来源候选不占用这部分配额。

### 7.3 为什么在 survivor 上分配

旧式做法先按原始像素分近中远预算，再做深度不确定性过滤。某个深度域若大量候选被过滤，最终有效预算会再次失衡。

当前顺序是：

```text
high-residual pool -> DepthCov -> uncertainty filter
                   -> Adaptive PBSD on survivors
```

因此 PBSD 平衡的是能够真正进入后续准入的候选，而不是未经验证的像素。

### 7.4 log-depth Lloyd 量化

对幸存者定义

\[
y_i=\log z_i.
\]

当前帧学习三个中心 (c_k)：

\[
\min_{\{c_k\},a_i}
\sum_i(y_i-c_{a_i})^2,
\qquad a_i\in\{1,2,3\}.
\]

实现细节：

1. 用 (1/6,3/6,5/6) 分位数初始化三个中心。
2. 交替执行最近中心分配和域内均值更新。
3. 每次更新后对中心排序。
4. 标签稳定时停止，最多 32 次。

使用 log depth 的直接意义是对全局尺度具有等变性。若所有深度同时乘以常数 (s)，则

\[
\log(sz_i)=\log z_i+\log s,
\]

所有中心只发生平移，候选所属深度域不变。算法不需要知道整段轨迹长度，也不依赖固定 20/50 m 边界。

### 7.5 capped water-filling

设三个域的实际容量为 (n_k)。当前实现通过 capped water-filling 分配配额 (b_k)：

\[
\sum_k b_k=\min(B,\sum_k n_k),
\qquad 0\le b_k\le n_k.
\]

算法反复把剩余预算均分给尚未饱和的域；小域被完整保留后，剩余预算自动流向其他域。它近似求解固定预算下的 max-min 域覆盖，不再保留旧的 25%/45%/30% 固定比例。

当前 `selection_mode=adaptive_log_depth_random`，因此域内随机选取，随机种子由配置固定。这里刻意不让 residual 再次主导域内排序，避免近场强纹理通过 residual 权重重新夺回预算。

### 7.6 自适应边界

用于日志的边界由相邻 log-depth 中心的中点得到：

\[
b_k=\exp\left(\frac{c_k+c_{k+1}}{2}\right).
\]

这些边界每个关键帧重新估计，只反映当前可用候选，不读取未来帧。

## 8. 可观测性类型化的 TSC/FPR 路由

### 8.1 模块动机

TSC 和 FPR 不是“近景地图”和“远景地图”。二者管理的是不同的出生责任坐标系：

- TSC 回答“已有世界空间表示是否已覆盖这个候选”。
- FPR 回答“当前低可观测候选池中，哪些投影和 log-depth 单元已经有代表”。

当前路由使用：

```yaml
FrontViewFarField:
  routing_mode: adaptive_observability
  responsibility_basis: source
```

### 8.2 来源责任基础

在当前 `source` 模式下：

\[
L_i=\neg\texttt{sparse\_valid}_i,
\]

即只有非稀疏来源候选缺少先验 metric owner。所有 sparse-source 候选都走 TSC，与其距离无关。

这不是 persistent-ID 路由。`sparse_track_identity_enabled` 在当前有效配置中为 false。

### 8.3 因果视差

对候选世界点 (\mathbf X_i)，在当前相机与历史参考相机之间计算单位视线夹角 (\theta_{ij})。投影视差近似为：

\[
p_i=\max_j \bar f\sin\theta_{ij}.
\]

只有候选在历史参考帧中可见时，该参考才贡献视差。

设候选当前投影半径为 (r_i)，DepthCov 的 log-depth 标准差为 (\sigma_{\log z,i})。当前 metric certificate 使用两个条件：

\[
\text{observable}_i: p_i\ge r_i,
\]

\[
\text{depth-precise}_i:
p_i\sigma_{\log z,i}\le r_i.
\]

因此：

\[
M_i=\text{observable}_i\land\text{depth-precise}_i.
\]

第一项要求基线视差至少达到一个 Gaussian footprint；第二项要求深度不确定性造成的视差扰动仍落在该 footprint 内。二者同时成立时，DepthCov 候选可在 metric 世界空间中接受 TSC 管理。

### 8.4 自适应远场责任

未获得 metric certificate 的候选为：

\[
U_i=L_i\land\neg M_i.
\]

当前方法不把所有 (U_i) 都交给 FPR，而是在全部 (L_i) 候选的当前帧 log-depth 上再次运行三类 Lloyd 量化，只把最远中心对应的 Voronoi 域记为 (F_i)。最终 FPR mask 为：

\[
R_i^{FPR}=U_i\land F_i.
\]

其余候选进入 TSC。这个设计避免把中近景中暂时视差不足的候选全部降级为投影责任，同时消除了固定 `depth >= 80 m` 的在线先验。

### 8.5 路由伪代码

```python
lacks_metric_owner = ~sparse_valid
observable = parallax_px >= projected_radius
depth_precise = parallax_px * sigma_logz <= projected_radius
metric_certified = observable & depth_precise

adaptive_far = farthest_lloyd_log_depth_regime(
    depths[lacks_metric_owner]
)
fpr = lacks_metric_owner & ~metric_certified & adaptive_far
tsc = ~fpr
```

## 9. TSC：连续世界尺度覆盖

### 9.1 模块定义

**名称**：Target-Scale Cover

**实现**：`utils_new/frontview_scale_cover.py::FrontViewScaleCover`

**位置**：候选责任路由之后、commit 之前。

### 9.2 为什么不用 HashBlock

固定 HashBlock 把三维空间离散到预设多层网格。其占用行为依赖场景尺度和 hash level，且同一个空间格中无法直接表达“已有 Gaussian 的尺度是否足以覆盖当前候选”。

TSC 把准入问题写成连续邻域覆盖：地图中每个已提交 Gaussian 在出生时登记世界位置、目标尺度和颜色；新候选只与附近连续坐标邻居比较。

当前配置为：

```yaml
HashBlock:
  use_hash: false
FrontViewScaleCover:
  enabled: true
  query_backend: scipy_kdtree
  radius_multiplier: 0.3
  scale_compatibility: 1.0
  neighbors: 32
  rebuild_rows: 8192
  color_distance_threshold: 0.15
  target_size_mode: view_scale
  distance_mode: fixed_radius
```

### 9.3 覆盖判定

设当前候选的目标尺度为 (s_i)。当前 `view_scale` 模式使用该相机视角对应的连续尺度；查询半径为：

\[
\rho_i=0.3s_i.
\]

从 KD-tree 取最多 32 个邻居。邻居 (j) 只有同时满足下列条件，才能覆盖候选 (i)：

\[
\|\boldsymbol\mu_i-\boldsymbol\mu_j\|_2\le\rho_i,
\]

\[
s_j\le 1.0s_i,
\]

\[
\|\mathbf c_i-\mathbf c_j\|_2\le0.15.
\]

若至少有一个兼容邻居，则拒绝该候选；否则允许出生。

尺度条件的含义是：一个比目标尺度更粗的旧 Gaussian 不能仅因中心靠近就阻止更细表示出生。这样 TSC 同时承担空间去重和连续尺度 refinement。

### 9.4 增量组织

新提交记录先进入 pending buffer；累计 8192 行后重建基础 `cKDTree`。查询同时覆盖基础树和 pending 点，因此不需要固定体素层级。TSC 的世界覆盖状态随 commit 增长，但不依赖序列总帧数。

### 9.5 当前没有启用的 TSC 扩展

当前最优配置没有启用：

- persistent sparse-track identity bypass；
- dynamic parent-child handoff；
- directional ownership；
- evidence quota routing；
- active occupancy lifecycle。

论文主方法不应把这些代码中存在但当前关闭的实验路径写入算法。

## 10. FPR：预算推导的投影责任

### 10.1 模块定义

**名称**：Far-field Projective Responsibility

**实现**：

- `utils_new/frontview_far_field.py::budget_cell_parameters`
- `utils_new/frontview_far_field.py::projective_survivor_mask`

### 10.2 投影单元不是手工像素阈值

设图像面积为 (HW)，每个关键帧 DepthCov 出生预算为 (B)，候选池倍率为 (m)。若把图像均分为约 (Bm) 个责任单元，则单元边长为：

\[
s_{xy}=\sqrt{\frac{HW}{Bm}}.
\]

当前 (B=3200,m=2)。图像分辨率变化时，单元大小自动变化，保持“每个候选池预算对应一个投影责任单元”的解释。

DepthCov 的最大 log-depth 标准差为 (\sigma_{\max})。两次独立深度估计的差异标准差为：

\[
s_{\log z}=\sqrt{2}\sigma_{\max}.
\]

因此每个 FPR cell 是：

\[
\left(
\left\lfloor\frac{u}{s_{xy}}\right\rfloor,
\left\lfloor\frac{v}{s_{xy}}\right\rfloor,
\left\lfloor\frac{\log z}{s_{\log z}}\right\rfloor
\right).
\]

### 10.3 单元竞争

在同一个 `(image cell, log-depth cell)` 内，按 residual 分数排序，只保留最强候选。这样远景不会因为微小 XYZ 波动在同一帧候选池中重复出生，也不会依赖固定世界体素大小。

当前配置为：

```yaml
FrontViewFarField:
  projective_nms_mode: budget_cells
  map_redundancy_gate: false
  ray_atlas_enabled: false
```

因此当前 FPR 是“出生时、当前 proposal batch 内的投影责任竞争”，不是跨帧持久 ray atlas，也不是动态远近层地图。通过 FPR 的 Gaussian 仍立即进入同一张永久世界 Gaussian 地图，后续不会因为相机接近而自动迁移到 TSC。

这一定义很重要：当前贡献是自适应责任路由和出生竞争，不应声称已经实现完整的动态远近 LOD handoff。

## 11. Footprint Trust：可观测性约束的尺度成长

### 11.1 动机

仅控制“是否出生”还不够。低视差远场点即使深度错误，也可能通过不断扩大 Gaussian 尺度覆盖大量像素，从而短期降低损失但形成厚重漂浮层和模糊山体。

Footprint trust 把候选的视差可分辨性、深度不确定性、出生预算和局部图像细节结合起来，限制新生 Gaussian 后续可以扩大多少。

当前模式：

```yaml
FrontViewFarField:
  footprint_trust_scope: all_depthcov
  footprint_trust_mode: certificate_residual_rd_visible_detail
```

### 11.2 几何信息瓶颈

对候选 (i)，定义：

\[
m_i^{parallax}=\frac{p_i}{r_i},
\]

\[
m_i^{precision}=\frac{r_i}{p_i\sigma_{\log z,i}},
\]

\[
I_i=\mathrm{clip}left(
\min(m_i^{parallax},m_i^{precision}),0,1
\right).
\]

- (p_i/r_i) 小，表示视差没有超过 Gaussian footprint，有限深度不可分辨。
- (r_i/(p_i\sigma_{\log z,i})) 小，表示深度不确定性在视差域中超过 footprint。
- 二者较小者决定几何信息瓶颈。

### 11.3 residual rate-distortion 半径

对局部 residual 梯度能量 (E_i)，分段常数重建的最优采样密度满足：

\[
\rho_i\propto\sqrt{E_i}.
\]

单个样本负责的面积与密度成反比，因此责任半径满足：

\[
a_i\propto E_i^{-1/4}.
\]

实现对 (a_i) 做 RMS 归一化，使候选总体责任面积不因重新分配而膨胀。`visible` 模式用当前 opacity 对能量加权；`detail` 模式把半径因子上限设为 1，防止高细节位置被扩大得比基础预算单元更粗。

FPR 候选的责任半径为：

\[
R_i=\frac{1}{2}s_{xy}a_i.
\]

非 FPR 的 DepthCov 候选仍接受几何信息约束，但其 (a_i=1)。

### 11.4 尺度扩张上限

设当前投影半径为 (r_i)，可用空间比为：

\[
\lambda_i=\max\left(\frac{R_i}{r_i},1\right).
\]

当前 certificate 模式把信息置信度转为 odds：

\[
h_i=\frac{I_i}{1-I_i}.
\]

最终尺度扩张倍数上限为：

\[
L_i=\exp(h_i\log\lambda_i).
\]

当 (I_i=0) 时 (L_i=1)，该 Gaussian 不允许扩张；当 (I_i\rightarrow1) 时约束逐步释放。该约束只限制尺度扩张，不通过修改深度来伪造更好的几何。

## 12. 有限深度证书

### 12.1 需要区分的两个假设

对当前候选像素 (\mathbf u_i) 和候选深度 (z_i)，历史参考帧中存在两种解释：

1. **有限深度假设**：按 (z_i) 反投影世界点，再用完整 SE(3) 投影到参考帧。
2. **无限远假设**：只保留当前视线方向，用相机旋转投影到参考帧，忽略平移。

若二者在参考帧中的位置几乎相同，则当前运动不足以区分有限深度和无限远，系统应 abstain，而不是强行判定深度错误。

### 12.2 图像误差

当前图像和参考图像先做曝光归一化和 (3\times3) 平均平滑。对参考帧 (j)：

- 有限投影颜色误差为 (e^f_{ij})；
- 无限远投影颜色误差为 (e^\infty_{ij})；
- 两种投影位置距离为 (\Delta u_{ij})。

### 12.3 可观测性和有限深度优势

像素噪声尺度当前为 (\sigma_{px}=1)：

\[
O_{ij}=1-\exp\left[-\frac12
\left(\frac{\Delta u_{ij}}{\sigma_{px}}\right)^2\right].
\]

有限深度相对优势为：

\[
A_{ij}=\mathrm{clip}\left(
\frac{e^\infty_{ij}-e^f_{ij}}
{\max(e^\infty_{ij},10^{-6})},0,1
\right).
\]

当前实现跨参考帧取：

\[
O_i=\max_j O_{ij},
\qquad
S_i=\max_j(O_{ij}A_{ij}).
\]

有限深度证书为：

\[
C_i^f=\mathrm{clip}(1-O_i+S_i,0,1).
\]

其行为具有明确解释：

- 若 (O_i\approx0)，有限和无限投影不可区分，(C_i^f\approx1)，证书 abstain 并保留先验。
- 若 (O_i\approx1) 且有限投影明显更好，(S_i\approx1)，保留有限深度。
- 若 (O_i\approx1) 但有限投影没有优势，(S_i\approx0)，抑制该候选的 metric 深度责任。
- 若没有有效参考视图，证书返回 1，不凭缺失证据惩罚候选。

实现位于 `utils_new/frontview_dual_responsibility.py::causal_finite_depth_certificates`。

## 13. 几何和外观责任解耦

### 13.1 先验责任

候选进入地图时先得到来源/深度后验责任 (q_i^{prior})：

\[
q_i^{prior}=
\begin{cases}
1, & \text{sparse or tracked metric source},\\
c_i, & \text{DepthCov source},\\
0, & \text{unresolved depth fallback}.
\end{cases}
\]

当前 depth fallback 关闭，所以第三类在最优配置中通常不出现。

### 13.2 两类责任

有限深度证书只作用于几何：

\[
q_i^g=q_i^{prior}C_i^f,
\]

\[
q_i^a=q_i^{prior}.
\]

`finite_depth_certificate_scope=all_unverified` 表示除明确 `tracked_metric` 外的未验证候选都可以接受证书；当前 source-based sparse proposal 也可能被证书检查。

### 13.3 单次 packed rasterization

当前实现没有为两类责任渲染两次。`heterogeneous_sh_rasterization` 在同一次 projection、tile intersection 和 alpha compositing 中附加多组通道：

\[
D_g=\frac{\sum_i T_i\alpha_i q_i^g z_i}
{\sum_i T_i\alpha_i q_i^g},
\qquad
M_g=\sum_iT_i\alpha_iq_i^g,
\]

\[
D_a=\frac{\sum_i T_i\alpha_i q_i^a z_i}
{\sum_i T_i\alpha_i q_i^a},
\qquad
M_a=\sum_iT_i\alpha_iq_i^a.
\]

此外保留普通 RGB、全体条件期望深度和 uncertainty mass。它们共享一次 `fully_fused_projection`、一次 `isect_tiles` 和一次 `rasterize_to_pixels`。

### 13.4 决策隔离

render package 同时返回：

```python
{
    "render": directional_composited_colors,
    "geometry_render": base_gaussian_colors,
    "metric_depth": D_g,
    "metric_opacity": M_g,
    "appearance_metric_depth": D_a,
    "appearance_metric_opacity": M_a,
}
```

所有 mapping 决策统一调用：

```python
geometry_decision_render(render_package)
```

因此下列过程只读取基础 Gaussian 解释，不读取方向层修改后的图像：

- photometric loss；
- 关键帧误差更新；
- residual 选点；
- DepthCov 补点区域；
- 覆盖恢复证书；
- post refinement loss。

方向层使用 (D_a,M_a) 决定外观所有权。这样有限深度证书可以清理错误 metric depth，又不会错误地把稳定天空外观标记成“完全未解释”。

## 14. K96 因果方向档案

### 14.1 定位

方向档案不是第二张 3D 地图，也不是 tracking 模块。它是一个有界的在线历史图像 sidecar，只在 Gaussian 几何责任不足且历史重投影一致时补充最终颜色。

实现：`utils_new/frontview_directional_layer.py::FrontViewDirectionalLayer`

当前配置：

```yaml
FrontViewDirectionalLayer:
  enabled: true
  sparse_point_threshold: 10
  max_anchors: 96
  min_anchors: 2
  anchor_selection_mode: episode_ordered_ward
  pose_score_mode: rendered_inverse_depth
  warp_mode: se3_fallback
  source_fusion: first
  geometry_gate_mode: metric_transmittance
  consistency_threshold: 0.12
  blend_weight: 1.0
```

### 14.2 稀疏掉线 episode

当当前帧稀疏点数小于 10 时，进入 sparse-dropout episode，并采集当前图像、位姿、内参和曝光为锚点。稀疏点恢复后，当前 episode 结束，常规锚点清空；后续掉线建立新 episode。

这使方向档案只描述连续的低几何可观测阶段，避免把两个相隔很远、场景内容已变化的掉线阶段混在一起。

### 14.3 时间有序 Ward 压缩

episode 内每个低稀疏帧都可成为候选锚点。超过 96 个时，只允许合并时间相邻的两个 segment。

位姿距离为：

\[
d(i,j)=\theta(R_iR_j^\top)
+\frac{\|\mathbf c_i-\mathbf c_j\|_2}{s_c},
\]

其中 (s_c) 是在线相机中心方差得到的平移尺度。相邻 segment (A,B) 的 Ward 代价为：

\[
\Delta(A,B)=\frac{|A||B|}{|A|+|B|}d(A,B)^2.
\]

每次合并代价最小的相邻 pair，并保留较大 segment 的代表帧。这样内存上界固定，又保留时间 episode 的结构。

### 14.4 不使用固定远深度的锚点选择

渲染目标帧时，以当前 appearance depth 的 opacity 加权平均逆深度：

\[
\bar\rho_t=
\frac{\sum_{u}M_a(u)/D_a(u)}{\sum_uM_a(u)}.
\]

锚点到目标的选择分数为：

\[
s(j,t)=\theta(R_jR_t^\top)
+\|\mathbf c_j-\mathbf c_t\|_2\bar\rho_t.
\]

这把平移换算成当前已渲染场景的角度漂移，不再需要手工设一个 80 m 远深度来平衡旋转和平移。

### 14.5 rotation-only 和 finite-depth 双 warp

对两个最匹配的历史锚点，分别计算：

- rotation-only warp：适合天空和近似无限远结构；
- finite-depth SE(3) warp：使用当前 (D_a)，适合可观测有限深度结构。

`se3_fallback` 的选择逻辑是：

```text
if two rotation-only warps agree:
    use rotation-only
elif two SE(3) warps agree:
    use SE(3)
else:
    reject the directional replacement
```

一致性阈值是两个历史源 warp 后 RGB 的平均绝对差不超过 0.12。

该逻辑有意把 rotation-only 作为低视差默认解释，只有它失败且有限深度 warp 通过双源一致性时才切换到 SE(3)。错误深度因此不能单独把方向层拖到错误位置。

### 14.6 双源认证，单源着色

两个锚点都用于构造 `certified` mask，但当前：

```yaml
source_fusion: first
```

颜色只来自位姿分数最优的第一个源，不对两个历史源取均值，也不做跨 segment crossfade。这样消除了“上一帧山头仍留在旧位置”的双源拖影机制。

### 14.7 metric transmittance ownership

常规方向档案的像素权重为：

\[
w(u)=\mathbb 1[\text{two-source consistent}]
\left(1-M_a(u)\right)w_b(u),
\]

其中 (w_b) 是边界支持 taper。最终：

\[
I_{out}(u)=(1-w(u))I_{GS}(u)+w(u)I_{anchor}(u).
\]

因此方向档案只接管 appearance metric coverage 不足的像素，不覆盖已有可靠 Gaussian 外观。

## 15. Uncertainty Bootstrap：起始阶段的自适应方向责任

### 15.1 问题

序列前段可能仍有足够稀疏点，因而尚未进入 sparse-dropout episode，但天空和极远山体已经具有大面积深度不确定性。如果方向层只由稀疏点数量触发，前几秒会保留错误近天空 Gaussian，形成靠近镜头的漂浮。

### 15.2 像素不确定性质量

在 packed rasterization 中，对 Gaussian (i) 定义不确定投影半径：

\[
r_i^u=r_i\sqrt{1-q_i^a}.
\]

当 (r_i^u>48\) px 时，该 packed Gaussian 对 uncertainty channel 贡献质量。alpha compositing 后得到每像素 uncertainty mass (U_t(u))。

### 15.3 自适应 episode 结束

系统从序列开始采集 bootstrap 锚点，并记录帧平均不确定性：

\[
\bar U_t=\frac{1}{HW}\sum_u U_t(u).
\]

当前实现把 (\bar U_t>0.5) 视为 uncertain run，把 (\bar U_t\le0.5) 视为 recovery run。结束条件不是固定帧号，而是 recovery 连续长度超过此前最长 uncertain run。系统回滚到 recovery 开始前的锚点快照，并把支持区间截止在 change point。

bootstrap 档案最多 48 个锚点，也使用时间相邻 Ward 合并。其 ownership profile 为 `uncertainty_mass`，混合权重 0.75，并启用边界 taper。它只负责已检测到的早期不确定 episode，不会永久接管后续稳定场景。

## 16. 覆盖恢复

### 16.1 动机

前视 UAV 在草地、天空或远山阶段可能短时缺少稀疏点。普通 frame checker 会拒绝这些帧，地图将停止增密，造成沿飞行方向不断积累的未覆盖区域。

覆盖恢复不是降低 frame checker 标准，而是只在稀疏点不足时，对被拒绝帧进行第二次认证。

### 16.2 当前认证条件

当前有效配置使用 `fixed_gap`：

- 距上一个关键帧至少 40 帧；
- 平移至少 1 m，或旋转至少 3 度；
- 当前基础 Gaussian 渲染中，至少 15% 像素满足以下任一条件：
  - RGB 平均绝对残差至少 0.08；
  - opacity 低于 0.5。

只有三类条件同时满足，才把该帧恢复为关键帧。

### 16.3 新生保护

恢复关键帧仍走同一套 DepthCov、Adaptive PBSD、TSC/FPR 和有限深度证书。commit 后执行 5 步新生组局部优化：

- means 冻结；
- 最大尺度扩张倍数为 1.0；
- 后续每 5 帧可执行 1 步轻量跟随优化。

这样覆盖恢复只能学习颜色、opacity 和允许的局部参数，不能用短时错误 residual 把新生点拖到任意深度或扩大为大块漂浮。

### 16.4 自适应边界说明

覆盖恢复模块当前仍含 40 帧、1 m、3 度和 residual 阈值。它不是当前方法中完全自适应的部分。代码还实现了 `projective_debt`，但当前最优配置没有启用，因此论文主方法不能用 projective-debt 公式替代当前真实设置。

## 17. TGBR-75：时间梯度证据的 SH 容量分配

### 17.1 模块定义

**名称**：Temporal Gradient-Band Routing

**实现**：

- `utils_new/streaming_appearance_lod.py`
- `utils_new/gaussian_models.py::observe_streaming_appearance_lod`

当前配置：

```yaml
StreamingAppearanceLOD:
  enabled: true
  birth_degree: 2
  target_degree: 3
  min_views: 2
  max_target_fraction: 0.75
  promotion_interval: 10
  selection_mode: gradient_agreement
  utility_ema_decay: 0.9
  compute_routing: false
```

### 17.2 未激活频带的证据

SH3 对应尚未激活的最高频带。即使其系数保持为 0，当前完整 SH3 计算图仍能提供“若允许该频带变化，loss 最希望往哪个方向变化”的梯度向量：

\[
\mathbf g_{i,t}=\nabla_{\mathbf c_i^{SH3}}\mathcal L_t.
\]

对可见 Gaussian 更新 EMA：

\[
\bar{\mathbf g}_{i,t}
=\beta\bar{\mathbf g}_{i,t-1}
+(1-\beta)\mathbf g_{i,t},
\qquad\beta=0.9.
\]

观测 (n_i) 次后的 bias-corrected 向量为：

\[
\hat{\mathbf g}_i=
\frac{\bar{\mathbf g}_i}{1-\beta^{n_i}}.
\]

晋升分数为：

\[
u_i=\|\hat{\mathbf g}_i\|_2^2.
\]

如果不同帧的高阶梯度方向相互抵消，EMA 向量范数会较小；只有跨时间方向持续一致的高阶需求才能得到高分。因此该分数区别于简单的单帧梯度能量。

### 17.3 单调预算晋升

每 10 次证据更新：

1. 只考虑可见次数至少 2 的 SH2 Gaussian。
2. 按 (u_i) 选择最高分对象。
3. 总 SH3 数量不超过当前 Gaussian 总数的 75%。
4. SH2 到 SH3 的晋升不可逆。
5. 未晋升 Gaussian 的 SH3 系数和 Adam 更新后结果都重新乘 mask，保持严格为 0。

### 17.4 当前收益边界

当前统一 launcher 强制 `compute_routing=false`。由于有限深度双责任需要 packed rasterizer，当前渲染会对所有行构造完整 SH3 路径，再用 mask 保证未晋升系数为零。因此 TGBR 当前主要贡献是高阶外观容量的证据分配和抑制无证据高频，而不是已验证的显著训练算力节省。

仓库中存在 sparse PLY 和 compute-routing 实验代码，但它们不是当前最优运行协议的一部分。论文不能用这些关闭路径的资源收益为当前 TGBR 背书。

## 18. 优化、剪枝和输出

### 18.1 在线优化

当前通用方法参数：

- 活动窗口 1；
- 全局窗口 4；
- 关键帧优化 10 步；
- 初始化 4 帧，每次 10 步；
- 相机 pose optimizer 保留，但每个步骤最多 4 个 pose 优化步；
- 当前方法核心模块不改变固定在线优化预算。

### 18.2 剪枝

每 100 个处理帧执行 opacity 剪枝：

\[
\alpha_i\le0.01
\]

的 Gaussian 被删除。TSC 当前不是 active-lifecycle cover，因此论文应把 opacity prune 描述为 Gaussian 地图清理，不应声称 TSC 覆盖表已经完整同步所有删除生命周期。

### 18.3 Post refinement

序列结束后运行最多 100 步优化，当前配置 `opt_cam=false`。该阶段可以改善颜色、opacity 和 Gaussian 参数，但不是 Adaptive PBSD/TSC/FPR novelty 的来源。实验必须同时给出 online 和 post 两组指标。

### 18.4 深度视频

`render.py` 默认用 `transmittance_far` 显示深度：

\[
D_{vis}=M_gD_g+(1-M_g)D_{far}.
\]

这把未解释的射线质量放到可视化远平面，使天空显示为远，而不是把低 opacity 条件命中深度误画成近处。该改动只修正诊断视频定义，不改变重建、PSNR 或 Gaussian 几何。

## 19. 哪些部分是自适应的

| 部分 | 自适应依据 | 是否使用未来帧 | 是否依赖固定米制远近边界 |
|---|---|---:|---:|
| Adaptive PBSD | 当前 DepthCov 幸存者的 log-depth Lloyd 分布 | 否 | 否 |
| TSC/FPR 路由 | 当前/历史可见视差、投影半径、log-depth 不确定性 | 否 | 否 |
| FPR 单元 | 图像面积、出生预算、候选池倍率、DepthCov 最大不确定性 | 否 | 否 |
| Footprint trust | 视差证书、深度精度、局部 residual 能量和 opacity | 否 | 否 |
| 有限深度责任 | finite-SE(3) 与 infinite-rotation 图像解释竞争 | 否 | 否 |
| 锚点位姿分数 | 当前渲染的 opacity 加权逆深度 | 否 | 否 |
| uncertainty bootstrap | 在线 uncertainty mass change point | 否 | 否 |
| TGBR | 在线高阶梯度方向和可见次数 | 否 | 否 |
| 覆盖恢复 | 固定间隔、位姿和 residual 阈值 | 否 | 部分依赖固定阈值 |

当前配置中仍然存在 `FrontViewFarField.depth_m=80` 和旧 PBSD `depth_edges_m` 字段，但在 `routing_mode=adaptive_observability` 与 `selection_mode=adaptive_log_depth_random` 下，它们不是当前路由和分配的决策边界。它们保留在兼容配置结构中，不能据此把当前方法误读成固定 80 m FPR。

## 20. 为什么这些设计针对前视 UAV

### 20.1 利用前视运动的连续视差

前视轨迹以连续向前运动为主，旋转通常比全向手持轨迹更平滑。视差随深度单调减弱，因此 (p/r) 和 (p\sigma_{\log z}/r) 能直接刻画 metric 深度是否可辨识。

### 20.2 处理强透视的预算偏置

前视图像近处占据更多像素，远处结构影响时间更长但像素面积更小。log-depth 分区和 capped water-filling 专门对抗这种“近处像素多、远处长期欠采样”的透视偏置。

### 20.3 区分有限地物与无限远外观

天空、极远山和地平线在前视 UAV 中经常占据大面积，却缺少足够平移视差。finite-vs-infinite certificate 与 rotation-first warp 正面处理这一弱可观测性，而不是简单把天空设为一个固定深度。

### 20.4 适应稀疏几何掉线

无人机进入草地、天空或低纹理山体时，稀疏点会成段减少。episode-based direction archive 和 residual-certified coverage recovery 都以这种成段掉线为结构假设，同时保持状态有界。

### 20.5 连续而狭窄的观察方向

前视轨迹的大量 Gaussian 只从相似方向被看到，SH2 通常足够。TGBR 不把高阶容量平均给所有点，而是让跨时间持续的高阶梯度证明 SH3 的必要性。

## 21. 与 baseline MODP 的数据流区别

| 阶段 | Baseline MODP | 当前方法 |
|---|---|---|
| 稀疏候选 | 有效稀疏深度直接候选 | 保留 sparse source 类别，进入责任路由 |
| DepthCov 补点 | 高残差未覆盖区域，过滤后取候选 | 相同候选来源，但过滤后先做 Adaptive PBSD |
| 候选预算 | 无当前帧自适应 log-depth max-min 分配 | 三类 Lloyd + capped water-filling |
| 重复抑制 | 多层 HashBlock 世界占用 | HashBlock 关闭；TSC 连续尺度覆盖 + FPR 投影责任 |
| 远近处理 | 同一世界哈希规则 | 按来源、视差和深度精度选择责任坐标系 |
| 远场单元 | 固定世界 hash cell | 由图像面积/预算/不确定性推导的 image-log-depth cell |
| 新生尺度 | 统一依赖优化 | 低信息 DepthCov 由 footprint trust 限制扩张 |
| 深度责任 | 所有 Gaussian 按 opacity 参与 | (q^g) 经有限深度证书衰减 |
| 外观责任 | 与几何绑定 | (q^a) 保留，和 (q^g) 解耦 |
| 低视差外观 | 只由 Gaussian 表示 | 有界因果方向档案在输出端补充 |
| SH 容量 | 统一最高阶或固定低阶 | TGBR 从 SH2 单调晋升到 SH3，最多 75% |
| commit | 通过后立即永久提交 | 仍立即提交，保持低延迟 |
| 剪枝 | 每 100 帧 opacity 剪枝 | 保持同一主干策略 |

当前方法仍使用 3DGS rasterization、DepthCov 深度预测、关键帧优化和立即 commit 这些通用组件。创新点不应表述为“完全不使用 MODP 的任何思想”，而应准确表述为：它替换了 MODP 的 HashBlock 出生骨架，并重新设计了候选预算、责任坐标、尺度信任、深度责任和低视差外观处理。

## 22. 与之前固定阈值方案的区别

### 22.1 旧 PBSD

旧方案使用固定深度边界 `[20, 50] m` 和固定比例 `[25%, 45%, 30%]`。当前方案使用当前帧三类 log-depth Lloyd 和 capped water-filling。旧字段仍在配置兼容层中，但不参与当前 `adaptive_log_depth_random` 决策。

### 22.2 旧 FPR-80

旧方案按 `depth >= 80 m` 把无身份候选交给 FPR。当前方案要求候选同时满足：

- 非 sparse source；
- 没有通过视差和深度精度 metric certificate；
- 位于当前帧自适应最远 log-depth 域。

因此当前没有固定 80 m 责任断点。

### 22.3 旧方向层

旧方向层使用较小 FIFO/固定间隔档案、固定远深度位姿分数、多个源平均或 crossfade。当前方案改为：

- sparse-dropout episode；
- K96 时间有序 Ward coreset；
- rendered inverse-depth pose score；
- rotation-first `se3_fallback`；
- 双源认证、第一源着色；
- 起始不确定性 change-point bootstrap。

这些变化直接针对先前山体拖影、边缘断层和前段近天空漂浮。

### 22.4 旧有限深度证书耦合错误

曾经把有限深度证书同时用于几何 metric mass 和方向层 appearance ownership。证书正确抑制错误深度的同时，也让天空区域看起来“无任何外观所有权”，方向层过度接管并造成跨场景回退。

当前版本保存证书前 `uncertainty_confidence` 作为 (q^a)，证书后 `metric_confidence` 作为 (q^g)，并在一次 rasterization 中同时输出两套深度/质量，从而修复该耦合错误。

## 23. 方法创新性应如何表述

### 23.1 核心方法贡献候选

#### 贡献 A：可观测性类型化的出生责任

当前方法不按固定米制距离分近远，而是用投影视差、Gaussian footprint 和 log-depth 不确定性判断候选能否承担世界空间责任；只有无法获得 metric certificate 且属于当前最远相对深度域的候选进入投影责任。

该设计把“远场”从场景尺度标签改写成在线可观测性类型，是当前最强的几何方法贡献。

#### 贡献 B：预算和不确定性共同定义的自适应责任尺度

Adaptive PBSD 的深度域由当前候选分布学习；FPR 的空间单元由图像面积和计算预算推导，log-depth 单元由成对深度不确定性推导；footprint trust 再把可观测信息和 residual rate-distortion 转为尺度扩张界。

这三者形成一致链条：预算决定“需要多少表示”，不确定性决定“哪些表示可区分”，局部图像能量决定“每个表示应负责多大面积”。

#### 贡献 C：有限深度几何与外观责任解耦

有限 SE(3) 和无限远 rotation-only 是前视远场的两个可证伪解释。当前证书只衰减 metric geometry responsibility，同时保留 appearance responsibility，并在一次 packed rasterization 中输出两套责任深度。

这不是简单 sky mask，也不需要语义模型；它由相机运动和历史图像一致性决定。

#### 贡献 D：有界因果方向档案

方向档案以 sparse-dropout episode 和 uncertainty change point 组织历史图像，使用时间有序 Ward 压缩；渲染时以 rotation-first/SE(3)-fallback 的双源一致性认证像素，但只用单源着色。它针对前视低视差区域提供外观补充，同时通过 `geometry_render` 隔离避免影响地图出生。

### 23.2 次要贡献

TGBR 是外观容量分配贡献：用 bias-corrected persistent gradient direction 在固定 75% 预算下选择 SH3。它可以作为方法完整性的一部分，但当前不应宣称显著训练时间或峰值显存收益。

覆盖恢复保证稀疏掉线时的连续建图，是重要鲁棒性模块；由于当前仍使用固定认证阈值，更适合作为系统设计而不是最主要理论创新。

### 23.3 novelty 的证据边界

代码层面的独立设计并不自动等于文献层面的新颖性。正式论文仍需要对以下相邻方向做 prior-art 检索：

- uncertainty-aware 3DGS densification；
- projective/ray-space Gaussian allocation；
- depth-at-infinity 和 rotation-only neural rendering；
- image cache/appearance memory for SLAM；
- per-Gaussian adaptive SH degree。

当前可以稳健声称“相对本项目 MODP baseline 的方法差异”和“代码实现中的独立机制”；在完成系统 prior-art 审计前，不应使用“首个”或“完全新颖”等绝对措辞。

## 24. 当前有效和关闭模块

### 24.1 有效模块

```text
FrontViewSampling.enabled = true
FrontViewSampling.selection_mode = adaptive_log_depth_random

FrontViewFarField.enabled = true
FrontViewFarField.routing_mode = adaptive_observability
FrontViewFarField.responsibility_basis = source
FrontViewFarField.projective_nms_mode = budget_cells
FrontViewFarField.footprint_trust_mode =
    certificate_residual_rd_visible_detail

FrontViewScaleCover.enabled = true
FrontViewCoverageRecovery.enabled = true

CausalDualResponsibility.enabled = true
CausalDualResponsibility.finite_depth_certificate_enabled = true
CausalDualResponsibility.finite_depth_preserve_appearance_ownership = true

FrontViewDirectionalLayer.enabled = true
FrontViewDirectionalLayer.max_anchors = 96
FrontViewDirectionalLayer.source_fusion = first
FrontViewDirectionalLayer.uncertainty_bootstrap_enabled = true

StreamingAppearanceLOD.enabled = true
StreamingAppearanceLOD.selection_mode = gradient_agreement
StreamingAppearanceLOD.max_target_fraction = 0.75
```

### 24.2 当前关闭或不属于主方法

```text
HashBlock.use_hash = false
preserve_sparse_track_geometry = false
sparse_track_identity_enabled = false
map_redundancy_gate = false
ray_atlas_enabled = false
coverage recovery depth fallback = false
TGBR compute_routing = false
TGBR bounded/spectral residency = false
ProgressiveMapping = disabled
AeroCommit = disabled
WorldTestGS = disabled
```

这些关闭路径不应出现在论文主流程图中。

## 25. 当前结果和正确解读

### 25.1 跨场景最终结果

当前统一方法的已完成 PSNR/SSIM 验证如下：

| 场景 | 帧数 | Online PSNR | Post PSNR | Post SSIM | Online FPS |
|---|---:|---:|---:|---:|---:|
| PanoAir | 2230 | 25.8122 | 26.2848 | 0.7890 | 3.100 |
| Pano360 NSC | 1859 | 27.6046 | 28.3943 | 0.8681 | 0.711 |
| MyData Village4 | 1329 | 25.8151 | 26.3087 | 0.8138 | 0.851 |
| 360DVO Mountains | 765 | 29.8586 | 30.3318 | 0.9167 | 1.738 |

证据位置：

- `Logs_cross_scene_current_best_8_15/finite_appearance_decision_decoupling_cross_scene_20260815/summary.json`
- `Logs_cross_scene_current_best_8_15/finite_appearance_decision_decoupling_mountains_20260815/summary.json`

### 25.2 不能误解的对比

Mountains 的证书修复版本相对错误耦合版本提升约 2.9238 dB，但这是消除实现回退，不是新增模块相对正确 baseline 的净收益。相对历史无证书 K96 最佳 30.3697 dB，当前 30.3318 dB 低约 0.0379 dB。

PanoAir 的 28.1121 dB 是 200 帧验证结果，不是完整 2230 帧序列，不能和当前完整序列 26.2848 dB 直接比较。

### 25.3 FPS 的含义

表中 FPS 为：

\[
\mathrm{FPS}=\frac{\text{输入帧数}}
{\text{online mapping wall time}}.
\]

不包含 online-stage 导出、post refinement 和离线视频评估。不同场景 FPS 差异主要来自：

- 图像分辨率；
- 关键帧比例；
- Gaussian 数量；
- 稀疏点和 DepthCov 候选规模；
- 方向证据和覆盖恢复触发次数。

因此 PanoAir 的 3.1 FPS 不能直接说明所有场景都达到该速度。

## 26. 可证伪的消融设计

为了让方法故事成立，消融不能只做“打开后指标变高”。建议至少保留下列因果对照：

### 26.1 Adaptive PBSD

- A：旧固定 20/50 m + 固定比例；
- B：Adaptive log-depth random；
- C：保持每个自适应域候选数不变但打乱域标签。

控制相同 DepthCov survivor、相同出生预算和随机种子。

### 26.2 可观测性路由

- A：固定 80 m FPR；
- B：adaptive observability routing；
- C：在每个 log-depth 域内等数量打乱 responsibility mask。

比较总 PSNR、远景 PSNR、深度稳定性、Gaussian 数量和运行时间。

### 26.3 Footprint trust

- A：关闭尺度限制；
- B：真实证书；
- C：在相同 log-depth 域内打乱证书/半径因子。

除尺度上限外，出生点、优化步数和预算保持一致。

### 26.4 有限深度双责任

- A：无有限深度证书；
- B：证书同时衰减 geometry 和 appearance；
- C：当前 geometry/appearance decoupling；
- D：等分布 shuffled certificate。

重点看天空/远山 metric depth、方向层接管比例、全图 PSNR 和远景 PSNR。

### 26.5 方向档案

- 固定小 FIFO；
- episode ordered Ward K96；
- 双源 mean；
- 当前双源认证、first-source 着色；
- 关闭 uncertainty bootstrap；
- 当前 uncertainty bootstrap。

除方向 sidecar 外使用完全相同的 Gaussian checkpoint，可以把外观档案收益与地图训练差异分离。

### 26.6 TGBR

- 全 SH2；
- 全 SH3；
- TGBR-75 gradient agreement；
- 相同 75% 数量的 shuffled promotion。

当前 `compute_routing=false`，所以应主要汇报质量、激活比例和模型有效容量，不应预设会出现训练速度收益。

## 27. 关键实现索引

| 模块 | 代码位置 |
|---|---|
| Adaptive PBSD | `utils_new/frontview_sampling.py` |
| 可观测性路由、FPR、footprint trust | `utils_new/frontview_far_field.py` |
| TSC | `utils_new/frontview_scale_cover.py` |
| 有限深度证书与决策隔离 | `utils_new/frontview_dual_responsibility.py` |
| 单次异构 SH/双责任 rasterization | `utils_new/heterogeneous_sh_rasterizer.py` |
| Gaussian proposal、commit、责任状态 | `utils_new/gaussian_models.py` |
| 覆盖恢复 | `utils_new/frontview_coverage_recovery.py` |
| 因果方向档案 | `utils_new/frontview_directional_layer.py` |
| TGBR 证据与晋升 | `utils_new/streaming_appearance_lod.py` |
| 每帧执行顺序、优化和保存 | `utils_new/scene_mapper.py` |
| 深度/合并视频诊断 | `render.py` |
| 当前统一验证 | `scripts/run_current_best_cross_scene_validation.py` |

## 28. 代码一致的每帧伪代码

```python
for frame in stream:
    camera = read_ordered_frame(frame)
    prune_low_opacity_if_due()

    is_keyframe = regular_keyframe_check(camera)
    if not is_keyframe and sparse_count(camera) < sparse_threshold:
        is_keyframe = residual_certified_coverage_recovery(camera)

    if is_keyframe:
        optimize_existing_gaussians(steps=10, use="geometry_render")
        render_pkg = render_current_map(camera)

        sparse = build_sparse_source_proposals(camera)
        pool = sample_high_residual_pixels(
            geometry_decision_render(render_pkg),
            camera.image,
            max_rows=2 * birth_budget,
        )
        depthcov = estimate_depth_and_uncertainty(pool)
        survivors = depthcov[depthcov_uncertainty_valid(depthcov)]

        selected = adaptive_log_depth_lloyd_waterfill(
            survivors,
            budget=birth_budget,
            within_regime="random",
        )
        proposals = concatenate(sparse, selected)
    else:
        proposals = build_sparse_source_proposals(camera)

    parallax = visible_parallax(proposals, causal_reference_views)
    projected_radius = projected_gaussian_radius(proposals)
    sigma_logz = proposals.log_depth_std

    metric_certified = (
        (parallax >= projected_radius)
        & (parallax * sigma_logz <= projected_radius)
    )
    adaptive_far = farthest_current_log_depth_regime(
        proposals.depth[~proposals.sparse_source]
    )
    fpr_mask = (
        ~proposals.sparse_source
        & ~metric_certified
        & adaptive_far
    )

    keep_tsc = target_scale_cover(proposals[~fpr_mask])
    keep_fpr = one_per_budget_image_logdepth_cell(proposals[fpr_mask])
    proposals = merge(keep_tsc, keep_fpr)

    proposals.scale_growth_limit = observability_residual_footprint_trust(
        proposals, render_pkg
    )
    proposals.finite_depth_certificate = finite_vs_infinite_certificate(
        proposals, causal_reference_views
    )

    q_prior = source_depth_confidence(proposals)
    proposals.q_geometry = q_prior * proposals.finite_depth_certificate
    proposals.q_appearance = q_prior
    commit_immediately(proposals)

    if is_coverage_recovery_keyframe:
        refine_newborn_group(
            steps=5,
            freeze_means=True,
            max_scale_expansion=1.0,
        )

    update_tgbr_gradient_evidence()
    update_bounded_directional_archives(camera)

save_online_stage()
post_refine(steps=100, optimize_camera=False)
activate_directional_archive_for_output()
save_final_map_and_sidecar()
```

## 29. 当前局限

1. FPR 是出生时的当前 batch 投影竞争，不是跨帧持久动态远近层；相机接近后不会自动把旧 FPR Gaussian 迁移到 TSC。
2. 当前 source-based 路径没有完整利用 persistent point ID。稀疏来源可靠性被保留，但无法强声称“同一世界实体只出生一次”。
3. TSC 使用 CPU `cKDTree`，虽然有增量重建，但超长序列的覆盖表仍随 Gaussian commit 增长。
4. Gaussian 地图只做 opacity prune，没有固定总 Gaussian 上限；超长序列仍可能因地图规模增长而 OOM。
5. 方向档案有固定锚点上限，但保存的是图像，内存开销仍与分辨率和 144 个最大锚点相关。
6. 覆盖恢复仍使用固定阈值，不是完全由投影预算自适应推导。
7. TGBR 当前不跳过未晋升 SH3 的实际计算，资源收益不能过度表述。
8. 方向 sidecar 改善的是输入轨迹上的输出渲染；严格 novel-view 泛化需要单独协议验证。

## 30. 推荐的论文叙事主线

论文不应把所有模块并列堆叠，而应围绕一个统一问题展开：

> 前视 UAV 的有限深度可观测性随视差、尺度和图像责任变化。在线 Gaussian 建图必须让候选预算、出生坐标系、尺度成长和外观补偿都服从同一可观测性，而不是服从固定米制远近阈值。

由此自然得到三层主线：

1. **Where to allocate**：Adaptive PBSD 以当前 log-depth 分布和固定计算预算分配出生机会。
2. **Where to own**：observability-typed TSC/FPR 与 footprint trust 决定世界空间或投影空间责任及可允许的尺度成长。
3. **What to trust**：finite-depth dual responsibility 与 causal directional archive 区分有限深度几何、无限远外观和高阶方向容量。

TGBR 和覆盖恢复分别作为外观容量控制和稀疏掉线鲁棒性补充。这样方法动机、数学定义、实现数据流和实验消融可以一一对应，而不是若干阈值模块的集合。
