# 面向前视无人机的范围感知在线 3D Gaussian 建图

> 代码审计分支：`research/hashless-frontview-lod-7-24`
>
> 代码审计提交：`0886bc906d1c0246b1fcb6d581a301c99bcd330e`
>
> 通用运行入口：`scripts/run_360dvo_scene_directional.py`
>
> 文档范围：只讨论 mapping。相机跟踪、位姿估计和数据预处理不属于本文方法。

本文描述的是一套独立的前视无人机在线 3D Gaussian 建图方法。方法接收按时间顺序到达的图像、相机内外参和当前帧稀疏世界点，在处理每一帧时更新 Gaussian 地图。它不是某个已有系统的“新版配置”；已有方法只在后文的基线比较中出现。

整套方法的核心不是简单地增加优化步数，而是回答三个前视 UAV 特有的问题：

1. 近处纹理产生的高残差像素很多，容易占满补点预算，远处结构长期得不到 Gaussian。
2. 不同来源、不同距离的候选，其几何可信度不同。带持久世界身份的稀疏点适合在世界坐标中判断重复；低视差远场的单帧深度候选不适合使用同样的三维占用规则。
3. 并非每个 Gaussian 都需要同样高的视角相关外观容量。统一使用高阶球谐会增加参数和过拟合风险，统一使用低阶球谐又会损失反光、阴影和方向性纹理。

方法据此形成三层设计：

- **PBSD** 在深度不确定性过滤之后分配近、中、远候选的出生预算。
- **TSC/FPR** 根据几何可观测性选择世界空间或投影空间的出生责任规则。
- **TGBR** 根据跨时间持续存在的高阶球谐梯度，为单个 Gaussian 分配 SH3 容量。

当稀疏世界点短时消失时，系统再通过一个残差认证的覆盖恢复分支继续建图。所有通过准入的候选都立即进入同一张 Gaussian 地图，随后参与统一渲染和在线优化。

---

## 1. 任务设定

### 1.1 输入

对时刻 (t=0,1,\ldots,T-1)，在线映射器依次接收：

- RGB 图像 (I_t\in[0,1]^{H\times W\times 3})；
- 相机内参 (K_t)；
- 世界到相机变换 (T_{cw,t})；
- 当前帧可见的稀疏世界点
  \(\mathcal P_t=\{(\mathbf X_j,\mathrm{id}_j)\}\)；
- 每个稀疏点在当前相机中的有效深度。

本文假设这些输入已经按图像顺序对齐。位姿怎样得到、稀疏点怎样生成，不在本文讨论范围内。

### 1.2 在线状态

处理到第 (t) 帧时，地图为

\[
\mathcal G_t=\{G_i\}_{i=1}^{N_t}.
\]

每个 Gaussian 包含：

\[
G_i=(\boldsymbol\mu_i,\mathbf s_i,\mathbf q_i,
\alpha_i,\mathbf c_i,d_i),
\]

其中：

- \(\boldsymbol\mu_i\)：世界坐标均值；
- \(\mathbf s_i\)：三轴尺度；
- \(\mathbf q_i\)：旋转四元数；
- \(\alpha_i\)：opacity；
- \(\mathbf c_i\)：球谐颜色系数；
- \(d_i\in\{2,3\}\)：当前允许使用的最高球谐阶数。

除了 Gaussian 参数，映射器还维护三个轻量状态：

- TSC 的连续世界空间覆盖表；
- TGBR 的逐 Gaussian 梯度证据；
- 覆盖恢复使用的有限长度远深度记忆。

### 1.3 输出

严格的 Gaussian 建图输出是当前时刻可渲染的 \(\mathcal G_t\)。序列结束后可保存为 `point_cloud.ply`，其中包含 means、scales、quaternions、opacity 和 SH 系数。

当前通用运行程序还可以保存一个 `frontview_directional_layer.pt`。它属于最终渲染扩展，不改变 Gaussian 地图，也不属于在线建图核心。第 12 节单独说明这一点。

### 1.4 在线边界

PBSD、TSC、FPR、TGBR、覆盖恢复、Gaussian 优化和 opacity 剪枝都只使用当前及历史信息。它们是因果在线的。

当前运行协议在序列结束后还有固定 100 步终止优化。这个步骤有固定上限，但不是逐帧 anytime 输出的一部分。论文应把“最后一帧处理完成时的在线地图”和“经过 100 步终止优化的最终地图”分开计时。

---

## 2. 为什么前视 UAV 需要不同的地图责任规则

前视无人机视频的主要困难不是简单的“大场景”，而是深度、视差和图像占比之间存在强耦合。

### 2.1 近场容易垄断候选预算

近处地面、树枝和建筑边缘在图像中占据大量像素，也更容易产生高频残差。如果只从所有高残差像素中随机或按残差取固定数量，补点预算会集中在近场。远处山体、道路尽头和天际线虽然每帧只占少量像素，却会持续影响整段飞行的渲染质量。

因此，候选选择不能只问“哪个像素误差最大”，还要问“不同透视深度是否都获得了稳定的出生机会”。这对应 PBSD。

### 2.2 远近候选的三维坐标可信度不同

对带持久 ID 的稀疏世界点，\(\mathbf X_j\) 已经是一个跨帧世界实体。即使点很远，也可以在世界空间判断它是否已被附近 Gaussian 表达。

对单帧 DepthCov 补点，情况不同。距离越远，微小的逆深度误差会被放大成很大的世界坐标位移。远场候选在图像中的投影位置可能稳定，但它的 metric XYZ 不一定稳定。若仍在世界空间用固定占用单元判断重复，就会出现两类错误：

- 错误深度把本应补充的远场候选判为已占用；
- 同一远场结构因深度波动落入不同空间位置，反复出生并形成漂浮层。

因此，本方法不按“近/远”机械地切成两个地图，而是按**几何可观测性**分配出生责任：

- 有持久世界身份，或 DepthCov 深度仍处于可度量范围：使用 TSC 世界空间责任；
- 没有持久身份且处于低视差远场：使用 FPR 投影空间责任。

### 2.3 外观复杂度也应由在线证据决定

高阶球谐不是对所有 Gaussian 都有价值。一个只被少数相似方向看到的远点，SH3 很可能只是在拟合噪声；一个反复从不同方向被看到、且高阶系数梯度持续指向相同下降方向的点，才真正需要更高外观容量。

TGBR 因此不根据距离直接指定 SH 阶数，也不在序列结束后重放整条轨迹，而是在正常在线反向传播中读取“尚未激活的 SH3 频带梯度”，用它决定哪些 Gaussian 从 SH2 晋升到 SH3。

---

## 3. 完整在线数据流

```text
按序到达的 RGB、K、W2C pose、当前帧稀疏世界点
                         |
                         v
              帧检查与在线关键帧选择
                         |
              +----------+-----------+
              |                      |
          普通帧                  关键帧
              |                      |
      稀疏点候选出生        先优化当前 Gaussian 地图
              |                      |
              |            渲染残差中的未覆盖像素
              |                      |
              |                DepthCov 深度预测
              |                      |
              |                不确定性过滤
              |                      |
              |          PBSD 固定预算近/中/远分配
              |                      |
              +-----------+----------+
                          候选合并
                              |
              +---------------+----------------+
              |                                |
       稀疏候选，任意深度             无稀疏身份的 DepthCov 候选
       DepthCov 深度 < 80 m                    深度 >= 80 m
              |                                |
       TSC 连续世界尺度覆盖             FPR 投影-对数深度责任
              |                                |
              +---------------+----------------+
                              |
                        立即写入地图
                              |
                    后续帧统一渲染与优化
                              |
             TGBR 根据时间梯度证据分配 SH3
                              |
                   每 100 帧 opacity 剪枝
```

稀疏点掉线时，覆盖恢复分支位于“帧检查”与“关键帧”之间。它只在普通检查拒绝当前帧后启动；通过残差、覆盖率和位姿新颖度认证后，该帧被当作恢复关键帧，并继续走同一套 DepthCov、PBSD、TSC/FPR 和立即提交路径。

---

## 4. 一帧在映射器中的处理顺序

为了理解各模块的位置，需要先明确当前代码不是“先补点再优化”，而是关键帧先优化已有地图，再从仍未解释的区域出生新 Gaussian。

### 4.1 普通帧

普通帧不运行 DepthCov 补点，只处理当前帧可见的稀疏世界点：

1. 读取当前稀疏点及其颜色、深度和持久 ID。
2. 用 TSC 检查这些世界点附近是否已经存在尺度和颜色兼容的表示。
3. 未被覆盖的稀疏点立即转成 Gaussian 并提交。
4. 若当前仍处于稀疏掉线阶段，按固定间隔对最近一次恢复出生的 Gaussian 组做一次轻量跟随优化。

因此，即使一帧不是关键帧，新的可靠稀疏世界点仍可进入地图。

### 4.2 关键帧

关键帧的处理顺序为：

1. 用当前局部窗口优化已有 Gaussian 地图，当前配置为 10 步。
2. 用优化后的地图渲染当前关键帧，得到 RGB、depth、opacity 和结构残差图。
3. 收集当前帧有效稀疏候选。当前 `semi_dense_err_threshold=0.00`，因此有效稀疏点基本全部保留。
4. 在非稀疏、高残差像素上生成 DepthCov 候选池。
5. 先执行深度不确定性过滤，再由 PBSD 从幸存者中取固定补点预算。
6. 合并稀疏候选与 DepthCov 候选。
7. 按来源和深度路由到 TSC 或 FPR。
8. 将最终幸存者立即写入永久 Gaussian 地图。

这意味着新生 Gaussian 不会反过来降低本帧用于选点的残差。它们主要在后续关键帧优化中学习；覆盖恢复出生是例外，会获得 5 步专用新生优化。

### 4.3 初始化

当前运行配置使用 4 个初始化帧、每次 10 个优化步。在地图尚不能提供有效残差时，补点像素从图像平面采样；一旦地图已初始化，补点只来自高残差且不与稀疏像素重合的区域。

---

## 5. 候选生成与 Gaussian 初始化

PBSD、TSC 和 FPR 都不负责预测深度。它们的输入是候选生成器已经得到的像素、深度、颜色和来源标签。

### 5.1 两类候选

#### 稀疏世界点候选

稀疏点已经带有世界坐标 \(\mathbf X_j\) 和持久 ID。映射器将其投影到当前图像采样颜色，但保留输入的精确世界坐标，不再通过像素和深度重复反投影。

稀疏候选具有：

- `sparse_depth_valid=True`；
- `track_id>=0`；
- 深度置信度固定为 1；
- 任意深度都由 TSC 管理。

#### DepthCov 补点候选

关键帧首先对当前渲染和真值图像做灰度高斯平滑，再计算 SSIM difference map：

\[
r_t(u,v)=1-\operatorname{SSIM}_{local}
(\hat I_t,I_t).
\]

当前阈值为 \(r_t(u,v)\ge 0.05\)。候选像素还必须不是已有稀疏深度像素。若本帧补点预算为 \(B\)，普通路径最多先采样 \(2B\) 个高残差像素。

DepthCov 根据当前帧最多 500 个稀疏深度锚点，对这些像素预测深度 \(z\) 和标准差 \(\sigma_z\)。只保留满足内部不确定性阈值的候选；当前 `std_valid_threshold=0.06`。幸存者的深度置信度为

\[
c_z=\operatorname{clip}
\left(1-\frac{\sigma_z}{0.06},0,1\right).
\]

DepthCov 候选具有：

- `sparse_depth_valid=False`；
- `track_id=-1`；
- 世界坐标由 \((u,v,z,K_t,T_{cw,t})\) 反投影得到；
- 深度小于 80 m 时进入 TSC，否则进入 FPR。

### 5.2 反投影

对 DepthCov 候选，先在相机坐标中恢复三维点：

\[
\mathbf x_c=zK_t^{-1}[u,v,1]^\top,
\]

再由输入的世界到相机变换得到世界坐标：

\[
\mathbf X=T_{cw,t}^{-1}[\mathbf x_c^\top,1]^\top.
\]

代码调用 `unproject_pts_tensor(...)` 完成该过程。

### 5.3 新生 Gaussian 参数

设平均焦距

\[
\bar f=\frac{f_x+f_y}{2}.
\]

当前实现的初始对数尺度为

\[
\ell_0=\log\left(\frac{0.5z}{\bar f}\right)+0.5.
\]

它对应“一个随距离线性增长的初始世界尺度”。颜色取当前像素的曝光校正 RGB，初始 opacity 为 0.5。模型最多支持 SH3，但 TGBR 将普通新生 Gaussian 的有效阶数初始化为 SH2；SH3 系数保持严格为零，直到该 Gaussian 被晋升。

候选一旦通过责任过滤就立即 commit，没有多帧候选缓存。这个选择保持了低延迟，但也意味着几何可靠性主要由输入稀疏点、DepthCov 不确定性、PBSD 和 TSC/FPR 共同保证。

---

## 6. PBSD：透视平衡的幸存者增密

**全名**：Perspective-Balanced Survivor Densification

**代码位置**：`utils_new/frontview_sampling.py::depth_stratified_indices`

**接入位置**：`utils_new/gaussian_models.py::propose_new_gaussians`

### 6.1 动机

只对“原始高残差像素”分深度配额是不可靠的，因为很多候选会在 DepthCov 不确定性过滤时消失。若先按深度分配、后过滤，最终预算仍会偏向某些深度。

PBSD 的关键顺序是：

```text
2B 个高残差候选像素
    -> DepthCov 预测
    -> 深度不确定性过滤
    -> 对过滤后的幸存者分配 B 个深度预算
```

因此“Survivor”不是命名修饰，而是算法边界：深度配额作用在可信候选上，而不是作用在尚未验证的像素上。

### 6.2 输入与输出

输入：

- 通过 DepthCov 不确定性过滤的深度数组 \(\mathbf z\in\mathbb R^M\)；
- 当前帧 DepthCov 出生预算 \(B\)；
- 深度边界 \([20,50]\) m；
- 三段预算比例 \([0.25,0.45,0.30]\)。

输出：

- 最多 \(B\) 个幸存者下标。

PBSD 只重分配 DepthCov 候选，不削减稀疏世界点候选。

### 6.3 算法逻辑

将幸存候选分为：

\[
\begin{aligned}
\mathcal B_0 &= \{i\mid z_i<20\},\\
\mathcal B_1 &= \{i\mid 20\le z_i<50\},\\
\mathcal B_2 &= \{i\mid z_i\ge 50\}.
\end{aligned}
\]

分配配额：

\[
(Q_0,Q_1,Q_2)\approx
(0.25B,0.45B,0.30B),
\]

其中最后一段吸收四舍五入误差，使三段总和严格等于 \(B\)。每段内部用“固定随机种子 + 当前帧编号”生成可复现随机顺序，并取前 \(Q_k\) 个。

某一深度段不足时，不浪费预算。系统从所有尚未选择的幸存者中随机补齐剩余名额。因此比例是目标配额，不是要求每一帧都严格满足的硬约束。

### 6.4 与代码一致的核心逻辑

```python
masks = (
    depths < 20.0,
    (depths >= 20.0) & (depths < 50.0),
    depths >= 50.0,
)
quotas = [round(B * 0.25), round(B * 0.45), round(B * 0.30)]
quotas[-1] += B - sum(quotas)

selected = empty_boolean_mask(len(depths))
for mask, quota in zip(masks, quotas):
    rows = indices_where(mask)
    selected[random_permutation(rows)[:quota]] = True

shortfall = B - selected.sum()
selected[random_permutation(indices_where(~selected))[:shortfall]] = True
```

### 6.5 容易误解的地方

1. PBSD 的远段从 50 m 开始，而 FPR 从 80 m 开始。50--80 m 候选获得远段预算，但仍由 TSC 管理。
2. PBSD 不是按残差从高到低取每段前若干个。残差负责定义候选池，段内当前是可复现随机选择。
3. 若过滤后幸存者不超过 \(B\)，全部保留，不再分层。
4. PBSD 不改变每个 Gaussian 的优化次数，也不增加候选总预算。

### 6.6 创新性定位

“按深度采样”本身不是新概念。PBSD 可辩护的具体点是：**在学习深度的不确定性过滤之后，对固定的在线出生预算做 metric-depth 分配**。它解决的是前视 UAV 候选预算被近场吞噬的问题。

PBSD 更适合作为完整方法中的预算组件，而不是单独作为主要创新。它是否有效应通过同一候选池、同一 \(B\)、同一随机种子下的 uniform-survivor 对照验证。

---

## 7. TSC：连续尺度覆盖

**代码名**：`FrontViewScaleCover`，本文简称 TSC

**代码位置**：`utils_new/frontview_scale_cover.py`

**接入位置**：候选 proposal 和最终 commit 都会检查 TSC

### 7.1 动机

世界空间占用的难点不是只判断“附近有没有点”，而是判断“附近已有表示是否足以覆盖当前观察尺度”。

一个远距离出生的粗 Gaussian 与一个近距离观察到的细节 Gaussian，即使均值很近，也不应无条件互相排斥。离散空间格子通常把空间位置和分辨率绑在固定层级上。TSC 改用连续的世界距离与出生视图尺度，让覆盖关系随当前观察尺度变化。

### 7.2 在 pipeline 中的位置

TSC 管理：

- 所有稀疏世界点候选，不论深度；
- 深度小于 80 m 的 DepthCov 候选。

它在 Gaussian 参数真正写入前执行。通过检查的候选提交后，位置、目标尺度和颜色立即登记到覆盖表，供后续帧查询。

### 7.3 输入与输出

对每个候选 (i)，TSC 输入：

- 世界坐标 \(\mathbf X_i\)；
- 当前帧目标覆盖尺度 \(s_t\)；
- 当前像素颜色 \(\mathbf c_i\)；
- 候选来源标签。

TSC 的持久表中，每一行保存已提交候选出生时的：

\[
(\mathbf X_j,s_j,\mathbf c_j).
\]

输出是布尔占用标记：

- `occupied=True`：已有表示足以负责该候选，拒绝出生；
- `occupied=False`：当前尺度或外观仍缺少表示，允许出生。

### 7.4 当前帧的连续目标尺度

相机对象根据当前稀疏深度中位数计算视图尺度：

\[
v_t=\frac{0.5\,\operatorname{median}(z_t^{sparse})}{\bar f_t}.
\]

当前配置再乘 `camera_scale_rescalar=0.25`：

\[
s_t=0.25v_t
=0.125\frac{\operatorname{median}(z_t^{sparse})}{\bar f_t}.
\]

当前主配置没有为每个候选单独计算 footprint，因此同一帧的候选共享一个 \(s_t\)。历史行则保留各自出生帧的尺度 \(s_j\)。

### 7.5 覆盖判定

TSC 查询候选附近最多 32 个历史行。候选 (i) 被已有行 (j) 覆盖，当且仅当同时满足：

\[
\|\mathbf X_i-\mathbf X_j\|_2
\le 0.3s_t,
\]

\[
s_j\le 1.0s_t,
\]

\[
\|\mathbf c_i-\mathbf c_j\|_2\le 0.15.
\]

只要存在一个兼容历史行，候选就被拒绝。

第二个条件使覆盖关系具有方向性：

- 已有表示与当前目标一样细或更细时，它可以覆盖当前候选；
- 已有表示比当前目标更粗时，新的细尺度候选仍可出生。

这就是“连续尺度覆盖”的核心。它允许无人机接近物体时逐渐补充细节，而不是被早期远距离出生的粗表示永久阻挡。

颜色条件则避免几何位置接近但外观明显不同的候选互相覆盖。当前 RGB 距离直接在三通道颜色上计算。

### 7.6 数据结构

TSC 使用 SciPy `cKDTree` 查询世界坐标近邻。新登记行先放入 pending 区；累计 8192 行后重建主树。查询时主树和 pending 行都会参与，因此刚提交的历史候选不必等待下一次重建。

proposal 阶段先查一次 TSC；commit 阶段再检查一次，避免 proposal 与永久写入之间地图状态变化导致重复提交。提交成功的非 FPR 行立即登记。

### 7.7 与代码一致的核心逻辑

```python
target_size = (
    0.5 * median_sparse_depth / mean_focal
) * 0.25
radius = 0.3 * target_size

distance_ok = world_distance <= radius
scale_ok = existing_size <= 1.0 * target_size
color_ok = rgb_l2_distance <= 0.15

occupied = any(distance_ok & scale_ok & color_ok)
```

### 7.8 当前实现的边界

TSC 是一个出生责任表，不是 Gaussian 参数的空间索引：

- Gaussian 后续优化移动了多少，TSC 行不会随 means 实时移动；
- 当前覆盖表是追加式的，opacity 剪枝不会释放普通空间覆盖行；
- 同一 proposal batch 内的近场候选不会依次写入并互相去重，它们主要与历史已提交行竞争；
- 当前同帧候选共享尺度，尚未使用逐候选投影 footprint。

这些边界不会改变当前算法行为，但限制了论文中可宣称的范围。当前 TSC 应称为“连续尺度的出生责任覆盖”，不能描述成完整的动态 Gaussian LOD 树。

### 7.9 创新性定位

KD-tree 和半径近邻查询不是创新。TSC 的方法价值来自两个更具体的设计：

1. 用出生视图导出的连续 metric 尺度，而不是固定离散 cell，定义世界空间责任半径；
2. 用非对称尺度兼容关系允许从粗表示继续出生细表示。

TSC 与 FPR 联合后，进一步形成“可度量候选使用世界尺度责任，低可观测远场使用投影责任”的统一设计。这一联合设计比单独的半径去重更适合作为论文主贡献。

---

## 8. FPR：远场投影责任

**全名**：Far-Field Projective Responsibility

**代码位置**：`utils_new/frontview_far_field.py`

**接入位置**：TSC 路由之后、Gaussian commit 之前

### 8.1 动机

对无持久世界身份的远场 DepthCov 候选，图像位置通常比世界 XYZ 更可靠。FPR 因而不问“它在三维空间中离已有点多近”，而问“同一帧中，哪些候选正在竞争同一个投影区域和相近的深度数量级”。

这避免使用不稳定的远场 metric 坐标进行出生去重。

### 8.2 精确归属条件

候选只有同时满足以下条件才进入 FPR：

```python
far_field = (not sparse_depth_valid) and (depth >= 80.0)
```

因此：

- 80 m 之外的稀疏世界点仍由 TSC 管理；
- 80 m 以内的 DepthCov 候选仍由 TSC 管理；
- FPR 只管理没有持久身份的远场 DepthCov 出生。

这种“来源 + 深度”的联合条件比纯深度切分重要，因为它保留了远距离可靠世界点的 metric 身份。

### 8.3 输入与输出

输入：

- 候选像素 \((u_i,v_i)\)；
- 预测深度 \(z_i\)；
- 当前地图在该像素的结构残差 \(r_i\)。

输出：

- 当前 proposal batch 中保留的远场候选 mask。

### 8.4 投影-对数深度键

每个远场候选被映射到键：

\[
k_i=\left(
\left\lfloor\frac{u_i}{12}\right\rfloor,
\left\lfloor\frac{v_i}{12}\right\rfloor,
\left\lfloor\frac{\log z_i}{\log 1.10}\right\rfloor
\right).
\]

前两维把图像划成 12 px 网格；第三维是比例为 1.10 的对数深度 bin。使用对数深度而不是等距米制 bin，是因为远场深度误差更接近相对误差：100 m 与 110 m 的差异，应和 200 m 与 220 m 的差异处在相似尺度。

### 8.5 batch 内责任竞争

FPR 按残差 \(r_i\) 从高到低排序。对每个键只保留第一个候选，即该投影区域和深度层中当前最需要修复的候选：

```python
order = argsort(-residual, stable=True)
occupied_keys = set()
for i in order:
    key = (
        floor(u[i] / 12),
        floor(v[i] / 12),
        floor(log(depth[i]) / log(1.10)),
    )
    if key not in occupied_keys:
        keep[i] = True
        occupied_keys.add(key)
```

FPR 不查询 TSC，也不把通过的远场行登记到 TSC。通过 FPR 的 Gaussian 仍具有世界坐标，并在同一 3DGS 地图中正常渲染和优化；“投影责任”只发生在出生准入阶段。

### 8.6 当前实现的边界

当前 FPR 是**单次 proposal batch 内的投影 NMS**：

- 没有跨帧 FPR occupancy ledger；
- 已出生的远场 Gaussian 不会在后续帧被 FPR 查询；
- 远场 Gaussian 随无人机接近后不会自动迁移到 TSC；
- 路由标签在出生时决定，后续所有 Gaussian 仍在同一永久地图中优化。

所以当前方法准确的说法是“出生时的远场责任分流”，不是动态远近层管理。

### 8.7 创新性定位

投影网格、对数深度 bin 和局部 NMS 单独看都属于常见工具。FPR 的可辩护点不是这些算子的发明，而是：**只把缺乏持久世界身份的低视差远场候选从 metric 世界占用中移出，并用投影责任约束其单帧出生密度。**

它与 TSC 共同组成方法主线。若只报告 FPR 本身，很容易被理解为普通远场 NMS；论文应围绕“按几何可观测性选择责任坐标系”组织贡献。

---

## 9. TSC 与 FPR 的统一解释

PBSD 决定“不同距离各有多少 DepthCov 候选能够走到准入阶段”；TSC/FPR 决定“这些候选用什么坐标系判断出生责任”。两者解决的是不同问题，不能混为一个深度采样模块。

当前路由表如下：

| 候选来源 | 深度 | 出生预算 | 责任规则 | 提交后的地图 |
|---|---:|---|---|---|
| 稀疏世界点 | 任意 | 不受 PBSD 限制 | TSC 世界空间连续尺度覆盖 | 统一 3DGS 地图 |
| DepthCov | `<20 m` | PBSD 近段 25% | TSC | 统一 3DGS 地图 |
| DepthCov | `20-50 m` | PBSD 中段 45% | TSC | 统一 3DGS 地图 |
| DepthCov | `50-80 m` | PBSD 远段的一部分 | TSC | 统一 3DGS 地图 |
| DepthCov | `>=80 m` | PBSD 远段的一部分 | FPR | 统一 3DGS 地图 |

这套设计背后的判断不是“远处一定不可靠”，而是：

\[
\text{responsibility space}=
\begin{cases}
\text{metric world space}, & \text{persistent sparse identity},\\
\text{metric world space}, & \text{DepthCov and } z<80\text{ m},\\
\text{projective-log-depth space}, & \text{DepthCov and } z\ge80\text{ m}.
\end{cases}
\]

这个路由保留了可靠远点的世界身份，又避免把不稳定的远场单帧深度强行塞入世界空间占用结构。它是当前几何组织部分最完整、最适合形成论文主张的设计。

---

## 10. TGBR：时间梯度频带路由

**全名**：Temporal Gradient Band Routing

**代码位置**：`utils_new/streaming_appearance_lod.py`

**接入位置**：`SceneMapper.optimize` 的最后一个标准优化 step，在 `loss.backward()` 之后、optimizer step 之前

### 10.1 动机

统一 SH2 容量不足以表达明显的视角相关颜色；统一 SH3 又会给每个 Gaussian 分配额外参数，即使它只在相似视角中出现或高阶梯度主要是噪声。

关键问题不是“这个点看过多少次”，而是：

> 如果允许该点使用尚未激活的 SH3 系数，不同时间的正常训练梯度是否持续要求相似的修正方向？

如果梯度方向随视角来回抵消，说明 SH3 需求不稳定；如果梯度长期同向，说明低阶外观在重复留下系统性误差。

### 10.2 输入与输出

输入：

- 正常在线优化最后一步产生的 SH3 梯度；
- 当前 batch 中可见的 Gaussian ID；
- 每个 Gaussian 的历史梯度 EMA 和观察次数；
- 全局 SH3 最大比例 0.75。

输出：

- 每个 Gaussian 的有效最高阶数 \(d_i\in\{2,3\}\)；
- 对 SH3 梯度和参数的逐 Gaussian mask。

TGBR 不增加一次渲染，也不增加一次反向传播。它复用本来就会产生的梯度。

### 10.3 反事实 SH3 梯度

模型张量为所有 Gaussian 预留 SH3 参数，但未晋升 Gaussian 的 SH3 系数始终为零。正常 loss 反向传播后，代码在清零未激活梯度之前读取 SH3 频带梯度。

SH3 有 7 个球谐基函数、3 个颜色通道，因此对 Gaussian (i) 得到 21 维向量：

\[
\mathbf g_i^t=
\frac{\partial\mathcal L_t}
{\partial\mathbf c_{i,\mathrm{SH3}}}
\in\mathbb R^{21}.
\]

因为当前 SH3 系数为零，这个梯度可以理解为“若开放 SH3，当前损失希望它首先朝哪个方向变化”的一阶反事实信号。

### 10.4 时间方向一致性

对每次被观察到的 Gaussian，维护向量 EMA：

\[
\mathbf m_i^t
=\beta\mathbf m_i^{t-1}
+(1-\beta)\mathbf g_i^t,
\qquad \beta=0.9.
\]

设该 Gaussian 已累计观察 \(n_i\) 次，做冷启动偏差修正：

\[
\hat{\mathbf m}_i
=\frac{\mathbf m_i}{1-\beta^{n_i}}.
\]

最终分数为：

\[
q_i=\|\hat{\mathbf m}_i\|_2^2.
\]

这里必须注意，代码计算的是**梯度向量 EMA 的平方范数**，不是“梯度能量的 EMA”。若两个时刻的梯度大小都很大但方向相反，它们会在 \(\mathbf m_i\) 中抵消，\(q_i\) 随之降低。这正是 TGBR 对时间一致性的定义。

### 10.5 晋升规则

每 10 次证据更新执行一次晋升：

1. 只考虑观察次数 \(n_i\ge2\) 且 \(q_i>0\) 的 SH2 Gaussian。
2. 按 \(q_i\) 排序。
3. 当前 SH3 总数最多为
   \(\lfloor0.75N_t\rfloor\)。
4. 从高分到低分填充剩余名额。
5. 晋升是单调的：SH3 不会降回 SH2。

随着地图增长，\(N_t\) 增大，75% 上限也会打开新的名额。晋升时只改变有效频带 mask，不改变 means、scales、quaternions 或 Gaussian 数量。

### 10.6 严格的频带约束

TGBR 在 optimizer step 前把未激活 SH3 的梯度清零，并在 Adam 更新后再次把未激活 SH3 系数乘零。第二次约束很重要，因为仅清零当前梯度不能消除 Adam 历史动量带来的参数漂移。

```python
# loss.backward() 已完成
observe_inactive_sh3_gradient()

if evidence_updates % 10 == 0:
    score = squared_norm(bias_corrected_vector_ema)
    promote_top_score_until_floor(0.75 * gaussian_count)

mask_inactive_sh3_gradients()
adam_step()
force_inactive_sh3_coefficients_to_zero()
```

### 10.7 创新性定位

自适应 SH 阶数和外观 LOD 已有相关工作，不能作为宽泛 novelty。TGBR 更具体的区别是：

- 完全因果地在在线 mapping 中分配外观频带；
- 读取未激活 SH3 的反事实梯度，不需要额外 render；
- 用向量梯度跨时间的一致方向，而不是单帧误差、可见次数或梯度能量，判断容量需求；
- 在固定 SH3 数量预算下单调晋升。

这是一个可独立消融的算法模块，但当前实验增益较小。它应作为第二贡献，而不是替代 TSC/FPR 的几何主线。

---

## 11. 稀疏掉线时的覆盖恢复

**代码名**：`FrontViewCoverageRecovery`

**代码位置**：`utils_new/frontview_coverage_recovery.py`

**接入位置**：普通帧检查拒绝当前帧之后、关键帧更新之前

### 11.1 动机

前视 UAV 会出现长距离低纹理飞行、强运动模糊或视野中可追踪结构骤减。此时当前帧稀疏点很少，常规关键帧检查会拒绝它；但地图可能已经无法覆盖新视野。如果系统一直等待稀疏点恢复，就会在图像中留下大面积空洞。

覆盖恢复不把所有稀疏掉线帧都变成关键帧。它要求三种证据同时成立：

- 稀疏观测确实不足且旋转速度仍可接受；
- 当前帧相对上一个关键帧有足够时间和位姿新颖度；
- 当前 Gaussian 地图确实无法解释大面积图像。

### 11.2 触发条件

当前实现只在以下条件下继续检查：

- 当前帧被普通帧检查拒绝；
- 稀疏点数小于 10；
- 旋转速度不超过 10 degree/frame；
- 距上一个关键帧至少 40 帧；
- 平移至少 1 m，或相对旋转至少 3 degree。

随后渲染当前地图。像素 \(p\) 被定义为失败像素，当：

\[
\operatorname{mean}_{RGB}|\hat I_t(p)-I_t(p)|\ge0.08
\]

或

\[
\hat\alpha_t(p)<0.50.
\]

记失败像素比例为 \(\rho_t\)。只有 \(\rho_t\ge0.15\) 时，当前帧才被提升为恢复关键帧。

### 11.3 恢复候选的图像覆盖

普通 DepthCov 候选池从高结构残差像素中随机采样。恢复帧仍以 SSIM difference 超过候选阈值的非稀疏像素作为补点集合，但改为把图像划成 18 px 网格，每个网格只取结构残差最强的像素；若全图网格代表仍超过候选池上限，再取残差最高的代表。

这里要区分两张 mask：RGB 残差与低 opacity 的并集用于判断“当前帧是否需要恢复”，而实际 DepthCov 像素由高 SSIM difference mask 产生。低 opacity 本身不会直接生成一个补点像素。

这样做的目标是覆盖大面积未建区域，避免恢复预算再次集中到一块高纹理区域。选出的像素之后仍进入正常 DepthCov 预测和不确定性过滤。

### 11.4 因果远深度记忆

稀疏掉线恰好也会使 DepthCov 缺少深度锚点。若 DepthCov 有效候选少于 128，系统可用历史稀疏深度构造 fallback。

每一帧只统计 \([20,120]\) m 内的稀疏深度，取该帧 90% 分位数。系统保留最近 240 帧的这些分位数，再对窗口取 90% 分位数，得到远深度先验 \(z_{prior}\)。整个记忆只使用已经到达的帧。

为了避免无人机平移较大时把 fallback 放得过近，根据当前帧相对上个关键帧的平移 \(\Delta x\) 计算运动下界：

\[
z_{motion}=\min
\left(500,
\frac{\Delta x\,\bar f}{8}
\right).
\]

最终 fallback 深度为：

\[
z_{fallback}=\max(z_{prior},z_{motion}).
\]

只有已经存在有效历史先验时才执行 fallback；没有历史样本时不会凭空使用运动下界生成深度。fallback 置信度固定为 0.5。

### 11.5 恢复 Gaussian 的保守优化

fallback 深度不是逐像素精确几何，因此新生 Gaussian 使用更大的初始尺度：

\[
s_{recovery}=10s_{normal}.
\]

提交后：

- 新生组的 means 冻结；
- scale expansion 上限为 1.0，不允许进一步膨胀；
- 只优化该新生组 5 步；
- 若后续仍处于稀疏掉线，每 5 帧对该组做 1 步当前视图优化。

这些限制让恢复分支优先填补投影覆盖，而不是让弱深度证据通过位置漂移主导已有地图。恢复候选最终仍通过 PBSD 和 TSC/FPR，不存在另一套永久地图。

### 11.6 核心伪代码

```python
if normal_frame_check_rejected and sparse_count < 10 and rot_speed <= 10:
    if frame_gap >= 40 and (translation >= 1.0 or rotation >= 3.0):
        failure = (rgb_l1_mean >= 0.08) | (opacity < 0.50)
        if failure.float().mean() >= 0.15:
            mark_as_recovery_keyframe()
            residual_pixels = high_ssim_difference_non_sparse_pixels()
            pixels = strongest_residual_per_18px_cell(residual_pixels)
            depth, valid = depthcov(pixels)
            if valid.sum() < 128 and causal_sparse_prior_exists:
                depth[~valid] = max(sparse_far_prior, motion_depth_floor)
                valid[~valid] = True
            run_pbsd_and_tsc_fpr()
            commit_and_refine_newborn_group(steps=5)
```

### 11.7 创新性定位

残差触发补点、深度先验和网格采样分别都不是全新算子。该模块的价值在于把“稀疏观测掉线”“地图覆盖失败”和“位姿新颖度”组合成一个因果恢复证书，并对弱深度新生组施加专门的冻结与优化约束。

它适合作为前视 UAV 长时在线运行的鲁棒性贡献，而不是整篇工作的唯一主创新。

---

## 12. 在线优化、提交与生命周期

### 12.1 立即提交

PBSD 只决定 DepthCov 预算，TSC/FPR 决定出生责任。候选通过后立即加入当前 Gaussian group，并为优化器扩展对应参数行。没有等待未来帧确认的 candidate bank。

立即提交的优点是延迟低、控制流简单；代价是错误出生只能依靠后续 opacity 优化和剪枝清理。因此当前方法把主要可靠性约束放在出生前，而不依赖长延迟晋升。

### 12.2 在线损失

当前标准图像损失为 L1 与 SSIM 的组合：

\[
\mathcal L_{img}
=0.8\,\|\hat I-I\|_1
+0.2\,(1-\operatorname{SSIM}(\hat I,I)).
\]

此外使用尺度最小约束。设一个 Gaussian 的最大、最小轴尺度为 \(s_{max},s_{min}\)，当前比例阈值为 10：

\[
\mathcal L_g=
\mathbb E_i\left[
\min(10s_{min,i},s_{max,i})-s_{max,i}
\right]^2.
\]

总损失中 \(\lambda_g=1\)。当前 3DGS 主配置不引入额外 depth、normal 或 distortion loss。

### 12.3 局部窗口

当前主配置在每个关键帧运行 10 个优化 step，关键帧图包含最多 4 个全局窗口视图。TGBR 只在一次标准优化的最后一个 step 读取证据，避免在每个 step 重复统计高度相关的梯度。

### 12.4 opacity 剪枝

每处理 100 帧执行一次剪枝，删除

\[
\alpha_i\le0.01
\]

的 Gaussian。剪枝减少永久地图中的低贡献表示，但当前普通 TSC 空间覆盖行不会随之释放。这一点是表示生命周期与责任生命周期尚未完全耦合的实现限制。

### 12.5 固定终止优化

序列结束后，当前运行协议执行最多 100 步终止优化，camera optimization 关闭。该阶段仍可更新 Gaussian 几何和外观，并非冻结几何的纯外观重放。

由于这 100 步使用序列结束状态，论文中的在线速度应至少报告：

- 逐帧 mapping 时间；
- 100 步终止优化时间；
- 两者相加的 end-to-end 时间。

---

## 13. 最终渲染扩展：方向图像层

这一部分在当前通用启动器中启用，但它不参与 Gaussian 出生、TSC/FPR 责任判断或在线参数优化。若论文只讨论纯 3DGS 在线地图，可以关闭该扩展并只报告 PLY 渲染结果。

### 13.1 作用

当远场视差极低、metric Gaussian 难以恢复高频纹理时，系统保留少量历史图像作为方向外观锚点。在最终渲染时，若两个历史锚点经过纯旋转 warp 后对目标像素给出一致颜色，就用历史颜色与 Gaussian 渲染混合。

### 13.2 锚点采集

只有稀疏点数小于 10 的帧可以成为锚点。当前设置：

- 相邻锚点至少间隔 20 帧；
- FIFO 最多保存 12 个锚点；
- 保存 uint8 RGB、pose、内参和曝光。

锚点采集是因果的，但该层在序列结束、终止优化完成后才被激活。

### 13.3 锚点选择与 warp

对目标帧，只允许选择帧号严格更早的锚点。每个锚点的排序分数为：

\[
s(a,t)=\theta(R_aR_t^\top)
+\frac{\|\mathbf C_a-\mathbf C_t\|_2}{80}.
\]

取分数最小的两个锚点。当前 warp 只使用旋转单应：

\[
H_{a\leftarrow t}=K_aR_aR_t^\top K_t^{-1}.
\]

两个 warp 都有效且 RGB 平均绝对差不超过 0.12 时，像素通过一致性认证：

\[
m(p)=m_1^{valid}(p)m_2^{valid}(p)
\left[
\operatorname{mean}_{RGB}|W_1(p)-W_2(p)|\le0.12
\right].
\]

当前最终 mask 就是这个认证 mask，不再与 Gaussian depth 或 opacity 条件相交。第一个锚点提供替换颜色，第二个锚点只用于一致性认证：

\[
C_{out}(p)=
\begin{cases}
0.25C_{GS}(p)+0.75W_1(p), & m(p)=1,\\
C_{GS}(p), & m(p)=0.
\end{cases}
\]

### 13.4 输出边界

该层是 image-based hybrid renderer，不是 Gaussian LOD。`point_cloud.ply` 本身不能复现它，必须同时保存 `frontview_directional_layer.pt`。

论文中应分别给出：

- 纯 Gaussian 地图指标；
- Gaussian + directional layer 指标。

否则无法判断提升来自在线 Gaussian 建图，还是来自最终图像复用。

### 13.5 创新性定位

多帧图像 warp 和一致性融合已有大量先例。当前方向层更适合作为极低视差场景的渲染扩展或附录模块，不宜作为在线 3DGS 建图的核心 novelty。

---

## 14. 整体算法伪代码

下面的伪代码省略输入准备，只保留当前 mapping 主线。

```python
G = empty_gaussian_map(max_sh_degree=3)
tsc = ContinuousScaleCover()
tgbr = TemporalGradientBandRouter(birth_degree=2, target_degree=3)
far_depth_memory = CausalSparseFarDepthMemory(window=240)

for frame in ordered_stream:
    I, K, T_cw, sparse_world_points = frame
    far_depth_memory.observe(frame.sparse_depth)

    is_keyframe = online_keyframe_decision(frame)
    recovery = False

    if not is_keyframe:
        recovery = certify_sparse_dropout_coverage_failure(frame, G)
        is_keyframe = recovery

    if is_keyframe:
        # Optimize the map that existed before this frame's births.
        for step in range(10):
            render = rasterize(G, local_window)
            loss = image_loss(render, local_window) + gaussian_scale_loss(G)
            loss.backward()

            if step == 9:
                tgbr.observe_inactive_sh3_gradients(render.projection_info, G)

            tgbr.mask_inactive_gradients(G)
            adam_step(G)
            tgbr.force_inactive_coefficients_to_zero(G)

        residual = local_ssim_difference(render_current(G), I)
        sparse_candidates = residual_valid_sparse_candidates(frame)

        pool_pixels = high_residual_non_sparse_pixels(residual)
        if recovery:
            pool_pixels = strongest_pixel_per_18px_cell(pool_pixels)
        else:
            pool_pixels = random_subset(pool_pixels, max_size=2 * B)

        depth, std = depthcov(I, sparse_depth_anchors, pool_pixels)
        if recovery and count_valid(std) < 128:
            depth, std = causal_far_depth_fallback(depth, std, frame.motion)

        depthcov_survivors = uncertainty_filter(pool_pixels, depth, std)
        depthcov_candidates = PBSD(
            depthcov_survivors,
            budget=B,
            bands=(20.0, 50.0),
            fractions=(0.25, 0.45, 0.30),
        )
        candidates = sparse_candidates + depthcov_candidates
    else:
        candidates = all_current_sparse_candidates(frame)

    metric = [
        c for c in candidates
        if c.is_sparse or c.depth < 80.0
    ]
    projective_far = [
        c for c in candidates
        if (not c.is_sparse) and c.depth >= 80.0
    ]

    accepted_metric = TSC_filter(
        metric,
        radius_multiplier=0.3,
        scale_compatibility=1.0,
        color_threshold=0.15,
    )
    accepted_far = FPR_filter(
        projective_far,
        cell_px=12,
        depth_bin_ratio=1.10,
    )

    newborns = initialize_gaussians(accepted_metric + accepted_far)
    G.commit(newborns, opacity=0.5, sh_degree=2)
    tsc.register(accepted_metric)

    if recovery:
        freeze_newborn_means(newborns)
        optimize_only_newborn_group(steps=5)

    if processed_frames % 100 == 0:
        G.prune(opacity_threshold=0.01)

terminal_refine(G, max_steps=100)
save_ply(G)
```

---

## 15. 当前主配置参数

下表只列现行 mapping 和最终渲染路径真正使用的参数。

| 模块 | 参数 | 当前值 | 含义 |
|---|---|---:|---|
| Mapping | `initialization_frames` | 4 | 初始化帧数 |
| Mapping | `optimization_iters` | 10 | 标准关键帧优化步数 |
| Mapping | `global_window_size` | 4 | 全局关键帧窗口上限 |
| Terminal | `max_steps` | 100 | 序列结束固定优化上限 |
| Candidate | `err_threshold` | 0.05 | SSIM difference 候选阈值 |
| Candidate | `std_valid_threshold` | 0.06 | DepthCov 不确定性阈值 |
| Candidate | `pool_multiplier` | 2 | DepthCov 初始池为预算的两倍 |
| PBSD | `depth_edges_m` | 20, 50 m | 近/中/远预算边界 |
| PBSD | `depth_fractions` | 0.25, 0.45, 0.30 | 三段目标预算比例 |
| TSC | `camera_scale_rescalar` | 0.25 | 视图尺度缩放 |
| TSC | `radius_multiplier` | 0.3 | 世界空间责任半径 |
| TSC | `scale_compatibility` | 1.0 | 已有尺度兼容上限 |
| TSC | `color_distance_threshold` | 0.15 | RGB L2 兼容阈值 |
| TSC | `neighbors` | 32 | 每个候选查询近邻数 |
| TSC | `rebuild_rows` | 8192 | KD-tree 重建 pending 行数 |
| FPR | `depth_m` | 80 m | 无身份 DepthCov 远场边界 |
| FPR | `projective_cell_px` | 12 px | 投影责任网格大小 |
| FPR | `depth_bin_ratio` | 1.10 | 对数深度 bin 比例 |
| TGBR | `birth_degree` | 2 | 新生有效 SH 阶数 |
| TGBR | `target_degree` | 3 | 晋升目标阶数 |
| TGBR | `min_views` | 2 | 最少梯度观察次数 |
| TGBR | `promotion_interval` | 10 | 证据更新间隔 |
| TGBR | `utility_ema_decay` | 0.9 | 梯度向量 EMA 衰减 |
| TGBR | `max_target_fraction` | 0.75 | SH3 最大比例 |
| Recovery | `min_frame_gap` | 40 | 恢复关键帧最小间隔 |
| Recovery | `min_translation_m` | 1 m | 位姿新颖度平移阈值 |
| Recovery | `min_rotation_deg` | 3 degree | 位姿新颖度旋转阈值 |
| Recovery | `min_failure_fraction` | 0.15 | 地图失败面积阈值 |
| Recovery | `depth_fallback_cell_px` | 18 px | 恢复候选网格 |
| Recovery | `depth_prior_window_frames` | 240 | 因果深度记忆长度 |
| Recovery | `newborn_optimization_iters` | 5 | 新生组专用优化步数 |
| Lifecycle | `prune_interval` | 100 frames | opacity 剪枝周期 |
| Lifecycle | `opacity_prune_threshold` | 0.01 | 删除阈值 |

通用启动器按序列长度调整地图容量和每帧 DepthCov 预算，以避免长序列显存无限增长：

| 序列长度 | `max_pts_num` | `extra_pts_num` 即 \(B\) |
|---:|---:|---:|
| `<=1000` 帧 | 5000 | 3200 |
| `1001-2000` 帧 | 1500 | 1200 |
| `>2000` 帧 | 800 | 600 |

这是一项运行容量策略，不是算法贡献。跨场景比较时应记录 \(B\) 和最终 Gaussian 数，避免把更大出生预算误当成方法增益。

---

## 16. 代码实现索引

| 方法部分 | 主要文件 | 关键函数或类 |
|---|---|---|
| 总体在线循环 | `utils_new/scene_mapper.py` | `SceneMapper.run`, `update_kf`, `densification`, `optimize` |
| 候选生成、路由和提交 | `utils_new/gaussian_models.py` | `propose_new_gaussians`, `propose_new_gaussians_pts_only`, `commit_proposals` |
| PBSD | `utils_new/frontview_sampling.py` | `depth_stratified_indices` |
| TSC | `utils_new/frontview_scale_cover.py` | `FrontViewScaleCover.occupied_with_parents`, `register` |
| FPR | `utils_new/frontview_far_field.py` | `projective_survivor_mask` |
| TGBR | `utils_new/streaming_appearance_lod.py` | `persistent_gradient_utility`, `select_gradient_agreement_promotions` |
| TGBR 接入 | `utils_new/gaussian_models.py` | `observe_streaming_appearance_lod`, `mask_sh_degree_gradients`, `constrain_sh_degree_masks` |
| 覆盖恢复 | `utils_new/frontview_coverage_recovery.py` | `coverage_recovery_certificate`, `SparseFarDepthPrior`, `residual_grid_indices` |
| 恢复优化 | `utils_new/scene_mapper.py` | `refine_frontview_coverage_newborns`, `refine_frontview_coverage_tracking` |
| 最终方向层 | `utils_new/frontview_directional_layer.py` | `FrontViewDirectionalLayer.observe`, `composite` |
| 通用运行入口 | `scripts/run_360dvo_scene_directional.py` | `runtime_config`, `main` |

---

## 17. 与 MODP baseline 的区别

这里只比较数据已经进入 mapper 之后的建图过程。

两者共同使用在线 3DGS 的基本求解框架：稀疏点出生、残差区域的 DepthCov 补点、局部关键帧优化、Gaussian rasterization、立即提交和 opacity 剪枝。这些通用部分不作为本文贡献。

真正的数据流差异如下：

| 环节 | MODP baseline | 本文方法 |
|---|---|---|
| DepthCov 幸存者预算 | 对过滤后的候选使用原有统一预算顺序 | PBSD 在固定总预算内分配近/中/远幸存者 |
| 世界空间出生责任 | 离散 HashBlock 对候选做统一占用判断 | TSC 用连续世界尺度、尺度兼容和颜色兼容管理 metric 候选 |
| 远场无身份候选 | 与其他候选使用相同世界空间占用骨架 | FPR 在 `(u,v,log-depth)` 中做 batch 责任竞争 |
| 远近关系 | 同一种准入坐标系 | 按来源和可观测性选择世界或投影责任坐标系 |
| 外观容量 | 全局统一 SH 容量 | TGBR 在固定比例预算内把单点从 SH2 晋升到 SH3 |
| 稀疏掉线 | 普通帧检查可能长期跳过 | 残差认证覆盖恢复，并约束弱深度新生组 |

因此，本方法不是“替换了一个采样比例的 MODP”。其独立算法主线是：

> 先按透视深度分配出生机会，再按几何可观测性选择出生责任坐标系，最后按时间一致的反事实梯度分配外观频带。

---

## 18. 创新点应如何表述

### 18.1 第一主贡献：范围异质的 Gaussian 出生责任

最完整的贡献是 PBSD、TSC 和 FPR 形成的联合出生机制：

1. PBSD 在不确定性过滤后保证各深度范围拥有固定预算份额；
2. TSC 对 metric-observable 候选使用连续尺度世界覆盖；
3. FPR 对无身份远场候选使用投影-对数深度竞争；
4. 两类候选提交后进入同一张可微 Gaussian 地图。

这里真正的新 insight 是“责任坐标系应由可观测性决定”，而不是“远处用另一组参数”。它针对前视 UAV 中远场低视差与近场大投影面积并存的问题，具有明确可证伪性。

### 18.2 第二贡献：TGBR 在线外观容量路由

TGBR 复用正常训练中未激活 SH3 的梯度，用跨时间梯度方向一致性分配有限的高阶外观槽位。它不需要全序列重放，也不增加额外渲染。

这部分应强调“counterfactual inactive-band gradient”和“vector temporal agreement”，不能泛化成“自适应 SH 是新的”。

### 18.3 支撑贡献：稀疏掉线覆盖恢复

覆盖恢复解决前视飞行中长期缺少稀疏锚点的问题。它的价值在于触发证书和对弱深度新生组的保守生命周期，而不是某个单独阈值。

### 18.4 不应作为核心创新的内容

- DepthCov 网络本身；
- Gaussian rasterizer 和标准局部优化；
- KD-tree 数据结构；
- 单独的投影网格 NMS；
- 100 步终止优化；
- 方向图像层中的普通旋转 warp。

### 18.5 当前创新性的诚实结论

当前方法已经摆脱“统一离散世界占用”的建图组织方式，并形成了一个针对前视 UAV 可观测性差异的独立数据流。它比单纯改输入、增加迭代或重分配少量候选更有方法性。

但当前 FPR 仍是出生时、batch 内的投影责任，没有跨帧 ledger 和远到近动态交接；TSC 也还不是随 Gaussian 生命周期实时更新的层级结构。因此最稳妥的论文定位是：

> 一种面向前视 UAV 的 range-heterogeneous online Gaussian birth and appearance routing 方法。

现阶段不宜声称已经实现完整动态 LOD 地图或通用远近层级场景图。

---

## 19. 已有实验信号与准确结论

下面的数字用于帮助论文作者理解当前证据强度，不代替最终统一实验表。

### 19.1 PBSD

完整 PanoAir 的已记录对照：

| 方法 | PSNR | SSIM | LPIPS | GS 数 | 时间 |
|---|---:|---:|---:|---:|---:|
| 高容量统一候选基线 | 26.2057 | 0.79493 | **0.29422** | 641,792 | 411.09 s |
| PBSD | **26.4303** | **0.79691** | 0.29885 | **600,072** | **389.34 s** |

这个结果支持：PBSD 在更少 Gaussian 和更短时间下提高了 PSNR/SSIM。它也明确显示 LPIPS 退化，因此不能声称所有视觉指标同时改善。

### 19.2 TGBR

完整 PanoAir 与等 SH3 数量 shuffled 对照：

| Seed | TGBR PSNR | Shuffled PSNR | 差值 |
|---:|---:|---:|---:|
| 43 | 26.92350 | 26.87867 | +0.04483 dB |
| 44 | 26.83500 | 26.83191 | +0.00309 dB |
| 平均 | - | - | +0.02396 dB |

两次实验方向一致，说明梯度位置比随机分配略好；但增益很小，尚不足以证明强跨场景优势。TGBR 需要更多场景、种子和 all-SH3 等参数量对照。

### 19.3 最终方向层

一次实现检查中，纯 Gaussian source evaluation 为 27.20654 dB；双锚点方向层为 27.38877 dB、SSIM 0.90593、LPIPS 0.21386。该数字只是单场景渲染扩展检查，不是通用 mapping 结果，也不是 PanoAir 指标。

### 19.4 速度

一份 765 帧记录的 `online_recon_time=491.827 s`，约为：

\[
0.643\ \text{s/frame}\approx1.56\ \text{FPS}.
\]

这说明当前实现处于每秒 1 帧以上的量级，但论文仍需要在 CPU 空闲、单卡独占、相同输出设置下重新计时，并分开记录逐帧 mapping、终止优化、评估和视频编码。

---

## 20. 论文必须完成的消融

要把当前设计变成 solid 的方法工作，实验不能只比较最终方法和一个 baseline。每个核心判断都需要等预算反证。

### 20.1 PBSD 消融

- PBSD 与 uniform-survivor 使用同一个不确定性过滤后候选池；
- 每帧 DepthCov 预算 \(B\) 完全相同；
- 报告最终 GS 数、近中远出生数量和运行时间；
- 再做 shuffled-depth-band 对照，检验深度位置本身是否有效。

### 20.2 TSC 消融

- 连续尺度覆盖与固定半径覆盖比较；
- 关闭尺度兼容，只保留世界距离和颜色；
- 打乱候选深度与目标尺度的对应关系；
- 保持每帧最终提交数量一致，排除单纯少点带来的速度变化。

### 20.3 FPR 消融

- TSC-only：所有候选都走世界空间责任；
- 正常 TSC/FPR 路由；
- 在保持 FPR 候选数量不变时随机打乱远场责任归属；
- 分别统计 `<80 m`、`>=80 m` 区域的 PSNR/SSIM/LPIPS 和 floater 比例。

这个消融最关键，因为它直接检验“无身份远场应使用投影责任”这一主假设。

### 20.4 TGBR 消融

- 全 SH2；
- 全 SH3；
- 75% SH3 随机位置；
- 75% SH3 按梯度能量分配；
- 75% SH3 按 TGBR 向量梯度一致性分配。

所有方法必须使用相同优化 step 和相同最大 SH3 数量。

### 20.5 覆盖恢复消融

- 完整方法关闭恢复；
- 只用失败面积，不用位姿新颖度；
- 完整恢复证书；
- 完整证书但不冻结 recovery means。

除全局指标外，应单独评估稀疏点少于 10 的连续区间，以及这些区间之后地图是否留下漂浮层。

### 20.6 输出协议

每个实验至少报告：

- 在线地图最后一帧指标；
- 100 步终止优化后的 PLY-only 指标；
- 使用方向层时的 full-renderer 指标；
- online reconstruction time；
- terminal refinement time；
- Gaussian 数量和 SH2/SH3 数量；
- PBSD 三段出生统计、TSC 拒绝数、FPR batch 拒绝数、恢复帧数。

这样才能区分增益来自地图组织、外观路由、额外终止优化，还是最终图像融合。

---

## 21. 当前实现的主要限制

1. **FPR 没有跨帧责任记忆。** 它能控制单批次远场出生密度，但不能阻止同一结构跨帧重复出生。
2. **没有远到近动态交接。** 一个以 FPR 身份出生的 Gaussian 不会在接近后转入 TSC 管理。
3. **TSC 覆盖表与实际 Gaussian 生命周期不同步。** 普通 opacity 剪枝不会释放空间覆盖行，Gaussian means 更新也不会移动覆盖行。
4. **TSC 同 batch 去重不严格。** 同一批近场候选主要与历史行竞争，批内候选之间可能同时提交。
5. **TSC 使用帧级尺度。** 同一帧所有候选共享由稀疏深度中位数导出的 \(s_t\)，没有逐点 footprint。
6. **PBSD 段内仍是随机选择。** 它保证深度预算，但没有进一步利用置信度、空间覆盖或多视图支持排序。
7. **80 m 是固定 metric 阈值。** 输入世界尺度必须一致；不同飞行高度和焦距下，固定阈值未必最优。
8. **立即提交缺少未来确认。** 错误候选主要依赖后续 opacity 下降和剪枝清理。
9. **TGBR 当前增益较弱。** 需要跨场景和更多 seed 证明梯度方向一致性优于等预算随机位置。
10. **最终方向层不是纯 3DGS。** 它需要图像 sidecar，且在序列结束后激活，不能与 PLY-only 在线地图混报。
11. **终止优化不是 anytime mapping。** 100 步上限可控，但必须与逐帧速度分开报告。
12. **方法依赖输入的 metric pose 与稀疏世界点质量。** 本文把它们视为给定输入，但 mapping 无法从根本上修复系统性错误的外部几何。

这些限制并不否定当前方法，但决定了论文应把贡献落在“出生时的可观测性条件责任路由”，而不是夸大为已经完成所有尺度和生命周期的动态场景组织。

---

## 22. 可直接用于论文写作的技术摘要

本文研究给定按序 RGB、相机参数和稀疏世界观测条件下的前视无人机在线 3D Gaussian 建图。前视飞行同时具有近场残差密集、远场长期占据视野以及远距离单帧深度低可观测的特点。统一的候选预算会使近场吞噬 Gaussian 出生机会，而统一的世界空间占用又会把远场深度噪声放大为错误的重复或抑制关系。

为此，我们提出范围异质的 Gaussian 出生责任机制。Perspective-Balanced Survivor Densification（PBSD）在 DepthCov 不确定性过滤之后，将固定补点预算分配给近、中、远深度幸存者，保证远场在不增加总候选数的情况下持续获得表示能力。随后，方法根据候选的几何可观测性选择责任坐标系：带持久世界身份的稀疏候选以及可度量的非远场候选进入连续尺度覆盖 TSC，利用世界距离、出生视图尺度和颜色兼容关系判断已有表示是否足够；无持久身份的远场 DepthCov 候选进入 Far-Field Projective Responsibility（FPR），在图像网格和对数深度 bin 中竞争，由局部残差最大的候选承担当前 batch 的出生责任。通过两条路径的候选最终写入同一张可微 Gaussian 地图。

在外观方面，我们提出 Temporal Gradient Band Routing（TGBR）。所有 Gaussian 以 SH2 出生，模型为 SH3 保留零值参数。在线优化最后一步读取未激活 SH3 的反事实梯度，并维护其跨时间向量 EMA；只有梯度在多次观察中保持一致下降方向的 Gaussian，才在固定 75% 容量预算内单调晋升到 SH3。该过程复用正常反向传播，不增加额外渲染。对于稀疏世界点短时掉线，映射器进一步利用位姿新颖度、渲染残差和 opacity 覆盖构成恢复证书，并通过因果远深度记忆、网格化残差采样和冻结新生 means 恢复未建区域。

与使用统一离散空间占用和统一外观容量的在线 3DGS baseline 相比，本文方法的核心区别不是增加优化时长，而是把有限的 Gaussian 出生和外观容量分配给前视 UAV 中真正缺少表示、且其几何证据适合相应责任坐标系的位置。当前实现保持立即提交和有界局部优化，目标是在维持在线速度量级的同时，减少近场预算垄断、远场错误世界占用和不必要的高阶外观参数。

---

## 23. 一句话方法定义

> 本方法是一套面向前视无人机的在线 3D Gaussian 建图系统：它在深度过滤后平衡远近出生预算，按几何可观测性在连续世界尺度责任与投影远场责任之间路由候选，并用跨时间反事实梯度按需分配高阶球谐外观容量。
