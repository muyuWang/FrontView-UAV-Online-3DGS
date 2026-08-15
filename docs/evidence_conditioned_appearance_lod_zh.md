# Evidence-Conditioned Appearance LOD 实现与实验结论

## 结论

本轮保留的是一个默认关闭的几何锁定外观模块，不替换已有的 WorldTest-GS
`q_g` admission，也不修改 P/M/S/A 的几何状态。在线阶段仍使用 SH0；序列结束后，
模块根据每个 Gaussian 已获得的多视图证据分配 SH0、SH1 或 SH2 上限，再进行固定
步数的外观优化。

Road 上该模块在 400 步等预算下优于相同 SH0、等数量 shuffled-LOD，并略高于
MODP PSNR。Red Sculpture 和 PanoAir 上 PSNR 提升但感知指标回退，因此不会替换
这两个场景的当前默认配置。对称 scale 和 shrink-only scale 两条 footprint 路线均已
被实验否决，最终代码不包含 scale 优化。

## 数据流

1. 在线 mapper 用 SH0、四视图窗口和 25 次迭代生成固定地图。
2. 从完整轨迹等间隔选最多 80 个历史观测帧，仅做一次无梯度投影。
3. 对 Gaussian `g` 累积可见视图数 `n_g`、平均投影半径 `r_g` 和单位观察方向。
4. 用方向集中度计算角度离散度：

   `d_g = 1 - ||sum_v u_gv|| / max(n_g, 1)`。

5. 排序分数为：

   `s_g = log(1+n_g) log(1+r_g) sqrt(d_g + 1e-8)`。

6. SH1 要求至少 3 个视图、平均半径至少 1 px、角度离散度至少 `1e-4`；SH2
   要求至少 6 个视图、平均半径至少 2 px、角度离散度至少 `5e-4`。
7. 在合格集合中，SH1/SH2 总预算分别不超过全部 GS 的 65%/30%。未获准的 SH
   band 在初始化、梯度和 Adam step 后都保持严格为零。
8. means、scales、quats 全程冻结并在输出时逐 bit 检查。

运行会在 `results.json` 的 `offline_appearance_refinement.appearance_lod` 中记录
SH0/SH1/SH2 数量、分位数、固定 bin 直方图、观测帧数和证据预计算时间。

## Road 对照

所有数值来自完整 158 帧 `render.py` 直接 PLY 重载评测。时间包含在线重建和本轮
外观 refinement，不包含视频编码。

| 设置 | PSNR | SSIM | LPIPS | 总时间 |
|---|---:|---:|---:|---:|
| MODP | 22.9115 | 0.6794 | 0.3499 | 253.78 s |
| 25-iter + SH0/L2 | 22.9023 | 0.6584 | 0.3755 | 251.65 s |
| 25-iter + global SH2/L2 | 22.9296 | 0.6585 | 0.3753 | 259.23 s |
| 25-iter + shuffled LOD/L2 | 22.9140 | 0.6584 | 0.3754 | 262.42 s |
| 25-iter + evidence LOD/L2 | **22.9236** | **0.6585** | **0.3753** | 259.17 s |

证据 LOD 使用 `SH0 461425 / SH1 461425 / SH2 395507`。它比 shuffled 对照高
`0.0097 dB`，并在相同代码路径下取得略好的 SSIM/LPIPS，说明分配位置有弱但方向
一致的作用。它与 global SH2 的 PSNR 差 `0.0060 dB`，因此当前结果支持“以更少
高阶有效系数接近全局 SH2”，不支持“证据 LOD 全面优于全局 SH2”。

压缩视频上的 near-side edge PSNR 为 21.7240 dB，MODP 为 21.6140 dB；但
Laplacian 方差仍明显低于 MODP，近景纹理问题尚未完全解决。

## 跨场景门控

| 场景 | 当前/基线 | evidence LOD | 判断 |
|---|---|---|---|
| PanoAir ORB-SLAM3 | 26.1174 / 0.7887 / 0.3297, 342.16 s | 26.3558 / 0.7773 / 0.3376, 373.19 s | PSNR 提升但感知回退，不替换当前配置 |
| Red Sculpture MODP map | 25.3770 / 0.8114 / 0.1552, 1347.16 s | 25.4046 / 0.7990 / 0.1649, 1379.62 s | PSNR 提升但感知回退，不替换当前配置 |

三组 LOD 结果的 means/scales/quats 都逐 bit 不变，因此它不会生成新的空间
floater。PanoAir 的完整 RGB 和 depth 视频已输出到对应实验目录。

## Novelty 边界

本轮 IdeaSpark 对 `Footprint Whitening` 候选给出 `abandon`：它属于已有的块对角
Gauss-Newton/Jacobi 预条件族，不能作为论文贡献。本实现没有使用或包装该候选。

当前论文主张仍应限制为已经独立设计和验证的 WorldTest-GS all-path world identity
evidence `q_g` admission。Evidence-Conditioned Appearance LOD 是一个有
global/shuffled/equal-budget 控制的质量模块，但当前增益很小、使用 end-of-stream
轨迹且尚未完成独立相关工作审计，不能单独宣称为 solid novelty。

## 运行与回滚

完整 Road 运行入口：

```bash
cd /home/wmy/workspace_vla/Online-3DGS-Monocular
GPU_ID=4 bash scripts/run_road_evidence_lod_full.sh
```

当前改动位于分支 `feature/evidence-conditioned-appearance-lod-7-22`。改动前版本固定
在提交 `d95fd17` 和标签 `pre_far_near_optimization_7_22`；切换回该标签即可完全
回到本轮优化前状态。旧日志没有被覆盖。
