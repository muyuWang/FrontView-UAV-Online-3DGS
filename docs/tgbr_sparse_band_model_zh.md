# TGBR 稀疏高阶频带模型

## 1. 动机

TGBR 为所有 Gaussian 分配 SH2 基础外观，只把具有持续一致 SH3 梯度方向的
Gaussian 晋升到 SH3。原实现虽然把未晋升 SH3 系数约束为零，但标准 PLY 仍为每个
Gaussian 保存完整 SH3，导致容量路由没有转化为持久化模型收益。

当前实现把 TGBR 的选择结果直接映射为模型结构：

- 所有 Gaussian 保存 SH0、SH1、SH2；
- 每个 Gaussian 使用 1 bit 表示是否具有 SH3；
- 只有已晋升 Gaussian 保存 7 个 RGB SH3 系数；
- 文件加载时恢复为渲染器原有的 dense `[N, 15, 3]` 张量。

因此该格式不改变在线优化、Gaussian 数量、几何或任何非零 SH 系数。它优化的是最终
Gaussian 模型大小，不宣称降低在线峰值显存或在线重建时间。

## 2. 单文件布局

自定义 binary PLY 包含三个 element：

```text
vertex[N]          geometry + opacity + SH0/SH1/SH2
tgbr_mask[ceil(N/8)]
                   packed SH3 activation bits
tgbr_high_sh[M]    SH3 RGB coefficients of M active Gaussians
```

`tgbr_high_sh` 的行顺序与 mask 中置位的 Gaussian 顺序一致，不需要为每个激活行额外
保存 32-bit index。PLY comment 保存格式版本、基础阶数、目标阶数和 bit order。

对于 SH2 到 SH3 的路由：

```text
dense SH3 bytes per Gaussian = 248
sparse base bytes per Gaussian = 164
SH3 bank bytes per active Gaussian = 84
mask bytes per Gaussian = 1/8
```

若 SH3 比例为 `r`，忽略固定 header 后的模型降幅为：

```text
1 - (164 + 0.125 + 84r) / 248
```

当 `r=0.75` 时理论降幅约为 8.42%；实际 PanoAir 最终有效 SH3 比例约为 73.18%，
降幅约为 9.03%。

## 3. 实现位置

- 编解码和大小统计：`utils_new/tgbr_sparse_model.py`
- 在线结束时按 degree 导出：`utils_new/gaussian_models.py`
- 输出路径选择：`utils_new/scene_mapper.py`
- Dense PLY 无损转换：`scripts/compact_tgbr_model.py`
- PanoAir 配置：`configs/frontview_uav/panoair_tgbr_sparse_band_full.yaml`
- 回归测试：`tests/test_tgbr_sparse_model.py`

配置开关：

```yaml
StreamingAppearanceLOD:
  enabled: true
  birth_degree: 2
  target_degree: 3
  max_target_fraction: 0.75
  sparse_model_export: true
```

导出前会检查所有未激活 Gaussian 的 SH3 系数严格为零。若不满足，程序报错而不是丢弃
系数。加载时检查 mask 长度、置位数量、频带宽度和 metadata，再恢复完整 SH tensor 及
`appearance_sh_degree`。

## 4. PanoAir 完整序列结果

源模型为 matched 实验中的最高质量 `dense TGBR-75`：

```text
Gaussian 数量:       672981
非零 SH3 Gaussian:   492520
标准 dense PLY:      166900819 bytes
TGBR sparse PLY:     151826403 bytes
模型文件降幅:        9.031960%
```

转换后 SH tensor 逐 bit 一致。对 2230 帧重新渲染且不计算 LPIPS：

```text
                      PSNR         SSIM
dense TGBR-75         26.541345    0.792501
sparse-band TGBR-75   26.541345    0.792501
```

两份 `render_metrics.json` 的全部逐帧记录相同。验证目录：

```text
Logs_frontview_uav/benchmarks/
  tgbr_sparse_band_full_seed43_v138_20260807/
```

## 5. 结论边界

当前可以声称 TGBR 的证据路由带来超过 8% 的最终 Gaussian 模型大小收益，并且该收益
对当前 PanoAir 模型是无损的。

当前不能声称：

- 在线峰值显存降低 8%；
- 在线重建时间加速 8%；
- 外部标准 3DGS 工具无需修改即可读取该多 element PLY；
- 单场景结果已经证明跨场景普适性。

仓库内 `render.py` 和 `GaussianModel.load_from_ply` 可以直接读取该格式。需要标准 PLY
兼容时，应继续保留默认 `sparse_model_export: false` 路径。
