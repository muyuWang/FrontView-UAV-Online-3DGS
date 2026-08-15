# LVBA Red_Sculpture 全量实验报告

日期：2026-07-21

## 1. 目的

使用 `data/Online3DGS_LVBA/Red_Sculpture` 的 FAST-LIVO2 位姿和 LiDAR 点云，检查准确、统一世界坐标下是否还会出现 PanoAir 中的地面 GS 漂浮，并评估当前 AeroCommit/PMSA 高效路径的重建速度、几何质量和外观质量。

数据共 1016 帧，图像分辨率为 640x512。相机内参为：

- `fx=646.784720`，`fy=646.657750`
- `cx=313.456795`，`cy=261.399612`
- `near=0.10 m`，`far=120.0 m`

## 2. 为 LVBA 增加的支持

1. `WorldTestGS` 坐标契约接受 `FAST-LIVO2 + persistent + normalized world`，并标记为 `geometry_mode=lidar_canonical`。
2. 新增 `scripts/build_lidar_voxel_point_ids.py`，以 10 cm 世界体素为每帧 LiDAR 点建立无碰撞持久 ID sidecar；不改变原始点坐标。
3. 新增 `scripts/evaluate_lidar_map_geometry.py`，双向计算 GS 地图与 LiDAR 样本的最近邻距离。
4. 新增 Red_Sculpture baseline、200 帧 smoke 和 1016 帧 full 配置。

全量当前方法使用：

- canonical-world `trusted_sparse_fast_path`
- 每帧最多提交 1200 个 LiDAR GS
- active map 预算 350k，超出后归档旧 GS
- 结束后对 active map 做 700 步 refinement
- 启用 far-field background

这里关闭了 `WorldTestGS.enabled`。原因不是绕过坐标检查，而是 `true_qg` 的逐点多视图再认证不适用于每帧独立重采样的准确 LiDAR 点：同一真实表面在相邻帧未必具有相同离散点 ID，特别是早期低运动阶段，三视图证据会错误拒绝大量有效点。200 帧 `true_qg` 结果作为负对照保留。

## 3. 使用配置

- 当前全量方法：`configs/lvba/Red_Sculpture_lidar_fastpath_full.yaml`
- 当前 200 帧 smoke：`configs/lvba/Red_Sculpture_lidar_fastpath_smoke200.yaml`
- `true_qg` 200 帧负对照：`configs/lvba/Red_Sculpture_worldtest_smoke200.yaml`
- baseline 全量配置：`configs/lvba/Red_Sculpture_baseline_full.yaml`

## 4. 可复现命令

在仓库根目录运行：

```bash
cd /home/wmy/workspace_vla/Online-3DGS-Monocular
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"

/home/wmy/anaconda3/envs/worldvln/bin/python slam_new.py \
  --config configs/lvba/Red_Sculpture_lidar_fastpath_full.yaml \
  --exp_name red_sculpture_lidar_fastpath_full_v1
```

运行会创建带时间戳的目录。以下命令使用本次已经完成的目录：

```bash
RUN_DIR=/home/wmy/workspace_vla/Online-3DGS-Monocular/Logs_lvba_worldtest/LVBA-Red_Sculpture-lidar-fastpath-full/2026-07-21-12-47-55_red_sculpture_lidar_fastpath_full_v1

/home/wmy/anaconda3/envs/worldvln/bin/python render.py \
  --run_dir "$RUN_DIR" \
  --output_dir "$RUN_DIR/videos_full" \
  --fps 24 \
  --device cuda:0 \
  --skip_novel \
  --save_opacity \
  --skip_primitives \
  --skip_view_detail \
  --skip_frequency_cache

/home/wmy/anaconda3/envs/worldvln/bin/python scripts/evaluate_render_vs_gt.py \
  --run_dir "$RUN_DIR" \
  --video "$RUN_DIR/videos_full/render_vs_gt.mp4" \
  --output "$RUN_DIR/validation_metrics.json"

/home/wmy/anaconda3/envs/worldvln/bin/python scripts/evaluate_lidar_map_geometry.py \
  --run-dir "$RUN_DIR"
```

## 5. 全量结果

图像指标来自解码后的同一 H.264 `render_vs_gt.mp4`，左右两半分别为 render 和 GT。该口径用于当前方法和 baseline 的公平对比。SSIM 在 0.25 倍分辨率计算。

| 方法 | 帧数 | 重建时间 | 秒/帧 | FPS | GS 数 | PSNR | SSIM | 非黑覆盖率 | 近侧 PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前 LiDAR fast-path | 1016 | 799.67 s | 0.787 | 1.27 | 359,364 active / 904,096 full | 14.937 | 0.558 | 95.815% | 15.990 |
| MODP baseline | 1016 | 1347.16 s | 1.326 | 0.754 | 2,871,438 | 25.051 | 0.925 | 100.000% | 26.539 |

当前方法比 baseline 重建快约 1.68 倍，总耗时减少约 40.6%，但 PSNR 低 10.11 dB。这里的 `359,364 active` 包含结束后的 24,576 个 stable detail split；`904,096 full` 包含渲染时重新合并的归档 GS。

当前方法分段结果：

| 帧区间 | PSNR | SSIM | 非黑覆盖率 | 近侧 PSNR |
|---|---:|---:|---:|---:|
| 0-199 | 14.471 | 0.556 | 91.709% | 15.561 |
| 200-399 | 14.971 | 0.526 | 96.835% | 15.959 |
| 400-599 | 14.502 | 0.578 | 97.113% | 14.818 |
| 600-799 | 17.054 | 0.588 | 99.373% | 18.946 |
| 800-1015 | 13.780 | 0.545 | 94.177% | 14.763 |

200 帧机制对照：

| 方法 | 时间 | GS 数 | PSNR | SSIM | 非黑覆盖率 |
|---|---:|---:|---:|---:|---:|
| `true_qg` | 336.36 s | 17,467 | 6.769 | 0.119 | 24.873% |
| LiDAR fast-path | 175.87 s | 191,221 | 12.196 | 0.625 | 79.066% |

fast-path smoke 的时间包含约 84 秒首次 GSplat 编译，不能直接作为稳定态吞吐。

## 6. 几何指标

| 方法 | Map->LiDAR median / P95 | LiDAR->Map median / P95 |
|---|---:|---:|
| 当前 LiDAR fast-path | 0.0487 m / 0.2757 m | 0.0239 m / 0.1835 m |
| baseline | 0.0562 m / 0.4899 m | 0.0163 m / 0.1200 m |

含义：

- 当前地图的 GS 到 LiDAR 距离更小，说明提交的点基本贴合输入几何，没有 PanoAir 那种系统性的地面抬升。
- baseline 的 LiDAR 到地图距离更小，说明 baseline 的表面覆盖更密。
- 该最近邻指标是几何代理；自适应 split 后的 GS 不必与某个原始 LiDAR 回波完全重合。

## 7. 视频与日志

当前方法：

- `videos_full/render_vs_gt.mp4`：RGB render/GT 对照
- `videos_full/render_depth.mp4`：深度视频
- `videos_full/render_opacity.mp4`：opacity 视频
- `videos_full/contact_sheets/`：三类视频的接触图
- `validation_metrics.json`：逐帧图像指标
- `lidar_geometry_metrics.json`：LiDAR 几何指标

baseline 的同口径视频位于：

`/home/wmy/workspace_vla/modp_raw/Logs_lvba_baseline/LVBA-Red_Sculpture-baseline/2026-07-21-02-02-20_red_sculpture_baseline_full_lio/videos_full_current_eval/`

## 8. 结论

Red_Sculpture 证明了准确 LiDAR 点云与位姿能消除 PanoAir 中最严重的世界坐标漂浮问题，因此 PanoAir 的输入几何确实是此前失败的主要瓶颈之一。但它同时证伪了“只要输入几何准确，当前高效 PMSA/AeroCommit 路径就能接近 baseline 画质”：当前方法几何稳定、速度更快，外观质量仍明显不足。

主要损失来自：

1. full map 只有 baseline GS 数量的 31.5%，表面覆盖和纹理采样不足。
2. 最终 700 步只优化 active map；544,732 个归档 GS 没有参加最终联合外观优化，重新合并渲染时产生颜色和曝光断层。
3. 当前 sparse-only densification 没有 baseline 的自适应 semi-dense 细节补点。
4. 常量式 far-field background 对蓝天拟合较差。

下一步最值得做的是轻量的 archive appearance fusion：固定归档 GS 几何，只联合更新其 SH/opacity/exposure；同时用 coverage/residual 触发局部细节 densification。这样针对本次已经定位的外观瓶颈，不重新引入几何漂浮，也比恢复 baseline 的 287 万 GS 更符合在线效率目标。

## 9. 质量模式后续实验

随后实现了完整地图的几何冻结外观重放：最终地图中的 `means`、`scales`、`quats` 不参与更新，只优化 SH 和 opacity；每帧曝光仅允许一个有界标量增益。所有重放都在结束时逐元素比较几何张量，只要任一元素变化就直接报错。

主要代码：

- `utils_new/scene_mapper.py::prepare_full_map_refinement`：恢复完整地图并冻结几何。
- `scripts/refine_saved_map_appearance.py`：从 PLY 和 tracked pose 重放完整轨迹，支持 L1/L2、uniform/hard replay、SSIM 权重和有界曝光。
- `render.py --ignore_cached_renders`：强制从最终 PLY 直接渲染，禁止复用 `eval/renders`。

对 PSNR 最有效的末段设置为：

- `color_loss_type=l2`
- `lambda_ssim=0.0`
- `sh_lr=0.002`
- `opacity_lr=0.02`
- `hard_fraction=0.0`，即全轨迹均匀重放
- `steps=1200`
- 重放后曝光比例范围 `0.9799-1.0070`

## 10. baseline 缓存问题与公平口径

旧 baseline 目录包含 `eval/renders/*.png`，旧版 `render.py` 会静默复用这些 RGB；新方法没有同类缓存，而是从最终 PLY 直接渲染。因此原先的 `25.051 dB` 不是严格统一的最终 PLY 口径。

| 结果 | RGB 来源 | PSNR | SSIM |
|---|---|---:|---:|
| 旧 baseline 报告值 | 缓存 `eval/renders` | 25.051 | 0.92476 |
| baseline 公平值 | 最终 PLY 直接渲染 | 24.665 | 0.91968 |
| 当前几何冻结质量模式 | 最终 PLY 直接渲染 | **25.292** | **0.93009** |

当前方法比旧缓存值高 `0.241 dB`，比公平 baseline 最终 PLY 高 `0.627 dB`。所有新旧 PLY 对比必须使用 `--ignore_cached_renders`。

## 11. 最终图像与几何结果

最终目录：

`Logs_lvba_quality/LVBA-Red_Sculpture-geometry-locked-sh2-l2-psnr-s0-lr002-op002-1200-v2`

1015 个可评测帧的最终指标：

| 指标 | 数值 |
|---|---:|
| PSNR | **25.29194 dB** |
| SSIM (0.25 scale) | **0.93009** |
| 非黑覆盖率 | 99.99998% |
| side-band PSNR | 24.66047 dB |
| near-side PSNR | **26.80533 dB** |
| edge PSNR | 23.63535 dB |
| near-side edge PSNR | 24.18815 dB |

几何冻结的精确检查：

- `means_max_abs_change = 0.0`
- `scales_max_abs_change = 0.0`
- `quats_max_abs_change = 0.0`

LiDAR 最近邻代理：

| 方向 | median | P95 |
|---|---:|---:|
| Map -> LiDAR | 0.05575 m | 0.49517 m |
| LiDAR -> Map | 0.01589 m | 0.11789 m |

抽帧检查中，RGB 的雕塑、楼体、道路和台阶保持连续；depth 没有出现随相机运动漂起的独立连续地面层；opacity 基本全覆盖。这里仍只把最近邻距离视为代理，几何稳定的强证据是重放前后几何张量精确相等。

## 12. 最终复现命令

从最佳前一阶段地图继续执行末段纯 L2 重放：

```bash
cd /home/wmy/workspace_vla/Online-3DGS-Monocular
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

/home/wmy/anaconda3/envs/worldvln/bin/python \
  scripts/refine_saved_map_appearance.py \
  --source-run Logs_lvba_quality/LVBA-Red_Sculpture-geometry-locked-sh2-l2-exposure-calibrated-v1 \
  --output-run Logs_lvba_quality/LVBA-Red_Sculpture-geometry-locked-sh2-l2-psnr-final-repro \
  --steps 1200 \
  --sh-lr 0.002 \
  --opacity-lr 0.02 \
  --hard-fraction 0.0 \
  --color-loss-type l2 \
  --ssim-weight 0.0 \
  --calibrate-exposure
```

最终直渲染必须显式禁用缓存：

```bash
RUN_DIR=Logs_lvba_quality/LVBA-Red_Sculpture-geometry-locked-sh2-l2-psnr-final-repro

/home/wmy/anaconda3/envs/worldvln/bin/python render.py \
  --run_dir "$RUN_DIR" \
  --output_dir "$RUN_DIR/videos_final_direct_ply" \
  --device cuda:0 \
  --ignore_cached_renders \
  --skip_novel \
  --save_opacity \
  --skip_primitives \
  --skip_view_detail \
  --skip_frequency_cache \
  --skip_background
```

## 13. 时间边界与下一步

最终实验目录记录的累计处理时间为 `1949.85 s`。它包含全量重建和为参数筛选依次执行的多段外观重放，因此不能宣称达到 baseline 的 `1347.16 s` 在线速度。单独最后 1200 步纯 L2 重放加曝光为 `197.61 s`。

本轮结论是：PSNR 追平/超过 baseline 且几何不动已经成立；“同等画质同时保持 baseline 级处理时间”仍未成立。下一步应把多段重放蒸馏为一次短程在线 appearance replay，例如按覆盖度维护帧 reservoir、只更新当前可见 SH/opacity，并用验证集早停，而不是继续增加全轨迹离线步数。

另外修复了 PLY 重载时每次累积 4 个初始化哑元 GS 的问题；连续两次读写点数保持不变，并由 `tests/test_gaussian_ply_reload.py` 覆盖。
