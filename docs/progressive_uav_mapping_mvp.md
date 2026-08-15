# Progressive UAV Mapping MVP

This prototype adds a causal, single-level hierarchy for forward-facing monocular UAV sequences. It is opt-in through `ProgressiveMapping.enabled`; the default is `false`, and the original MODP densification path remains unchanged when disabled.

## States and storage

| State | Representation | Rendering | Optimizer/storage |
| --- | --- | --- | --- |
| `PROJECTIVE` | Cross-frame `ProjectiveAnchor` with four inverse-depth modes | Detached fronto-parallel proxies for visualization only | No hash entry and no optimizer |
| `METRIC` | One coarse 2DGS or 3DGS root | Active; no children are rendered | One independently removable GPU group with Adam |
| `SURFACE` | A configurable square 2DGS or 3DGS grid selected from depth, projected size, and residual | Active; the parent group has been removed | One variable-row GPU group with Adam |
| `ARCHIVED` | Coarse A proxy plus saved fine children | Frozen A proxy is passed as an external splat | Fine rows are CPU FP16; no gradients or Adam state |

Each persistent root has a stable tree `node_id`. A metric or surface root owns one Gaussian group, while the registry maps every surface child ID to its row in that group. Removing a group destroys the complete parameter and optimizer objects, avoiding mutable global-row indices.

## Per-frame data flow

After bootstrap, `SceneMapper` performs the following strictly causal sequence:

1. Render the stable map from legacy seed Gaussians, active M/S groups, and A proxies. P proxies are excluded from this coverage render.
2. Build a candidate mask from low opacity or high RGB residual, apply optional invalid-mask hooks, and select at most one patch per grid cell.
3. Associate observations to visible S/M roots, then projected P anchors using a 2D bin index. Only unmatched new-region observations create P anchors.
4. Update all four P inverse-depth likelihoods, normalized entropy, relative uncertainty, descriptor/color EMA, and maximum parallax.
5. Promote observable P tracks to M roots after 3D/appearance deduplication.
6. Update visible root statistics, split eligible M roots into S children, reactivate visible archives, and archive stale or over-budget S roots.
7. Clamp optimized S scales against their creation scale and, when visible, a maximum projected sigma so dense children cannot grow back into coarse image-sized blobs.
8. Apply hard P, M, S, and total-active budgets and append `progressive_stats.jsonl`.
9. Every debug interval, append P posterior histograms, per-promotion-condition
   failure counts, and P/M/S/A depth-band counts to `progressive_p_histograms.jsonl`.

P mode projection, persistent-root projection, patch description, and residual
support evaluation are batched on the GPU. The registry keeps an explicit root
ID list, so root queries do not scan every S child. These are behavioral
equivalents of the original per-item calculations, but avoid thousands of CUDA
scalar synchronizations and prevent per-frame cost from growing with the total
number of child nodes.

Near-depth observations receive a bounded sampling quota rather than taking the
entire frame budget. On `road_street1`, 70% of extracted observations and 85%
of newly spawned P anchors are reserved for valid depth at or below 30 m. Those
anchors use the near-only P-to-M thresholds; P without a valid near sparse-depth
sample continues to use the global thresholds.

The defaults use 2x2, 3x3, and 4x4 M-to-S grids. The denser `road_street1`
profile uses 3x3, 7x7, and 9x9 grids, with the two fine modes selected at 60 m
and 30 m. Each child receives a bilinearly sampled color from the observation's
appearance grid instead of copying one patch mean. Near children use smaller
scale ratios so the additional points increase local detail instead of
reproducing the same coarse splat.

A P anchor is a scene track, not a per-frame object. A successful association updates the existing anchor and prevents a duplicate spawn. Greedy matching enforces at most one observation per anchor and one anchor per observation in a frame.

## Bootstrap and baseline behavior

For `bootstrap_frames`, MODP's existing sparse/semi-dense densification creates
a legacy seed map. With
`replace_original_densification_after_bootstrap: true`, both original keyframe
and point-only additions stop after bootstrap and new regions must follow P to
M to S to A. Legacy seed Gaussians continue rendering and optimizing but do not
enter the tree.

The `road_street1` quality profile sets this option to `false`. After bootstrap,
the original causal densification remains as a stable support layer while P/M/S
processes the same current frame and fills residual or low-opacity regions. This
hybrid is still online and uses no future frames, but it should be treated as a
separate ablation from the pure replacement mode.

The hybrid profile also performs one current-view optimization immediately
after each densification pass, including bootstrap and non-key frames. This is
important for the large rotations near the start of `road_street1`: newly added
legacy or progressive Gaussians no longer wait for a later key frame before
receiving their first image gradient. Regular key-frame optimization uses 16
steps, assigns 10 of them to the current camera, and optimizes at most 768
visible progressive roots per step.

Global Gaussian count is not a coverage metric. A root may be behind the
camera, outside the frustum after the early viewpoint change, or frozen by the
per-step optimization budget. Use `num_visible_roots`, `num_optimized_roots`,
and `num_frozen_roots` from `progressive_stats.jsonl` together with rendered
opacity when diagnosing a sparse current view.

Set the feature off without changing any original config:

```yaml
ProgressiveMapping:
  enabled: false
```

## Promotion, refinement, and archive

P to M requires observation count, posterior best weight, normalized entropy, relative inverse-depth standard deviation, parallax, and match-error thresholds. A candidate first queries existing M roots using world distance, scale, and RGB descriptor similarity. A match merges support instead of creating a second root.

M to S requires sufficient observations, projected size, residual, confidence,
and budget. Children are offset along the parent's rotated tangent axes. The M
group is removed before the S group becomes active, so parent and children never
fully render together. Center regularization and a hard per-child scale bound
constrain optimized children to their original parent support.

S to A copies raw fine parameters to CPU FP16, persists them in `progressive_archive/root_XXXXXXXX.pt`, and removes the GPU group and its Adam objects. The original M snapshot becomes a frozen coarse A proxy. Re-entry above the configured projected radius reloads the FP16 children to a new GPU group and initializes fresh Adam state.

`point_cloud_progressive_full.ply` contains legacy seeds, active M/S rows, and fine CPU archive rows. An A proxy is used only when its detailed archive is unavailable. Archived detail is concatenated on CPU and does not need to be restored to GPU.

## Running

The replacement-mode configuration uses the existing 2DGS rasterizer:

```bash
python slam_new.py \
  --config configs/progressive_uav/progressive_uav_road_street1.yaml \
  --exp_name progressive_uav_hybrid_scale
```

The baseline-quality profile uses the native HorizonGS 3DGS renderer, preserves
baseline densification and multi-view optimization, and adds P/M/S/A as an
incremental layer:

```bash
python run_slam_worldvln.py \
  --config configs/progressive_uav/progressive_uav_road_street1_baseline_quality_short.yaml \
  --exp_name progressive_baseline_quality_road40
```

Measure an emitted trajectory video with the same metric used below:

```bash
python scripts/evaluate_render_vs_gt.py --run_dir <run-directory>
```

CPU unit and state-machine smoke tests:

```bash
/home/wmy/anaconda3/bin/python -m pytest -q tests/test_progressive_mapping.py tests/test_render.py
python scripts/run_progressive_smoke_test.py
```

## road_street1 benchmark

### Baseline-quality 3DGS profile

The quality gap was not primarily caused by too few P/M/S points. The earlier
profile changed four variables at once: it forced 2DGS, replaced HorizonGS
multi-view optimization with mostly current-view updates, reduced online
optimization, and constrained S opacity and scale so erroneous surfels could
not fade out. The baseline-quality profile restores native 3DGS, 30 online
steps, local/global camera batches, camera optimization, and 1000-step final
refinement. P/M/S opacity floors are zero, scale bounds are relaxed, and the
original densified map remains the quality-bearing support layer.

Matched 40-frame results use the same input frames, 3DGS renderer, online
optimization count, camera optimization, final-refinement count, tracked poses,
and decoded-video metric. The run used physical GPU 3 because the requested GPU
4 was occupied; the matched baseline ran on physical GPU 6.

| Run | PSNR | SSIM (0.25 scale) | Nonblack coverage | Gaussians | Reconstruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| HorizonGS matched baseline | 20.370 | 0.6591 | 99.841% | 223,070 | 170.96 s |
| Progressive v24 quality | 20.064 | 0.6492 | 99.825% | 271,403 | 647.48 s |
| Progressive v22 | 15.018 | 0.3517 | 97.679% | 352,050 | 376.76 s |

Progressive v24 is within 0.305 dB PSNR, 0.010 SSIM, and 0.016 percentage
points of matched baseline coverage. It improves over v22 by 5.046 dB PSNR,
0.298 SSIM, and 2.146 coverage points. Visual comparison shows that the
remaining gap is light blur and incorrect surfels around road and vegetation,
not missing scene coverage.

The v24 final-refinement mode freezes progressive M/S parameters while keeping
them in the render. On the 20-frame smoke test this reduced 200-step refinement
from 59.67 to 21.31 seconds and total reconstruction from 248.85 to 207.62
seconds. On 40 frames, however, 1000-step refinement still took 211.36 seconds
because 47,176 S rows remain split across about 1000 root groups and must be
assembled for every render. The next performance optimization is to compact
frozen M/S rows into one read-only render group before final refinement; simply
removing their Adam state is not enough.

### v31 all-frame replay quality profile

The remaining blur was concentrated in non-keyframes rather than caused by an
insufficient Gaussian count. On the 20-frame sequence, most keyframes already
rendered at 22--23 dB, while several non-keyframes were only 17--20 dB because
the final refinement sampled only the 11 keyframes. v31 keeps every input frame
on CPU and replays four-frame batches during final refinement. Half of each
batch is sampled uniformly and half is sampled in proportion to an EMA of the
per-frame L1 error. All replayed cameras receive SE(3) pose optimization except
the first camera, which fixes the world-coordinate gauge.

Before replay, all progressive M/S groups are compacted into the managed
baseline group. This removes thousands of per-root render/optimizer groups and
makes their position, scale, opacity, SH color, and rotation jointly trainable.
The final loss adds a near/high-gradient color term and an explicit grayscale
gradient-matching term. Low-opacity high-gradient pixels are included so that
missing near-detail regions are not excluded by an invalid rendered depth.
SH degree 1 provides view-dependent color, and progressive candidate processing
starts at frame zero on every input frame while still requiring valid depth for
new candidates.

The selected full configuration is
`configs/progressive_uav/progressive_uav_road_street1_quality_full40.yaml`.
Its decoded 40-frame trajectory result on physical GPU 6 is:

| Run | PSNR | SSIM | Bottom-half PSNR | Edge PSNR | Coverage | Gaussians | Reconstruction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Progressive v31 | 22.213 | 0.7605 | 21.071 | 19.948 | 99.796% | 279,217 | 748.70 s |
| Progressive v24 | 20.064 | 0.6492 | 19.186 | 18.466 | 99.825% | 271,403 | 647.48 s |

The v31 render Laplacian variance is 142.2 versus 107.4 for v24 and 566.8 for
GT. The nearby bicycles, railings, vegetation layers, tree trunks, road texture,
and facade windows are structurally recognizable, although still smoother than
GT. Reconstruction time is 378.43 seconds main optimization, 78.72 seconds
progressive processing, 16.71 seconds newborn optimization, 53.58 seconds debug
rendering, and 206.94 seconds final all-frame refinement. This result meets the
22 dB trajectory goal but is not real-time.

The emitted result directory also contains 40-frame trajectory RGB, depth, and
primitive videos and 240-frame novel-orbit RGB, depth, and primitive videos.
The trajectory result is the validated target. The 360-degree orbit moves well
outside the observed camera manifold and still exposes floaters and incorrect
surfaces, so it should not be treated as a completed omnidirectional model.

### Earlier 2DGS profiles

On physical RTX 4090 GPU 4, the 40-frame short sequence with the quality profile
completed online reconstruction in 269.59 seconds and produced 352,343
Gaussians. The previous unbatched 424,711-Gaussian run took 2,620.78 seconds.
The final render-vs-GT video measured PSNR 14.93, downscaled SSIM 0.332, and
97.65% nonblack coverage. These measurements include debug generation in the
reconstruction time but exclude the separate final video-rendering command.

The v22 profile keeps the same 16 regular optimization iterations and Gaussian
budgets, but increases `newborn_optimization_iters` from one to two. This gives
newly densified baseline points and promoted/refined progressive roots one extra
causal current-view update before the next input frame. On the same 40-frame
sequence it produced 352,050 Gaussians with PSNR 15.018, downscaled SSIM 0.352,
and 97.679% nonblack coverage (maximum RGB channel greater than 8/255). The
matched v15 measurements are 14.929, 0.332, and 97.655%. The v22 reconstruction
took 376.76 seconds in this run: 223.95 seconds regular optimization, 74.42
seconds progressive processing, 44.47 seconds newborn optimization, and 18.73
seconds debug rendering. The quality gain therefore comes with a measured
latency increase and remains far from real time.

Guarded sparse-depth M-root correction is implemented as an optional, disabled
ablation. It checks depth support, reprojection error, descriptor agreement,
relative depth error, and depth uncertainty before updating an unrefined root.
On this sequence, both hard-wait and opportunistic variants failed to improve
PSNR, SSIM, and coverage together, so the road profile leaves
`enable_metric_depth_correction: false`.

The v15 run's 269.59 seconds are dominated by full-resolution optimization:
143.23 seconds for regular map optimization, 71.16 seconds for progressive
processing, 22.67 seconds for immediate post-densification optimization, and
18.24 seconds for debug rendering. This is substantially faster but is not a
real-time result.

## MVP limitations

- The legacy seed remains outside the hierarchy.
- Resized RGB patches are used instead of learned descriptors or semantic/dynamic masks.
- Root visibility uses center/frustum tests and does not estimate full occlusion.
- The A proxy is the saved coarse M root rather than a learned distilled primitive.
- Archive files use FP16 tensors without entropy coding.
- There is one adaptive square-grid split and no loop closure or recursive tree.
- S children still start from a reference-frame fronto-parallel patch and a
  discrete monocular depth hypothesis. More points or optimization steps cannot
  fully correct wrong depth and plane orientation; multi-view plane/normal or
  depth-aware root refinement is required for clean street geometry.
- The second newborn step improves trajectory-view structure but does not remove
  large floaters or make the 360-degree novel orbit geometrically complete.
- The 3DGS quality profile reaches baseline trajectory quality, but its
  progressive groups still add substantial online and final-refinement latency.
