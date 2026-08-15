# Canonical WorldTest-GS MVP

## 1. Scope

This implementation separates two claims that must not be conflated:

1. The PanoAir `seq1` conversion mixed RTK poses with COLMAP camera-frame depth. This is a deterministic world-frame bug and is fixed by using one canonical reconstruction.
2. A proposal can remain shadow-only until a held-out multi-view world-identity score allows permanent Gaussian birth. This is the research mechanism tested by matched controls.

P/M/S/A, archive, frequency refinement, weak-parallax abstention, and coordinate conversion are not claimed as novelty here.

## 2. World-Frame Contract

`WorldFrameContract` is built from `conversion_stats.json` and allows permanent birth only for:

- `colmap_canonical`: COLMAP poses, track IDs, and sparse world points from the same reconstruction.
- `rtk_canonical`: RTK poses and sparse points triangulated under those same RTK poses.

The original hybrid `seq1` is marked `hybrid_frame_local`, with `permanent_birth_valid=false`. It can only commit when the explicit diagnostic stress override is enabled. Its certificates remain marked diagnostic-only.

Every proposal/group carries:

```text
world_frame_id, geometry_mode, calibration_version, source_frame_id,
track_id, pose_source, depth_source, source_kind
```

`CertificateAuthority` is process-local. `GaussianModel.commit_proposals`, detail birth, and archive restore fail closed whenever WorldTest is active and no authority-issued `AdmissionCertificate` is supplied. A child/detail certificate must refer to an authority-issued parent.

## 3. Data Flow

```text
frame + pose + sparse/depth/residual proposals
  -> proposal batch with track ID and source kind
  -> ShadowGroup (detached host state, no Parameter/Adam/hash/archive entry)
  -> at least 3 distinct observations
  -> shared nuisance estimate or cached sparse-track calibration
  -> held-out world-identity evidence
  -> q_g > log(19)
  -> AdmissionCertificate
  -> commit_proposals
  -> permanent Gaussian + optimizer/hash/coverage state
```

Untracked proposals receive negative track IDs from forward/backward pyramidal LK association. The cycle threshold is 1.5 px. Sparse tracks already containing at least three observations may run the same held-out test from the cached COLMAP track record at first appearance.

Shadow splats are rendered only for debug. They are detached tensors, rendered in a separate pass, and capped to 0.1 accumulated alpha. Formal PLY, PSNR, SSIM, opacity, primitive, and ground evaluations use permanent splats only.

An optional far-field sidecar is fitted after mapping from already observed frames. Bright, low-saturation depthless pixels vote in a 256 x 512 world-direction grid, with at most one vote per frame/bin. A direction needs support from at least 20 distinct frames before it can composite behind permanent opacity. This state is not a Gaussian birth and never enters the PLY, certificate authority, optimizer, hash, or GS budget. Permanent-only and permanent-plus-background metrics are always reported separately.

## 4. Evidence Computation

For a group `g`, every support view is held out in turn. The remaining observations predict `(u, v, inverse_depth)` under a shared world point. The independent alternative gives the held-out view its own latent point. The implemented decision is:

```math
e_{gh} = \log p(D_{gh} | H_W, D_{g,-h}) - \log p(D_{gh} | H_F)
```

```math
q_g = \min_{h,\kappa\in\{0.5,1,2\}}
      [e_{gh} - 1.645\sqrt{v_{gh}}]
```

Birth requires `q_g > log(19)`. Fewer than three views, weak parallax, rank failure, invalid projection, or non-finite terms return `-inf`.

The online nuisance solver uses one window-shared cubic B-spline pose correction and affine inverse-depth correction. The first pose coefficient is fixed for gauge, the knot stride is four frames, and the solve uses normalized Huber residuals with covariance-derived priors.

### MVP approximation boundary

The current real-data fast implementation is not the complete formal derivation in the design document:

- cached sparse tracks use a fixed calibrated nuisance state rather than re-solving the global spline posterior;
- the held-out predictor uses empirical point covariance and an analytic frustum-volume independent density;
- nuisance covariance is not fully propagated through a joint point/nuisance Schur complement.

The experiment therefore validates a narrower cached held-out world-consistency gate. It does not yet validate the stronger claim of an exact nuisance-marginalized Bayes factor under a Gaussian scene prior.

## 5. Falsification Controls

The true run writes `worldtest_commit_schedule.json`. Each control reads that schedule and matches the number of committed groups on every frame:

- `matched_delay`: closest source/age bucket to each true commit;
- `equal_count_random`: random ready group, exact per-frame count;
- `shuffled_qg`: score assignment shuffled within source/age buckets, exact per-frame count;
- `npo_lite`: lowest normalized world-point dispersion, exact per-frame count.

Cached but unselected offline groups remain ready in controls. This is required for exact matching and is covered by a regression test.

## 6. Complexity and Runtime

For `G` ready groups, at most `V` views, and `K=3` prior scales, held-out evaluation is approximately `O(G V^2 K)` small-matrix work. Shadow storage is `O(G V)`. Offline cache construction is `O(F N)` over frames and sparse points and occurs before the CUDA reconstruction timer, so both process wall time and internal online time are reported.

On 200 COLMAP-canonical frames:

- coordinate fast path: 147.67 s internal online time;
- true gate: 214.12 s internal online time and 235.48 s process wall time;
- overhead relative to the coordinate fast path: 1.45x internal time.

On the hybrid stress input, true gate takes 163.08 s versus 158.71 s for the unsafe coordinate fast path because most incompatible births are rejected.

## 7. Reproduction

```bash
cd /home/wmy/workspace_vla/Online-3DGS-Monocular

/home/wmy/anaconda3/envs/worldvln/bin/python -m pytest -q

/home/wmy/anaconda3/envs/worldvln/bin/python scripts/benchmark_worldtest_gs.py \
  --suite coordinate --frames 200 --gpu-ids 0,1,2 --seed 42 \
  --tag coordinate200_repro

/home/wmy/anaconda3/envs/worldvln/bin/python scripts/benchmark_worldtest_gs.py \
  --suite qg_true --frames 200 --gpu-ids 0,1 --seed 42 \
  --tag qg_true200_repro
```

Run each control suite with the corresponding true run's `worldtest_commit_schedule.json`, then use `scripts/postprocess_worldtest_gs.py` on the three manifests. The postprocessor creates full and frames 120-160 permanent-only RGB/depth/opacity/primitive videos, contact sheets, image metrics, and lawn diagnostics.

## 8. Limitations

- The canonical maps are sparse and do not recover high-quality dense texture; this experiment tests geometry admission, not a final reconstruction system.
- The fixed frame-120 bottom-half lawn ROI is configurable, but the low-count hybrid maps make 3 cm plane statistics noisy. Future-invalid rate and primitive side views are more reliable here.
- Cached COLMAP tracks may use observations later than their first online appearance. This is allowed by the offline-track protocol but is not a fully causal learned-depth path.
- DepthCov/residual/detail/archive bypasses are guarded and tested, but the first-round real experiment disables detail and archive as required.
- The complete joint nuisance covariance and proper Gaussian scene-prior evidence remain future work before a paper-level claim.
- The directional far field restores depthless sky but does not sharpen finite-depth grass, windows, trees, or bicycles. It must not be presented as a geometry or admission gain.
