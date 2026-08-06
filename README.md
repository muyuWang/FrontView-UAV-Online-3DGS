# Range-Aware Front-View UAV Online 3D Gaussian Mapping

This repository contains the current code snapshot of a monocular online 3D
Gaussian mapping system for forward-looking UAV video. The mapping method is
designed around the depth imbalance and weak far-field parallax of front-view
flight, while keeping a single incrementally optimized Gaussian map.

The repository is a clean source export. Datasets, checkpoints, rendered
videos, evaluation outputs, and experiment logs are intentionally excluded.

## Mapping Contract

At frame `t`, the mapper consumes:

- the RGB image and camera intrinsics;
- the ordered world-to-camera pose;
- currently visible sparse points with persistent world identities.

Tracking and pose estimation are outside the method boundary. Mapping is
causal: online admission and optimization use only the current and previous
frames. The output is an incrementally updated 3D Gaussian map.

## Active Method

The current pipeline combines five mapping components:

1. **PBSD** (`utils_new/frontview_sampling.py`) reserves a fixed DepthCov birth
   budget across near, middle, and far depth ranges after uncertainty filtering.
2. **TSC** (`utils_new/frontview_scale_cover.py`) admits metrically observable
   candidates through continuous world-space scale and color coverage.
3. **FPR** (`utils_new/frontview_far_field.py`) assigns low-parallax far-field
   DepthCov candidates by projective cells and logarithmic depth bins instead
   of a Cartesian hash grid.
4. **Residual-certified coverage recovery**
   (`utils_new/frontview_coverage_recovery.py`) restores candidate birth during
   temporary sparse-point dropout using residual, coverage, and motion gates.
5. **TGBR** (`utils_new/streaming_appearance_lod.py`) promotes selected
   Gaussians from SH2 to SH3 from persistent online gradient agreement.

`utils_new/gaussian_models.py` implements candidate generation and admission;
`utils_new/scene_mapper.py` integrates the components into the online mapper.
The default launcher sets `HashBlock.use_hash: false`.

The optional causal directional far-field layer is implemented in
`utils_new/frontview_directional_layer.py`. It is a final rendering extension
and does not alter the Gaussian map or the online birth rules.

For the full task definition, module inputs/outputs, equations, execution
order, configuration, and implementation references, see
[docs/frontview_uav_online_reconstruction_paper_handoff_zh.md](docs/frontview_uav_online_reconstruction_paper_handoff_zh.md).

## Environment

The code targets Linux, Python 3.9, PyTorch, and CUDA 11.8. DepthCov is included
as a git submodule.

```bash
git clone --recurse-submodules <repository-url>
cd FrontView-UAV-Online-3DGS
conda create -n frontview-3dgs python=3.9
conda activate frontview-3dgs
bash setup_env.sh
```

The first run may compile CUDA extensions. Make sure the CUDA version used by
PyTorch matches the local toolkit.

## Running One 360DVO Scene

The generic entry point prepares one supported scene, runs online mapping,
renders the full sequence, and writes metrics and videos under `--save-dir`:

```bash
python scripts/run_360dvo_scene_directional.py \
  --scene mountains \
  --save-dir /path/to/output \
  --gpu 0 \
  --data-root /path/to/Online3DGS_360DVO \
  --prepared-root /path/to/prepared \
  --work-root /path/to/work \
  --cache-root /path/to/cache \
  --python /path/to/environment/bin/python \
  --cuda-home /usr/local/cuda-11.8 \
  --orb-binary /path/to/ORB_SLAM3/Examples/Monocular/mono_tum_vi \
  --orb-vocabulary /path/to/ORB_SLAM3/Vocabulary/ORBvoc.txt
```

Use `--dry-run` to validate paths and generate the runtime configuration without
starting reconstruction. The data and ORB-SLAM3 binaries are not distributed in
this repository.

For an already prepared stream, the lower-level mapping entry is:

```bash
python slam_new.py --config <config.yaml> --exp_name <name> --seed 43
```

## Validation

Focused mapping tests can be run with:

```bash
pytest -q \
  tests/test_frontview_sampling.py \
  tests/test_frontview_scale_cover.py \
  tests/test_frontview_far_field.py \
  tests/test_frontview_coverage_recovery.py \
  tests/test_streaming_appearance_lod.py \
  tests/test_frontview_directional_layer.py
```

## Upstream And License

This implementation started from
[facebookresearch/Online-3DGS-Monocular](https://github.com/facebookresearch/Online-3DGS-Monocular)
and retains its MonoGS, DepthCov, and gsplat foundations. The active front-view
mapping design and integrations are described above and in the method document.

The repository remains under the included Creative Commons
Attribution-NonCommercial 4.0 license. See [LICENSE](LICENSE).
