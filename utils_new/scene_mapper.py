# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
from os.path import join as pjoin

import numpy as np
import torch
import torch.multiprocessing as mp
from torchvision.transforms import v2
from tqdm import tqdm
from utils_new.dataset import load_dataset
from utils_new.gaussian_models import GaussianModel

try:
    from utils_new.stream_loader import load_streamer
except:
    print("Warning: Stream loader not found. Please check the dependency.")
# from torchmetrics import StructuralSimilarityIndexMeasure  # Removed unused import

from utils_new.camera_utils import (
    # Camera_Optimizer,  # Removed unused import
    # in_frustum_region,  # Removed unused import
    SE3_Camera_Optimizer,
    unproject_pts_tensor,
)
from utils_new.appearance_lod import AppearanceLODEvidence
from utils_new.appearance_anchor import AppearanceProximalAnchor
from utils_new.frame_checker import FrameChecker
from utils_new.frontview_observability import (
    validate_front_view_observability_config,
)
from utils_new.frontview_coverage_recovery import (
    coverage_recovery_certificate,
    pose_novelty,
    validate_front_view_coverage_recovery_config,
)
from utils_new.frontview_directional_layer import (
    DIRECTIONAL_LAYER_FILENAME,
    validate_front_view_directional_layer_config,
)
from utils_new.frontview_sampling import validate_front_view_sampling_config
from utils_new.frontview_depth_transport import (
    validate_front_view_depth_transport_config,
)
from utils_new.frontview_birth import validate_front_view_birth_config
from utils_new.frontview_far_field import validate_front_view_far_field_config
from utils_new.frontview_identity_lod import (
    validate_front_view_identity_lod_config,
)
from utils_new.frontview_residual_cover import (
    validate_front_view_residual_cover_config,
)
from utils_new.frontview_scale_cover import validate_front_view_scale_cover_config
from utils_new.frontview_sparse_scale_map import (
    validate_front_view_sparse_scale_map_config,
)
from utils_new.frontview_track_fusion import (
    validate_front_view_track_fusion_config,
)
from utils_new.streaming_appearance_lod import (
    validate_streaming_appearance_lod_config,
)
from utils_new.kf_graph import cal_cams_covisibility, KFGraph
from utils_new.logging_utils import Log
from utils_new.progressive_mapping import ProgressiveManager, validate_progressive_config
from utils_new.aerocommit import AeroCommitManager, validate_aerocommit_config
from utils_new.aerocommit.dataset_geometry import guard_sparse_fast_path
from utils_new.aerocommit.frequency_responsibility import (
    frequency_weighted_pose_loss,
    laplacian_pyramid_reconstruction_loss,
)
from utils_new.aerocommit.stable_detail_split import stable_detail_split_scores
from utils_new.aerocommit.sparse_track_geometry import zbuffer_sparse_tracks
from utils_new.aerocommit.sparse_flow_detail import triangulate_multiview_flow_detail
from utils_new.aerocommit.track_detail import (
    SparseTrackDetailAccumulator,
    StableSurfaceDetailSampler,
)
from utils_new.worldtest_gs import (
    CertificateAuthority,
    WorldFrameContract,
    validate_worldtest_config,
)
from utils_new.worldtest_gs.controller import WorldTestController
from utils_new.loss_utils import (
    DepthLoss,
    DistortionLoss,
    GaussianLoss,
    NormalLoss,
    NormalRegularizationLoss,
    RGBLoss,
    masked_ssim_loss,
    blend_mse_tail_loss,
)
from utils_new.tool_utils import (
    convert_depth_to_normal,
    mkdir_p,
    saveTensorAsEXR,
    saveTensorAsPNG,
    ssim_map,
)


class SceneMapper(mp.Process):
    def __init__(self, configs):
        super(SceneMapper, self).__init__()
        self.configs = configs
        self.frontview_observability_config = (
            validate_front_view_observability_config(
                configs.get("FrontViewObservability")
            )
        )
        configs["FrontViewObservability"] = self.frontview_observability_config
        self.frontview_coverage_recovery_config = (
            validate_front_view_coverage_recovery_config(
                configs.get("FrontViewCoverageRecovery")
            )
        )
        configs["FrontViewCoverageRecovery"] = (
            self.frontview_coverage_recovery_config
        )
        self.frontview_directional_layer_config = (
            validate_front_view_directional_layer_config(
                configs.get("FrontViewDirectionalLayer")
            )
        )
        configs["FrontViewDirectionalLayer"] = (
            self.frontview_directional_layer_config
        )
        self.frontview_coverage_recovery_stats = {
            "sparse_dropout_frames": 0,
            "interval_eligible_frames": 0,
            "pose_novel_frames": 0,
            "rendered_certificates": 0,
            "admitted_keyframes": 0,
            "first_admitted_frame": -1,
            "last_admitted_frame": -1,
            "failure_fraction_sum": 0.0,
            "mean_residual_sum": 0.0,
            "translation_sum_m": 0.0,
            "rotation_sum_deg": 0.0,
            "last_admitted_translation_m": 0.0,
            "newborn_refinement_calls": 0,
            "newborn_refinement_steps": 0,
            "newborn_refinement_rows": 0,
            "newborn_refinement_seconds": 0.0,
            "tracking_refinement_calls": 0,
            "tracking_refinement_steps": 0,
            "tracking_refinement_seconds": 0.0,
        }
        self.frontview_sampling_config = validate_front_view_sampling_config(
            configs.get("FrontViewSampling")
        )
        configs["FrontViewSampling"] = self.frontview_sampling_config
        self.frontview_depth_transport_config = (
            validate_front_view_depth_transport_config(
                configs.get("FrontViewDepthTransport")
            )
        )
        configs["FrontViewDepthTransport"] = self.frontview_depth_transport_config
        self.frontview_birth_config = validate_front_view_birth_config(
            configs.get("FrontViewBirth")
        )
        configs["FrontViewBirth"] = self.frontview_birth_config
        self.frontview_track_fusion_config = (
            validate_front_view_track_fusion_config(
                configs.get("FrontViewTrackFusion")
            )
        )
        configs["FrontViewTrackFusion"] = self.frontview_track_fusion_config
        self.frontview_far_field_config = validate_front_view_far_field_config(
            configs.get("FrontViewFarField")
        )
        configs["FrontViewFarField"] = self.frontview_far_field_config
        self.frontview_identity_lod_config = (
            validate_front_view_identity_lod_config(
                configs.get("FrontViewIdentityLOD")
            )
        )
        configs["FrontViewIdentityLOD"] = self.frontview_identity_lod_config
        self.frontview_residual_cover_config = (
            validate_front_view_residual_cover_config(
                configs.get("FrontViewResidualCover")
            )
        )
        configs["FrontViewResidualCover"] = self.frontview_residual_cover_config
        self.frontview_scale_cover_config = validate_front_view_scale_cover_config(
            configs.get("FrontViewScaleCover")
        )
        configs["FrontViewScaleCover"] = self.frontview_scale_cover_config
        self.streaming_appearance_lod_config = (
            validate_streaming_appearance_lod_config(
                configs.get("StreamingAppearanceLOD")
            )
        )
        configs["StreamingAppearanceLOD"] = self.streaming_appearance_lod_config
        self.frontview_sparse_scale_map_config = (
            validate_front_view_sparse_scale_map_config(
                configs.get("FrontViewSparseScaleMap")
            )
        )
        configs["FrontViewSparseScaleMap"] = (
            self.frontview_sparse_scale_map_config
        )
        if (
            self.frontview_birth_config["enabled"]
            and self.frontview_sampling_config["enabled"]
        ):
            raise ValueError(
                "FrontViewBirth replaces FrontViewSampling survivor allocation; "
                "the two methods cannot be enabled together"
            )
        if (
            self.frontview_birth_config["enabled"]
            and self.frontview_far_field_config["enabled"]
        ):
            raise ValueError(
                "FrontViewBirth and FrontViewFarField cannot be enabled together"
            )
        if self.frontview_identity_lod_config["enabled"] and (
            self.frontview_birth_config["enabled"]
            or self.frontview_far_field_config["enabled"]
        ):
            raise ValueError(
                "FrontViewIdentityLOD must exclusively own HashBlock-free admission"
            )
        if self.frontview_residual_cover_config["enabled"] and (
            self.frontview_birth_config["enabled"]
            or self.frontview_far_field_config["enabled"]
            or self.frontview_identity_lod_config["enabled"]
        ):
            raise ValueError(
                "FrontViewResidualCover must exclusively own HashBlock-free admission"
            )
        if self.frontview_scale_cover_config["enabled"]:
            if (
                self.frontview_birth_config["enabled"]
                or self.frontview_identity_lod_config["enabled"]
                or self.frontview_residual_cover_config["enabled"]
            ):
                raise ValueError(
                    "FrontViewScaleCover must exclusively own spatial admission"
                )
            if bool(configs.get("HashBlock", {}).get("use_hash", False)):
                raise ValueError(
                    "FrontViewScaleCover requires HashBlock.use_hash=false"
                )
        if self.frontview_sparse_scale_map_config["enabled"]:
            if not self.frontview_far_field_config["enabled"]:
                raise ValueError(
                    "FrontViewSparseScaleMap currently requires FrontViewFarField"
                )
            if self.frontview_scale_cover_config["enabled"]:
                raise ValueError(
                    "FrontViewScaleCover and FrontViewSparseScaleMap are exclusive"
                )
            if bool(configs.get("HashBlock", {}).get("use_hash", False)):
                raise ValueError(
                    "FrontViewSparseScaleMap requires HashBlock.use_hash=false"
                )
        self.gaussians = GaussianModel(configs)
        self.cur_idx = 0
        self.cur_view = None
        self.current_frame_coverage_recovered = False
        self.current_frame_coverage_recovery_translation_m = None
        self.last_coverage_recovery_commit = None
        self.active_coverage_recovery_group_id = None
        self.active_coverage_recovery_birth_frame = -1
        self.coverage_recovery_tracking_views = []

        self.last_record_time = torch.cuda.Event(enable_timing=True)

        self.last_record_time.record()

        self.initialization_frames = configs["Mapper"]["initialization_frames"]
        self.processed_frames = 0

        self.scene_exposure_gain = configs["Mapper"]["scene_exposure_gain"]

        # key frames methods
        # ------------------------- Legacy code -------------------------
        # self.kf_selection_method = configs["Mapper"]["kf_selection_method"]
        # self.last_kf_info = None
        # self.kf_interval = configs["Mapper"]["kf_interval"]
        # self.kf_overlap_ratio = configs["Mapper"]["kf_overlap_ratio"]
        # self.kf_cameras = []
        # self.num_additional_views = configs["Mapper"]["num_additional_views"]
        # ----------------------------------------------------------------
        self.kf_graph = KFGraph(configs["Mapper"]["KFGraph"])
        self.global_window_size = configs["Mapper"]["KFGraph"]["global_window_size"]
        self.force_keyframes_through_frame = int(
            configs["Mapper"].get("force_keyframes_through_frame", -1)
        )

        self.last_add_gaussians = -1000000000

        self.cur_group_gaussian_frames = 0
        self.group_max_gaussian_frames = configs["Mapper"]["group_max_gaussian_frames"]

        self.device = configs["Mapper"]["device"]

        self.progressive_config = validate_progressive_config(
            configs.get("ProgressiveMapping")
        )
        configs["ProgressiveMapping"] = self.progressive_config
        self.progressive_manager = (
            ProgressiveManager(
                self.progressive_config,
                gaussian_model=self.gaussians,
                output_dir=configs["Results"]["save_dir"],
            )
            if self.progressive_config["enabled"]
            else None
        )
        self.aerocommit_config = validate_aerocommit_config(
            configs.get("AeroCommit")
        )
        if (
            (
                self.frontview_identity_lod_config["enabled"]
                or self.frontview_residual_cover_config["enabled"]
            )
            and self.aerocommit_config["enabled"]
        ):
            raise ValueError(
                "HashBlock-free front-view admission currently requires immediate online commit"
            )
        sparse_geometry_guard = guard_sparse_fast_path(
            self.aerocommit_config, configs["Dataset"]
        )
        if sparse_geometry_guard.get("diagnostic_override", False):
            Log(
                "UNSAFE COORDINATE-STRESS OVERRIDE: frame-local sparse geometry "
                "may enter the permanent map; never use this run as a method result. "
                "Reason: {}".format(sparse_geometry_guard["reason"]),
                tag="AeroCommit",
            )
        if sparse_geometry_guard["changed"]:
            Log(
                "Disabled trusted sparse admission and limited bootstrap to one "
                "frame: {}".format(sparse_geometry_guard["reason"]),
                tag="AeroCommit",
            )
        configs["AeroCommit"] = self.aerocommit_config
        if self.progressive_manager is not None and self.aerocommit_config["enabled"]:
            raise ValueError(
                "ProgressiveMapping and AeroCommit cannot own densification together"
            )
        self.aerocommit_manager = (
            AeroCommitManager(
                self.aerocommit_config,
                gaussian_model=self.gaussians,
                output_dir=configs["Results"]["save_dir"],
                device=self.device,
            )
            if self.aerocommit_config["enabled"]
            else None
        )
        self.worldtest_config = validate_worldtest_config(
            configs.get("WorldTestGS")
        )
        configs["WorldTestGS"] = self.worldtest_config
        self.worldtest_contract = None
        self.worldtest_authority = None
        self.worldtest_controller = None
        if self.worldtest_config["enabled"]:
            if self.aerocommit_manager is None:
                raise ValueError("WorldTestGS requires AeroCommit map ownership")
            self.worldtest_contract = WorldFrameContract.from_dataset(
                configs["Dataset"]["dataset_path"],
                calibration_version=self.worldtest_config["calibration_version"],
            )
            self.worldtest_contract.require_permanent_birth(
                self.worldtest_config["allow_invalid_world_stress"]
            )
            self.worldtest_authority = CertificateAuthority(
                self.worldtest_contract,
                self.worldtest_config["qg_threshold"],
                allow_invalid_stress=self.worldtest_config[
                    "allow_invalid_world_stress"
                ],
            )
            self.gaussians.configure_worldtest_certificate_authority(
                self.worldtest_authority
            )
            self.worldtest_controller = WorldTestController(
                self.worldtest_config,
                self.worldtest_contract,
                self.worldtest_authority,
                self.gaussians,
                configs["Results"]["save_dir"],
            )
            if (
                self.aerocommit_config["admission"]["trusted_sparse_fast_path"]
                or self.aerocommit_config["admission"][
                    "trusted_depthcov_fast_path"
                ]
            ):
                Log(
                    "WorldTestGS ignores source-based permanent fast-path fields; "
                    "all proposals use certificate admission.",
                    tag="WorldTestGS",
                )
        track_detail_config = self.aerocommit_config["track_detail"]
        self.track_detail_accumulator = (
            SparseTrackDetailAccumulator(track_detail_config)
            if self.aerocommit_config["enabled"]
            and bool(track_detail_config["enabled"])
            else None
        )
        self.track_detail_group_id = None
        self.track_detail_batches = []
        surface_detail_config = self.aerocommit_config["surface_detail"]
        self.surface_detail_sampler = (
            StableSurfaceDetailSampler(surface_detail_config)
            if self.aerocommit_config["enabled"]
            and bool(surface_detail_config["enabled"])
            else None
        )
        self.surface_detail_group_id = None
        flow_detail_config = self.aerocommit_config["flow_detail"]
        self.flow_detail_sampler = (
            StableSurfaceDetailSampler(
                {
                    "voxel_size": flow_detail_config["voxel_size"],
                    "max_commits_per_keyframe": flow_detail_config[
                        "max_commits_per_frame"
                    ],
                    "max_total_gaussians": flow_detail_config[
                        "max_total_gaussians"
                    ],
                }
            )
            if self.aerocommit_config["enabled"]
            and bool(flow_detail_config["enabled"])
            else None
        )
        self.flow_detail_group_id = None
        self.flow_detail_cams = []
        self.last_aerocommit_stats = None
        self.aerocommit_runtime = {
            "pose_prepass_seconds": 0.0,
            "post_refinement_seconds": 0.0,
            "newborn_optimization_seconds": 0.0,
            "newborn_optimization_steps": 0,
            "track_detail_seconds": 0.0,
            "track_detail_gaussians": 0,
            "track_detail_tracks": 0,
            "surface_detail_seconds": 0.0,
            "surface_detail_gaussians": 0,
            "flow_detail_seconds": 0.0,
            "flow_detail_gaussians": 0,
            "full_map_restore_seconds": 0.0,
            "full_map_restore_groups": 0,
            "full_map_restore_gaussians": 0,
        }
        self.appearance_lod_stats = None
        self.appearance_anchor_stats = None
        self.progressive_runtime = {
            "main_optimization_seconds": 0.0,
            "baseline_densification_seconds": 0.0,
            "stable_render_seconds": 0.0,
            "progressive_processing_seconds": 0.0,
            "newborn_optimization_seconds": 0.0,
            "debug_render_seconds": 0.0,
            "visualization_seconds": 0.0,
            "post_refinement_seconds": 0.0,
            "newborn_optimization_steps": 0,
        }

        # Key frames and active window
        self.active_window_size = configs["Mapper"]["active_window_size"]
        self.active_window = []

        self.coarse_active_window_size = configs["Mapper"]["coarse_active_window_size"]
        self.coarse_active_window = []
        self.coarse_pool_size = configs["Mapper"]["coarse_pool_size"]
        self.coarse_pool = []
        self.coarse_level_interval = configs["Mapper"]["coarse_level_interval"]

        self.add_gaussians_interval = configs["Mapper"]["add_gaussians_interval"]

        self.prune_interval = configs["Mapper"]["prune_interval"]

        self.frame_checker = FrameChecker(configs["Mapper"]["FrameChecker"])

        self.use_multi_reso = configs["Mapper"]["use_multi_reso"]

        self.pin_kf_gpu = (
            configs["Mapper"]["pin_kf_gpu"]
            if "pin_kf_gpu" in configs["Mapper"]
            else False
        )  # whether pin key frames to gpu

        self.save_exr = self.configs["Results"]["save_exr"]

        mkdir_p(pjoin(self.configs["Results"]["save_dir"], "online", "point_cloud"))
        self.save_model_interval = configs["Mapper"]["save_model_interval"]

        # save optimization log info
        self.opt_log = {}
        # save tracked poses
        self.opt_log["tracked_poses"] = {}
        # save orig poses
        self.opt_log["poses_pair"] = {}
        # cam opt iterations
        self.opt_log["cam_opt_iterations"] = {}
        # gaussian opt iterations
        self.opt_log["gaussian_opt_iterations"] = {}
        # loss info
        self.opt_log["l1_err"] = {}
        # is key frame
        self.opt_log["is_key_frame"] = {}

        # Dataset
        self.use_dataset = configs["Mapper"]["use_dataset"]
        if self.use_dataset:
            configs["Dataset"]["scene_exposure_gain"] = self.scene_exposure_gain
            self.dataset = load_dataset(configs["Dataset"])
            vignette_img = self.dataset.dataset.get_vignette
            self.sample_cam = self.dataset.dataset.get_sample_cam()
            self.dataset = iter(self.dataset)
            self.gaussians.set_vignette_img(vignette_img)
        else:
            raise NotImplementedError(
                "Stream loader is not implemented for this mapper."
            )  # Songyin: Modify this if needed
            configs["Streamer"]["scene_exposure_gain"] = self.scene_exposure_gain
            self.dataset = load_streamer(configs["Streamer"])
            self.first_frame = True

        # Optimizer
        self.optimization_iters = configs["Mapper"]["optimization_iters"]
        self.intialization_iters = configs["Mapper"]["initialization_iters"]
        self.camera_optimizer = SE3_Camera_Optimizer(
            configs["Mapper"]["CameraOptimizer"]
        )
        self.pose_opt_steps = configs["Mapper"]["CameraOptimizer"]["pose_opt_steps"]
        self.pose_refine_init_steps = configs["Mapper"]["CameraOptimizer"][
            "pose_refine_init_steps"
        ]

        self.post_refinement_config = configs["Mapper"]["post_refinement"]
        self.post_refinement_frames = []

        self.use_random_bg = (
            configs["Mapper"]["use_random_bg"]
            if "use_random_bg" in configs["Mapper"]
            else False
        )

        # Loss function
        self.image_loss = RGBLoss(configs["Loss"])
        self.gaussian_loss = GaussianLoss(configs["Loss"])
        self.depth_loss = DepthLoss(configs["Loss"])
        self.normal_loss = NormalLoss(configs["Loss"])
        self.distortion_loss = DistortionLoss(configs["Loss"])
        self.normal_reg_loss = NormalRegularizationLoss(configs["Loss"])

        # ssim func
        # self.ssim_func = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True)
        self.ssim_func = ssim_map(self.device)
        self.gaussian_blurrer = v2.GaussianBlur(kernel_size=(5, 5), sigma=0.5)

        # some gaussian initial parameters
        self.err_threshold = configs["Model"]["err_threshold"]
        self.semi_dense_err_threshold = configs["Model"]["semi_dense_err_threshold"]

    def update_track_detail(self, cam):
        """Commit causal, geometry-frozen detail carriers from repeated tracks."""
        if self.track_detail_accumulator is None:
            return 0
        start_time = time.perf_counter()
        sparse_rows = cam.get_color_pts_depth()
        if sparse_rows is None or len(sparse_rows) == 0:
            return 0
        world_points = np.asarray(sparse_rows[:, :3], dtype=np.float32)
        observations = zbuffer_sparse_tracks(
            world_points,
            cam.get_raw_pose().detach().cpu().numpy(),
            cam.get_int_mat(0).detach().cpu().numpy(),
            cam.get_width(0),
            cam.get_height(0),
        )
        if len(observations.world_points) == 0:
            return 0

        image = cam.get_gt_image(0)
        gray = image.mean(dim=-1)
        gradient = torch.zeros_like(gray)
        horizontal = torch.abs(gray[:, 1:] - gray[:, :-1])
        vertical = torch.abs(gray[1:, :] - gray[:-1, :])
        gradient[:, 1:] = torch.maximum(gradient[:, 1:], horizontal)
        gradient[:, :-1] = torch.maximum(gradient[:, :-1], horizontal)
        gradient[1:, :] = torch.maximum(gradient[1:, :], vertical)
        gradient[:-1, :] = torch.maximum(gradient[:-1, :], vertical)
        pixel_indices = torch.as_tensor(
            observations.pixel_indices, device=image.device, dtype=torch.long
        )
        colors = image.reshape(-1, 3)[pixel_indices]
        colors = colors * float(self.scene_exposure_gain) / max(
            float(cam.exposure_gain), 1.0e-8
        )
        gradients = gradient.reshape(-1)[pixel_indices]
        side_scores = np.abs(
            2.0
            * observations.uv[:, 0]
            / max(float(cam.get_width(0) - 1), 1.0)
            - 1.0
        ).astype(np.float32)
        batch = self.track_detail_accumulator.observe(
            frame_id=int(cam.cam_idx),
            world_points=observations.world_points,
            colors=colors.detach().cpu().numpy(),
            depths=observations.depths,
            gradients=gradients.detach().cpu().numpy(),
            side_scores=side_scores,
            focal=0.5 * (float(cam.get_fx(0)) + float(cam.get_fy(0))),
        )
        committed = 0
        if len(batch) > 0:
            config = self.aerocommit_config["track_detail"]
            if self.worldtest_controller is not None:
                committed = 0
                self.worldtest_controller.observe_external_geometry(
                    cam,
                    batch.world_points,
                    batch.colors,
                    batch.log_scales,
                    source_kind="track_detail",
                )
            elif config["mode"] == "reassign":
                self.track_detail_batches.append(batch)
                committed = len(batch)
                self.aerocommit_runtime["track_detail_tracks"] += committed
            else:
                self.track_detail_group_id, committed = (
                    self.gaussians.add_track_detail_gaussians(
                        batch.world_points,
                        batch.colors,
                        batch.log_scales,
                        initial_opacity=float(config["initial_opacity"]),
                        max_scale_expansion=float(config["max_scale_expansion"]),
                        group_id=self.track_detail_group_id,
                        freeze_geometry=bool(config["freeze_geometry"]),
                    )
                )
                self.aerocommit_runtime["track_detail_gaussians"] += committed
            Log(
                "Track detail accepted {} (tracks {}, carriers {})".format(
                    committed,
                    self.aerocommit_runtime["track_detail_tracks"],
                    self.aerocommit_runtime["track_detail_gaussians"],
                ),
                tag="SceneMapper",
            )
        self.aerocommit_runtime["track_detail_seconds"] += (
            time.perf_counter() - start_time
        )
        return committed

    def apply_track_detail_reassignment(self):
        if not self.track_detail_batches:
            return
        start_time = time.perf_counter()
        world_points = np.concatenate(
            [batch.world_points for batch in self.track_detail_batches], axis=0
        )
        colors = np.concatenate(
            [batch.colors for batch in self.track_detail_batches], axis=0
        )
        log_scales = np.concatenate(
            [batch.log_scales for batch in self.track_detail_batches], axis=0
        )
        scores = np.concatenate(
            [batch.scores for batch in self.track_detail_batches], axis=0
        )
        stats = self.gaussians.reassign_track_detail_responsibility(
            world_points,
            colors,
            log_scales,
            scores,
            self.aerocommit_config["track_detail"],
        )
        self.aerocommit_runtime["track_detail_reassignment"] = stats
        self.aerocommit_runtime["track_detail_seconds"] += (
            time.perf_counter() - start_time
        )
        Log(
            "Track detail reassigned {}/{} rows, mean distance {:.4f}".format(
                stats["matched"], stats["tracks"], stats["distance_mean"]
            ),
            tag="SceneMapper",
        )

    def update_surface_detail(self, cam, render_pkg):
        """Replay side-view RGB detail onto high-confidence stable surfaces."""
        if self.surface_detail_sampler is None or render_pkg is None:
            return 0
        start_time = time.perf_counter()
        config = self.aerocommit_config["surface_detail"]
        image = cam.get_gt_image(0)
        rendered = render_pkg["render"]
        depth = render_pkg["depth"]
        opacity = render_pkg["opacity"]
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        if opacity.ndim == 3:
            opacity = opacity.squeeze(-1)

        gray = image.mean(dim=-1)
        gradient = torch.zeros_like(gray)
        horizontal = torch.abs(gray[:, 1:] - gray[:, :-1])
        vertical = torch.abs(gray[1:, :] - gray[:-1, :])
        gradient[:, 1:] = torch.maximum(gradient[:, 1:], horizontal)
        gradient[:, :-1] = torch.maximum(gradient[:, :-1], horizontal)
        gradient[1:, :] = torch.maximum(gradient[1:, :], vertical)
        gradient[:-1, :] = torch.maximum(gradient[:-1, :], vertical)
        residual = torch.abs(image - rendered).mean(dim=-1)
        width = cam.get_width(0)
        side = torch.linspace(
            -1.0, 1.0, width, device=image.device, dtype=image.dtype
        ).abs().view(1, width)
        candidate = (
            (opacity >= float(config["opacity_threshold"]))
            & (gradient >= float(config["gradient_threshold"]))
            & (residual >= float(config["residual_threshold"]))
            & (side >= float(config["side_start"]))
        )
        if config["depth_source"] == "stable":
            candidate &= (
                torch.isfinite(depth)
                & (depth > 0.0)
                & (depth <= float(config["near_depth_m"]))
            )
        flat_candidates = torch.nonzero(candidate.reshape(-1), as_tuple=False).reshape(-1)
        if flat_candidates.numel() == 0:
            return 0
        side_weight = 1.0 + float(config["side_score_boost"]) * side.expand_as(
            gradient
        )
        importance = gradient * torch.clamp(residual, min=0.01) * side_weight
        prelimit = min(
            int(flat_candidates.numel()),
            int(config["max_commits_per_keyframe"]) * (
                4 if config["depth_source"] in ("depthcov", "precomputed") else 8
            ),
        )
        if flat_candidates.numel() > prelimit:
            top = torch.topk(
                importance.reshape(-1)[flat_candidates], prelimit, sorted=False
            ).indices
            flat_candidates = flat_candidates[top]
        x = flat_candidates % width
        y = flat_candidates // width
        uv = torch.stack((x + 0.5, y + 0.5), dim=-1).to(torch.float32)
        if config["depth_source"] == "depthcov":
            sparse_depth = cam.get_sparse_depth(0).reshape(-1)
            sparse_indices = torch.nonzero(
                sparse_depth > 0.0, as_tuple=False
            ).reshape(-1)
            if sparse_indices.numel() == 0:
                return 0
            if sparse_indices.numel() > 500:
                sparse_indices = sparse_indices[
                    torch.randperm(sparse_indices.numel(), device=image.device)[:500]
                ]
            sparse_uv = torch.stack(
                (
                    sparse_indices % width + 0.5,
                    sparse_indices // width + 0.5,
                ),
                dim=-1,
            ).to(torch.float32)
            estimated, valid_depthcov, depth_std = (
                self.gaussians.depth_cov_estimator.query_tensor(
                    image,
                    sparse_depth[sparse_indices],
                    sparse_uv,
                    uv,
                    return_std=True,
                )
            )
            confidence = torch.clamp(
                1.0
                - depth_std
                / max(
                    float(self.gaussians.depth_cov_estimator.std_valid_threshold),
                    1.0e-8,
                ),
                min=0.0,
                max=1.0,
            )
            stable_depth = depth.reshape(-1)[flat_candidates]
            consistent = (
                ~torch.isfinite(stable_depth)
                | (stable_depth <= 0.0)
                | (
                    torch.abs(estimated - stable_depth)
                    / torch.clamp(estimated, min=1.0e-6)
                    <= float(config["stable_depth_consistency_ratio"])
                )
            )
            valid_depthcov &= (
                torch.isfinite(estimated)
                & (estimated > 0.0)
                & (estimated <= float(config["near_depth_m"]))
                & (confidence >= float(config["depth_confidence_threshold"]))
                & consistent
            )
            if not torch.any(valid_depthcov):
                return 0
            flat_candidates = flat_candidates[valid_depthcov]
            uv = uv[valid_depthcov]
            selected_depth = estimated[valid_depthcov]
        elif config["depth_source"] == "precomputed":
            depth_path = pjoin(
                config["precomputed_depth_directory"],
                "depth_{:04d}.npz".format(int(cam.cam_idx)),
            )
            with np.load(depth_path) as depth_data:
                precomputed_depth = torch.from_numpy(
                    depth_data["depth"].astype(np.float32)
                ).to(image.device)
            selected_depth = precomputed_depth.reshape(-1)[flat_candidates]
            stable_depth = depth.reshape(-1)[flat_candidates]
            valid_precomputed = (
                torch.isfinite(selected_depth)
                & (selected_depth > 0.0)
                & (selected_depth <= float(config["near_depth_m"]))
                & (
                    ~torch.isfinite(stable_depth)
                    | (stable_depth <= 0.0)
                    | (
                        torch.abs(selected_depth - stable_depth)
                        / torch.clamp(selected_depth, min=1.0e-6)
                        <= float(config["stable_depth_consistency_ratio"])
                    )
                )
            )
            if not torch.any(valid_precomputed):
                return 0
            flat_candidates = flat_candidates[valid_precomputed]
            uv = uv[valid_precomputed]
            selected_depth = selected_depth[valid_precomputed]
        else:
            selected_depth = depth.reshape(-1)[flat_candidates]
        world_points = unproject_pts_tensor(
            uv,
            selected_depth,
            cam.get_int_mat(0),
            cam.get_raw_pose().detach(),
        )
        focal = 0.5 * (float(cam.get_fx(0)) + float(cam.get_fy(0)))
        scales = (
            float(config["projected_scale_px"]) * selected_depth / focal
        )
        log_scales = torch.log(torch.clamp(scales, min=1.0e-8)).reshape(-1, 1)
        colors = image.reshape(-1, 3)[flat_candidates]
        colors = colors * float(self.scene_exposure_gain) / max(
            float(cam.exposure_gain), 1.0e-8
        )
        batch = self.surface_detail_sampler.select(
            world_points.detach().cpu().numpy(),
            colors.detach().cpu().numpy(),
            log_scales.detach().cpu().numpy(),
            importance.reshape(-1)[flat_candidates].detach().cpu().numpy(),
        )
        committed = 0
        if len(batch) > 0:
            if self.worldtest_controller is not None:
                self.worldtest_controller.observe_external_geometry(
                    cam,
                    batch.world_points,
                    batch.colors,
                    batch.log_scales,
                    source_kind="surface_detail",
                )
                committed = 0
            else:
                self.surface_detail_group_id, committed = (
                    self.gaussians.add_track_detail_gaussians(
                        batch.world_points,
                        batch.colors,
                        batch.log_scales,
                        initial_opacity=float(config["initial_opacity"]),
                        max_scale_expansion=float(config["max_scale_expansion"]),
                        group_id=self.surface_detail_group_id,
                        freeze_geometry=bool(config["freeze_geometry"]),
                    )
                )
            self.aerocommit_runtime["surface_detail_gaussians"] += committed
            Log(
                "Stable-surface detail committed {} (total {})".format(
                    committed,
                    self.aerocommit_runtime["surface_detail_gaussians"],
                ),
                tag="SceneMapper",
            )
        self.aerocommit_runtime["surface_detail_seconds"] += (
            time.perf_counter() - start_time
        )
        return committed

    def update_flow_detail(self, cam):
        """Triangulate causal sparse RGB tracks into fixed side-detail carriers."""
        if self.flow_detail_sampler is None:
            return 0
        if self.flow_detail_sampler.full:
            return 0
        config = self.aerocommit_config["flow_detail"]
        self.flow_detail_cams.append(cam)
        track_views = int(config["track_views"])
        if len(self.flow_detail_cams) > track_views:
            self.flow_detail_cams.pop(0)
        if (
            len(self.flow_detail_cams) < track_views
            or int(cam.cam_idx) < int(config["start_frame"])
        ):
            return 0
        start_time = time.perf_counter()
        triangulated = triangulate_multiview_flow_detail(
            [view.get_gt_image(0).detach().cpu().numpy() for view in self.flow_detail_cams],
            [
                view.get_raw_pose().detach().cpu().numpy()
                for view in self.flow_detail_cams
            ],
            cam.get_int_mat(0).detach().cpu().numpy(),
            config,
        )
        committed = 0
        if len(triangulated.world_points) > 0:
            focal = 0.5 * (float(cam.get_fx(0)) + float(cam.get_fy(0)))
            log_scales = np.log(
                np.maximum(
                    float(config["projected_scale_px"])
                    * triangulated.depths
                    / focal,
                    1.0e-8,
                )
            ).reshape(-1, 1)
            batch = self.flow_detail_sampler.select(
                triangulated.world_points,
                triangulated.colors,
                log_scales,
                triangulated.scores,
            )
            if len(batch) > 0:
                if self.worldtest_controller is not None:
                    self.worldtest_controller.observe_external_geometry(
                        cam,
                        batch.world_points,
                        batch.colors,
                        batch.log_scales,
                        source_kind="flow_detail",
                    )
                    committed = 0
                else:
                    self.flow_detail_group_id, committed = (
                        self.gaussians.add_track_detail_gaussians(
                            batch.world_points,
                            batch.colors,
                            batch.log_scales,
                            initial_opacity=float(config["initial_opacity"]),
                            max_scale_expansion=float(config["max_scale_expansion"]),
                            group_id=self.flow_detail_group_id,
                            freeze_geometry=bool(config["freeze_geometry"]),
                        )
                    )
                self.aerocommit_runtime["flow_detail_gaussians"] += committed
                Log(
                    "Sparse-flow detail committed {} (total {})".format(
                        committed,
                        self.aerocommit_runtime["flow_detail_gaussians"],
                    ),
                    tag="SceneMapper",
                )
        self.aerocommit_runtime["flow_detail_seconds"] += (
            time.perf_counter() - start_time
        )
        return committed

    def get_stable_external_splats(self, cameras=None):
        """Return frozen A proxies; P proxies do not count as stable coverage."""
        if self.progressive_manager is None:
            return None
        return self.progressive_manager.stable_external_splats(
            torch.device(self.device), torch.float32, cameras=cameras
        )

    # Optimize gaussians (do not densify or prune)
    def optimize(
        self,
        is_last_frame=False,
        is_key_frame=False,
        level=0,
        steps=-1,
        current_view_only=False,
        optimize_pose=True,
        supplemental=False,
        views_override=None,
    ):
        if len(self.active_window) == 0:
            return

        if self.gaussians.get_num_gaussians == 0:
            return

        color_mask = None

        # t1.record()
        optimized_local_window = []
        used_cams = []
        exist_idx = []
        cur_res_idx = 0
        if views_override is not None:
            for cam in views_override:
                if cam.cam_idx not in exist_idx:
                    cam.to_device(self.device)
                    optimized_local_window.append(cam)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)
            current_matches = [
                index
                for index, cam in enumerate(optimized_local_window)
                if int(cam.cam_idx) == int(self.cur_view.cam_idx)
            ]
            cur_res_idx = current_matches[-1] if current_matches else len(
                optimized_local_window
            ) - 1
        elif current_view_only:
            self.cur_view.to_device(self.device)
            optimized_local_window = [self.cur_view]
            used_cams = [self.cur_view]
            exist_idx = [self.cur_view.cam_idx]
        else:
            for cam in self.active_window:
                if cam.cam_idx not in exist_idx:
                    cam.to_device(self.device)
                    optimized_local_window.append(cam)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)
                    if cam.cam_idx == self.cur_idx:
                        cur_res_idx = len(optimized_local_window) - 1
            for cam in self.coarse_active_window:
                if cam.cam_idx not in exist_idx:
                    if not self.pin_kf_gpu:
                        cam.to_device(self.device)
                    optimized_local_window.append(cam)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)

        gt_imgs = torch.stack(
            [cam.get_gt_image(level) for cam in optimized_local_window]
        )
        gt_depths = torch.stack(
            [cam.get_sparse_depth(level) for cam in optimized_local_window]
        )

        # Supplemental current-view steps must not consume initialization state.
        if current_view_only or supplemental:
            optimization_steps = max(0, steps)
        else:
            if self.initialization_frames == 0:
                optimization_steps = self.intialization_iters
                Log("Initialize map with 300 iterations", tag="SceneMapper")
            elif is_last_frame:
                optimization_steps = self.intialization_iters * 2
            else:
                optimization_steps = self.optimization_iters
            self.initialization_frames -= 1
            if steps >= 0:
                optimization_steps = steps

        for step in range(optimization_steps):
            # t3.record()

            # t1.record()
            global_windows = (
                (
                    []
                    if current_view_only or views_override is not None
                    else self.kf_graph.get_and_update_global_window()
                )
            )
            for cam in global_windows:
                if cam.cam_idx not in exist_idx:
                    if not self.pin_kf_gpu:
                        cam.to_device(self.device)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)
            global_gt_imgs = (
                torch.stack([cam.get_gt_image(level) for cam in global_windows])
                if len(global_windows) > 0
                else []
            )
            global_gt_depths = (
                torch.stack([cam.get_sparse_depth(level) for cam in global_windows])
                if len(global_windows) > 0
                else []
            )

            if len(global_windows) > 0:
                iter_window = optimized_local_window + global_windows
                iter_gt_imgs = torch.cat([gt_imgs, global_gt_imgs], dim=0)
                iter_gt_depths = torch.cat([gt_depths, global_gt_depths], dim=0)
            else:
                iter_window = optimized_local_window
                iter_gt_imgs = gt_imgs
                iter_gt_depths = gt_depths

            if (
                self.progressive_manager is not None
                and self.gaussians.gaussian_type == "2dgs"
                and len(iter_window) > 1
            ):
                current_idx = next(
                    (i for i, cam in enumerate(iter_window) if cam.cam_idx == self.cur_idx),
                    None,
                )
                current_fraction = float(
                    self.progressive_config["current_view_optimization_fraction"]
                )
                choose_current = (
                    current_idx is not None
                    and (
                        step == optimization_steps - 1
                        or (
                            current_fraction > 0.0
                            and (step * current_fraction) % 1.0 < current_fraction
                        )
                    )
                )
                if choose_current:
                    selected_idx = current_idx
                elif current_fraction > 0.0:
                    history_indices = sorted(
                        (
                            i for i, cam in enumerate(iter_window)
                            if i != current_idx
                        ),
                        key=lambda i: int(iter_window[i].cam_idx),
                        reverse=True,
                    )
                    selected_idx = history_indices[step % len(history_indices)]
                else:
                    selected_idx = step % len(iter_window)
                iter_window = [iter_window[selected_idx]]
                iter_gt_imgs = iter_gt_imgs[selected_idx : selected_idx + 1]
                iter_gt_depths = iter_gt_depths[selected_idx : selected_idx + 1]
                cur_res_idx = 0

            if self.progressive_manager is not None:
                self.progressive_manager.configure_optimization_visibility(iter_window)

            if optimize_pose and step < self.pose_opt_steps:
                # only opt pose for limited steps to avoid bad performance
                for cam in iter_window:
                    cam.set_opt_pose(True)

            # t1.record()
            batch_render_pkg = self.gaussians.render_batch(
                iter_window,
                self.use_random_bg,
                level=level,
                external_splats=self.get_stable_external_splats(iter_window),
            )
            batch_render = batch_render_pkg["render"]
            batch_render_depth = batch_render_pkg["depth"].squeeze(-1)

            # t1.record()
            gt_mask = (
                torch.sum(iter_gt_imgs, dim=-1, keepdim=True) <= 0.0000001
            ).float()  # ignore invalid pixels in gt
            iter_gt_imgs = iter_gt_imgs * (1.0 - gt_mask) + batch_render * gt_mask

            # t1.record()
            if self.normal_reg_loss.normal_reg_weight > 0.0:
                render_normal = convert_depth_to_normal(
                    batch_render_depth.view(
                        batch_render_depth.shape[0],
                        1,
                        batch_render_depth.shape[1],
                        batch_render_depth.shape[2],
                    ),
                    iter_window[0].get_int_mat(),
                )
            else:
                render_normal = None

            # t1.record()
            loss_img, l1_values = self.image_loss(batch_render, iter_gt_imgs)
            loss_depth = self.depth_loss(batch_render_depth, iter_gt_depths)
            loss_gaussians = self.gaussian_loss(self.gaussians)
            loss_progressive = (
                self.progressive_manager.surface_regularization_loss()
                if self.progressive_manager is not None
                else 0.0
            )

            loss_normal_reg, color_mask = self.normal_reg_loss(
                render_normal, iter_gt_imgs.permute(0, 3, 1, 2)
            )

            if (
                "normal" in batch_render_pkg
                and "normal_from_depth" in batch_render_pkg
                and "opacity" in batch_render_pkg
            ):
                loss_normal = self.normal_loss(
                    batch_render_pkg["normal"],
                    batch_render_pkg["normal_from_depth"],
                    batch_render_pkg["opacity"],
                )
            else:
                loss_normal = 0.0

            if "distortion" in batch_render_pkg:
                loss_distortion = self.distortion_loss(batch_render_pkg["distortion"])
            else:
                loss_distortion = 0.0

            loss = (
                loss_img
                + loss_gaussians
                + loss_depth
                + loss_normal
                + loss_distortion
                + loss_normal_reg
                + loss_progressive
            )

            # t1.record()
            if len(global_windows) > 0:
                for cam, err in zip(
                    iter_window[-self.global_window_size :],
                    l1_values[-self.global_window_size :],
                ):
                    self.kf_graph.update_err(cam.cam_idx, err)

            loss.backward()

            if (
                self.streaming_appearance_lod_config["enabled"]
                and step == optimization_steps - 1
                and not supplemental
            ):
                current_camera_index = next(
                    (
                        index
                        for index, camera in enumerate(iter_window)
                        if int(camera.cam_idx) == int(self.cur_idx)
                    ),
                    None,
                )
                if current_camera_index is not None:
                    self.gaussians.observe_streaming_appearance_lod(
                        batch_render_pkg.get("projection_info"),
                        iter_window,
                        current_camera_index,
                    )

            self.gaussians.mask_sh_degree_gradients()

            self.gaussians.precondition_frontview_mean_gradients(
                iter_window,
                self.frontview_observability_config,
                update_evidence=(
                    step
                    % int(
                        self.frontview_observability_config[
                            "evidence_update_interval"
                        ]
                    )
                    == 0
                ),
            )

            self.gaussians.update()  # Update gaussians, will zero the gradients inside this function

            if optimize_pose and step < self.pose_opt_steps:
                self.camera_optimizer.step()  # Update cameras, will zero the gradients inside this function

            # t1.record()
            if optimize_pose and step < self.pose_opt_steps:
                # only opt pose for limited steps to avoid bad performance
                for cam in iter_window:
                    cam.set_opt_pose(False)

            # save tracked poses
            for i, cam in enumerate(iter_window):
                if cam.name is not None:
                    self.opt_log["tracked_poses"][cam.name] = (
                        cam.get_pose().detach().cpu().numpy()
                    )
                    orig_data = self.opt_log["poses_pair"][cam.cam_idx]
                    self.opt_log["poses_pair"][cam.cam_idx] = (
                        orig_data[0],
                        cam.get_pose().detach().cpu().numpy(),
                        orig_data[2],
                    )

                    if cam.cam_idx not in self.opt_log["l1_err"]:
                        self.opt_log["l1_err"][cam.cam_idx] = {}
                    err = l1_values[i]
                    self.opt_log["l1_err"][cam.cam_idx][self.cur_view.cam_idx] = (
                        float(err.detach().item())
                        if torch.is_tensor(err)
                        else float(err)
                    )

                    if cam.name not in self.opt_log["gaussian_opt_iterations"]:
                        self.opt_log["gaussian_opt_iterations"][cam.name] = 1
                    else:
                        self.opt_log["gaussian_opt_iterations"][cam.name] += 1

                    if step < self.pose_opt_steps:
                        if cam.name not in self.opt_log["cam_opt_iterations"]:
                            self.opt_log["cam_opt_iterations"][cam.name] = 1
                        else:
                            self.opt_log["cam_opt_iterations"][cam.name] += 1

            if step == optimization_steps - 1:
                cur_view_render_pkg = {}
                cur_view_render_pkg["gt"] = iter_gt_imgs[cur_res_idx]
                cur_view_render_pkg["sparse_depth"] = optimized_local_window[
                    cur_res_idx
                ].get_sparse_depth(level)
                cur_view_render_pkg["render"] = batch_render_pkg["render"][cur_res_idx]
                cur_view_render_pkg["opacity"] = batch_render_pkg["opacity"][
                    cur_res_idx
                ]
                if "depth" in batch_render_pkg:
                    cur_view_render_pkg["depth"] = batch_render_pkg["depth"][
                        cur_res_idx
                    ]
                if "normal" in batch_render_pkg:
                    cur_view_render_pkg["normal"] = batch_render_pkg["normal"][
                        cur_res_idx
                    ]
                if "normal_from_depth" in batch_render_pkg:
                    cur_view_render_pkg["normal_from_depth"] = batch_render_pkg[
                        "normal_from_depth"
                    ][cur_res_idx]
                if "distortion" in batch_render_pkg:
                    cur_view_render_pkg["distortion"] = batch_render_pkg["distortion"][
                        cur_res_idx
                    ]

                if color_mask is not None:
                    cur_view_render_pkg["color_mask"] = color_mask[
                        cur_res_idx
                    ].unsqueeze(-1)

        # t1.record()
        if not self.pin_kf_gpu:
            for cam in used_cams:
                cam.to_device("cpu")

        if self.pin_kf_gpu and (not is_key_frame):  # do not keep non kfs in gpu
            self.cur_view.to_device("cpu")

        if not current_view_only and not supplemental:
            self.kf_graph.add_new_cam_for_global(optimized_local_window[cur_res_idx])

        return cur_view_render_pkg

    def get_frame(self):
        if self.use_dataset:
            cur_view = next(self.dataset, None)

            # if cur_view is not None:
            #     cur_view.to_device(self.device)
        else:
            cur_view = self.dataset.get_data()
            if self.first_frame:
                self.first_frame = False
                self.gaussians.set_vignette_img(self.dataset.get_vignette)
                self.sample_cam = self.dataset.get_sample_cam()

        return cur_view

    # Add new gaussians
    def densification(self, level=0, is_key_frame=True):
        cur_level = level
        self.last_coverage_recovery_commit = None
        # optimize() offloads cameras when keyframes are not pinned. The same
        # current view is still needed below for DepthCov proposal generation.
        self.cur_view.to_device(self.device)
        temporal_birth_enabled = bool(
            self.frontview_birth_config["enabled"]
            and self.frontview_birth_config["temporal_map_competition"]
        )
        reference_count = max(
            int(self.frontview_sampling_config["reference_frames"]),
            (
                int(self.frontview_birth_config["temporal_reference_frames"])
                if temporal_birth_enabled
                else 0
            ),
        )
        reference_cameras = [
            camera
            for camera in self.kf_graph.kf_cameras
            if int(camera.cam_idx) < int(self.cur_view.cam_idx)
        ][-reference_count:]
        if not self.frontview_sampling_config["enabled"] and not temporal_birth_enabled:
            reference_cameras = None
        elif reference_cameras is not None:
            for reference in reference_cameras:
                reference.to_device(self.device)
        if self.initialization_frames <= 0:
            with torch.no_grad():
                attemp_cur_render_pkg = self.gaussians.render(
                    self.cur_view,
                    level=cur_level,
                    return_info=self.frontview_identity_lod_config["enabled"],
                )

                rendered_img = (
                    attemp_cur_render_pkg["render"]
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                    .mean(dim=1, keepdims=True)
                )
                gt_img = (
                    self.cur_view.get_gt_image(level=cur_level)
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                    .mean(dim=1, keepdims=True)
                )

                rendered_img = torch.clamp(
                    self.gaussian_blurrer(rendered_img), 0.0, 1.0
                )
                gt_img = torch.clamp(self.gaussian_blurrer(gt_img), 0.0, 1.0)
                # _, diff_img = self.ssim_func(rendered_img, gt_img)
                # attemp_cur_render_pkg["diff"] = 1.0 - diff_img.squeeze(0).permute(1, 2, 0).mean(dim=-1).cpu()
                diff_img = self.ssim_func.cal_ssim_map(rendered_img, gt_img)
                attemp_cur_render_pkg["diff"] = (
                    diff_img.squeeze(0).permute(1, 2, 0).mean(dim=-1)
                )

                # Log("SSIM Diff: Max/Min: {:.4f}/{:.4f}".format(attemp_cur_render_pkg["diff"].max().item(), attemp_cur_render_pkg["diff"].min().item()), tag="SceneMapper")
                attemp_cur_render_pkg["depth"][
                    attemp_cur_render_pkg["opacity"] < 0.1
                ] = -1

                # if not self.pin_kf_gpu:
                #     self.cur_view.to_device("cpu")
        else:
            attemp_cur_render_pkg = None

        if is_key_frame:
            self.update_surface_detail(self.cur_view, attemp_cur_render_pkg)

        if (
            self.cur_view.cam_idx - self.last_add_gaussians
            > self.add_gaussians_interval
        ):
            self.last_add_gaussians = self.cur_view.cam_idx
            create_new_gaussians = (
                self.cur_group_gaussian_frames >= self.group_max_gaussian_frames
                or (
                    self.current_frame_coverage_recovered
                    and (
                        int(
                            self.frontview_coverage_recovery_config[
                                "newborn_optimization_iters"
                            ]
                        )
                        > 0
                        or int(
                            self.frontview_coverage_recovery_config[
                                "tracking_update_interval"
                            ]
                        )
                        > 0
                    )
                )
            )
            if create_new_gaussians:
                self.cur_group_gaussian_frames = 0
            if self.aerocommit_manager is None:
                commit_result = self.gaussians.add_new_gaussians(
                    self.cur_view,
                    create_new_group=create_new_gaussians,
                    render_pkg=attemp_cur_render_pkg,
                    level=cur_level,
                    reference_cameras=reference_cameras,
                    coverage_recovery=self.current_frame_coverage_recovered,
                    coverage_recovery_translation_m=(
                        self.current_frame_coverage_recovery_translation_m
                    ),
                )
                if self.current_frame_coverage_recovered:
                    self.last_coverage_recovery_commit = commit_result
            else:
                proposal_start = time.perf_counter()
                if self.gaussians.densification_mode == "sparse_points_only":
                    proposals = self.gaussians.propose_new_gaussians_pts_only(
                        self.cur_view
                    )
                else:
                    proposals = self.gaussians.propose_new_gaussians(
                        self.cur_view,
                        create_new_group=create_new_gaussians,
                        render_pkg=attemp_cur_render_pkg,
                        level=cur_level,
                        reference_cameras=reference_cameras,
                        coverage_recovery=self.current_frame_coverage_recovered,
                        coverage_recovery_translation_m=(
                            self.current_frame_coverage_recovery_translation_m
                        ),
                    )
                proposal_ms = (time.perf_counter() - proposal_start) * 1000.0
                if self.worldtest_controller is not None:
                    self.last_aerocommit_stats = self.worldtest_controller.process(
                        self.cur_view,
                        proposals,
                        is_key_frame=is_key_frame,
                        proposal_ms=proposal_ms,
                    )
                    self.worldtest_controller.save_shadow_debug(self.cur_view)
                else:
                    self.last_aerocommit_stats = self.aerocommit_manager.process_proposals(
                        self.cur_view,
                        proposals,
                        is_key_frame=is_key_frame,
                        proposal_ms=proposal_ms,
                    )
            self.cur_group_gaussian_frames += 1
        elif self.initialization_frames >= 0:
            raise NotImplementedError("Not implemented")
            self.last_add_gaussians = self.cur_view.cam_idx
            self.gaussians.add_new_gaussians(
                self.cur_view, render_pkg=attemp_cur_render_pkg, level=cur_level
            )  # Add new gaussians for every frames before initialization
            self.cur_group_gaussian_frames += 1

        if reference_cameras is not None and not self.pin_kf_gpu:
            for reference in reference_cameras:
                reference.to_device("cpu")
        return attemp_cur_render_pkg

    def refine_frontview_coverage_newborns(self):
        config = self.frontview_coverage_recovery_config
        steps = int(config["newborn_optimization_iters"])
        result = self.last_coverage_recovery_commit
        if (
            not self.current_frame_coverage_recovered
            or result is None
            or int(result.committed) <= 0
            or result.group_id is None
        ):
            return

        group_id = int(result.group_id)
        self.active_coverage_recovery_group_id = group_id
        self.active_coverage_recovery_birth_frame = int(self.cur_view.cam_idx)
        self.coverage_recovery_tracking_views = [self.cur_view]
        if bool(config["newborn_freeze_positions"]):
            self.gaussians.freeze_group_positions(group_id)
        if steps <= 0:
            return
        elapsed = self._optimize_frontview_coverage_group(group_id, steps)
        stats = self.frontview_coverage_recovery_stats
        stats["newborn_refinement_calls"] += 1
        stats["newborn_refinement_steps"] += steps
        stats["newborn_refinement_rows"] += int(result.committed)
        stats["newborn_refinement_seconds"] += elapsed

    def _optimize_frontview_coverage_group(self, group_id, steps, views=None):
        if self.gaussians.get_group_num_gaussians(group_id) <= 0:
            return 0.0
        config = self.frontview_coverage_recovery_config
        self.gaussians.bound_group_scale_expansion(
            group_id, config["newborn_max_scale_expansion"]
        )
        disabled_groups = self.gaussians.isolate_optimization_to_group(group_id)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        try:
            self.optimize(
                is_last_frame=False,
                is_key_frame=False,
                level=0,
                steps=int(steps),
                current_view_only=views is None,
                optimize_pose=False,
                supplemental=True,
                views_override=views,
            )
        finally:
            self.gaussians.restore_group_optimization(disabled_groups)
        torch.cuda.synchronize()
        return time.perf_counter() - start_time

    def refine_frontview_coverage_tracking(self):
        config = self.frontview_coverage_recovery_config
        interval = int(config["tracking_update_interval"])
        group_id = self.active_coverage_recovery_group_id
        frame_gap = int(self.cur_view.cam_idx) - int(
            self.active_coverage_recovery_birth_frame
        )
        if (
            interval <= 0
            or group_id is None
            or frame_gap <= 0
            or len(self.cur_view.get_pts())
            >= int(self.frame_checker.pts_num_threshold)
        ):
            return
        self.coverage_recovery_tracking_views.append(self.cur_view)
        window_size = int(config["tracking_window_frames"])
        self.coverage_recovery_tracking_views = (
            self.coverage_recovery_tracking_views[-window_size:]
        )
        if frame_gap % interval != 0:
            return
        steps = int(config["tracking_optimization_iters"])
        views = (
            self.coverage_recovery_tracking_views
            if window_size > 1
            else None
        )
        elapsed = self._optimize_frontview_coverage_group(
            group_id, steps, views=views
        )
        if elapsed <= 0.0:
            return
        stats = self.frontview_coverage_recovery_stats
        stats["tracking_refinement_calls"] += 1
        stats["tracking_refinement_steps"] += steps
        stats["tracking_refinement_seconds"] += elapsed

    # add new gaussians simpler
    def densification_pts_only(self):
        if self.aerocommit_manager is None:
            self.gaussians.add_new_gaussians_pts_only(self.cur_view)
            return
        proposal_start = time.perf_counter()
        proposals = self.gaussians.propose_new_gaussians_pts_only(self.cur_view)
        proposal_ms = (time.perf_counter() - proposal_start) * 1000.0
        if self.worldtest_controller is not None:
            self.last_aerocommit_stats = self.worldtest_controller.process(
                self.cur_view,
                proposals,
                is_key_frame=False,
                proposal_ms=proposal_ms,
            )
        else:
            self.last_aerocommit_stats = self.aerocommit_manager.process_proposals(
                self.cur_view,
                proposals,
                is_key_frame=False,
                proposal_ms=proposal_ms,
            )

    @torch.no_grad()
    def _should_recover_frontview_keyframe(self, cur_view):
        config = self.frontview_coverage_recovery_config
        if not config["enabled"] or self.kf_graph.last_kf_info is None:
            return False
        sparse_count = len(cur_view.get_pts())
        if sparse_count >= int(self.frame_checker.pts_num_threshold):
            return False
        if float(cur_view.rot_speed) > float(self.frame_checker.rot_speed_threshold):
            return False
        stats = self.frontview_coverage_recovery_stats
        stats["sparse_dropout_frames"] += 1
        last = self.kf_graph.last_kf_info
        frame_gap = int(cur_view.cam_idx) - int(last["cam_idx"])
        if frame_gap < int(config["min_frame_gap"]):
            return False
        stats["interval_eligible_frames"] += 1
        current_pose = cur_view.get_raw_pose().detach().cpu().numpy()
        translation, rotation = pose_novelty(last["pose"], current_pose)
        if (
            translation < float(config["min_translation_m"])
            and rotation < float(config["min_rotation_deg"])
        ):
            return False
        stats["pose_novel_frames"] += 1
        if self.gaussians.get_num_gaussians == 0:
            return False
        cur_view.to_device(self.device)
        render_pkg = self.gaussians.render(cur_view, level=0)
        certificate = coverage_recovery_certificate(
            frame_id=int(cur_view.cam_idx),
            last_keyframe_id=int(last["cam_idx"]),
            last_world_to_camera=last["pose"],
            current_world_to_camera=current_pose,
            rendered=render_pkg["render"],
            target=cur_view.get_gt_image(0),
            opacity=render_pkg["opacity"],
            config=config,
        )
        stats["rendered_certificates"] += 1
        stats["failure_fraction_sum"] += certificate["failure_fraction"]
        stats["mean_residual_sum"] += certificate["mean_residual"]
        stats["translation_sum_m"] += certificate["translation_m"]
        stats["rotation_sum_deg"] += certificate["rotation_deg"]
        if not certificate["admitted"]:
            return False
        stats["admitted_keyframes"] += 1
        stats["last_admitted_translation_m"] = float(
            certificate["translation_m"]
        )
        stats["last_admitted_frame"] = int(cur_view.cam_idx)
        if stats["first_admitted_frame"] < 0:
            stats["first_admitted_frame"] = int(cur_view.cam_idx)
        Log(
            "Coverage recovery frame {} | sparse {} | failure {:.3f} | "
            "motion {:.2f}m/{:.2f}deg".format(
                cur_view.cam_idx,
                sparse_count,
                certificate["failure_fraction"],
                certificate["translation_m"],
                certificate["rotation_deg"],
            ),
            tag="SceneMapper",
        )
        return True

    def frontview_coverage_recovery_summary(self):
        result = dict(self.frontview_coverage_recovery_stats)
        checked = int(result["rendered_certificates"])
        result.update(
            {
                "enabled": bool(self.frontview_coverage_recovery_config["enabled"]),
                "mean_failure_fraction": (
                    result["failure_fraction_sum"] / checked if checked else None
                ),
                "mean_residual": (
                    result["mean_residual_sum"] / checked if checked else None
                ),
                "mean_translation_m": (
                    result["translation_sum_m"] / checked if checked else None
                ),
                "mean_rotation_deg": (
                    result["rotation_sum_deg"] / checked if checked else None
                ),
            }
        )
        result.update(self.gaussians.frontview_coverage_recovery_depth_summary())
        return result

    # Update keyframe cameras
    # also add new gaussians here if needed
    def update_kf(self):
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t3 = torch.cuda.Event(enable_timing=True)
        t4 = torch.cuda.Event(enable_timing=True)

        t1.record()
        cur_view = self.get_frame()
        t2.record()
        torch.cuda.synchronize()
        get_frame_time = t1.elapsed_time(t2) / 1000.0
        t3.record()

        if cur_view is None:  # End of the sequence
            return True, False, 0, 0

        self.cur_view = cur_view

        # update the current index
        self.cur_idx = cur_view.cam_idx

        # update the number of processed frames
        self.processed_frames += 1

        self.gaussians.observe_frontview_raw_pose(cur_view)

        # prune gaussians
        if self.prune_interval > 0 and self.processed_frames % self.prune_interval == 0:
            self.gaussians.prune_w_opacity(
                current_frame_id=int(cur_view.cam_idx),
                processed_frames=int(self.processed_frames),
            )

        # Check frame's quality: (rotation speed, pts num)
        # is_good_frame = self.frame_checker.check(cur_view) or self.initialization_frames >= 0
        is_good_frame = self.frame_checker.check(cur_view)
        self.current_frame_coverage_recovered = False
        self.current_frame_coverage_recovery_translation_m = None

        # Preserve every view during the configured high-motion bootstrap window.
        if cur_view.cam_idx <= self.force_keyframes_through_frame:
            self.kf_graph.add_new_cam_to_kf(cur_view)
            if self.kf_graph.get_kf_num == 1:
                self.kf_graph.init_global_window()
            is_key_frame = True
        elif is_good_frame:  # Do not add key frames if the rotation speed is too high
            is_key_frame = self.kf_graph.update_frame(cur_view)
        elif self._should_recover_frontview_keyframe(cur_view):
            self.kf_graph.add_new_cam_to_kf(cur_view)
            if self.kf_graph.get_kf_num == 1:
                self.kf_graph.init_global_window()
            is_key_frame = True
            self.current_frame_coverage_recovered = True
            self.current_frame_coverage_recovery_translation_m = float(
                self.frontview_coverage_recovery_stats[
                    "last_admitted_translation_m"
                ]
            )
        else:
            is_key_frame = False

        self.update_flow_detail(cur_view)
        self.update_track_detail(cur_view)

        if self.pin_kf_gpu:
            self.cur_view.to_device(self.device)

        if bool(self.post_refinement_config.get("use_all_frames", False)) or bool(
            self.post_refinement_config.get("pose_prepass_enabled", False)
        ):
            self.post_refinement_frames.append(cur_view)

        # Use all frames for training (but not all of them are key frames)
        if len(self.active_window) < self.active_window_size:
            self.active_window.append(cur_view)
        else:
            self.active_window.pop(0)
            self.active_window.append(cur_view)

        # Naively way to add coarse key frames
        if is_key_frame:
            if len(self.coarse_pool) <= self.coarse_active_window_size:
                self.coarse_active_window = self.coarse_pool
            else:
                covis = (
                    np.array(
                        [
                            cal_cams_covisibility(cur_view, cam)
                            for cam in self.coarse_pool
                        ]
                    )
                    + 1e-6
                )

                prob = covis / np.sum(covis)

                self.coarse_active_window = np.random.choice(
                    self.coarse_pool,
                    self.coarse_active_window_size,
                    replace=False,
                    p=prob,
                )

            if len(self.coarse_pool) < self.coarse_pool_size:
                self.coarse_pool.append(cur_view)
            elif self.coarse_pool_size != 0:
                self.coarse_pool.pop(0)
                self.coarse_pool.append(cur_view)

        # Only add key frame camera to camera optimizer
        if is_key_frame:
            self.camera_optimizer.add_cam(cur_view)

        Log(
            "Process frame: {} | total gaussians {} | kf size {}".format(
                cur_view.cam_idx,
                self.gaussians.get_num_gaussians,
                self.kf_graph.get_kf_num,
            ),
            tag="SceneMapper",
        )
        # Log("Active window {} | Coarse active window {}".format([cam.cam_idx for cam in self.active_window], [cam.cam_idx for cam in self.coarse_active_window]), tag="SceneMapper")

        t4.record()
        torch.cuda.synchronize()
        frame_process_time = t3.elapsed_time(t4) / 1000.0

        self.opt_log["is_key_frame"][cur_view.name] = is_key_frame

        return False, is_key_frame, get_frame_time, frame_process_time

    def camera_refinement(self):
        if len(self.active_window) == 0:
            return

        if self.gaussians.get_num_gaussians == 0:
            return

        optimized_local_window = []
        exist_idx = []
        # cur_res_idx is removed as it was assigned but never used
        for cam in self.active_window:
            if cam.cam_idx not in exist_idx:
                cam.to_device(self.device)
                optimized_local_window.append(cam)
                exist_idx.append(cam.cam_idx)
                # Removed assignment to cur_res_idx
        for cam in self.coarse_active_window:
            if cam.cam_idx not in exist_idx:
                cam.to_device(self.device)
                optimized_local_window.append(cam)
                exist_idx.append(cam.cam_idx)

        gt_imgs = torch.stack([cam.get_gt_image() for cam in optimized_local_window])
        gt_mask = (
            torch.sum(gt_imgs, dim=-1, keepdim=True) <= 0.0000001
        ).float()  # ignore invalid pixels in

        for cam in optimized_local_window:
            cam.set_opt_pose(True)

        for _ in range(self.pose_refine_init_steps):
            batch_render_pkg = self.gaussians.render_batch(
                optimized_local_window,
                self.use_random_bg,
                detach_gaussians=True,
                external_splats=self.get_stable_external_splats(
                    optimized_local_window
                ),
            )
            batch_render = batch_render_pkg["render"]

            iter_gt_imgs = gt_imgs * (1.0 - gt_mask) + batch_render * gt_mask

            loss_img, _ = self.image_loss(batch_render, iter_gt_imgs)

            loss = loss_img
            loss.backward()

            self.camera_optimizer.step()  # Update cameras, will zero the gradients inside this function

            # save tracked poses
            for cam in optimized_local_window:
                if cam.name is not None:
                    self.opt_log["tracked_poses"][cam.name] = (
                        cam.get_pose().detach().cpu().numpy()
                    )
                    orig_data = self.opt_log["poses_pair"][cam.cam_idx]
                    self.opt_log["poses_pair"][cam.cam_idx] = (
                        orig_data[0],
                        cam.get_pose().detach().cpu().numpy(),
                        orig_data[2],
                    )

                    if cam.name not in self.opt_log["cam_opt_iterations"]:
                        self.opt_log["cam_opt_iterations"][cam.name] = 1
                    else:
                        self.opt_log["cam_opt_iterations"][cam.name] += 1

        for cam in optimized_local_window:
            cam.set_opt_pose(False)
        # Don't need to set them back to cpu, they will be used again in the map optimization
        # for cam in optimized_local_window:
        #     cam.to_device("cpu")

    def pose_pre_refinement(self):
        """Refine observed poses at low resolution while the map stays frozen."""

        if not bool(self.post_refinement_config.get("pose_prepass_enabled", False)):
            return
        if not self.post_refinement_frames:
            return
        steps = max(
            0, int(self.post_refinement_config.get("pose_prepass_steps", 0))
        )
        if steps == 0:
            return

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        level = min(
            max(int(self.post_refinement_config.get("pose_prepass_level", 2)), 0),
            self.cur_view.MAX_LEVEL - 1,
        )
        batch_size = min(
            max(
                1,
                int(
                    self.post_refinement_config.get("pose_prepass_batch_size", 8)
                ),
            ),
            len(self.post_refinement_frames),
        )
        anchor_idx = (
            min(cam.cam_idx for cam in self.post_refinement_frames)
            if bool(self.post_refinement_config.get("anchor_first_pose", True))
            else None
        )
        for cam in self.post_refinement_frames:
            if cam.camera_opt is None:
                self.camera_optimizer.add_cam(cam)

        optimizer = self.camera_optimizer.optimizer
        original_lrs = [group["lr"] for group in optimizer.param_groups]
        pose_lr = float(
            self.post_refinement_config.get("pose_prepass_lr", original_lrs[0])
        )
        for group in optimizer.param_groups:
            group["lr"] = pose_lr

        prior_weight = float(
            self.post_refinement_config.get("pose_prepass_prior_weight", 0.01)
        )
        max_translation = float(
            self.post_refinement_config.get("pose_prepass_max_translation", 0.05)
        )
        used_cams = []
        used_ids = set()
        for _ in range(steps):
            indices = np.random.choice(
                len(self.post_refinement_frames),
                size=batch_size,
                replace=False,
            ).tolist()
            cameras = [self.post_refinement_frames[index] for index in indices]
            for cam in cameras:
                cam.to_device(self.device)
                cam.set_opt_pose(cam.cam_idx != anchor_idx)
                if self.pin_kf_gpu and cam.cam_idx not in used_ids:
                    used_ids.add(cam.cam_idx)
                    used_cams.append(cam)

            gt = torch.stack([cam.get_gt_image(level) for cam in cameras])
            render_pkg = self.gaussians.render_batch(
                cameras,
                self.use_random_bg,
                detach_gaussians=True,
                level=level,
                external_splats=self.get_stable_external_splats(cameras),
            )
            render = render_pkg["render"]
            invalid = (torch.sum(gt, dim=-1, keepdim=True) <= 1.0e-7).float()
            gt = gt * (1.0 - invalid) + render * invalid
            loss, _ = self.image_loss(render, gt)
            frequency_pose_weight = float(
                self.post_refinement_config.get(
                    "pose_prepass_frequency_weight", 0.0
                )
            )
            if frequency_pose_weight > 0.0:
                loss = loss + frequency_pose_weight * frequency_weighted_pose_loss(
                    render,
                    gt,
                    valid_mask=1.0 - invalid,
                    gradient_threshold=float(
                        self.post_refinement_config.get(
                            "pose_prepass_gradient_threshold", 0.04
                        )
                    ),
                    edge_weight=float(
                        self.post_refinement_config.get(
                            "pose_prepass_edge_weight", 1.0
                        )
                    ),
                    side_start=float(
                        self.post_refinement_config.get(
                            "pose_prepass_side_start", 0.45
                        )
                    ),
                    side_boost=float(
                        self.post_refinement_config.get(
                            "pose_prepass_side_boost", 2.0
                        )
                    ),
                )

            if prior_weight > 0.0:
                pose_prior = render.new_zeros(())
                prior_count = 0
                identity = torch.eye(3, device=render.device, dtype=render.dtype)
                for cam in cameras:
                    if cam.cam_idx == anchor_idx:
                        continue
                    delta = self.camera_optimizer.get_delta_transform(
                        cam.cam_opt_idx
                    ).matrix()
                    pose_prior = pose_prior + delta[:3, 3].square().sum()
                    pose_prior = pose_prior + 0.25 * (
                        delta[:3, :3] - identity
                    ).square().sum()
                    prior_count += 1
                if prior_count:
                    loss = loss + prior_weight * pose_prior / prior_count

            loss.backward()
            self.camera_optimizer.step()

            if max_translation > 0.0:
                with torch.no_grad():
                    embed_indices = torch.as_tensor(
                        [
                            cam.cam_opt_idx
                            for cam in cameras
                            if cam.cam_idx != anchor_idx
                        ],
                        device=self.camera_optimizer.embeds.weight.device,
                        dtype=torch.long,
                    )
                    if embed_indices.numel():
                        translations = self.camera_optimizer.embeds.weight[
                            embed_indices, :3
                        ]
                        norms = torch.linalg.norm(translations, dim=1, keepdim=True)
                        translations.mul_(
                            torch.clamp(max_translation / torch.clamp(norms, min=1.0e-8), max=1.0)
                        )

            for cam in cameras:
                pose = cam.get_pose().detach()
                self.opt_log["tracked_poses"][cam.name] = pose.cpu().numpy()
                cam.set_opt_pose(False)
                if not self.pin_kf_gpu:
                    cam.to_device("cpu")

        for group, lr in zip(optimizer.param_groups, original_lrs):
            group["lr"] = lr
        for cam in used_cams:
            cam.set_opt_pose(False)
            cam.to_device("cpu")
        torch.cuda.synchronize()
        self.aerocommit_runtime["pose_prepass_seconds"] = (
            time.perf_counter() - start_time
        )
        Log(
            "Low-resolution pose prepass: {:.3f} s for {} frames".format(
                self.aerocommit_runtime["pose_prepass_seconds"],
                len(self.post_refinement_frames),
            ),
            tag="SceneMapper",
        )

    def split_stable_side_detail(self):
        config = self.post_refinement_config.get("stable_detail_split", {})
        if not bool(config.get("enabled", False)) or not self.post_refinement_frames:
            return
        start_time = time.perf_counter()
        max_views = max(1, int(config.get("max_views", 24)))
        if len(self.post_refinement_frames) <= max_views:
            view_indices = list(range(len(self.post_refinement_frames)))
        else:
            view_indices = np.linspace(
                0, len(self.post_refinement_frames) - 1, max_views, dtype=np.int64
            ).tolist()
        active_count = self.gaussians.get_num_gaussians
        per_view_limit = max(1, int(config.get("max_candidates_per_view", 4096)))
        best = {}
        support_counts = {}

        for view_index in view_indices:
            cam = self.post_refinement_frames[view_index]
            cam.to_device(self.device)
            with torch.no_grad():
                render_pkg = self.gaussians.render(
                    cam,
                    external_splats=self.get_stable_external_splats([cam]),
                    return_info=True,
                )
                gaussian_ids, scores = stable_detail_split_scores(
                    cam.get_gt_image(),
                    render_pkg["projection_info"],
                    config,
                    rendered=render_pkg["render"],
                )
                valid = torch.isfinite(scores) & (gaussian_ids < active_count)
                candidate_positions = torch.nonzero(valid, as_tuple=False).reshape(-1)
                if candidate_positions.numel() > per_view_limit:
                    top = torch.topk(
                        scores[candidate_positions], per_view_limit, sorted=False
                    ).indices
                    candidate_positions = candidate_positions[top]
                camera_to_world = torch.linalg.inv(cam.get_pose())
                tangent_x = camera_to_world[:3, 0].detach().cpu().numpy()
                tangent_y = camera_to_world[:3, 1].detach().cpu().numpy()
                ids = gaussian_ids[candidate_positions].detach().cpu().tolist()
                values = scores[candidate_positions].detach().cpu().tolist()
            for gaussian_id, score in zip(ids, values):
                gaussian_id = int(gaussian_id)
                support_counts[gaussian_id] = support_counts.get(gaussian_id, 0) + 1
                if gaussian_id not in best or score > best[gaussian_id][0]:
                    best[gaussian_id] = (float(score), tangent_x.copy(), tangent_y.copy())
            cam.to_device("cpu")

        min_views = max(1, int(config.get("min_support_views", 2)))
        eligible = [
            (record[0], gaussian_id, record[1], record[2])
            for gaussian_id, record in best.items()
            if support_counts.get(gaussian_id, 0) >= min_views
        ]
        eligible.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected = eligible[: max(0, int(config.get("max_splits", 512)))]
        if selected:
            parents, added = self.gaussians.split_stable_detail_gaussians(
                [item[1] for item in selected],
                np.stack([item[2] for item in selected]),
                np.stack([item[3] for item in selected]),
                config,
            )
        else:
            parents, added = 0, 0
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        if self.aerocommit_manager is not None:
            self.aerocommit_runtime["stable_detail_split_seconds"] = elapsed
            self.aerocommit_runtime["stable_detail_split_parents"] = parents
            self.aerocommit_runtime["stable_detail_split_added"] = added
        Log(
            "Stable side detail split: {} parents / {} added in {:.3f} s".format(
                parents, added, elapsed
            ),
            tag="SceneMapper",
        )

    def prepare_full_map_refinement(self):
        """Make the complete map visible to replay while preserving world geometry."""

        freeze_geometry = bool(
            self.post_refinement_config.get("freeze_geometry", False)
        )
        restore_archives = bool(
            self.post_refinement_config.get(
                "restore_aerocommit_archives", False
            )
        )
        if restore_archives and self.aerocommit_manager is not None:
            restored = self.aerocommit_manager.restore_all_archives_for_refinement(
                freeze_geometry=freeze_geometry
            )
            self.aerocommit_runtime["full_map_restore_seconds"] = restored[
                "seconds"
            ]
            self.aerocommit_runtime["full_map_restore_groups"] = restored[
                "groups"
            ]
            self.aerocommit_runtime["full_map_restore_gaussians"] = restored[
                "gaussians"
            ]
            Log(
                "Restored {} archived groups / {} Gaussians for full-map replay in {:.3f} s".format(
                    restored["groups"],
                    restored["gaussians"],
                    restored["seconds"],
                ),
                tag="SceneMapper",
            )
        if freeze_geometry:
            for group_id in list(self.gaussians.valid_groups):
                self.gaussians.freeze_group_geometry(group_id)
            Log(
                "Full-map replay freezes means/scales/quats for {} groups".format(
                    len(self.gaussians.valid_groups)
                ),
                tag="SceneMapper",
            )

    def post_refinement(self):
        if self.kf_graph.global_widndow_size == 0:
            return

        self.prepare_full_map_refinement()
        self.apply_track_detail_reassignment()
        self.pose_pre_refinement()
        self.split_stable_side_detail()

        use_all_frames = bool(
            self.post_refinement_config.get("use_all_frames", False)
        ) and bool(self.post_refinement_frames)
        replay_batch_size = min(
            max(
                1,
                int(
                    self.post_refinement_config.get(
                        "all_frame_batch_size", self.kf_graph.global_widndow_size
                    )
                ),
            ),
            (
                len(self.post_refinement_frames)
                if use_all_frames
                else self.kf_graph.global_widndow_size
            ),
        )

        if self.progressive_manager is not None:
            self.progressive_manager.configure_post_refinement_optimization()
        self.gaussians.reset_optimizer(
            self.post_refinement_config, replay_batch_size
        )
        self.configure_post_refinement_appearance_lod()
        appearance_anchor = AppearanceProximalAnchor(
            self.gaussians,
            self.post_refinement_config.get("appearance_anchor"),
        )
        origin_kf_method = self.kf_graph.get_get_next_camera_method()
        self.kf_graph.set_get_next_camera_method(
            self.post_refinement_config.get("sampling_method", "random")
        )

        opt_cam = self.post_refinement_config["opt_cam"]
        anchor_first_pose = bool(
            self.post_refinement_config.get("anchor_first_pose", True)
        )
        anchor_cam_idx = (
            min(cam.cam_idx for cam in self.post_refinement_frames)
            if use_all_frames and anchor_first_pose
            else None
        )

        if use_all_frames and opt_cam:
            for cam in self.post_refinement_frames:
                if cam.camera_opt is None:
                    self.camera_optimizer.add_cam(cam)

        if use_all_frames:
            Log(
                "Post refinement replays all {} frames in batches of {}".format(
                    len(self.post_refinement_frames), replay_batch_size
                ),
                tag="SceneMapper",
            )

        used_cams = []
        exist_idx = []
        frame_errors = {
            int(cam.cam_idx): 1.0 for cam in self.post_refinement_frames
        }
        uniform_order = (
            np.random.permutation(len(self.post_refinement_frames)).tolist()
            if use_all_frames
            else []
        )
        uniform_cursor = 0
        hard_fraction = float(
            self.post_refinement_config.get("all_frame_hard_fraction", 0.0)
        )
        hard_fraction = min(max(hard_fraction, 0.0), 1.0)
        hard_warmup_steps = max(
            0,
            int(self.post_refinement_config.get("all_frame_hard_warmup_steps", 50)),
        )
        min_uniform_replays = max(
            0,
            int(self.post_refinement_config.get("all_frame_min_uniform_replays", 0)),
        )
        uniform_counts = np.zeros(len(self.post_refinement_frames), dtype=np.int64)

        def take_uniform_indices(count, excluded):
            nonlocal uniform_cursor, uniform_order
            selected = []
            while len(selected) < count:
                if uniform_cursor >= len(uniform_order):
                    uniform_order = np.random.permutation(
                        len(self.post_refinement_frames)
                    ).tolist()
                    uniform_cursor = 0
                candidate = uniform_order[uniform_cursor]
                uniform_cursor += 1
                if candidate not in excluded and candidate not in selected:
                    selected.append(candidate)
            return selected

        pbar = tqdm(range(self.post_refinement_config["max_steps"]))
        appearance_only_start = int(
            self.post_refinement_config.get("appearance_only_start_step", -1)
        )
        appearance_mse_start = int(
            self.post_refinement_config.get("appearance_mse_start_step", -1)
        )
        appearance_mse_mix = float(
            self.post_refinement_config.get("appearance_mse_mix", 0.0)
        )
        detail_loss_start = int(
            self.post_refinement_config.get("detail_loss_start_step", 0)
        )
        far_structure_tail_steps = max(
            0,
            int(self.post_refinement_config.get("far_structure_tail_steps", 0)),
        )
        far_structure_tail_start = max(
            0,
            self.post_refinement_config["max_steps"] - far_structure_tail_steps,
        )
        far_structure_weight = max(
            0.0,
            float(self.post_refinement_config.get("far_structure_weight", 0.0)),
        )
        far_structure_depth_m = float(
            self.post_refinement_config.get("far_structure_depth_m", 25.0)
        )
        far_structure_opacity = float(
            self.post_refinement_config.get("far_structure_opacity", 0.1)
        )

        for step in pbar:
            if appearance_only_start >= 0 and step >= appearance_only_start:
                self.gaussians.set_parameter_optimizer_lr("means", 0.0)
                self.gaussians.set_parameter_optimizer_lr("quats", 0.0)
                if bool(
                    self.post_refinement_config.get(
                        "appearance_freeze_scales", False
                    )
                ):
                    self.gaussians.set_parameter_optimizer_lr("scales", 0.0)
            if use_all_frames:
                num_hard = (
                    min(
                        replay_batch_size - 1,
                        int(round(replay_batch_size * hard_fraction)),
                    )
                    if step >= hard_warmup_steps
                    and (
                        min_uniform_replays == 0
                        or bool(np.all(uniform_counts >= min_uniform_replays))
                    )
                    else 0
                )
                hard_indices = []
                if num_hard > 0:
                    error_weights = np.asarray(
                        [
                            max(frame_errors[int(cam.cam_idx)], 1.0e-6)
                            for cam in self.post_refinement_frames
                        ],
                        dtype=np.float64,
                    )
                    error_weights /= error_weights.sum()
                    hard_indices = np.random.choice(
                        len(self.post_refinement_frames),
                        size=num_hard,
                        replace=False,
                        p=error_weights,
                    ).tolist()
                replay_indices = hard_indices + take_uniform_indices(
                    replay_batch_size - len(hard_indices), set(hard_indices)
                )
                for replay_index in replay_indices[len(hard_indices) :]:
                    uniform_counts[replay_index] += 1
                global_windows = [
                    self.post_refinement_frames[idx] for idx in replay_indices
                ]
            else:
                global_windows = self.kf_graph.get_and_update_global_window()
            for cam in global_windows:
                if not self.pin_kf_gpu:
                    cam.to_device(self.device)
                    if opt_cam:
                        cam.set_opt_pose(cam.cam_idx != anchor_cam_idx)
                elif cam.cam_idx not in exist_idx:
                    cam.to_device(self.device)
                    if opt_cam:
                        cam.set_opt_pose(cam.cam_idx != anchor_cam_idx)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)
            global_gt_imgs = (
                torch.stack([cam.get_gt_image() for cam in global_windows])
                if len(global_windows) > 0
                else []
            )
            global_gt_depths = (
                torch.stack([cam.get_sparse_depth() for cam in global_windows])
                if len(global_windows) > 0
                else []
            )

            iter_window = global_windows
            iter_gt_imgs = global_gt_imgs
            iter_gt_depths = global_gt_depths

            batch_render_pkg = self.gaussians.render_batch(
                iter_window,
                self.use_random_bg,
                external_splats=self.get_stable_external_splats(iter_window),
            )
            batch_render = batch_render_pkg["render"]
            batch_render_depth = batch_render_pkg["depth"].squeeze(-1)

            gt_mask = (
                torch.sum(iter_gt_imgs, dim=-1, keepdim=True) <= 0.0000001
            ).float()  # ignore invalid pixels in gt
            iter_gt_imgs = iter_gt_imgs * (1.0 - gt_mask) + batch_render * gt_mask

            loss_img, l1_values = self.image_loss(batch_render, iter_gt_imgs)
            if appearance_mse_start >= 0 and step >= appearance_mse_start:
                loss_img = blend_mse_tail_loss(
                    loss_img,
                    batch_render,
                    iter_gt_imgs,
                    appearance_mse_mix,
                    self.post_refinement_config.get("appearance_mse_scale", 1.0),
                )
            loss_depth = self.depth_loss(batch_render_depth, iter_gt_depths)
            loss_gaussians = self.gaussian_loss(self.gaussians)
            detail_loss = (
                self.post_refinement_detail_loss(
                    batch_render,
                    iter_gt_imgs,
                    batch_render_depth,
                    batch_render_pkg.get("opacity"),
                )
                if step >= detail_loss_start
                else batch_render.new_zeros(())
            )
            far_structure_loss = batch_render.new_zeros(())
            if (
                far_structure_weight > 0.0
                and step >= far_structure_tail_start
            ):
                opacity = batch_render_pkg.get("opacity")
                far_mask = (
                    torch.isfinite(batch_render_depth)
                    & (batch_render_depth > far_structure_depth_m)
                )
                if opacity is not None:
                    far_mask &= opacity.squeeze(-1) > far_structure_opacity
                far_structure_loss = far_structure_weight * masked_ssim_loss(
                    batch_render,
                    iter_gt_imgs,
                    far_mask,
                )
            appearance_anchor_loss = batch_render.new_zeros(())
            if appearance_anchor.config["enabled"]:
                appearance_anchor_loss, _ = appearance_anchor.loss(self.gaussians)

            if (
                "normal" in batch_render_pkg
                and "normal_from_depth" in batch_render_pkg
                and "opacity" in batch_render_pkg
            ):
                loss_normal = self.normal_loss(
                    batch_render_pkg["normal"],
                    batch_render_pkg["normal_from_depth"],
                    batch_render_pkg["opacity"],
                )
            else:
                loss_normal = 0.0

            if "distortion" in batch_render_pkg:
                loss_distortion = self.distortion_loss(batch_render_pkg["distortion"])
            else:
                loss_distortion = 0.0

            loss = (
                loss_img
                + loss_gaussians
                + loss_depth
                + loss_normal
                + loss_distortion
                + detail_loss
                + far_structure_loss
                + appearance_anchor_loss
            )

            for batch_index, (cam, err) in enumerate(zip(iter_window, l1_values)):
                if cam.cam_idx in self.kf_graph.camIdx_to_kfIdx:
                    self.kf_graph.update_err(cam.cam_idx, err)
                if use_all_frames:
                    previous = frame_errors[int(cam.cam_idx)]
                    current_error = (
                        float(err.detach().item())
                        if torch.is_tensor(err)
                        else float(err)
                    )
                    frequency_error_weight = float(
                        self.post_refinement_config.get(
                            "all_frame_frequency_error_weight", 0.0
                        )
                    )
                    if frequency_error_weight > 0.0:
                        frequency_error = self.post_refinement_frequency_error(
                            batch_render[batch_index : batch_index + 1].detach(),
                            iter_gt_imgs[batch_index : batch_index + 1].detach(),
                        )[0]
                        current_error += frequency_error_weight * float(
                            frequency_error.item()
                        )
                    frame_errors[int(cam.cam_idx)] = (
                        0.8 * previous + 0.2 * current_error
                    )

            loss.backward()

            self.gaussians.mask_sh_degree_gradients()
            if bool(self.frontview_observability_config["apply_post_refinement"]):
                self.gaussians.precondition_frontview_mean_gradients(
                    iter_window,
                    self.frontview_observability_config,
                    update_evidence=(
                        step
                        % int(
                            self.frontview_observability_config[
                                "evidence_update_interval"
                            ]
                        )
                        == 0
                    ),
                )

            self.gaussians.update()  # Update gaussians, will zero the gradients inside this function

            if opt_cam:
                self.camera_optimizer.step()  # Update cameras, will zero the gradients inside this function

            self.gaussians.step_all_lr()  # Update learning rate

            # save tracked poses
            for i, cam in enumerate(iter_window):
                if cam.name is not None:
                    self.opt_log["tracked_poses"][cam.name] = (
                        cam.get_pose().detach().cpu().numpy()
                    )
                    orig_data = self.opt_log["poses_pair"][cam.cam_idx]
                    self.opt_log["poses_pair"][cam.cam_idx] = (
                        orig_data[0],
                        cam.get_pose().detach().cpu().numpy(),
                        orig_data[2],
                    )

                    if cam.cam_idx not in self.opt_log["l1_err"]:
                        self.opt_log["l1_err"][cam.cam_idx] = {}
                    err = l1_values[i]
                    self.opt_log["l1_err"][cam.cam_idx][self.cur_view.cam_idx] = (
                        float(err.detach().item())
                        if torch.is_tensor(err)
                        else float(err)
                    )

                    if cam.name not in self.opt_log["gaussian_opt_iterations"]:
                        self.opt_log["gaussian_opt_iterations"][cam.name] = 1
                    else:
                        self.opt_log["gaussian_opt_iterations"][cam.name] += 1

                    if opt_cam and cam.cam_idx != anchor_cam_idx:
                        if cam.name not in self.opt_log["cam_opt_iterations"]:
                            self.opt_log["cam_opt_iterations"][cam.name] = 1
                        else:
                            self.opt_log["cam_opt_iterations"][cam.name] += 1

            pbar.set_description(
                "Loss: {:.4f} | Mean Lr: {:.7f}".format(
                    loss.item(), self.gaussians.get_avg_pos_lr()
                )
            )

            if not self.pin_kf_gpu:
                for cam in global_windows:
                    if opt_cam:
                        cam.set_opt_pose(False)
                    cam.to_device("cpu")

        for cam in used_cams:
            if opt_cam:
                cam.set_opt_pose(False)
            cam.to_device("cpu")

        if appearance_anchor.config["enabled"]:
            self.appearance_anchor_stats = appearance_anchor.report(self.gaussians)
        self.kf_graph.set_get_next_camera_method(origin_kf_method)

    @torch.no_grad()
    def configure_post_refinement_appearance_lod(self):
        config = self.post_refinement_config.get("appearance_lod", {})
        if not bool(config.get("enabled", False)):
            return
        if self.gaussians.max_sh_degree < 1:
            raise ValueError("appearance_lod requires Model.sh_degree >= 1")
        if not self.post_refinement_frames:
            raise ValueError("appearance_lod requires replay frames")

        frame_stride = max(1, int(config.get("observation_stride", 1)))
        evidence_frames = self.post_refinement_frames[::frame_stride]
        max_frames = int(config.get("max_observation_frames", 0))
        if max_frames > 0 and len(evidence_frames) > max_frames:
            frame_indices = np.linspace(
                0, len(evidence_frames) - 1, max_frames, dtype=np.int64
            )
            evidence_frames = [evidence_frames[index] for index in frame_indices]

        means = torch.cat(
            [
                self.gaussians.get_xyz(level=level)
                for level in range(self.gaussians.MAX_LEVEL)
            ],
            dim=0,
        ).detach()
        evidence = AppearanceLODEvidence(
            gaussian_count=means.shape[0], device=means.device, dtype=means.dtype
        )
        started = time.perf_counter()
        projection_records = 0
        for camera in evidence_frames:
            camera.to_device(self.device)
            render_pkg = self.gaussians.render(camera, return_info=True)
            projection_records += evidence.observe(
                render_pkg["projection_info"],
                means,
                camera.get_pose().unsqueeze(0),
            )
            if not self.pin_kf_gpu:
                camera.to_device("cpu")

        degrees, stats = evidence.select_degrees(config)
        self.gaussians.configure_sh_degree_masks(
            degrees, zero_inactive=bool(config.get("zero_inactive", True))
        )
        stats.update(
            {
                "observation_frames": len(evidence_frames),
                "observation_stride": frame_stride,
                "projection_records": projection_records,
                "seconds": time.perf_counter() - started,
            }
        )
        self.appearance_lod_stats = stats
        Log(
            "Appearance LOD SH0/SH1/SH2: {}/{}/{} from {} frames in {:.3f} s".format(
                stats["degree_counts"]["sh0"],
                stats["degree_counts"]["sh1"],
                stats["degree_counts"]["sh2"],
                stats["observation_frames"],
                stats["seconds"],
            ),
            tag="SceneMapper",
        )

    def post_refinement_frequency_error(self, render, gt):
        """Per-frame side/high-frequency residual used only for replay sampling."""
        gray = gt.mean(dim=-1)
        gradient = torch.zeros_like(gray)
        gradient[:, :, 1:] = torch.maximum(
            gradient[:, :, 1:],
            torch.abs(gray[:, :, 1:] - gray[:, :, :-1]),
        )
        gradient[:, :, :-1] = torch.maximum(
            gradient[:, :, :-1],
            torch.abs(gray[:, :, 1:] - gray[:, :, :-1]),
        )
        gradient[:, 1:, :] = torch.maximum(
            gradient[:, 1:, :],
            torch.abs(gray[:, 1:, :] - gray[:, :-1, :]),
        )
        gradient[:, :-1, :] = torch.maximum(
            gradient[:, :-1, :],
            torch.abs(gray[:, 1:, :] - gray[:, :-1, :]),
        )
        threshold = float(
            self.post_refinement_config.get(
                "all_frame_frequency_gradient_threshold", 0.035
            )
        )
        side_start = float(
            self.post_refinement_config.get(
                "all_frame_frequency_side_start", 0.35
            )
        )
        width = gray.shape[-1]
        side = torch.linspace(
            -1.0, 1.0, width, device=gray.device, dtype=gray.dtype
        ).abs()
        mask = (gradient >= threshold) & (side[None, None, :] >= side_start)
        residual = torch.abs(render - gt).mean(dim=-1)
        normalization = torch.clamp(mask.sum(dim=(1, 2)), min=1)
        return (residual * mask.to(residual.dtype)).sum(dim=(1, 2)) / normalization

    def post_refinement_detail_loss(self, render, gt, depth, opacity=None):
        """Refine stable near-field appearance and footprint at image edges."""
        l1_weight = float(self.post_refinement_config.get("detail_l1_weight", 0.0))
        gradient_weight = float(
            self.post_refinement_config.get("detail_gradient_weight", 0.0)
        )
        laplacian_fine_weight = float(
            self.post_refinement_config.get("detail_laplacian_fine_weight", 0.0)
        )
        laplacian_coarse_weight = float(
            self.post_refinement_config.get("detail_laplacian_coarse_weight", 0.0)
        )
        if (
            l1_weight <= 0.0
            and gradient_weight <= 0.0
            and laplacian_fine_weight <= 0.0
            and laplacian_coarse_weight <= 0.0
        ):
            return render.new_zeros(())

        gradient_threshold = float(
            self.post_refinement_config.get("detail_gradient_threshold", 0.04)
        )
        near_depth = float(
            self.post_refinement_config.get("detail_near_depth_m", 0.0)
        )
        gray = gt.mean(dim=-1)
        gradient = torch.zeros_like(gray)
        gradient[:, :, 1:] += torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
        gradient[:, 1:, :] += torch.abs(gray[:, 1:, :] - gray[:, :-1, :])
        detail_weight = torch.clamp(
            gradient / max(gradient_threshold, 1.0e-6), min=0.0, max=1.0
        )

        side_boost = max(
            1.0, float(self.post_refinement_config.get("detail_side_boost", 1.0))
        )
        side_start = min(
            max(
                float(
                    self.post_refinement_config.get("detail_side_start", 1.0)
                ),
                0.0,
            ),
            1.0,
        )
        if side_boost > 1.0 and side_start < 1.0:
            width = detail_weight.shape[-1]
            columns = torch.linspace(
                -1.0, 1.0, width, device=detail_weight.device, dtype=detail_weight.dtype
            ).abs()
            side_ramp = torch.clamp(
                (columns - side_start) / max(1.0 - side_start, 1.0e-6),
                min=0.0,
                max=1.0,
            )
            detail_weight = detail_weight * (
                1.0 + (side_boost - 1.0) * side_ramp.view(1, 1, width)
            )

        if near_depth > 0.0:
            near_mask = (depth > 0.0) & (depth <= near_depth)
            low_opacity_threshold = float(
                self.post_refinement_config.get(
                    "detail_low_opacity_threshold", 0.0
                )
            )
            if opacity is not None and low_opacity_threshold > 0.0:
                if opacity.ndim == depth.ndim + 1:
                    opacity = opacity.squeeze(-1)
                near_mask = near_mask | (opacity < low_opacity_threshold)
            detail_weight = detail_weight * near_mask.to(detail_weight.dtype)
        min_opacity = float(
            self.post_refinement_config.get("detail_min_opacity", 0.0)
        )
        if opacity is not None and min_opacity > 0.0:
            if opacity.ndim == depth.ndim + 1:
                opacity = opacity.squeeze(-1)
            detail_weight = detail_weight * (opacity >= min_opacity).to(
                detail_weight.dtype
            )
        detail_weight = detail_weight.detach()
        color_error = torch.abs(render - gt).mean(dim=-1)
        normalization = torch.clamp(detail_weight.sum(), min=1.0)
        detail_loss = l1_weight * (color_error * detail_weight).sum() / normalization

        if gradient_weight > 0.0:
            render_gray = render.mean(dim=-1)
            render_dx = render_gray[:, :, 1:] - render_gray[:, :, :-1]
            render_dy = render_gray[:, 1:, :] - render_gray[:, :-1, :]
            gt_dx = gray[:, :, 1:] - gray[:, :, :-1]
            gt_dy = gray[:, 1:, :] - gray[:, :-1, :]
            weight_dx = torch.maximum(
                detail_weight[:, :, 1:], detail_weight[:, :, :-1]
            )
            weight_dy = torch.maximum(
                detail_weight[:, 1:, :], detail_weight[:, :-1, :]
            )
            grad_error = (
                torch.abs(render_dx - gt_dx) * weight_dx
            ).sum() + (torch.abs(render_dy - gt_dy) * weight_dy).sum()
            grad_normalization = torch.clamp(
                weight_dx.sum() + weight_dy.sum(), min=1.0
            )
            detail_loss = detail_loss + gradient_weight * (
                grad_error / grad_normalization
            )

        if laplacian_fine_weight > 0.0 or laplacian_coarse_weight > 0.0:
            detail_loss = detail_loss + laplacian_pyramid_reconstruction_loss(
                render,
                gt,
                pixel_weight=detail_weight,
                fine_weight=laplacian_fine_weight,
                coarse_weight=laplacian_coarse_weight,
            )

        return detail_loss

    def additional_optimization(self):
        used_cams = []
        exist_idx = []

        for step in range(5):
            global_windows = self.kf_graph.get_and_update_global_window()
            for cam in global_windows:
                if cam.cam_idx not in exist_idx:
                    cam.to_device(self.device)
                    used_cams.append(cam)
                    exist_idx.append(cam.cam_idx)
            global_gt_imgs = (
                torch.stack([cam.gt_img for cam in global_windows])
                if len(global_windows) > 0
                else []
            )
            global_gt_depths = (
                torch.stack([cam.sparse_depth for cam in global_windows])
                if len(global_windows) > 0
                else []
            )

            iter_window = global_windows
            iter_gt_imgs = global_gt_imgs
            iter_gt_depths = global_gt_depths

            batch_render_pkg = self.gaussians.render_batch(
                iter_window,
                self.use_random_bg,
                external_splats=self.get_stable_external_splats(iter_window),
            )
            batch_render = batch_render_pkg["render"]
            batch_render_depth = batch_render_pkg["depth"].squeeze(-1)

            gt_mask = (
                torch.sum(iter_gt_imgs, dim=-1, keepdim=True) <= 0.0000001
            ).float()  # ignore invalid pixels in gt
            iter_gt_imgs = iter_gt_imgs * (1.0 - gt_mask) + batch_render * gt_mask

            loss_img, l1_values = self.image_loss(batch_render, iter_gt_imgs)
            loss_depth = self.depth_loss(batch_render_depth, iter_gt_depths)
            # loss_gaussians = self.gaussian_loss(self.gaussians)

            est_depths, est_vars = self.depth_cov_estimator.estimate_depth(
                iter_gt_imgs.permute(0, 3, 1, 2),
                batch_render_depth.view(
                    batch_render_depth.shape[0],
                    1,
                    batch_render_depth.shape[1],
                    batch_render_depth.shape[2],
                ),
            )

            est_normal = convert_depth_to_normal(
                est_depths, global_windows[0].get_int_mat()
            )

            render_normal = convert_depth_to_normal(
                batch_render_depth.view(
                    batch_render_depth.shape[0],
                    1,
                    batch_render_depth.shape[1],
                    batch_render_depth.shape[2],
                ),
                global_windows[0].get_int_mat(),
            )

            print(est_depths.shape)
            loss_normal = (
                (1.0 - torch.abs(est_normal * render_normal).sum(1)) * (est_vars < 0.01)
            ).mean()
            loss_depth = (
                (est_depths.squeeze(-1) - batch_render_depth) ** 2 * (est_vars < 0.01)
            ).mean()

            if step == 0:
                cur_idx = self.cur_view.cam_idx

                saveTensorAsEXR(
                    est_depths[0],
                    pjoin("tmp/visualize_normals", "est_depth_{}.exr".format(cur_idx)),
                )
                saveTensorAsEXR(
                    est_vars[0],
                    pjoin("tmp/visualize_normals", "est_var_{}.exr".format(cur_idx)),
                )
                saveTensorAsEXR(
                    est_normal[0].permute(1, 2, 0),
                    pjoin("tmp/visualize_normals", "est_normal_{}.exr".format(cur_idx)),
                )
                saveTensorAsEXR(
                    render_normal[0].permute(1, 2, 0),
                    pjoin(
                        "tmp/visualize_normals", "render_normal_{}.exr".format(cur_idx)
                    ),
                )
                #     saveTensorAsEXR(iter_gt_imgs[0], pjoin("tmp/visualize_normals", "gt_{}.exr".format(cur_idx)))
                # saveTensorAsEXR(batch_render[0], pjoin("tmp/visualize_normals", "render_{}.exr".format(cur_idx)))
                saveTensorAsEXR(
                    batch_render_depth[0],
                    pjoin(
                        "tmp/visualize_normals", "render_depth_{}.exr".format(cur_idx)
                    ),
                )

            Log(
                "Additional optimization: loss_normal: {}, loss_depth: {}, loss_img: {}".format(
                    loss_normal, loss_depth, loss_img
                ),
                tag="SceneMapper",
            )

            loss = loss_img + loss_normal + loss_depth

            loss.backward()

            self.gaussians.precondition_frontview_mean_gradients(
                iter_window,
                self.frontview_observability_config,
                update_evidence=(
                    step
                    % int(
                        self.frontview_observability_config[
                            "evidence_update_interval"
                        ]
                    )
                    == 0
                ),
            )

            self.gaussians.update()  # Update gaussians, will zero the gradients inside this function

        for cam in used_cams:
            cam.to_device("cpu")

    def run(self):  # Main process of scene mapper
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        now = torch.cuda.Event(enable_timing=True)

        infos = {
            "fps": 0,
            "Num gaussians": 0,
        }

        tracked_info = {
            "scenes": {
                "scene_exposure_gain": self.gaussians.scene_exposure_gain,
                "use_anti_aliasing": self.gaussians.use_anti_aliasing,
                "sh_degree": self.gaussians.max_sh_degree,
                "radius_clip": self.gaussians.radius_clip,
            },
            "cameras": [],
        }

        while True:
            # cur_view_render_pkg = None  # Removed unused variable

            is_last_frame, is_key_frame, get_frame_time, frame_process_time = (
                self.update_kf()
            )

            if not is_last_frame:
                self.opt_log["poses_pair"][self.cur_view.cam_idx] = (
                    self.cur_view.get_raw_pose().detach().cpu().numpy(),
                    self.cur_view.get_pose().detach().cpu().numpy(),
                    is_key_frame,
                )

            if self.use_multi_reso:
                LEVEL_VALUE = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            else:
                LEVEL_VALUE = 0

            start.record()
            if is_key_frame:
                self.optimize(
                    is_last_frame=is_last_frame,
                    is_key_frame=is_key_frame,
                    level=LEVEL_VALUE,
                )

            if self.progressive_manager is not None and is_key_frame:
                self.progressive_manager.constrain_active_surface_scales(
                    self.cur_view
                )

            # uncomment this part if going to save online recon results
            # self.optimize(is_last_frame=is_last_frame, is_key_frame=is_key_frame, level=0, steps=1)

            end.record()
            torch.cuda.synchronize()
            map_time = start.elapsed_time(end) / 1000.0
            if self.progressive_manager is not None:
                self.progressive_runtime["main_optimization_seconds"] += map_time

            if is_last_frame:
                break

            # after initialization, we add gaussians after the optimization
            # if self.initialization_frames < 0:
            start.record()
            baseline_end = torch.cuda.Event(enable_timing=True)
            stable_end = torch.cuda.Event(enable_timing=True)
            process_end = torch.cuda.Event(enable_timing=True)
            newborn_end = torch.cuda.Event(enable_timing=True)
            debug_end = torch.cuda.Event(enable_timing=True)
            baseline_render_pkg = None
            use_baseline_densification = (
                self.progressive_manager is None
                or self.progressive_manager.should_use_baseline_densification(
                    self.processed_frames
                )
            )
            if use_baseline_densification and is_key_frame:
                baseline_render_pkg = self.densification(
                    level=LEVEL_VALUE, is_key_frame=True
                )
                self.refine_frontview_coverage_newborns()
            elif use_baseline_densification:
                self.densification_pts_only()
                self.refine_frontview_coverage_tracking()
            self.gaussians.observe_frontview_directional_layer(self.cur_view)
            baseline_end.record()
            if self.aerocommit_manager is not None:
                newborn_iters = int(
                    self.aerocommit_config["commit_refinement"][
                        "newborn_optimization_iters"
                    ]
                )
                newborn_multiview = bool(
                    self.aerocommit_config["commit_refinement"][
                        "newborn_multiview"
                    ]
                )
                if (
                    newborn_iters > 0
                    and self.last_aerocommit_stats is not None
                    and self.last_aerocommit_stats.num_committed_gaussians > 0
                    and (is_key_frame or not newborn_multiview)
                ):
                    newborn_start = time.perf_counter()
                    self.optimize(
                        is_last_frame=False,
                        is_key_frame=False,
                        level=0,
                        steps=newborn_iters,
                        current_view_only=not newborn_multiview,
                        optimize_pose=False,
                        supplemental=True,
                    )
                    self.aerocommit_runtime["newborn_optimization_seconds"] += (
                        time.perf_counter() - newborn_start
                    )
                    self.aerocommit_runtime["newborn_optimization_steps"] += (
                        newborn_iters
                    )
                    self.cur_view.to_device(self.device)
            if (
                self.progressive_manager is not None
                and self.progressive_manager.should_process_progressive_frame(
                    self.processed_frames, is_key_frame
                )
            ):
                self.cur_view.to_device(self.device)
                with torch.no_grad():
                    stable_render_pkg = baseline_render_pkg
                    if stable_render_pkg is None or LEVEL_VALUE != 0:
                        stable_render_pkg = self.gaussians.render(
                            self.cur_view,
                            level=0,
                            external_splats=self.get_stable_external_splats(
                                self.cur_view
                            ),
                        )
                    stable_render_pkg["diff"] = torch.mean(
                        torch.abs(
                            self.cur_view.get_gt_image(0)
                            - stable_render_pkg["render"]
                        ),
                        dim=-1,
                    )
                stable_end.record()
                progressive_stats = self.progressive_manager.process_frame(
                    self.cur_view, stable_render_pkg, is_key_frame
                )
                process_end.record()
                newborn_steps = int(
                    self.progressive_config["newborn_optimization_iters"]
                )
                num_new_persistent_roots = (
                    progressive_stats.num_promoted_P_to_M
                    - progressive_stats.num_merged_into_existing_M
                    + progressive_stats.num_refined_M_to_S
                    + progressive_stats.num_reactivated_A
                )
                if newborn_steps > 0 and (
                    num_new_persistent_roots > 0 or use_baseline_densification
                ):
                    self.optimize(
                        is_last_frame=False,
                        is_key_frame=False,
                        level=0,
                        steps=newborn_steps,
                        current_view_only=True,
                    )
                    self.progressive_runtime["newborn_optimization_steps"] += newborn_steps
                    self.cur_view.to_device(self.device)
                newborn_end.record()
                proxy_render = None
                if (
                    self.progressive_config["debug"]
                    and self.cur_view.cam_idx
                    % self.progressive_config["debug_save_interval"]
                    == 0
                ):
                    with torch.no_grad():
                        visualization_splats = (
                            self.progressive_manager.visualization_external_splats(
                                torch.device(self.device),
                                torch.float32,
                                cameras=self.cur_view,
                            )
                        )
                        proxy_render = self.gaussians.render(
                            self.cur_view,
                            level=0,
                            external_splats=visualization_splats,
                        )["render"]
                self.progressive_manager.save_debug_frame(
                    self.cur_view, proxy_render=proxy_render
                )
                debug_end.record()
                Log(
                    "Progressive P/M/S/A: {}/{}/{}/{} | promoted {} | refined {} | archived {} | visible/optimized/frozen roots {}/{}/{}".format(
                        progressive_stats.num_active_P,
                        progressive_stats.num_active_M,
                        progressive_stats.num_active_S,
                        progressive_stats.num_active_A,
                        progressive_stats.num_promoted_P_to_M,
                        progressive_stats.num_refined_M_to_S,
                        progressive_stats.num_archived_S_to_A,
                        progressive_stats.num_visible_roots,
                        progressive_stats.num_optimized_roots,
                        progressive_stats.num_frozen_roots,
                    ),
                    tag="ProgressiveMapping",
                )
            elif use_baseline_densification and self.progressive_manager is not None:
                self.progressive_manager.record_bootstrap_frame(self.cur_view.cam_idx)
                stable_end.record()
                process_end.record()
                newborn_steps = int(
                    self.progressive_config["newborn_optimization_iters"]
                )
                if newborn_steps > 0:
                    self.optimize(
                        is_last_frame=False,
                        is_key_frame=False,
                        level=0,
                        steps=newborn_steps,
                        current_view_only=True,
                    )
                    self.progressive_runtime[
                        "newborn_optimization_steps"
                    ] += newborn_steps
                    self.cur_view.to_device(self.device)
                newborn_end.record()
                debug_end.record()
            else:
                stable_end.record()
                process_end.record()
                newborn_end.record()
                debug_end.record()
            end.record()
            torch.cuda.synchronize()
            densification_time = start.elapsed_time(end) / 1000.0
            if self.progressive_manager is not None:
                stage_seconds = {
                    "baseline_densification_seconds": start.elapsed_time(baseline_end) / 1000.0,
                    "stable_render_seconds": baseline_end.elapsed_time(stable_end) / 1000.0,
                    "progressive_processing_seconds": stable_end.elapsed_time(process_end) / 1000.0,
                    "newborn_optimization_seconds": process_end.elapsed_time(newborn_end) / 1000.0,
                    "debug_render_seconds": newborn_end.elapsed_time(debug_end) / 1000.0,
                }
                for name, seconds in stage_seconds.items():
                    self.progressive_runtime[name] += seconds

            tracked_info["cameras"].append(
                {
                    "uid": self.cur_view.cam_idx,
                    "name": self.cur_view.name
                    if self.cur_view.name is not None
                    else "",
                    # "pose": self.cur_view.get_pose().detach().cpu().numpy().tolist(), # pose here is not the final pose
                    "raw_pose": self.cur_view.get_raw_pose()
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "fx": self.cur_view.fx,
                    "fy": self.cur_view.fy,
                    "cx": self.cur_view.cx,
                    "cy": self.cur_view.cy,
                    "width": self.cur_view.width,
                    "height": self.cur_view.height,
                    "exposure_gain": self.cur_view.exposure_gain,
                    "near": self.cur_view.near,
                    "far": self.cur_view.far,
                    "timestamp": time.time(),
                    "is_key_frame": is_key_frame,
                }
            )

            start.record()

            if self.processed_frames % self.save_model_interval == 0:
                self.gaussians.save_as_ply(
                    pjoin(
                        self.configs["Results"]["save_dir"],
                        "online",
                        "point_cloud",
                        f"{self.cur_idx:06d}.ply",
                    )
                )

            if not self.pin_kf_gpu:
                self.cur_view.to_device("cpu")

            end.record()
            torch.cuda.synchronize()
            visualization_time = start.elapsed_time(end) / 1000.0
            if self.progressive_manager is not None:
                self.progressive_runtime["visualization_seconds"] += visualization_time
            # Log("Visualization time: ", visualization_time, "s", tag="SceneMapper")
            Log(
                "Time | Get: {:.2f} ms | Process: {:2f} ms | Optimization: {:.2f} ms | Densification: {:.2f} ms | Visualize: {:.2f} ms".format(
                    get_frame_time * 1000,
                    frame_process_time * 1000,
                    map_time * 1000,
                    densification_time * 1000,
                    visualization_time * 1000,
                ),
                tag="SceneMapper",
            )

            now.record()
            torch.cuda.synchronize()
            total_time = self.last_record_time.elapsed_time(now) / 1000.0
            self.last_record_time.record()

            infos["fps"] = 1.0 / total_time
            infos["Num gaussians"] = self.gaussians.get_num_gaussians
            infos["Exposure*Gain"] = self.cur_view.exposure_gain
            infos["Get frame time"] = get_frame_time * 1000
            infos["Frame process time"] = frame_process_time * 1000
            infos["Optimization time"] = map_time * 1000
            infos["Densification time"] = densification_time * 1000
            infos["Visualization time"] = visualization_time * 1000
            infos["GPU Memory Usage"] = "{:03f} GB".format(
                torch.cuda.memory_allocated(self.device) / (1024**3)
            )

        # Post Refinement
        start.record()
        self.post_refinement()
        if self.progressive_manager is not None:
            self.progressive_manager.constrain_active_surface_scales()
        end.record()
        torch.cuda.synchronize()
        post_refinement_time = start.elapsed_time(end) / 1000.0
        if self.aerocommit_manager is not None:
            self.aerocommit_runtime["post_refinement_seconds"] = post_refinement_time
        if self.progressive_manager is not None:
            self.progressive_runtime["post_refinement_seconds"] = post_refinement_time
        Log("Post Refinement time: ", post_refinement_time, "s", tag="SceneMapper")

        if self.frontview_directional_layer_config["enabled"]:
            self.gaussians.activate_frontview_directional_layer(True)
            self.gaussians.save_frontview_directional_layer(
                pjoin(
                    self.configs["Results"]["save_dir"],
                    DIRECTIONAL_LAYER_FILENAME,
                )
            )

        # Save the final model and other parameters
        self.gaussians.save_as_ply(
            pjoin(self.configs["Results"]["save_dir"], "point_cloud.ply")
        )
        if self.progressive_manager is not None:
            self.progressive_manager.export_full_progressive_map(
                pjoin(
                    self.configs["Results"]["save_dir"],
                    "point_cloud_progressive_full.ply",
                )
            )
        if self.aerocommit_manager is not None:
            self.aerocommit_manager.export_full_map(
                pjoin(
                    self.configs["Results"]["save_dir"],
                    "point_cloud_aerocommit_full.ply",
                )
            )
            aerocommit_summary = self.aerocommit_manager.finalize()
            worldtest_summary = (
                self.worldtest_controller.finalize()
                if self.worldtest_controller is not None
                else None
            )
            if worldtest_summary is not None:
                aerocommit_summary = worldtest_summary
        else:
            aerocommit_summary = None
            worldtest_summary = None

        for i in range(len(tracked_info["cameras"])):
            name = tracked_info["cameras"][i]["name"]
            if name is not None and name in self.opt_log["tracked_poses"]:
                tracked_info["cameras"][i]["pose"] = self.opt_log["tracked_poses"][
                    name
                ].tolist()

            if name is not None and name in self.opt_log["cam_opt_iterations"]:
                tracked_info["cameras"][i]["cam_opt_iterations"] = self.opt_log[
                    "cam_opt_iterations"
                ][name]

            if name is not None and name in self.opt_log["gaussian_opt_iterations"]:
                tracked_info["cameras"][i]["gaussian_opt_iterations"] = self.opt_log[
                    "gaussian_opt_iterations"
                ][name]

        json.dump(
            tracked_info,
            open(
                pjoin(self.configs["Results"]["save_dir"], "tracked_info.json"),
                "w",
                encoding="utf-8",
            ),
            indent=4,
        )

        json.dump(
            self.opt_log["l1_err"],
            open(
                pjoin(self.configs["Results"]["save_dir"], "l1_err.json"),
                "w",
                encoding="utf-8",
            ),
            indent=4,
        )

        if self.gaussians.get_vignette_img() is not None:
            saveTensorAsPNG(
                self.gaussians.get_vignette_img()[0],
                pjoin(self.configs["Results"]["save_dir"], "vignette.png"),
            )

        # meta info
        meta_info = {}
        meta_info["num_gaussians"] = self.gaussians.get_num_gaussians
        meta_info["num_keyframes"] = self.kf_graph.get_kf_num
        meta_info["num_processed_frames"] = self.processed_frames
        meta_info["kf_ids"] = self.kf_graph.get_kf_cam_idx
        if self.progressive_manager is not None:
            meta_info["progressive_runtime"] = self.progressive_runtime
        if aerocommit_summary is not None:
            meta_info["aerocommit_summary"] = aerocommit_summary
            meta_info["aerocommit_runtime"] = self.aerocommit_runtime
            meta_info["num_gaussians_full"] = (
                self.gaussians.get_num_gaussians
                + self.aerocommit_manager.archive_store.gaussian_count
            )
        if worldtest_summary is not None:
            meta_info["worldtest_summary"] = worldtest_summary
        if self.appearance_lod_stats is not None:
            meta_info["appearance_lod"] = self.appearance_lod_stats
        if self.frontview_observability_config["enabled"]:
            meta_info["frontview_observability"] = (
                self.gaussians.frontview_observability_summary()
            )
        if self.frontview_coverage_recovery_config["enabled"]:
            meta_info["frontview_coverage_recovery"] = (
                self.frontview_coverage_recovery_summary()
            )
        if self.frontview_sampling_config["enabled"]:
            meta_info["frontview_sampling"] = (
                self.gaussians.frontview_sampling_summary()
            )
        if self.frontview_depth_transport_config["enabled"]:
            meta_info["frontview_depth_transport"] = (
                self.gaussians.frontview_depth_transport_summary()
            )
        if self.frontview_birth_config["enabled"]:
            meta_info["frontview_birth"] = self.gaussians.frontview_birth_summary()
        if self.frontview_far_field_config["enabled"]:
            meta_info["frontview_far_field"] = (
                self.gaussians.frontview_far_field_summary()
            )
        if self.frontview_directional_layer_config["enabled"]:
            meta_info["frontview_directional_layer"] = (
                self.gaussians.frontview_directional_layer_summary()
            )
        if self.frontview_identity_lod_config["enabled"]:
            meta_info["frontview_identity_lod"] = (
                self.gaussians.frontview_identity_lod_summary()
            )
        if self.frontview_residual_cover_config["enabled"]:
            meta_info["frontview_residual_cover"] = (
                self.gaussians.frontview_residual_cover_summary()
            )
        if self.frontview_scale_cover_config["enabled"]:
            meta_info["frontview_scale_cover"] = (
                self.gaussians.frontview_scale_cover_summary()
            )
        if self.streaming_appearance_lod_config["enabled"]:
            meta_info["streaming_appearance_lod"] = (
                self.gaussians.streaming_appearance_lod_summary()
            )
        if self.frontview_track_fusion_config["enabled"]:
            meta_info["frontview_track_fusion"] = (
                self.gaussians.frontview_track_fusion_summary()
            )
        if self.frontview_sparse_scale_map_config["enabled"]:
            meta_info["frontview_sparse_scale_map"] = (
                self.gaussians.frontview_sparse_scale_map_summary()
            )
        if self.kf_graph.get_next_camera_method.startswith("frontview_"):
            meta_info["frontview_replay"] = self.kf_graph.frontview_replay_summary()
        return meta_info, self.opt_log
