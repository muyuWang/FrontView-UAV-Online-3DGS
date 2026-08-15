# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from depth_cov.depth_cov_estimator import DepthCovEstimator
from gsplat.rendering import rasterization, rasterization_2dgs
from plyfile import PlyData, PlyElement
from torch import nn
from utils_new.camera_utils import Camera, unproject_pts_tensor
from utils_new.aerocommit.types import CommitResult, GaussianProposalBatch
from utils_new.frontview_observability import (
    parallax_learning_scale,
    matched_events_within_log_depth_regimes,
    posterior_information_scale,
    precondition_raywise_gradient,
    project_raywise_update,
    release_owned_scale_caps,
    resolved_footprint_mask,
    resolution_information_scale,
    shuffle_within_log_depth_regimes,
)
from utils_new.frontview_coverage_recovery import (
    SparseFarDepthPrior,
    apply_sparse_depth_prior_fallback,
    apply_visible_surface_depth_fallback,
    motion_conditioned_depth_floor,
    posterior_inverse_depth_fusion,
    residual_grid_indices,
    validate_front_view_coverage_recovery_config,
)
from utils_new.frontview_causal_metric_birth import (
    CausalMetricBirth,
    bind_tracks_to_proxy_slots,
    causal_birth_replaces_depth_fallback,
    fuse_candidate_log_depth_posteriors,
    validate_causal_metric_birth_config,
)
from utils_new.frontview_track_depth_gauge import cross_fitted_track_depth_gauge
from utils_new.frontview_causal_landmark_memory import (
    CausalPersistentLandmarkMemory,
    information_gain_transport,
    validate_causal_landmark_memory_config,
)
from utils_new.frontview_dual_responsibility import (
    causal_finite_depth_certificates,
    geometry_decision_render,
    nearest_unique_replacement_positions,
    proposal_metric_confidences,
    validate_causal_dual_responsibility_config,
)
from utils_new.frontview_sampling import (
    adaptive_log_depth_indices,
    depth_stratified_indices,
    evidence_balanced_indices,
    projective_coverage_indices,
    rate_distortion_density_weights,
    residual_rate_distortion_radius_factors,
    residual_importance_indices,
)
from utils_new.frontview_depth_transport import (
    split_depth_anchors,
    transport_candidate_depths,
)
from utils_new.frontview_birth import (
    TrackResponsibilityLedger,
    layered_projective_birth_indices,
    layered_scale_expansion_limits,
    multi_layer_projective_occupancy,
    responsibility_initial_opacities,
    temporal_responsibility_rejections,
    validate_front_view_birth_config,
)
from utils_new.frontview_far_field import (
    CausalRayResponsibilityAtlas,
    budgeted_fallback_radius,
    budget_cell_parameters,
    far_field_responsibility_mask,
    matched_responsibility_shuffle,
    observability_footprint_trust_limits,
    projected_gaussian_radii,
    projective_map_redundancy_mask,
    projective_map_posterior_log_odds,
    posterior_budget_refill_mask,
    projective_radial_scale_factors,
    projective_survivor_mask,
    ray_aligned_quaternions,
    validate_front_view_far_field_config,
    visible_parallax_pixels,
)
from utils_new.frontview_inverse_depth_certificate import (
    causal_inverse_depth_posterior,
    validate_front_view_inverse_depth_certificate_config,
)
from utils_new.frontview_projective_structure import (
    budget_normalized_information_radii,
    structure_aligned_covariances,
)
from utils_new.frontview_directional_layer import (
    FrontViewDirectionalLayer,
    validate_front_view_directional_layer_config,
)
from utils_new.frontview_identity_lod import (
    FrontViewIdentityLOD,
    validate_front_view_identity_lod_config,
)
from utils_new.frontview_residual_cover import (
    FrontViewResidualCover,
    validate_front_view_residual_cover_config,
)
from utils_new.frontview_scale_cover import (
    FrontViewScaleCover,
    validate_front_view_scale_cover_config,
)
from utils_new.frontview_sparse_scale_map import (
    FrontViewSparseScaleMap,
    validate_front_view_sparse_scale_map_config,
)
from utils_new.frontview_track_fusion import (
    robust_color_ema,
    validate_front_view_track_fusion_config,
)
from utils_new.streaming_appearance_lod import (
    select_gradient_agreement_promotions,
    select_gradient_promotions,
    select_monotonic_promotions,
    sh_band_bounds,
    validate_streaming_appearance_lod_config,
)
from utils_new.tgbr_sparse_model import (
    decode_tgbr_sparse_sh,
    is_tgbr_sparse_ply,
    write_tgbr_sparse_ply,
)
from utils_new.aerocommit.frequency_sampling import (
    frequency_evidence_map,
    frequency_footprint_log_offset,
    sample_frequency_balanced_indices,
)
from utils_new.aerocommit.sparse_track_geometry import (
    conditional_scale_expansion_limits,
    zbuffer_sparse_tracks,
)
from utils_new.aerocommit.stable_detail_split import split_gaussian_parameters
from utils_new.hash_utils import HashBlock
from utils_new.heterogeneous_sh_rasterizer import heterogeneous_sh_rasterization
from utils_new.logging_utils import Log
from utils_new.tool_utils import BasicPointCloud, inverse_sigmoid, rgb_to_sh_np


def get_expon_lr_func(lr_init, lr_final, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper


class Gaussians:
    def __init__(
        self,
        BS: int = 6,
        scene_scale: float = 1.0,
        init_config=None,
        max_sh_degree: int = 2,
        managed_optimizers: bool = True,
    ):
        _xyz = torch.zeros((1, 3), dtype=torch.float32)
        _sh0 = torch.zeros((1, 1, 3), dtype=torch.float32)
        _shN = torch.zeros((1, (max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32)
        _scaling = torch.log(torch.ones((1, 3), dtype=torch.float32) * 1e-8)
        _rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float()
        _opacity = inverse_sigmoid(
            0.1 * torch.ones((_xyz.shape[0],), dtype=torch.float32)
        )

        self.is_optimize = True
        self.max_sh_degree = max_sh_degree
        self.default_max_scale_expansion = float(
            (init_config or {}).get("max_scale_expansion", 0.0)
        )
        self.default_max_mean_displacement = float("inf")

        self.scene_scale = scene_scale
        self.BS = BS
        self.lr_step = 0

        self.scheduler_args = {}

        self.non_trainable_params = {
            "init_scales": torch.ones((1,), dtype=torch.float32) * 1e-8,
            "max_scale_expansions": torch.full(
                (1,), float("inf"), dtype=torch.float32
            ),
            "footprint_trust_mask": torch.zeros((1,), dtype=torch.bool),
            "footprint_target_scales": torch.full(
                (1,), float("inf"), dtype=torch.float32
            ),
            "footprint_release_scale_expansions": torch.full(
                (1,), float("inf"), dtype=torch.float32
            ),
            "footprint_evidence_pending_mask": torch.zeros(
                (1,), dtype=torch.bool
            ),
            "mean_anchors": _xyz.detach().clone(),
            "max_mean_displacements": torch.full(
                (1,), float("inf"), dtype=torch.float32
            ),
            "directional_observability_mask": torch.zeros((1,), dtype=torch.bool),
            "reference_camera_centers": torch.zeros((1, 3), dtype=torch.float32),
            "reference_rays": torch.zeros((1, 3), dtype=torch.float32),
            "max_parallax_sin2": torch.zeros((1,), dtype=torch.float32),
            "birth_log_depth_stds": torch.zeros((1,), dtype=torch.float32),
            "track_ids": torch.full((1,), -1, dtype=torch.long),
            "gaussian_uids": torch.full((1,), -1, dtype=torch.long),
            "birth_frame_ids": torch.full((1,), -1, dtype=torch.long),
            "metric_confidences": torch.ones((1,), dtype=torch.float32),
            "uncertainty_confidences": torch.ones((1,), dtype=torch.float32),
            "gradient_utility_ema": torch.zeros((1,), dtype=torch.float32),
            "appearance_view_count": torch.zeros((1,), dtype=torch.int32),
            "appearance_radius_sum": torch.zeros((1,), dtype=torch.float32),
            "appearance_direction_sum": torch.zeros((1, 3), dtype=torch.float32),
            "appearance_band_utility_ema": torch.zeros((1,), dtype=torch.float32),
            "appearance_sh_degree": torch.zeros((1,), dtype=torch.uint8),
        }

        if init_config is not None:
            self.raw_lr = {
                "means": init_config["means_lr_init"],
                "scales": init_config["scales_lr_init"],
                "quats": init_config["quats_lr_init"],
            }
            self.scheduler_args["means"] = get_expon_lr_func(
                lr_init=init_config["means_lr_init"],
                lr_final=init_config["means_lr_final"],
                max_steps=init_config["lr_final_step"],
            )
            self.scheduler_args["scales"] = get_expon_lr_func(
                lr_init=init_config["scales_lr_init"],
                lr_final=init_config["scales_lr_final"],
                max_steps=init_config["lr_final_step"],
            )
            self.scheduler_args["quats"] = get_expon_lr_func(
                lr_init=init_config["quats_lr_init"],
                lr_final=init_config["quats_lr_final"],
                max_steps=init_config["lr_final_step"],
            )

            self.max_steps = init_config["lr_final_step"]
            params = [
                # name, value, lr
                (
                    "means",
                    torch.nn.Parameter(_xyz),
                    init_config["means_lr_init"] * scene_scale,
                ),
                ("scales", torch.nn.Parameter(_scaling), init_config["scales_lr_init"]),
                ("quats", torch.nn.Parameter(_rotation), init_config["quats_lr_init"]),
                (
                    "opacities",
                    torch.nn.Parameter(_opacity),
                    init_config["opacities_lr"],
                ),
                ("sh0", torch.nn.Parameter(_sh0), init_config["sh_lr"]),
                ("shN", torch.nn.Parameter(_shN), init_config["sh_lr"] / 20),
            ]
        else:
            self.raw_lr = {
                "means": 1.6e-4,
                "scales": 0.005,
                "quats": 0.001,
            }
            self.scheduler_args["means"] = get_expon_lr_func(
                lr_init=1.6e-4, lr_final=1.6e-4, max_steps=1
            )
            self.scheduler_args["scales"] = get_expon_lr_func(
                lr_init=0.005, lr_final=0.005, max_steps=1
            )
            self.scheduler_args["quats"] = get_expon_lr_func(
                lr_init=0.001, lr_final=0.001, max_steps=1
            )
            self.max_steps = 1
            params = [
                # name, value, lr
                ("means", torch.nn.Parameter(_xyz), 1.6e-4 * scene_scale),
                ("scales", torch.nn.Parameter(_scaling), 5e-3),
                ("quats", torch.nn.Parameter(_rotation), 1e-3),
                ("opacities", torch.nn.Parameter(_opacity), 5e-2),
                ("sh0", torch.nn.Parameter(_sh0), 2.5e-3),
                ("shN", torch.nn.Parameter(_shN), 2.5e-3 / 20),
            ]

        self.splats = torch.nn.ParameterDict({n: v for n, v, _ in params})

        self.optimizers = (
            {
                name: torch.optim.Adam(
                    [{"params": self.splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                    eps=1e-15 / math.sqrt(BS),
                    # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
                    betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                )
                for name, _, lr in params
            }
            if managed_optimizers
            else {}
        )

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.device = "cpu"

    @property
    def get_scaling(self):
        return self.scaling_activation(self.splats["scales"])

    @property
    def get_rotation(self):
        return self.splats[
            "quats"
        ]  # gsplats will normalize the quat inside the rasterizer

    @property
    def get_xyz(self):
        return self.splats["means"]

    @property
    def get_features(self):
        return torch.cat((self.splats["sh0"], self.splats["shN"]), dim=1)  # [N, K, 3]

    @property
    def get_opacity(self):
        return self.opacity_activation(self.splats["opacities"])

    @property
    def get_num(self):
        return int(self.splats["means"].shape[0])

    @property
    def get_init_scales(self):
        return self.non_trainable_params["init_scales"]

    @property
    def get_params(self):
        return self.splats

    def reset_optimizer(self, config, BS):
        self.max_steps = config["max_steps"]
        self.lr_step = 0

        self.scheduler_args = {}

        self.raw_lr = {
            "means": config["means_lr_init"],
            "scales": config["scales_lr_init"],
        }
        self.scheduler_args["means"] = get_expon_lr_func(
            lr_init=config["means_lr_init"],
            lr_final=config["means_lr_final"],
            max_steps=config["max_steps"],
        )
        self.scheduler_args["scales"] = get_expon_lr_func(
            lr_init=config["scales_lr_init"],
            lr_final=config["scales_lr_final"],
            max_steps=config["max_steps"],
        )

        params = [
            # name, value, lr
            ("means", config["means_lr_init"] * self.scene_scale),
            ("scales", config["scales_lr_init"]),
            ("quats", config["quats_lr"]),
            ("opacities", config["opacities_lr"]),
            ("sh0", config["sh_lr"]),
            ("shN", config["sh_lr"] / 20),
        ]

        self.optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, lr in params
        }

    def update_opt_lr(self, new_lr, data_type):  # update lr in optimizer
        self.optimizers[data_type].param_groups[0]["lr"] = new_lr

    def update_lr(self, new_lr, data_type):  # update lr
        self.raw_lr[data_type] = new_lr
        scene_scale = self.scene_scale if data_type != "quats" else 1.0

        self.update_opt_lr(new_lr * scene_scale * math.sqrt(self.BS), data_type)

    def update_scene_scale(self, new_scale):
        self.scene_scale = new_scale

        self.update_opt_lr(
            self.raw_lr["means"] * new_scale * math.sqrt(self.BS), "means"
        )
        self.update_opt_lr(self.raw_lr["scales"] * math.sqrt(self.BS), "scales")

    def step_lr(self):
        self.lr_step += 1

        for data_type in self.scheduler_args:
            new_lr = self.scheduler_args[data_type](self.lr_step)
            self.update_lr(new_lr, data_type)

        return self.lr_step < self.max_steps

    def get_pos_lr(self):
        return self.optimizers["means"].param_groups[0]["lr"]

    def clean(self):
        self.splats = None
        self.optimizers = None
        self.is_optimize = False

    def to_device(self, device):
        self.splats = self.splats.to(device)
        for name in self.non_trainable_params:
            self.non_trainable_params[name] = self.non_trainable_params[name].to(device)
        self.device = device

    def disable_grad(self):
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = False
        self.splats.requires_grad_(False)
        self.is_optimize = False

    def enable_grad(self):
        for optimizer in self.optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["params"][0].requires_grad = True
        self.splats.requires_grad_(True)
        self.is_optimize = True

    @torch.no_grad()
    def cat_tensors_to_optimizer(self, new_params):
        optimizable_tensors = {}
        for name, optimizer in self.optimizers.items():
            assert len(optimizer.param_groups) == 1
            group = optimizer.param_groups[0]
            assert len(group["params"]) == 1
            extension_tensor = new_params[name]  # newly added tensor
            p = group["params"][0]
            stored_state = optimizer.state.get(p, None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del optimizer.state[p]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors

    @torch.no_grad()
    def extend_gaussians(self, new_params):
        new_optimized_tensors = self.cat_tensors_to_optimizer(new_params)
        self.splats["means"] = new_optimized_tensors["means"]
        self.splats["scales"] = new_optimized_tensors["scales"]
        self.splats["quats"] = new_optimized_tensors["quats"]
        self.splats["opacities"] = new_optimized_tensors["opacities"]
        self.splats["sh0"] = new_optimized_tensors["sh0"]
        self.splats["shN"] = new_optimized_tensors["shN"]

        self.non_trainable_params["init_scales"] = torch.cat(
            [self.non_trainable_params["init_scales"], new_params["init_scales"]], dim=0
        )
        self.non_trainable_params["max_scale_expansions"] = torch.cat(
            [
                self.non_trainable_params["max_scale_expansions"],
                new_params["max_scale_expansions"],
            ],
            dim=0,
        )
        footprint_mask = new_params.get("footprint_trust_mask")
        if footprint_mask is None:
            footprint_mask = torch.zeros(
                (new_params["means"].shape[0],),
                device=new_params["means"].device,
                dtype=torch.bool,
            )
        footprint_targets = new_params.get("footprint_target_scales")
        if footprint_targets is None:
            footprint_targets = torch.full(
                (new_params["means"].shape[0],),
                float("inf"),
                device=new_params["means"].device,
                dtype=torch.float32,
            )
        self.non_trainable_params["footprint_trust_mask"] = torch.cat(
            (
                self.non_trainable_params["footprint_trust_mask"],
                footprint_mask.detach().to(self.device, dtype=torch.bool),
            )
        )
        self.non_trainable_params["footprint_target_scales"] = torch.cat(
            (
                self.non_trainable_params["footprint_target_scales"],
                footprint_targets.detach().to(self.device, dtype=torch.float32),
            )
        )
        footprint_release_limits = new_params.get(
            "footprint_release_scale_expansions"
        )
        if footprint_release_limits is None:
            footprint_release_limits = torch.full(
                (new_params["means"].shape[0],),
                float("inf"),
                device=new_params["means"].device,
                dtype=torch.float32,
            )
        self.non_trainable_params["footprint_release_scale_expansions"] = (
            torch.cat(
                (
                    self.non_trainable_params[
                        "footprint_release_scale_expansions"
                    ],
                    footprint_release_limits.detach().to(
                        self.device, dtype=torch.float32
                    ),
                )
            )
        )
        footprint_pending = new_params.get("footprint_evidence_pending_mask")
        if footprint_pending is None:
            footprint_pending = footprint_mask
        self.non_trainable_params["footprint_evidence_pending_mask"] = torch.cat(
            (
                self.non_trainable_params["footprint_evidence_pending_mask"],
                footprint_pending.detach().to(self.device, dtype=torch.bool),
            )
        )
        mean_anchors = new_params.get("mean_anchors", new_params["means"])
        max_mean_displacements = new_params.get("max_mean_displacements")
        if max_mean_displacements is None:
            max_mean_displacements = torch.full(
                (new_params["means"].shape[0],),
                self.default_max_mean_displacement,
                device=new_params["means"].device,
                dtype=torch.float32,
            )
        self.non_trainable_params["mean_anchors"] = torch.cat(
            [
                self.non_trainable_params["mean_anchors"],
                mean_anchors.detach().to(self.device, dtype=torch.float32),
            ],
            dim=0,
        )
        self.non_trainable_params["max_mean_displacements"] = torch.cat(
            [
                self.non_trainable_params["max_mean_displacements"],
                max_mean_displacements.detach().to(self.device, dtype=torch.float32),
            ],
            dim=0,
        )
        directional_mask = new_params.get("directional_observability_mask")
        if directional_mask is None:
            directional_mask = torch.zeros(
                (new_params["means"].shape[0],),
                device=new_params["means"].device,
                dtype=torch.bool,
            )
        reference_centers = new_params.get("reference_camera_centers")
        if reference_centers is None:
            reference_centers = torch.zeros_like(new_params["means"])
        reference_rays = new_params.get("reference_rays")
        if reference_rays is None:
            reference_rays = torch.zeros_like(new_params["means"])
        max_parallax_sin2 = new_params.get("max_parallax_sin2")
        if max_parallax_sin2 is None:
            max_parallax_sin2 = torch.zeros(
                (new_params["means"].shape[0],),
                device=new_params["means"].device,
                dtype=torch.float32,
            )
        birth_log_depth_stds = new_params.get("birth_log_depth_stds")
        if birth_log_depth_stds is None:
            birth_log_depth_stds = torch.zeros(
                (new_params["means"].shape[0],),
                device=new_params["means"].device,
                dtype=torch.float32,
            )
        self.non_trainable_params["directional_observability_mask"] = torch.cat(
            (
                self.non_trainable_params["directional_observability_mask"],
                directional_mask.detach().to(self.device, dtype=torch.bool),
            )
        )
        self.non_trainable_params["reference_camera_centers"] = torch.cat(
            (
                self.non_trainable_params["reference_camera_centers"],
                reference_centers.detach().to(self.device, dtype=torch.float32),
            )
        )
        self.non_trainable_params["reference_rays"] = torch.cat(
            (
                self.non_trainable_params["reference_rays"],
                reference_rays.detach().to(self.device, dtype=torch.float32),
            )
        )
        self.non_trainable_params["max_parallax_sin2"] = torch.cat(
            (
                self.non_trainable_params["max_parallax_sin2"],
                max_parallax_sin2.detach().to(self.device, dtype=torch.float32),
            )
        )
        self.non_trainable_params["birth_log_depth_stds"] = torch.cat(
            (
                self.non_trainable_params["birth_log_depth_stds"],
                birth_log_depth_stds.detach().to(self.device, dtype=torch.float32),
            )
        )
        track_ids = new_params.get("track_ids")
        if track_ids is None:
            track_ids = torch.full(
                (new_params["means"].shape[0],),
                -1,
                device=new_params["means"].device,
                dtype=torch.long,
            )
        self.non_trainable_params["track_ids"] = torch.cat(
            (
                self.non_trainable_params["track_ids"],
                track_ids.detach().to(self.device, dtype=torch.long),
            )
        )
        gaussian_uids = new_params.get("gaussian_uids")
        if gaussian_uids is None:
            gaussian_uids = torch.full(
                (new_params["means"].shape[0],),
                -1,
                device=new_params["means"].device,
                dtype=torch.long,
            )
        self.non_trainable_params["gaussian_uids"] = torch.cat(
            (
                self.non_trainable_params["gaussian_uids"],
                gaussian_uids.detach().to(self.device, dtype=torch.long),
            )
        )
        birth_frame_ids = new_params.get("birth_frame_ids")
        if birth_frame_ids is None:
            birth_frame_ids = torch.full(
                (new_params["means"].shape[0],),
                -1,
                device=new_params["means"].device,
                dtype=torch.long,
            )
        self.non_trainable_params["birth_frame_ids"] = torch.cat(
            (
                self.non_trainable_params["birth_frame_ids"],
                birth_frame_ids.detach().to(self.device, dtype=torch.long),
            )
        )
        metric_confidences = new_params.get("metric_confidences")
        if metric_confidences is None:
            metric_confidences = torch.ones(
                (new_params["means"].shape[0],),
                device=new_params["means"].device,
                dtype=torch.float32,
            )
        self.non_trainable_params["metric_confidences"] = torch.cat(
            (
                self.non_trainable_params["metric_confidences"],
                metric_confidences.detach().to(self.device, dtype=torch.float32),
            )
        )
        uncertainty_confidences = new_params.get("uncertainty_confidences")
        if uncertainty_confidences is None:
            uncertainty_confidences = metric_confidences
        self.non_trainable_params["uncertainty_confidences"] = torch.cat(
            (
                self.non_trainable_params["uncertainty_confidences"],
                uncertainty_confidences.detach().to(
                    self.device, dtype=torch.float32
                ),
            )
        )
        gradient_utility_ema = new_params.get("gradient_utility_ema")
        if gradient_utility_ema is None:
            gradient_utility_ema = torch.zeros(
                (new_params["means"].shape[0],),
                device=new_params["means"].device,
                dtype=torch.float32,
            )
        self.non_trainable_params["gradient_utility_ema"] = torch.cat(
            (
                self.non_trainable_params["gradient_utility_ema"],
                gradient_utility_ema.detach().to(self.device, dtype=torch.float32),
            )
        )
        count = new_params["means"].shape[0]
        self.non_trainable_params["appearance_view_count"] = torch.cat(
            (
                self.non_trainable_params["appearance_view_count"],
                torch.zeros(count, device=self.device, dtype=torch.int32),
            )
        )
        self.non_trainable_params["appearance_radius_sum"] = torch.cat(
            (
                self.non_trainable_params["appearance_radius_sum"],
                torch.zeros(count, device=self.device, dtype=torch.float32),
            )
        )
        self.non_trainable_params["appearance_direction_sum"] = torch.cat(
            (
                self.non_trainable_params["appearance_direction_sum"],
                torch.zeros((count, 3), device=self.device, dtype=torch.float32),
            )
        )
        self.non_trainable_params["appearance_band_utility_ema"] = torch.cat(
            (
                self.non_trainable_params["appearance_band_utility_ema"],
                torch.zeros(count, device=self.device, dtype=torch.float32),
            )
        )
        if "appearance_band_gradient_ema" in self.non_trainable_params:
            gradient_ema = self.non_trainable_params[
                "appearance_band_gradient_ema"
            ]
            width = gradient_ema.shape[1]
            self.non_trainable_params["appearance_band_gradient_ema"] = torch.cat(
                (
                    gradient_ema,
                    torch.zeros(
                        (count, width), device=self.device, dtype=gradient_ema.dtype
                    ),
                )
            )
        self.non_trainable_params["appearance_sh_degree"] = torch.cat(
            (
                self.non_trainable_params["appearance_sh_degree"],
                torch.zeros(count, device=self.device, dtype=torch.uint8),
            )
        )

    @torch.no_grad()
    def replace_gaussians(self, params):
        """Replace the full group and reset Adam state for a stable group identity."""
        required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
        if set(params) < required:
            raise ValueError("Missing Gaussian parameters: {}".format(sorted(required - set(params))))
        count = params["means"].shape[0]
        if any(params[name].shape[0] != count for name in required):
            raise ValueError("All Gaussian parameter tensors must have the same row count")
        if self.optimizers:
            for name, optimizer in self.optimizers.items():
                value = params[name].detach().to(self.device, dtype=torch.float32).contiguous()
                new_parameter = nn.Parameter(value.requires_grad_(True))
                optimizer.state.clear()
                optimizer.param_groups[0]["params"] = [new_parameter]
                self.splats[name] = new_parameter
        else:
            self.splats = nn.ParameterDict(
                {
                    name: nn.Parameter(
                        params[name]
                        .detach()
                        .to(self.device, dtype=torch.float32)
                        .contiguous()
                        .requires_grad_(True)
                    )
                    for name in required
                }
            )
        self.non_trainable_params["init_scales"] = torch.exp(
            self.splats["scales"].detach()
        ).mean(dim=1)
        expansion = params.get("max_scale_expansions")
        if expansion is None:
            expansion = torch.full(
                (count,), float("inf"), device=self.device, dtype=torch.float32
            )
        self.non_trainable_params["max_scale_expansions"] = expansion.detach().to(
            self.device, dtype=torch.float32
        )
        footprint_mask = params.get("footprint_trust_mask")
        if footprint_mask is None:
            footprint_mask = torch.zeros(count, device=self.device, dtype=torch.bool)
        footprint_targets = params.get("footprint_target_scales")
        if footprint_targets is None:
            footprint_targets = torch.full(
                (count,), float("inf"), device=self.device, dtype=torch.float32
            )
        self.non_trainable_params["footprint_trust_mask"] = footprint_mask.detach().to(
            self.device, dtype=torch.bool
        )
        self.non_trainable_params["footprint_target_scales"] = (
            footprint_targets.detach().to(self.device, dtype=torch.float32)
        )
        footprint_release_limits = params.get(
            "footprint_release_scale_expansions"
        )
        if footprint_release_limits is None:
            footprint_release_limits = torch.full(
                (count,), float("inf"), device=self.device, dtype=torch.float32
            )
        self.non_trainable_params["footprint_release_scale_expansions"] = (
            footprint_release_limits.detach().to(
                self.device, dtype=torch.float32
            )
        )
        footprint_pending = params.get("footprint_evidence_pending_mask")
        if footprint_pending is None:
            footprint_pending = footprint_mask
        self.non_trainable_params["footprint_evidence_pending_mask"] = (
            footprint_pending.detach().to(self.device, dtype=torch.bool)
        )
        mean_anchors = params.get("mean_anchors", self.splats["means"].detach())
        self.non_trainable_params["mean_anchors"] = mean_anchors.detach().to(
            self.device, dtype=torch.float32
        )
        displacement = params.get("max_mean_displacements")
        if displacement is None:
            displacement = torch.full(
                (count,),
                self.default_max_mean_displacement,
                device=self.device,
                dtype=torch.float32,
            )
        self.non_trainable_params["max_mean_displacements"] = displacement.detach().to(
            self.device, dtype=torch.float32
        )
        directional_mask = params.get("directional_observability_mask")
        if directional_mask is None:
            directional_mask = torch.zeros(count, device=self.device, dtype=torch.bool)
        reference_centers = params.get("reference_camera_centers")
        if reference_centers is None:
            reference_centers = torch.zeros_like(self.splats["means"])
        reference_rays = params.get("reference_rays")
        if reference_rays is None:
            reference_rays = torch.zeros_like(self.splats["means"])
        max_parallax_sin2 = params.get("max_parallax_sin2")
        if max_parallax_sin2 is None:
            max_parallax_sin2 = torch.zeros(
                count, device=self.device, dtype=torch.float32
            )
        birth_log_depth_stds = params.get("birth_log_depth_stds")
        if birth_log_depth_stds is None:
            birth_log_depth_stds = torch.zeros(
                count, device=self.device, dtype=torch.float32
            )
        self.non_trainable_params["directional_observability_mask"] = (
            directional_mask.detach().to(self.device, dtype=torch.bool)
        )
        self.non_trainable_params["reference_camera_centers"] = (
            reference_centers.detach().to(self.device, dtype=torch.float32)
        )
        self.non_trainable_params["reference_rays"] = reference_rays.detach().to(
            self.device, dtype=torch.float32
        )
        self.non_trainable_params["max_parallax_sin2"] = (
            max_parallax_sin2.detach().to(self.device, dtype=torch.float32)
        )
        self.non_trainable_params["birth_log_depth_stds"] = (
            birth_log_depth_stds.detach().to(self.device, dtype=torch.float32)
        )
        track_ids = params.get("track_ids")
        if track_ids is None:
            track_ids = torch.full(
                (count,), -1, device=self.device, dtype=torch.long
            )
        self.non_trainable_params["track_ids"] = track_ids.detach().to(
            self.device, dtype=torch.long
        )
        gaussian_uids = params.get("gaussian_uids")
        if gaussian_uids is None:
            gaussian_uids = torch.full(
                (count,), -1, device=self.device, dtype=torch.long
            )
        self.non_trainable_params["gaussian_uids"] = gaussian_uids.detach().to(
            self.device, dtype=torch.long
        )
        birth_frame_ids = params.get("birth_frame_ids")
        if birth_frame_ids is None:
            birth_frame_ids = torch.full(
                (count,), -1, device=self.device, dtype=torch.long
            )
        self.non_trainable_params["birth_frame_ids"] = birth_frame_ids.detach().to(
            self.device, dtype=torch.long
        )
        metric_confidences = params.get("metric_confidences")
        if metric_confidences is None:
            metric_confidences = torch.ones(
                (count,), device=self.device, dtype=torch.float32
            )
        self.non_trainable_params["metric_confidences"] = (
            metric_confidences.detach().to(self.device, dtype=torch.float32)
        )
        uncertainty_confidences = params.get("uncertainty_confidences")
        if uncertainty_confidences is None:
            uncertainty_confidences = metric_confidences
        self.non_trainable_params["uncertainty_confidences"] = (
            uncertainty_confidences.detach().to(self.device, dtype=torch.float32)
        )
        gradient_utility_ema = params.get("gradient_utility_ema")
        if gradient_utility_ema is None:
            gradient_utility_ema = torch.zeros(
                (count,), device=self.device, dtype=torch.float32
            )
        self.non_trainable_params["gradient_utility_ema"] = (
            gradient_utility_ema.detach().to(self.device, dtype=torch.float32)
        )
        self.non_trainable_params["appearance_view_count"] = torch.zeros(
            count, device=self.device, dtype=torch.int32
        )
        self.non_trainable_params["appearance_radius_sum"] = torch.zeros(
            count, device=self.device, dtype=torch.float32
        )
        self.non_trainable_params["appearance_direction_sum"] = torch.zeros(
            (count, 3), device=self.device, dtype=torch.float32
        )
        self.non_trainable_params["appearance_band_utility_ema"] = torch.zeros(
            count, device=self.device, dtype=torch.float32
        )
        if "appearance_band_gradient_ema" in self.non_trainable_params:
            gradient_ema = self.non_trainable_params[
                "appearance_band_gradient_ema"
            ]
            width = gradient_ema.shape[1]
            self.non_trainable_params["appearance_band_gradient_ema"] = torch.zeros(
                (count, width), device=self.device, dtype=gradient_ema.dtype
            )
        self.non_trainable_params["appearance_sh_degree"] = torch.zeros(
            count, device=self.device, dtype=torch.uint8
        )

    @torch.no_grad()
    def update_gaussians_data(self, params):
        """Update equal-shaped group values while retaining its Adam moments."""
        for name in self.splats:
            value = params[name].detach().to(self.device, dtype=self.splats[name].dtype)
            if value.shape != self.splats[name].shape:
                raise ValueError("Cannot change progressive group shape in-place")
            self.splats[name].data.copy_(value)

    @torch.no_grad()
    def extend_gaussians_from_color_points(
        self,
        pts,
        colors,
        init_scale=1e-4,
        initial_opacity=0.5,
        max_scale_expansion=None,
        footprint_trust_mask=None,
        footprint_target_scales=None,
        footprint_release_scale_expansions=None,
        directional_observability_mask=None,
        directional_log_depth_stds=None,
        directional_initial_max_parallax_sin2=None,
        reference_camera_center=None,
        track_ids=None,
        gaussian_uids=None,
        birth_frame_ids=None,
        metric_confidences=None,
        uncertainty_confidences=None,
        initial_quaternions=None,
    ):
        xyz = torch.from_numpy(np.asarray(pts)).float().to(self.device)
        sh0 = (
            torch.from_numpy(np.asarray(rgb_to_sh_np(colors)))
            .float()
            .reshape(-1, 1, 3)
            .to(self.device)
        )
        shN = torch.zeros(
            (sh0.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3),
            device=self.device,
            dtype=torch.float32,
        )
        if np.isscalar(initial_opacity):
            opacity_values = torch.full(
                (xyz.shape[0],),
                float(initial_opacity),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            opacity_values = torch.as_tensor(
                initial_opacity, device=self.device, dtype=torch.float32
            ).reshape(-1)
            if opacity_values.shape != (xyz.shape[0],):
                raise ValueError("Per-Gaussian initial opacity has the wrong shape")
        opacity = self.inverse_opacity_activation(
            torch.clamp(opacity_values, min=1.0e-4, max=1.0 - 1.0e-4)
        )

        if isinstance(init_scale, float):
            init_scale_param = (
                torch.ones((xyz.shape[0],), device=self.device, dtype=torch.float32)
                * init_scale
            )
            scales = (
                torch.ones((xyz.shape[0], 3), device=self.device, dtype=torch.float32)
                * init_scale
            )
        elif isinstance(init_scale, np.ndarray):
            if init_scale.shape[1] == 1:
                init_scale_param = (
                    torch.from_numpy(init_scale).float().to(self.device).reshape(-1)
                )
                init_scale = np.repeat(init_scale, 3, axis=1)
            elif init_scale.shape[1] == 3:
                init_scale_param = (
                    torch.from_numpy(np.mean(init_scale, axis=1))
                    .float()
                    .to(self.device)
                    .reshape(-1)
                )
            else:
                raise NotImplementedError(
                    "init_scale should be either float or np.ndarray"
                )
            scales = torch.from_numpy(init_scale).float().to(self.device)
        else:
            raise NotImplementedError("init_scale should be either float or np.ndarray")

        if initial_quaternions is None:
            quats = torch.zeros(
                (xyz.shape[0], 4), device=self.device, dtype=torch.float32
            )
            quats[:, 0] = 1
        else:
            quats = torch.as_tensor(
                initial_quaternions, device=self.device, dtype=torch.float32
            )
            if quats.shape != (xyz.shape[0], 4) or not bool(
                torch.all(torch.isfinite(quats))
            ):
                raise ValueError("Initial quaternions must be finite [N, 4]")
            norms = torch.linalg.vector_norm(quats, dim=1, keepdim=True)
            if bool(torch.any(norms <= 1.0e-8)):
                raise ValueError("Initial quaternions must be nonzero")
            quats = quats / norms

        new_params = {
            "means": xyz,
            "scales": scales,
            "quats": quats,
            "opacities": opacity,
            "sh0": sh0,
            "shN": shN,
            "init_scales": torch.exp(
                init_scale_param
            ),  # actual scales are after exp op
        }
        if track_ids is not None:
            track_id_tensor = torch.as_tensor(
                track_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
            if track_id_tensor.shape != (xyz.shape[0],):
                raise ValueError("Track IDs must match Gaussian rows")
            new_params["track_ids"] = track_id_tensor
        if gaussian_uids is not None:
            uid_tensor = torch.as_tensor(
                gaussian_uids, device=self.device, dtype=torch.long
            ).reshape(-1)
            if uid_tensor.shape != (xyz.shape[0],):
                raise ValueError("Gaussian UIDs must match Gaussian rows")
            new_params["gaussian_uids"] = uid_tensor
        if birth_frame_ids is not None:
            birth_tensor = torch.as_tensor(
                birth_frame_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
            if birth_tensor.shape != (xyz.shape[0],):
                raise ValueError("Birth frame IDs must match Gaussian rows")
            new_params["birth_frame_ids"] = birth_tensor
        if metric_confidences is not None:
            metric_tensor = torch.as_tensor(
                metric_confidences, device=self.device, dtype=torch.float32
            ).reshape(-1)
            if metric_tensor.shape != (xyz.shape[0],) or bool(
                torch.any(~torch.isfinite(metric_tensor))
            ) or bool(torch.any((metric_tensor < 0.0) | (metric_tensor > 1.0))):
                raise ValueError("Metric confidences must lie in [0, 1]")
            new_params["metric_confidences"] = metric_tensor
        if uncertainty_confidences is not None:
            uncertainty_tensor = torch.as_tensor(
                uncertainty_confidences, device=self.device, dtype=torch.float32
            ).reshape(-1)
            if uncertainty_tensor.shape != (xyz.shape[0],) or bool(
                torch.any(~torch.isfinite(uncertainty_tensor))
            ) or bool(
                torch.any((uncertainty_tensor < 0.0) | (uncertainty_tensor > 1.0))
            ):
                raise ValueError("Uncertainty confidences must lie in [0, 1]")
            new_params["uncertainty_confidences"] = uncertainty_tensor
        if directional_observability_mask is not None:
            directional_mask = torch.as_tensor(
                directional_observability_mask, device=self.device, dtype=torch.bool
            ).reshape(-1)
            if directional_mask.shape != (xyz.shape[0],):
                raise ValueError("Directional observability mask has the wrong shape")
            if reference_camera_center is None:
                raise ValueError(
                    "Directional observability rows require a reference camera center"
                )
            reference_center = torch.as_tensor(
                reference_camera_center, device=self.device, dtype=torch.float32
            ).reshape(1, 3)
            reference_centers = reference_center.expand(xyz.shape[0], -1).clone()
            reference_rays = torch.nn.functional.normalize(
                xyz - reference_centers, dim=1, eps=1.0e-8
            )
            log_depth_stds = torch.as_tensor(
                directional_log_depth_stds,
                device=self.device,
                dtype=torch.float32,
            ).reshape(-1)
            if log_depth_stds.shape != (xyz.shape[0],):
                raise ValueError("Directional log-depth priors must match Gaussian rows")
            if not bool(torch.all(torch.isfinite(log_depth_stds))) or bool(
                torch.any(log_depth_stds < 0.0)
            ):
                raise ValueError("Directional log-depth priors must be finite")
            initial_parallax = (
                torch.zeros(
                    xyz.shape[0], device=self.device, dtype=torch.float32
                )
                if directional_initial_max_parallax_sin2 is None
                else torch.as_tensor(
                    directional_initial_max_parallax_sin2,
                    device=self.device,
                    dtype=torch.float32,
                ).reshape(-1)
            )
            if initial_parallax.shape != (xyz.shape[0],) or bool(
                torch.any(~torch.isfinite(initial_parallax))
            ) or bool(torch.any((initial_parallax < 0.0) | (initial_parallax > 1.0))):
                raise ValueError("Initial parallax evidence must lie in [0, 1]")
            new_params.update(
                {
                    "directional_observability_mask": directional_mask,
                    "reference_camera_centers": reference_centers,
                    "reference_rays": reference_rays,
                    "max_parallax_sin2": initial_parallax,
                    "birth_log_depth_stds": log_depth_stds,
                }
            )
        if max_scale_expansion is None:
            expansion = (
                self.default_max_scale_expansion
                if self.default_max_scale_expansion > 0.0
                else float("inf")
            )
            new_params["max_scale_expansions"] = torch.full(
                (xyz.shape[0],), expansion, device=self.device, dtype=torch.float32
            )
        else:
            expansion = torch.as_tensor(
                max_scale_expansion, device=self.device, dtype=torch.float32
            ).reshape(-1)
            if expansion.numel() == 1:
                expansion = expansion.expand(xyz.shape[0]).clone()
            if expansion.shape != (xyz.shape[0],):
                raise ValueError("Scale expansion limits must match Gaussian rows")
            new_params["max_scale_expansions"] = expansion
        if footprint_trust_mask is not None:
            trust_mask = torch.as_tensor(
                footprint_trust_mask, device=self.device, dtype=torch.bool
            ).reshape(-1)
            targets = torch.as_tensor(
                footprint_target_scales, device=self.device, dtype=torch.float32
            ).reshape(-1)
            if trust_mask.shape != (xyz.shape[0],) or targets.shape != (xyz.shape[0],):
                raise ValueError("Footprint trust state must match Gaussian rows")
            if bool(torch.any(trust_mask & ~torch.isfinite(targets))) or bool(
                torch.any(targets <= 0.0)
            ):
                raise ValueError("Certified footprint targets must be positive")
            new_params["footprint_trust_mask"] = trust_mask
            new_params["footprint_target_scales"] = targets
            new_params["footprint_evidence_pending_mask"] = trust_mask.clone()
            release_limits = torch.as_tensor(
                footprint_release_scale_expansions,
                device=self.device,
                dtype=torch.float32,
            ).reshape(-1)
            if release_limits.shape != (xyz.shape[0],) or bool(
                torch.any(release_limits <= 0.0)
            ):
                raise ValueError("Footprint release limits must match Gaussian rows")
            new_params["footprint_release_scale_expansions"] = release_limits

        self.extend_gaussians(new_params)

    @torch.no_grad()
    def prune_w_opacity(self, threshold):
        opacity = self.get_opacity
        return self.prune_with_mask(opacity > threshold)

    @torch.no_grad()
    def prune_with_mask(self, valid_mask):
        valid_mask = torch.as_tensor(
            valid_mask, device=self.device, dtype=torch.bool
        ).reshape(-1)
        if valid_mask.shape != (self.get_num,):
            raise ValueError("Gaussian prune mask has the wrong shape")
        removed_track_ids = self.non_trainable_params["track_ids"][~valid_mask]
        self.last_pruned_gaussian_uids = self.non_trainable_params[
            "gaussian_uids"
        ][~valid_mask].detach().cpu().numpy()

        new_params = {}
        for name, optimizer in self.optimizers.items():
            assert len(optimizer.param_groups) == 1
            group = optimizer.param_groups[0]
            assert len(group["params"]) == 1
            p = group["params"][0]
            stored_state = optimizer.state.get(p, None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]

                del optimizer.state[p]
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid_mask].requires_grad_(True)
                )
                optimizer.state[group["params"][0]] = stored_state

                new_params[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid_mask].requires_grad_(True)
                )
                new_params[name] = group["params"][0]

        for name in self.splats:
            self.splats[name] = new_params[name]

        for name in self.non_trainable_params:
            self.non_trainable_params[name] = self.non_trainable_params[name][
                valid_mask
            ]
        return removed_track_ids.detach().cpu().numpy()

    def optimizer_step(self):
        opacity_gradient = self.splats["opacities"].grad
        if opacity_gradient is not None:
            signal = opacity_gradient.detach().abs().reshape(-1)
            utility = self.non_trainable_params["gradient_utility_ema"]
            utility.mul_(0.95).add_(signal, alpha=0.05)
        for optimizer in self.optimizers.values():
            optimizer.step()
        self.constrain_scale_expansion()
        self.constrain_mean_displacement()

    @torch.no_grad()
    def scale_optimizer_gradients(self, factor):
        for parameter in self.splats.values():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)

    def zero_optimizer_gradients(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def update(self):
        self.optimizer_step()
        self.zero_optimizer_gradients()

    @torch.no_grad()
    def set_mean_trust_region(self, max_displacement):
        """Anchor previously unbounded rows once and constrain future extensions."""
        radius = float(max_displacement)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("Mean trust-region radius must be finite and positive")
        self.default_max_mean_displacement = radius
        limits = self.non_trainable_params["max_mean_displacements"]
        unbounded = ~torch.isfinite(limits)
        if torch.any(unbounded):
            self.non_trainable_params["mean_anchors"][unbounded] = (
                self.splats["means"].detach()[unbounded]
            )
            limits[unbounded] = radius

    @torch.no_grad()
    def constrain_mean_displacement(self):
        """Project optimized means back into their canonical birth trust regions."""
        if self.get_num == 0:
            return 0
        limits = self.non_trainable_params["max_mean_displacements"].reshape(-1, 1)
        bounded = torch.isfinite(limits.reshape(-1))
        if not torch.any(bounded):
            return 0
        anchors = self.non_trainable_params["mean_anchors"]
        delta = self.splats["means"] - anchors
        distance = torch.linalg.norm(delta, dim=1, keepdim=True)
        factor = torch.clamp(limits / torch.clamp(distance, min=1.0e-12), max=1.0)
        projected = anchors + delta * factor
        changed = int(torch.count_nonzero(bounded & (distance.reshape(-1) > limits.reshape(-1))).item())
        self.splats["means"][bounded] = projected[bounded]
        return changed

    @torch.no_grad()
    def constrain_scale_expansion(self):
        """Keep optimized footprints inside their initialization trust region."""
        if self.get_num == 0:
            return 0
        factors = self.non_trainable_params["max_scale_expansions"].reshape(-1, 1)
        if not torch.isfinite(factors).any():
            return 0
        initial = self.get_init_scales.reshape(-1, 1)
        maximum = torch.clamp(
            initial * factors,
            min=torch.finfo(self.splats["scales"].dtype).tiny,
        )
        current = self.get_scaling
        clamped = torch.minimum(current, maximum)
        changed = int(torch.count_nonzero(current > maximum).item())
        self.splats["scales"].copy_(torch.log(clamped))
        return changed

    @torch.no_grad()
    def load_from_ply(self, path, max_sh_degree=2):
        plydata = PlyData.read(path)
        sparse_degrees = None

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])
        vertex_property_names = {
            prop.name for prop in plydata.elements[0].properties
        }
        metric_confidences = (
            np.asarray(
                plydata.elements[0]["metric_confidence"], dtype=np.float32
            )
            if "metric_confidence" in vertex_property_names
            else np.ones((xyz.shape[0],), dtype=np.float32)
        )
        uncertainty_confidences = (
            np.asarray(
                plydata.elements[0]["uncertainty_confidence"], dtype=np.float32
            )
            if "uncertainty_confidence" in vertex_property_names
            else metric_confidences.copy()
        )

        sh0 = np.zeros((xyz.shape[0], 1, 3))
        sh0[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        sh0[:, 0, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        sh0[:, 0, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        if is_tgbr_sparse_ply(plydata):
            shN, sparse_degrees, _ = decode_tgbr_sparse_sh(
                plydata, max_sh_degree
            )
        else:
            shN = np.zeros(
                (xyz.shape[0], ((max_sh_degree + 1) ** 2 - 1) * 3)
            )
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(
                extra_f_names, key=lambda x: int(x.split("_")[-1])
            )
            for idx, attr_name in enumerate(extra_f_names):
                shN[:, idx] = np.asarray(plydata.elements[0][attr_name])
            shN = shN.reshape(
                (shN.shape[0], 3, (max_sh_degree + 1) ** 2 - 1)
            ).transpose(0, 2, 1)

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.from_numpy(xyz).float().to(self.device)
        sh0 = torch.from_numpy(sh0).float().to(self.device)
        shN = torch.from_numpy(shN).float().to(self.device)
        opacities = torch.from_numpy(opacities).float().to(self.device)
        scales = torch.from_numpy(scales).float().to(self.device)
        rots = torch.from_numpy(rots).float().to(self.device)
        init_scales = (
            torch.ones((xyz.shape[0],), dtype=torch.float32, device=self.device) * 1e-4
        )

        new_params = {
            "means": xyz,
            "scales": scales,
            "quats": rots,
            "opacities": opacities,
            "sh0": sh0,
            "shN": shN,
            "init_scales": torch.exp(init_scales),
            "max_scale_expansions": torch.full(
                (xyz.shape[0],), float("inf"), device=self.device, dtype=torch.float32
            ),
            "metric_confidences": torch.from_numpy(metric_confidences)
            .to(self.device, dtype=torch.float32),
            "uncertainty_confidences": torch.from_numpy(
                uncertainty_confidences
            ).to(self.device, dtype=torch.float32),
        }

        self.replace_gaussians(new_params)
        if sparse_degrees is not None:
            self.non_trainable_params["appearance_sh_degree"].copy_(
                torch.from_numpy(sparse_degrees).to(
                    self.device, dtype=torch.uint8
                )
            )


class GaussianModel:
    def __init__(self, configs):
        self.configs = configs
        self.worldtest_certificate_authority = None
        self.worldtest_group_certificates = {}

        self.active_sh_degree = configs["Model"]["sh_degree"]
        self.max_sh_degree = configs["Model"]["sh_degree"]

        self.gaussian_pos_schedule_steps = configs["Model"][
            "gaussian_pos_schedule_steps"
        ]

        self.use_anti_aliasing = configs["Model"]["use_anti_aliasing"]
        self.render_mode = configs["Model"]["render_mode"]
        self.device = configs["Model"]["device"]

        self.radius_clip = configs["Model"][
            "radius_clip"
        ]  # Do not render gaussians smaller than this radius
        self.init_scale_size = float(
            configs["Model"]["init_scale_size"]
        )  # Initial scale of the gaussians
        self.camera_scale_rescalar = (
            float(configs["Model"]["camera_scale_rescalar"])
            if "camera_scale_rescalar" in configs["Model"]
            else 1.0
        )  # Rescale the camera scale to avoid gaussians too small to render

        self.BS = (
            configs["Mapper"]["active_window_size"]
            + configs["Mapper"]["coarse_active_window_size"]
            + configs["Mapper"]["KFGraph"]["global_window_size"]
        )
        self.scene_scale = configs["Model"]["scene_scale"]
        self.init_gaussian_config = configs["Model"]["init_gaussian_config"]
        self.init_gaussian_config["lr_final_step"] = self.gaussian_pos_schedule_steps

        self.MAX_LEVEL = 4

        self.init_gaussian_groups()

        # Additional groups owned by ProgressiveGaussianStore. Empty when the
        # feature is disabled, so baseline group lifecycle is unchanged.
        self.progressive_group_ids = set()
        self.progressive_optimizers = {}
        self.progressive_optimizer_group_ids = {}
        self.progressive_optimizer_indices = {}
        self.progressive_scheduler_args = {}
        self.progressive_lr_step = 0
        self.frozen_geometry_group_ids = set()
        self.frozen_position_group_ids = set()
        self.sh_degree_masks = {}
        self.frontview_observability_config = dict(
            configs.get("FrontViewObservability", {})
        )
        self._frontview_post_step_projection = {}
        self.frontview_observability_stats = {
            "calls": 0,
            "rows": 0,
            "learning_scale_mode": None,
            "optimization_mode": None,
            "responsibility_scope": None,
            "radial_scale_sum": 0.0,
            "last_rows": 0,
            "last_mean_radial_scale": None,
            "last_unlocked_fraction": None,
            "evidence_updates": 0,
            "post_step_projection_calls": 0,
            "post_step_projected_rows": 0,
        }
        self.frontview_coverage_recovery_config = (
            validate_front_view_coverage_recovery_config(
                configs.get("FrontViewCoverageRecovery")
            )
        )
        self.frontview_coverage_depth_prior = SparseFarDepthPrior(
            self.frontview_coverage_recovery_config
        )
        self.causal_metric_birth_config = validate_causal_metric_birth_config(
            configs.get("CausalMetricBirth")
        )
        self.causal_metric_birth = CausalMetricBirth(
            self.causal_metric_birth_config, self.device
        )
        self.causal_landmark_memory_config = (
            validate_causal_landmark_memory_config(
                configs.get("CausalPersistentLandmarkMemory")
            )
        )
        self.causal_landmark_memory = CausalPersistentLandmarkMemory(
            self.causal_landmark_memory_config
        )
        self.causal_dual_responsibility_config = (
            validate_causal_dual_responsibility_config(
                configs.get("CausalDualResponsibility")
            )
        )
        self.causal_dual_responsibility_stats = {
            "metric_rows": 0,
            "proxy_rows": 0,
            "partial_metric_rows": 0,
            "metric_confidence_sum": 0.0,
            "metric_depth_render_calls": 0,
            "finite_certificate_calls": 0,
            "finite_certificate_rows": 0,
            "finite_certificate_valid_rows": 0,
            "finite_certificate_observability_sum": 0.0,
            "finite_certificate_support_sum": 0.0,
            "finite_certificate_value_sum": 0.0,
        }
        self.frontview_coverage_depth_stats = {
            "depth_fallback_calls": 0,
            "depth_fallback_rows": 0,
            "depth_fallback_map_rows": 0,
            "depth_fallback_prior_rows": 0,
            "depth_fallback_motion_floor_calls": 0,
            "depth_fallback_motion_floor_sum_m": 0.0,
            "last_depth_fallback_motion_floor_m": None,
            "depth_fallback_skipped_no_prior": 0,
            "depth_fallback_scaled_rows": 0,
            "last_depth_fallback_frame": -1,
            "multiview_depth_calls": 0,
            "multiview_depth_rows": 0,
            "multiview_depth_supported_rows": 0,
            "multiview_depth_concentrated_rows": 0,
            "multiview_depth_score_sum": 0.0,
            "multiview_depth_concentration_sum": 0.0,
            "multiview_depth_selected_sum_m": 0.0,
            "multiview_depth_hypotheses_sum": 0,
            "multiview_depth_shuffled_calls": 0,
        }
        self.frontview_sampling_config = dict(configs.get("FrontViewSampling", {}))
        self.frontview_inverse_depth_certificate_config = (
            validate_front_view_inverse_depth_certificate_config(
                configs.get("FrontViewInverseDepthCertificate")
            )
        )
        self.frontview_inverse_depth_certificate_stats = {
            "calls": 0,
            "rows": 0,
            "certified_rows": 0,
            "conflicted_rows": 0,
            "rejected_conflicted_rows": 0,
            "projective_uncertified_rows": 0,
            "information_gain_sum": 0.0,
            "absolute_log_depth_shift_sum": 0.0,
            "posterior_log_std_sum": 0.0,
            "valid_view_sum": 0,
            "baseline_information_sum": 0.0,
            "shuffle_calls": 0,
        }
        self.frontview_sampling_stats = {
            "calls": 0,
            "pool_rows": 0,
            "selected_rows": 0,
            "priority_selected_rows": 0,
            "coverage_selected_rows": 0,
            "fallback_selected_rows": 0,
            "valid_reference_rows": 0,
            "score_sum": 0.0,
            "last_mean_score": None,
            "pool_depth_counts": [0, 0, 0],
            "selected_depth_counts": [0, 0, 0],
            "adaptive_calls": 0,
            "adaptive_iterations": 0,
            "adaptive_objective_sum": 0.0,
            "adaptive_boundary_sum_m": [0.0, 0.0],
            "last_adaptive_boundaries_m": [],
            "adaptive_assigned_pool_counts": [0, 0, 0],
            "adaptive_quotas": [0, 0, 0],
            "adaptive_coverage_calls": 0,
            "adaptive_coverage_cell_pixel_sum": 0.0,
            "adaptive_coverage_representatives": [0, 0, 0],
            "rate_distortion_calls": 0,
            "rate_distortion_cell_pixel_sum": 0.0,
            "rate_distortion_density_min": None,
            "rate_distortion_density_max": None,
            "rate_distortion_shuffled_calls": 0,
            "posterior_refill_pool_rows": 0,
            "posterior_refill_primary_rows": 0,
        }
        self.frontview_depth_transport_config = dict(
            configs.get("FrontViewDepthTransport", {})
        )
        self.frontview_depth_transport_stats = {
            "calls": 0,
            "calibrated_calls": 0,
            "skipped_calls": 0,
            "training_anchors": 0,
            "calibration_anchors": 0,
            "valid_calibration_anchors": 0,
            "corrected_rows": 0,
            "absolute_log_residual_sum": 0.0,
            "absolute_log_correction_sum": 0.0,
        }
        self.frontview_birth_config = validate_front_view_birth_config(
            configs.get("FrontViewBirth")
        )
        self.frontview_track_ledger = TrackResponsibilityLedger(
            self.frontview_birth_config["track_refinement_ratio"]
        )
        self.frontview_birth_stats = {
            "calls": 0,
            "pool_rows": 0,
            "selected_rows": 0,
            "priority_selected_rows": 0,
            "coverage_selected_rows": 0,
            "fallback_selected_rows": 0,
            "map_rejected_rows": 0,
            "atlas_projected_rows": 0,
            "atlas_rejected_rows": 0,
            "anchor_rejected_rows": 0,
            "near_hash_query_rows": 0,
            "near_hash_set_rows": 0,
            "far_hash_bypass_rows": 0,
            "cell_rejected_rows": 0,
            "pool_layer_counts": [0, 0, 0],
            "selected_layer_counts": [0, 0, 0],
            "last_layer_edges_m": [],
            "scale_capped_rows": 0,
            "responsibility_opacity_rows": 0,
            "responsibility_opacity_sum": 0.0,
            "temporal_calls": 0,
            "temporal_tested_rows": 0,
            "temporal_free_space_rows": 0,
            "temporal_duplicate_rows": 0,
            "temporal_rejected_rows": 0,
        }
        self.frontview_far_field_config = validate_front_view_far_field_config(
            configs.get("FrontViewFarField")
        )
        self.frontview_ray_atlas = CausalRayResponsibilityAtlas(
            enabled=self.frontview_far_field_config["ray_atlas_enabled"],
            shuffle_evidence=self.frontview_far_field_config[
                "ray_atlas_shuffle_evidence"
            ],
            seed=self.frontview_far_field_config["shuffle_seed"],
            coordinate_mode=self.frontview_far_field_config[
                "ray_atlas_coordinate_mode"
            ],
            competition_mode=self.frontview_far_field_config[
                "ray_atlas_competition_mode"
            ],
        )
        self.frontview_far_field_stats = {
            "host_rows": 0,
            "commit_rows": 0,
            "hash_bypass_rows": 0,
            "projective_rejected_rows": 0,
            "unobservable_rejected_rows": 0,
            "hash_query_rows": 0,
            "hash_set_rows": 0,
            "exact_sparse_world_rows": 0,
            "identity_route_calls": 0,
            "identity_route_rows": 0,
            "persistent_identity_rows": 0,
            "sparse_missing_identity_rows": 0,
            "identity_far_field_rows": 0,
            "causal_route_calls": 0,
            "causal_route_rows": 0,
            "causal_visible_rows": 0,
            "causal_projective_rows": 0,
            "causal_metric_depthcov_rows": 0,
            "causal_parallax_pixel_sum": 0.0,
            "causal_support_pixel_sum": 0.0,
            "causal_log_depth_std_sum": 0.0,
            "responsibility_shuffle_calls": 0,
            "responsibility_shuffled_rows": 0,
            "footprint_trust_calls": 0,
            "footprint_trust_rows": 0,
            "footprint_trust_bounded_rows": 0,
            "footprint_trust_information_sum": 0.0,
            "footprint_trust_limit_sum": 0.0,
            "footprint_trust_cell_pixel_sum": 0.0,
            "footprint_trust_shuffled_calls": 0,
            "footprint_trust_projective_scope_calls": 0,
            "footprint_trust_owner_area_calls": 0,
            "footprint_trust_owner_area_rows": 0,
            "footprint_trust_residual_rd_calls": 0,
            "footprint_trust_residual_rd_rows": 0,
            "footprint_trust_residual_rd_radius_sum": 0.0,
            "footprint_trust_dynamic_calls": 0,
            "footprint_trust_dynamic_rows": 0,
            "footprint_trust_dynamic_released_rows": 0,
            "footprint_trust_dynamic_shuffled_calls": 0,
            "adaptive_route_calls": 0,
            "adaptive_route_far_rows": 0,
            "adaptive_route_iterations": 0,
            "adaptive_route_objective_sum": 0.0,
            "adaptive_route_boundary_sum_m": [0.0, 0.0],
            "adaptive_route_regime_counts": [0, 0, 0],
            "last_adaptive_route_boundaries_m": [],
            "map_gate_calls": 0,
            "map_gate_rows": 0,
            "map_gate_rejected_rows": 0,
            "map_gate_residual_scale_sum": 0.0,
            "map_gate_photometric_calls": 0,
            "map_gate_shuffled_calls": 0,
            "posterior_refill_calls": 0,
            "posterior_refill_requested_rows": 0,
            "posterior_refill_reserve_rows": 0,
            "posterior_refill_selected_rows": 0,
            "posterior_refill_shuffled_calls": 0,
            "adaptive_nms_calls": 0,
            "adaptive_nms_rows": 0,
            "adaptive_nms_rejected_rows": 0,
            "budget_nms_calls": 0,
            "budget_nms_cell_pixel_sum": 0.0,
            "budget_nms_log_depth_width_sum": 0.0,
            "projective_covariance_rows": 0,
            "projective_radial_factor_sum": 0.0,
            "projective_radial_factor_min": None,
            "projective_radial_factor_max": None,
            "fallback_support_rows": 0,
            "fallback_information_rows": 0,
            "fallback_information_radius_factor_sum": 0.0,
            "fallback_information_radius_factor_min": None,
            "fallback_information_radius_factor_max": None,
            "fallback_information_density_sum": 0.0,
            "fallback_information_shuffled_calls": 0,
            "fallback_structure_rows": 0,
            "fallback_anisotropy_sum": 0.0,
            "fallback_anisotropy_max": None,
        }
        self.frontview_directional_layer_config = (
            validate_front_view_directional_layer_config(
                configs.get("FrontViewDirectionalLayer")
            )
        )
        self.frontview_directional_layer = FrontViewDirectionalLayer(
            self.frontview_directional_layer_config
        )
        self.frontview_scale_cover_config = validate_front_view_scale_cover_config(
            configs.get("FrontViewScaleCover")
        )
        self.frontview_scale_cover = FrontViewScaleCover(
            self.frontview_scale_cover_config
        )
        self.streaming_appearance_lod_config = (
            validate_streaming_appearance_lod_config(
                configs.get("StreamingAppearanceLOD")
            )
        )
        if self.streaming_appearance_lod_config["enabled"]:
            if self.max_sh_degree < self.streaming_appearance_lod_config["target_degree"]:
                raise ValueError(
                    "StreamingAppearanceLOD target_degree exceeds Model.sh_degree"
                )
            if (
                self.streaming_appearance_lod_config["sparse_model_export"]
                and self.max_sh_degree
                != self.streaming_appearance_lod_config["target_degree"]
            ):
                raise ValueError(
                    "TGBR sparse export requires Model.sh_degree == target_degree"
                )
        self.streaming_appearance_lod_stats = {
            "evidence_updates": 0,
            "observed_rows": 0,
            "promotion_updates": 0,
            "promoted_rows": 0,
            "gradient_signal_rows": 0,
            "gradient_positive_rows": 0,
            "gradient_signal_sum": 0.0,
            "gradient_signal_max": 0.0,
            "current_target_fraction": 0.0,
        }
        self.streaming_compute_routing_stats = {
            "render_calls": 0,
            "rasterization_calls": 0,
            "packed_rows": 0,
            "base_rows": 0,
            "promoted_target_rows": 0,
            "probe_rows": 0,
            "target_rows": 0,
            "skipped_target_band_rows": 0,
        }
        self.tgbr_sparse_model_stats = None
        self.frontview_sparse_scale_map_config = (
            validate_front_view_sparse_scale_map_config(
                configs.get("FrontViewSparseScaleMap")
            )
        )
        self.frontview_sparse_scale_map = FrontViewSparseScaleMap(
            self.frontview_sparse_scale_map_config
        )
        self.frontview_identity_lod_config = validate_front_view_identity_lod_config(
            configs.get("FrontViewIdentityLOD")
        )
        self.frontview_identity_lod = FrontViewIdentityLOD(
            self.frontview_identity_lod_config
        )
        if self.frontview_identity_lod.enabled:
            self._ensure_frontview_identity_uids()
        self.frontview_residual_cover_config = (
            validate_front_view_residual_cover_config(
                configs.get("FrontViewResidualCover")
            )
        )
        self.frontview_residual_cover = FrontViewResidualCover(
            self.frontview_residual_cover_config
        )
        self.frontview_track_fusion_config = (
            validate_front_view_track_fusion_config(
                configs.get("FrontViewTrackFusion")
            )
        )
        self.frontview_track_lookup = {}
        self.frontview_track_lookup_dirty = False
        self.frontview_track_fusion_stats = {
            "calls": 0,
            "observed_tracks": 0,
            "matched_tracks": 0,
            "updated_gaussians": 0,
            "shuffled_tracks": 0,
            "lookup_rebuilds": 0,
        }
        progressive_config = configs.get("ProgressiveMapping", {})
        self.progressive_lr_multipliers = {
            "means": float(progressive_config.get("means_lr_multiplier", 1.0)),
            "scales": float(progressive_config.get("scales_lr_multiplier", 1.0)),
            "quats": float(progressive_config.get("quats_lr_multiplier", 1.0)),
            "opacities": 1.0,
            "sh0": float(progressive_config.get("appearance_lr_multiplier", 1.0)),
            "shN": float(progressive_config.get("appearance_lr_multiplier", 1.0)),
        }

        self.hash_block = HashBlock(configs["HashBlock"])

        self.densification_mode = configs["Model"]["densification_mode"]
        self.init_scale_offset = float(configs["Model"].get("init_scale_offset", 0.0))
        self.frequency_sampling_config = {}
        if "semi-dense_extra-pts" in self.densification_mode:
            self.extra_pts_num = configs["Model"]["extra_pts_num"]
            self.err_threshold = configs["Model"]["err_threshold"]
            self.semi_dense_err_threshold = configs["Model"]["semi_dense_err_threshold"]
            self.frequency_sampling_config = dict(
                configs["Model"].get("frequency_sampling", {})
            )

            if "adaptive" in configs["Model"]["densification_mode"]:
                self.init_scale_offset = (
                    configs["Model"]["init_scale_offset"]
                    if "init_scale_offset" in configs["Model"]
                    else 0.0
                )

        self.opacity_prune_threshold = configs["Model"]["opacity_prune_threshold"]
        self.opacity_pruning_audit_enabled = bool(
            configs.get("CausalDepthAudit", {}).get("audit_opacity_pruning", False)
        )
        self.opacity_pruning_audit_calls = []

        self.gaussian_type = configs["Model"]["gaussian_type"]

        Log("Gaussian Type: {}".format(self.gaussian_type), tag="GaussianModel")

        self.scene_exposure_gain = configs["Mapper"]["scene_exposure_gain"]

        self.gaussians_size_filter = (
            configs["Model"]["gaussians_size_filter"]
            if "gaussians_size_filter" in configs["Model"]
            else False
        )

        self.vignette_imgs = None

        if "DepthCovEstimator" in configs["Model"]:
            configs["Model"]["DepthCovEstimator"]["device"] = self.device
            self.depth_cov_estimator = DepthCovEstimator(
                configs["Model"]["DepthCovEstimator"]
            )

    def configure_worldtest_certificate_authority(self, authority):
        """Enable fail-closed permanent birth for this model instance."""
        self.worldtest_certificate_authority = authority

    def _frontview_birth_enabled(self):
        return bool(
            getattr(self, "frontview_birth_config", {}).get("enabled", False)
        )

    def _frontview_far_field_enabled(self):
        return bool(
            getattr(self, "frontview_far_field_config", {}).get("enabled", False)
        )

    def _frontview_scale_cover_enabled(self):
        return bool(
            getattr(self, "frontview_scale_cover_config", {}).get("enabled", False)
        )

    def observe_frontview_raw_pose(self, cam):
        if self.frontview_coverage_recovery_config["depth_fallback_enabled"]:
            self.frontview_coverage_depth_prior.observe(
                int(cam.cam_idx), cam.get_sparse_depth(0)
            )
        if not self._frontview_scale_cover_enabled():
            return True
        return self.frontview_scale_cover.observe_raw_pose(
            cam.get_raw_pose().detach().cpu().numpy(), int(cam.cam_idx)
        )

    def _frontview_sparse_scale_map_enabled(self):
        return bool(
            getattr(self, "frontview_sparse_scale_map_config", {}).get(
                "enabled", False
            )
        )

    def _frontview_identity_lod_enabled(self):
        return bool(
            getattr(self, "frontview_identity_lod_config", {}).get("enabled", False)
        )

    def _frontview_residual_cover_enabled(self):
        return bool(
            getattr(self, "frontview_residual_cover_config", {}).get(
                "enabled", False
            )
        )

    def _frontview_track_fusion_enabled(self):
        return bool(
            getattr(self, "frontview_track_fusion_config", {}).get(
                "enabled", False
            )
        )

    @torch.no_grad()
    def _ensure_frontview_identity_uids(self):
        if not self._frontview_identity_lod_enabled():
            return
        for group in self.gaussian_groups:
            if group.splats is None:
                continue
            uids = group.non_trainable_params["gaussian_uids"]
            missing = uids < 0
            count = int(missing.sum().item())
            if count:
                assigned = torch.from_numpy(
                    self.frontview_identity_lod.allocate_uids(count)
                ).to(device=uids.device, dtype=torch.long)
                uids[missing] = assigned
            self.frontview_identity_lod.register_existing_roots(
                uids.detach().cpu().numpy()
            )

    @torch.no_grad()
    def _ensure_frontview_scale_cover_uids(self):
        if not (
            self._frontview_scale_cover_enabled()
            and self.frontview_scale_cover.tracks_uids
        ):
            return
        for group in self.gaussian_groups:
            if group.splats is None:
                continue
            uids = group.non_trainable_params["gaussian_uids"]
            missing = uids < 0
            count = int(missing.sum().item())
            if count:
                assigned = torch.from_numpy(
                    self.frontview_scale_cover.allocate_uids(count)
                ).to(device=uids.device, dtype=torch.long)
                uids[missing] = assigned
            self.frontview_scale_cover.register_uids(
                uids.detach().cpu().numpy()
            )

    def get_gaussian_uids(self):
        self._ensure_frontview_identity_uids()
        self._ensure_frontview_scale_cover_uids()
        return torch.cat(
            [
                self.gaussian_groups[group_id].non_trainable_params["gaussian_uids"]
                for level in range(self.MAX_LEVEL)
                for group_id in self.active_gaussian_groups[level]
            ],
            dim=0,
        )

    def frontview_identity_lod_summary(self):
        return self.frontview_identity_lod.summary()

    def frontview_residual_cover_summary(self):
        return self.frontview_residual_cover.summary()

    def frontview_track_fusion_summary(self):
        result = dict(getattr(self, "frontview_track_fusion_stats", {}))
        config = getattr(self, "frontview_track_fusion_config", {})
        result["enabled"] = self._frontview_track_fusion_enabled()
        result["shuffle_identity"] = bool(
            config.get("shuffle_identity", False)
        )
        return result

    def _rebuild_frontview_track_lookup(self):
        lookup = {}
        for group_id in self.valid_groups:
            group = self.gaussian_groups[group_id]
            if group.splats is None:
                continue
            tracks = group.non_trainable_params["track_ids"].detach().cpu().numpy()
            for row, track_id in enumerate(tracks.tolist()):
                if track_id >= 0:
                    lookup.setdefault(int(track_id), []).append(
                        (int(group_id), int(row))
                    )
        self.frontview_track_lookup = lookup
        self.frontview_track_lookup_dirty = False
        self.frontview_track_fusion_stats["lookup_rebuilds"] += 1

    @torch.no_grad()
    def fuse_frontview_track_appearance(self, track_ids, colors, depths, frame_id):
        if not self._frontview_track_fusion_enabled():
            return 0
        config = self.frontview_track_fusion_config
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        depths = np.asarray(depths, dtype=np.float32).reshape(-1)
        if not (len(track_ids) == len(colors) == len(depths)):
            raise ValueError("Track-fusion observations must align")
        valid = (track_ids >= 0) & np.isfinite(colors).all(axis=1)
        observation_rows = np.flatnonzero(valid)
        stats = self.frontview_track_fusion_stats
        stats["calls"] += 1
        stats["observed_tracks"] += len(observation_rows)
        if len(observation_rows) == 0:
            return 0
        if self.frontview_track_lookup_dirty:
            self._rebuild_frontview_track_lookup()

        matched_rows = np.asarray(
            [
                row
                for row in observation_rows.tolist()
                if int(track_ids[row]) in self.frontview_track_lookup
            ],
            dtype=np.int64,
        )
        stats["matched_tracks"] += len(matched_rows)
        if len(matched_rows) == 0:
            return 0

        observed_colors = colors[matched_rows].copy()
        if bool(config["shuffle_identity"]):
            bands = np.digitize(
                depths[matched_rows],
                np.asarray(config["shuffle_depth_edges_m"], dtype=np.float32),
            )
            rng = np.random.default_rng(int(config["shuffle_seed"]) + int(frame_id))
            for band in range(len(config["shuffle_depth_edges_m"]) + 1):
                rows = np.flatnonzero(bands == band)
                if len(rows) > 1:
                    observed_colors[rows] = observed_colors[
                        rows[rng.permutation(len(rows))]
                    ]
                    stats["shuffled_tracks"] += len(rows)

        updates = {}
        for observation_index, source_row in enumerate(matched_rows.tolist()):
            for group_id, gaussian_row in self.frontview_track_lookup[
                int(track_ids[source_row])
            ]:
                group_updates = updates.setdefault(group_id, ([], []))
                group_updates[0].append(gaussian_row)
                group_updates[1].append(observed_colors[observation_index])

        c0 = 0.28209479177387814
        updated = 0
        for group_id, (rows, targets) in updates.items():
            group = self.gaussian_groups[group_id]
            row_tensor = torch.as_tensor(rows, device=self.device, dtype=torch.long)
            target_rgb = torch.as_tensor(
                np.asarray(targets), device=self.device, dtype=torch.float32
            )
            current_rgb = group.splats["sh0"][row_tensor, 0] * c0 + 0.5
            fused_rgb = robust_color_ema(
                current_rgb,
                target_rgb,
                config["color_ema"],
                config["max_color_step"],
            )
            group.splats["sh0"][row_tensor, 0] = (fused_rgb - 0.5) / c0
            updated += len(rows)
        stats["updated_gaussians"] += updated
        return updated

    def _frontview_handoff_opacities(self, opacities, means, scales, cameras, level):
        identity_handoff = (
            self._frontview_identity_lod_enabled()
            and self.frontview_identity_lod_config["render_handoff_enabled"]
        )
        scale_handoff = (
            self._frontview_scale_cover_enabled()
            and self.frontview_scale_cover.dynamic_handoff_enabled
        )
        if not identity_handoff and not scale_handoff:
            return opacities
        cameras = list(cameras)
        poses = torch.stack([camera.get_pose() for camera in cameras], dim=0)
        focal_pixels = means.new_tensor(
            [
                0.5 * (camera.get_fx(level) + camera.get_fy(level))
                for camera in cameras
            ]
        )
        manager = (
            self.frontview_identity_lod
            if identity_handoff
            else self.frontview_scale_cover
        )
        multipliers = manager.render_handoff_multipliers(
            self.get_gaussian_uids(),
            means,
            scales,
            poses,
            focal_pixels,
            frame_id=max(int(camera.cam_idx) for camera in cameras),
        )
        return opacities * multipliers.reshape(
            (len(multipliers),) + (1,) * (opacities.ndim - 1)
        )

    @torch.no_grad()
    def _current_projective_map_occupancy(
        self, camera, candidate_pixels, candidate_depths, level
    ):
        if not bool(self.frontview_birth_config["multi_layer_map_competition"]):
            return None, 0
        means = self.get_xyz().detach()
        opacities = self.get_opacity().detach().reshape(-1)
        if len(means) == 0:
            return torch.zeros_like(candidate_depths, dtype=torch.bool), 0
        pose = camera.get_raw_pose().detach().to(means)
        camera_points = means @ pose[:3, :3].T + pose[:3, 3]
        depths = camera_points[:, 2]
        projected = camera_points @ camera.get_int_mat(level).detach().to(means).T
        uv = projected[:, :2] / torch.clamp(depths.reshape(-1, 1), min=1.0e-8)
        valid = (
            (depths > float(camera.near))
            & (depths < float(camera.far))
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < float(camera.get_width(level)))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < float(camera.get_height(level)))
            & (opacities >= float(self.frontview_birth_config["atlas_min_opacity"]))
        )
        map_uv = uv[valid]
        map_depths = depths[valid]
        occupied = multi_layer_projective_occupancy(
            candidate_pixels,
            candidate_depths,
            map_uv,
            map_depths,
            self.frontview_birth_config,
        )
        return occupied, int(valid.sum().item())

    @torch.no_grad()
    def _temporal_birth_rejections(
        self, camera, reference_cameras, pixels, depths, level
    ):
        count = int(depths.numel())
        if count == 0 or not reference_cameras:
            return torch.zeros(count, device=depths.device, dtype=torch.bool), {
                "tested_rows": 0,
                "free_space_rows": 0,
                "duplicate_rows": 0,
                "rejected_rows": 0,
            }
        world_points = unproject_pts_tensor(
            pixels,
            depths,
            camera.get_int_mat(level),
            camera.get_raw_pose().detach(),
        )
        candidate_depths = []
        map_depths = []
        map_opacities = []
        validities = []
        for reference in reference_cameras:
            render_pkg = self.render(reference, level=level)
            pose = reference.get_raw_pose().detach().to(world_points)
            camera_points = world_points @ pose[:3, :3].T + pose[:3, 3]
            z = camera_points[:, 2]
            projected = camera_points @ reference.get_int_mat(level).to(
                world_points
            ).T
            uv = projected[:, :2] / torch.clamp(z.reshape(-1, 1), min=1.0e-8)
            width = reference.get_width(level)
            height = reference.get_height(level)
            x = torch.floor(uv[:, 0]).to(dtype=torch.long)
            y = torch.floor(uv[:, 1]).to(dtype=torch.long)
            valid = (
                (z > float(reference.near))
                & (z < float(reference.far))
                & (x >= 0)
                & (x < width)
                & (y >= 0)
                & (y < height)
            )
            safe_x = torch.clamp(x, 0, width - 1)
            safe_y = torch.clamp(y, 0, height - 1)
            candidate_depths.append(z.detach().cpu().numpy())
            map_depths.append(
                self._geometry_depth(render_pkg)[safe_y, safe_x]
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
            map_opacities.append(
                self._geometry_opacity(render_pkg)[safe_y, safe_x]
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
            validities.append(valid.detach().cpu().numpy())
        reject, stats = temporal_responsibility_rejections(
            np.stack(candidate_depths),
            np.stack(map_depths),
            np.stack(map_opacities),
            np.stack(validities),
            self.frontview_birth_config,
        )
        return torch.from_numpy(reject).to(device=depths.device), stats

    def _require_worldtest_certificate(self, certificate, proposals=None, path="unknown"):
        authority = getattr(self, "worldtest_certificate_authority", None)
        if authority is None:
            return None
        return authority.require(certificate, proposals=proposals, path=path)

    def init_gaussian_groups(self):
        self.current_gaussian_group = {}
        self.active_gaussian_groups = {}
        self.gaussian_groups = []
        self.valid_groups = []

        for i in range(self.MAX_LEVEL):
            self.current_gaussian_group[i] = i
            self.active_gaussian_groups[i] = [i]
            self.gaussian_groups.append(
                Gaussians(
                    BS=self.BS,
                    scene_scale=self.scene_scale,
                    init_config=self.init_gaussian_config,
                    max_sh_degree=self.max_sh_degree,
                )
            )
            self.gaussian_groups[-1].to_device(self.device)
            self.valid_groups.append(i)

    @property
    def get_num_gaussians(self):
        total_num = 0
        for j in range(self.MAX_LEVEL):
            total_num += int(
                np.sum(
                    [
                        self.gaussian_groups[i].get_num
                        for i in self.active_gaussian_groups[j]
                    ]
                )
            )
        return total_num

    def get_xyz(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_xyz
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_xyz
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_xyz for i in self.active_gaussian_groups], dim=0)

    def get_scaling(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_scaling
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_scaling
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_scaling for i in self.active_gaussian_groups], dim=0)

    def get_rotation(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_rotation
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_rotation
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_rotation for i in self.active_gaussian_groups], dim=0)

    def get_opacity(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_opacity
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_opacity
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

        # return torch.cat([self.gaussian_groups[i].get_opacity for i in self.active_gaussian_groups], dim=0)

    def get_features(self, level=-1):
        if level == -1:
            res = []
            for j in range(self.MAX_LEVEL):
                res += [
                    torch.cat(
                        [
                            self.gaussian_groups[i].get_features
                            for i in self.active_gaussian_groups[j]
                        ],
                        dim=0,
                    )
                ]
            return torch.cat(res, dim=0)
        elif level < self.MAX_LEVEL:
            return torch.cat(
                [
                    self.gaussian_groups[i].get_features
                    for i in self.active_gaussian_groups[level]
                ],
                dim=0,
            )
        else:
            raise ValueError("level should be in range [-1, MAX_LEVEL)")

    def get_metric_confidences(self, level=-1):
        levels = range(self.MAX_LEVEL) if level == -1 else (level,)
        values = [
            self.gaussian_groups[group_id].non_trainable_params[
                "metric_confidences"
            ]
            for gaussian_level in levels
            for group_id in self.active_gaussian_groups[gaussian_level]
        ]
        if not values:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        return torch.cat(values, dim=0)

    def get_uncertainty_confidences(self, level=-1):
        levels = range(self.MAX_LEVEL) if level == -1 else (level,)
        values = [
            self.gaussian_groups[group_id].non_trainable_params[
                "uncertainty_confidences"
            ]
            for gaussian_level in levels
            for group_id in self.active_gaussian_groups[gaussian_level]
        ]
        if not values:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        return torch.cat(values, dim=0)

    def _geometry_depth(self, render_pkg):
        if (
            self.causal_dual_responsibility_config["enabled"]
            and self.causal_dual_responsibility_config[
                "geometry_use_metric_depth"
            ]
            and "metric_depth" in render_pkg
        ):
            return render_pkg["metric_depth"]
        return render_pkg["depth"]

    def _geometry_opacity(self, render_pkg):
        if (
            self.causal_dual_responsibility_config["enabled"]
            and self.causal_dual_responsibility_config[
                "geometry_use_metric_depth"
            ]
            and "metric_opacity" in render_pkg
        ):
            return render_pkg["metric_opacity"]
        return render_pkg["opacity"]

        # return torch.cat([self.gaussian_groups[i].get_features for i in self.active_gaussian_groups], dim=0)

    def get_num_active_groups(self, level):
        return len(self.active_gaussian_groups[level])

    def get_num_optimized_groups(self, level):
        return np.sum(
            [
                self.gaussian_groups[i].is_optimize
                for i in self.active_gaussian_groups[level]
            ]
        )

    def get_current_group_size(self):
        return self.gaussian_groups[self.current_gaussian_group].get_num

    def set_vignette_img(self, vignette_img):
        if vignette_img is not None:
            vignette_img = torch.from_numpy(vignette_img).float().to(self.device)
            self.vignette_imgs = [vignette_img.unsqueeze(0)]

            for _ in range(1, Camera.MAX_LEVEL):
                vignette_img = Camera.downsample(vignette_img, mode="color")
                self.vignette_imgs.append(vignette_img.unsqueeze(0))
        else:
            self.vignette_imgs = []

    def get_vignette_img(self, level=0):
        if self.vignette_imgs is None or len(self.vignette_imgs) == 0:
            return None
        return self.vignette_imgs[level]

    def render_3dgs(self, cam, level=0, external_splats=None, return_info=False):
        # means = torch.cat([self.gaussian_groups[i].get_xyz for i in self.active_gaussian_groups], dim=0)
        # scales = torch.cat([self.gaussian_groups[i].get_scaling for i in self.active_gaussian_groups], dim=0)
        # rotations = torch.cat([self.gaussian_groups[i].get_rotation for i in self.active_gaussian_groups], dim=0)
        # opacities = torch.cat([self.gaussian_groups[i].get_opacity for i in self.active_gaussian_groups], dim=0)
        # shs = torch.cat([self.gaussian_groups[i].get_features for i in self.active_gaussian_groups], dim=0)

        means = []
        scales = []
        rotations = []
        opacities = []
        shs = []

        # for i in range(level, self.MAX_LEVEL):
        for i in range(0, self.MAX_LEVEL):
            _means = self.get_xyz(level=i)
            _scales = self.get_scaling(level=i)
            _rotations = self.get_rotation(level=i)
            _opacities = self.get_opacity(level=i)
            _shs = self.get_features(level=i)

            means.append(_means)
            scales.append(_scales)
            rotations.append(_rotations)
            opacities.append(_opacities)
            shs.append(_shs)

            # if i - level <= 1:
            #     means.append(_means)
            #     scales.append(_scales)
            #     rotations.append(_rotations)
            #     opacities.append(_opacities)
            #     shs.append(_shs)
            # else:
            #     means.append(_means.detach())
            #     scales.append(_scales.detach())
            #     rotations.append(_rotations.detach())
            #     opacities.append(_opacities.detach())
            #     shs.append(_shs.detach())

        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)
        opacities = torch.cat(opacities, dim=0)
        shs = torch.cat(shs, dim=0)
        opacities = self._frontview_handoff_opacities(
            opacities, means, scales, [cam], level
        )

        external_splats = self._validated_external_splats(
            external_splats, means.device, means.dtype
        )
        if external_splats is not None:
            means = torch.cat((means, external_splats["means"]), dim=0)
            scales = torch.cat((scales, external_splats["scales"]), dim=0)
            rotations = torch.cat((rotations, external_splats["quats"]), dim=0)
            opacities = torch.cat((opacities, external_splats["opacities"]), dim=0)
            shs = torch.cat((shs, external_splats["shs"]), dim=0)

        rasterize_mode = "antialiased" if self.use_anti_aliasing else "classic"

        route_degrees = self._streaming_appearance_render_degrees(level)
        routed_existing_rows = route_degrees is not None
        metric_confidences = None
        uncertainty_confidences = None
        appearance_confidences = None
        if self.causal_dual_responsibility_config["enabled"]:
            metric_confidences = self.get_metric_confidences()
            uncertainty_confidences = self.get_uncertainty_confidences()
            if external_splats is not None:
                metric_confidences = torch.cat(
                    (
                        metric_confidences,
                        torch.ones(
                            len(external_splats["means"]),
                            device=means.device,
                            dtype=means.dtype,
                        ),
                    )
                )
                uncertainty_confidences = torch.cat(
                    (
                        uncertainty_confidences,
                        torch.ones(
                            len(external_splats["means"]),
                            device=means.device,
                            dtype=means.dtype,
                        ),
                    )
                )
            if (
                level == 0
                and self.causal_dual_responsibility_config[
                    "finite_depth_certificate_enabled"
                ]
                and self.causal_dual_responsibility_config[
                    "finite_depth_preserve_appearance_ownership"
                ]
            ):
                appearance_confidences = uncertainty_confidences
            if route_degrees is None:
                route_degrees = torch.full(
                    (len(means),),
                    int(self.active_sh_degree),
                    device=means.device,
                    dtype=torch.uint8,
                )
            elif external_splats is not None and routed_existing_rows:
                route_degrees = torch.cat(
                    (
                        route_degrees,
                        torch.full(
                            (len(external_splats["means"]),),
                            int(self.active_sh_degree),
                            device=means.device,
                            dtype=torch.uint8,
                        ),
                    )
                )
        if route_degrees is not None and (
            external_splats is None
            or self.causal_dual_responsibility_config["enabled"]
        ):
            render_colors, render_alphas, projection_info = (
                heterogeneous_sh_rasterization(
                    means=means,
                    quats=rotations,
                    scales=scales,
                    opacities=opacities,
                    sh_coefficients=shs,
                    sh_degrees=route_degrees,
                    metric_confidences=metric_confidences,
                    appearance_confidences=appearance_confidences,
                    uncertainty_confidences=uncertainty_confidences,
                    uncertainty_cell_px=(
                        self.frontview_directional_layer.uncertainty_cell_px_for_camera(
                            cam
                        )
                    ),
                    viewmats=cam.get_pose()[None, :, :],
                    Ks=cam.get_int_mat(level)[None, ...],
                    width=cam.get_width(level),
                    height=cam.get_height(level),
                    rasterize_mode=rasterize_mode,
                    near_plane=cam.near,
                    far_plane=cam.far,
                    radius_clip=self.radius_clip,
                    render_mode=self.render_mode,
                    base_degree=self.streaming_appearance_lod_config[
                        "birth_degree"
                    ],
                    target_degree=self.streaming_appearance_lod_config[
                        "target_degree"
                    ],
                )
            )
            self._record_streaming_compute_routing(projection_info)
        else:
            render_colors, render_alphas, projection_info = rasterization(
                means=means,
                quats=rotations,
                scales=scales,
                opacities=opacities,
                colors=shs,
                viewmats=cam.get_pose()[
                    None, :, :
                ],  # pose is already world2cam
                Ks=cam.get_int_mat(level)[None, ...],
                width=cam.get_width(level),
                height=cam.get_height(level),
                rasterize_mode=rasterize_mode,
                near_plane=cam.near,
                far_plane=cam.far,
                radius_clip=self.radius_clip,
                render_mode=self.render_mode,
                sh_degree=self.active_sh_degree,
            )

        assert render_colors.shape[0] == 1, "batch size should be 1"

        colors = render_colors[0]
        metric_depths = None
        if self.causal_dual_responsibility_config["enabled"]:
            if colors.shape[2] != 5:
                raise RuntimeError(
                    "Dual responsibility requires RGB, full depth, and metric depth"
                )
            colors, depths, metric_depths = (
                colors[..., 0:3],
                colors[..., 3:4],
                colors[..., 4:5],
            )
        elif colors.shape[2] == 4:
            colors, depths = colors[..., 0:3], colors[..., 3:4]
        elif colors.shape[2] == 3:
            depths = None
        else:
            assert False, "render_colors should be 3 or 4 channel"

        vig_img = self.get_vignette_img(level)
        if vig_img is not None:
            colors = colors * vig_img[0]

        render_alphas = render_alphas[0]
        metric_opacity = None
        appearance_depth = None
        appearance_opacity = None
        uncertainty_opacity = None
        if self.causal_dual_responsibility_config["enabled"]:
            metric_opacity = projection_info["metric_depth_mass"][0]
            metric_depths = torch.where(
                metric_opacity
                >= float(
                    self.causal_dual_responsibility_config[
                        "minimum_metric_opacity"
                    ]
                ),
                metric_depths,
                torch.zeros_like(metric_depths),
            )
            self.causal_dual_responsibility_stats[
                "metric_depth_render_calls"
            ] += 1
            uncertainty_opacity = projection_info.get("uncertainty_mass")
            if uncertainty_opacity is not None:
                uncertainty_opacity = uncertainty_opacity[0]
            appearance_depth = projection_info.get("appearance_depth")
            appearance_opacity = projection_info.get("appearance_depth_mass")
            if appearance_depth is not None and appearance_opacity is not None:
                appearance_depth = appearance_depth[0]
                appearance_opacity = appearance_opacity[0]
                appearance_depth = torch.where(
                    appearance_opacity
                    >= float(
                        self.causal_dual_responsibility_config[
                            "minimum_metric_opacity"
                        ]
                    ),
                    appearance_depth,
                    torch.zeros_like(appearance_depth),
                )

        colors = colors * cam.exposure_gain / self.scene_exposure_gain
        geometry_colors = colors
        if level == 0:
            geometry_directional_depth = (
                metric_depths
                if self.causal_dual_responsibility_config["enabled"]
                and self.causal_dual_responsibility_config[
                    "directional_use_metric_depth"
                ]
                else depths
            )
            preserve_appearance = (
                self.causal_dual_responsibility_config["enabled"]
                and self.causal_dual_responsibility_config[
                    "finite_depth_certificate_enabled"
                ]
                and self.causal_dual_responsibility_config[
                    "finite_depth_preserve_appearance_ownership"
                ]
            )
            directional_depth = (
                appearance_depth
                if preserve_appearance and appearance_depth is not None
                else geometry_directional_depth
            )
            if metric_opacity is None:
                colors = self.frontview_directional_layer.composite(
                    cam, colors, directional_depth, render_alphas
                )
            else:
                directional_metric_opacity = (
                    appearance_opacity
                    if preserve_appearance and appearance_opacity is not None
                    else metric_opacity
                )
                colors = self.frontview_directional_layer.composite(
                    cam,
                    colors,
                    directional_depth,
                    render_alphas,
                    metric_opacity=directional_metric_opacity,
                    uncertainty_opacity=uncertainty_opacity,
                )

        if depths is not None:
            render_pkg = {
                "render": colors,
                "geometry_render": geometry_colors,
                "depth": depths,
                "opacity": render_alphas,
            }
            if metric_opacity is not None:
                render_pkg["metric_opacity"] = metric_opacity
                render_pkg["metric_depth"] = metric_depths
            if uncertainty_opacity is not None:
                render_pkg["uncertainty_mass"] = uncertainty_opacity
            if appearance_opacity is not None:
                render_pkg["appearance_metric_depth"] = appearance_depth
                render_pkg["appearance_metric_opacity"] = appearance_opacity
        else:
            render_pkg = {
                "render": colors,
                "geometry_render": geometry_colors,
                "opacity": render_alphas,
            }
        if return_info:
            render_pkg["projection_info"] = projection_info

        return render_pkg

    def _validated_external_splats(self, external_splats, device, dtype):
        if external_splats is None:
            return None
        required = {"means", "scales", "quats", "opacities", "shs"}
        if set(external_splats) != required:
            raise ValueError("external_splats must contain exactly {}".format(sorted(required)))
        count = external_splats["means"].shape[0]
        if any(external_splats[name].shape[0] != count for name in required):
            raise ValueError("external_splats tensors must share the first dimension")
        if any(external_splats[name].requires_grad for name in required):
            raise ValueError("external_splats must be detached and non-trainable")
        return {
            name: external_splats[name].detach().to(device=device, dtype=dtype)
            for name in required
        }

    def render_2dgs(self, cam, level=0, external_splats=None):
        # means = torch.cat([self.gaussian_groups[i].get_xyz for i in self.active_gaussian_groups], dim=0)
        # scales = torch.cat([self.gaussian_groups[i].get_scaling for i in self.active_gaussian_groups], dim=0)
        # rotations = torch.cat([self.gaussian_groups[i].get_rotation for i in self.active_gaussian_groups], dim=0)
        # opacities = torch.cat([self.gaussian_groups[i].get_opacity for i in self.active_gaussian_groups], dim=0)
        # shs = torch.cat([self.gaussian_groups[i].get_features for i in self.active_gaussian_groups], dim=0)

        means = []
        scales = []
        rotations = []
        opacities = []
        shs = []

        # for i in range(level, self.MAX_LEVEL):
        for i in range(0, self.MAX_LEVEL):
            _means = self.get_xyz(level=i)
            _scales = self.get_scaling(level=i)
            _rotations = self.get_rotation(level=i)
            _opacities = self.get_opacity(level=i)
            _shs = self.get_features(level=i)

            means.append(_means)
            scales.append(_scales)
            rotations.append(_rotations)
            opacities.append(_opacities)
            shs.append(_shs)

            # if i - level <= 1:
            #     means.append(_means)
            #     scales.append(_scales)
            #     rotations.append(_rotations)
            #     opacities.append(_opacities)
            #     shs.append(_shs)
            # else:
            #     means.append(_means.detach())
            #     scales.append(_scales.detach())
            #     rotations.append(_rotations.detach())
            #     opacities.append(_opacities.detach())
            #     shs.append(_shs.detach())

        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)
        opacities = torch.cat(opacities, dim=0)
        shs = torch.cat(shs, dim=0)
        external_splats = self._validated_external_splats(
            external_splats, means.device, means.dtype
        )
        if external_splats is not None:
            means = torch.cat((means, external_splats["means"]), dim=0)
            scales = torch.cat((scales, external_splats["scales"]), dim=0)
            rotations = torch.cat((rotations, external_splats["quats"]), dim=0)
            opacities = torch.cat((opacities, external_splats["opacities"]), dim=0)
            shs = torch.cat((shs, external_splats["shs"]), dim=0)

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=shs,
            viewmats=cam.get_pose()[
                None, :, :
            ],  # we don't need to inverse the matrix here because the pose is already world2cam
            Ks=cam.get_int_mat(level)[None, ...],  # [1, 3, 3]
            width=cam.get_width(level),
            height=cam.get_height(level),
            near_plane=cam.near,
            far_plane=cam.far,
            radius_clip=self.radius_clip,
            render_mode=self.render_mode,
            sh_degree=self.active_sh_degree,
            distloss=True,
        )

        assert render_colors.shape[0] == 1, "batch size should be 1"

        colors = render_colors[0]
        if colors.shape[2] == 4:
            colors, depths = colors[..., 0:3], colors[..., 3:4]
        elif colors.shape[2] == 3:
            depths = None
        else:
            assert False, "render_colors should be 3 or 4 channel"

        vig_img = self.get_vignette_img(level)
        if vig_img is not None:
            colors = colors * vig_img[0]

        render_alphas = render_alphas[0]

        colors = colors * cam.exposure_gain / self.scene_exposure_gain

        if depths is not None:
            render_pkg = {
                "render": colors,
                "depth": depths,
                "normal": render_normals,
                "normal_from_depth": normals_from_depth,
                "distortion": render_distort,
                "opacity": render_alphas,
            }
        else:
            render_pkg = {
                "render": colors,
                "normal": render_normals,
                "normal_from_depth": normals_from_depth,
                "distortion": render_distort,
                "opacity": render_alphas,
            }

        return render_pkg

    def render(self, cam, level=0, external_splats=None, return_info=False):
        if self.gaussian_type == "2dgs":
            if return_info:
                raise NotImplementedError("Stable detail projection info requires 3DGS")
            return self.render_2dgs(cam, level, external_splats=external_splats)
        elif self.gaussian_type == "3dgs":
            return self.render_3dgs(
                cam,
                level,
                external_splats=external_splats,
                return_info=return_info,
            )
        else:
            raise NotImplementedError

    def render_batch_3dgs(
        self,
        cams,
        random_bg=False,
        detach_gaussians=False,
        level=0,
        external_splats=None,
        appearance_probe=False,
        appearance_sh_degree_override=None,
    ):
        means = []
        scales = []
        rotations = []
        opacities = []
        shs = []

        for i in range(0, self.MAX_LEVEL):
            _means = self.get_xyz(level=i)
            _scales = self.get_scaling(level=i)
            _rotations = self.get_rotation(level=i)
            _opacities = self.get_opacity(level=i)
            _shs = self.get_features(level=i)

            means.append(_means)
            scales.append(_scales)
            rotations.append(_rotations)
            opacities.append(_opacities)
            shs.append(_shs)

        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)
        opacities = torch.cat(opacities, dim=0)
        shs = torch.cat(shs, dim=0)
        opacities = self._frontview_handoff_opacities(
            opacities, means, scales, cams, level
        )

        external_splats = self._validated_external_splats(
            external_splats, means.device, means.dtype
        )
        if external_splats is not None:
            means = torch.cat((means, external_splats["means"]), dim=0)
            scales = torch.cat((scales, external_splats["scales"]), dim=0)
            rotations = torch.cat((rotations, external_splats["quats"]), dim=0)
            opacities = torch.cat((opacities, external_splats["opacities"]), dim=0)
            shs = torch.cat((shs, external_splats["shs"]), dim=0)

        if detach_gaussians:
            means = means.detach()
            scales = scales.detach()
            rotations = rotations.detach()
            opacities = opacities.detach()
            shs = shs.detach()

        rasterize_mode = "antialiased" if self.use_anti_aliasing else "classic"

        poses = torch.stack([cam.get_pose() for cam in cams], dim=0)
        Ks = torch.stack([cam.get_int_mat(level) for cam in cams], dim=0)

        if appearance_sh_degree_override is not None:
            appearance_sh_degree_override = int(appearance_sh_degree_override)
            if not 0 <= appearance_sh_degree_override <= self.active_sh_degree:
                raise ValueError("Appearance SH override exceeds the active degree")
        route_degrees = self._streaming_appearance_render_degrees(level)
        if (
            appearance_sh_degree_override is None
            and route_degrees is not None
            and external_splats is None
        ):
            render_colors, render_alphas, projection_info = (
                heterogeneous_sh_rasterization(
                    means=means,
                    quats=rotations,
                    scales=scales,
                    opacities=opacities,
                    sh_coefficients=shs,
                    sh_degrees=route_degrees,
                    viewmats=poses,
                    Ks=Ks,
                    width=cams[0].get_width(level),
                    height=cams[0].get_height(level),
                    rasterize_mode=rasterize_mode,
                    near_plane=cams[0].near,
                    far_plane=cams[0].far,
                    radius_clip=self.radius_clip,
                    render_mode=self.render_mode,
                    base_degree=self.streaming_appearance_lod_config[
                        "birth_degree"
                    ],
                    target_degree=self.streaming_appearance_lod_config[
                        "target_degree"
                    ],
                    probe_inactive=bool(appearance_probe),
                )
            )
            self._record_streaming_compute_routing(projection_info)
        else:
            render_colors, render_alphas, projection_info = rasterization(
                means=means,
                quats=rotations,
                scales=scales,
                opacities=opacities,
                colors=shs,
                viewmats=poses,  # pose is already world2cam
                Ks=Ks,
                width=cams[0].get_width(level),
                height=cams[0].get_height(level),
                rasterize_mode=rasterize_mode,
                near_plane=cams[0].near,
                far_plane=cams[0].far,
                radius_clip=self.radius_clip,
                render_mode=self.render_mode,
                sh_degree=(
                    self.active_sh_degree
                    if appearance_sh_degree_override is None
                    else appearance_sh_degree_override
                ),
            )

        colors = render_colors
        if colors.shape[3] == 4:
            colors, depths = colors[..., 0:3], colors[..., 3:4]
        elif colors.shape[3] == 3:
            depths = None
        else:
            assert False, "render_colors should be 3 or 4 channel"

        if random_bg:
            bgc = torch.rand((colors.shape[0], 1, 1, 3)).float().to(colors.device)
            colors = colors + bgc * (1 - render_alphas)

        vig_img = self.get_vignette_img(level)
        if vig_img is not None:
            colors = colors * vig_img[0]

        exposure_gain = (
            torch.from_numpy(
                np.array([cam.exposure_gain for cam in cams]).reshape(-1, 1, 1, 1)
            )
            .float()
            .to(self.device)
        )
        colors = colors * exposure_gain / self.scene_exposure_gain

        if depths is not None:
            render_pkg = {
                "render": colors,
                "depth": depths,
                "opacity": render_alphas,
                "projection_info": projection_info,
            }
        else:
            render_pkg = {
                "render": colors,
                "opacity": render_alphas,
                "projection_info": projection_info,
            }

        return render_pkg

    def render_batch_2dgs(
        self, cams, random_bg=False, detach_gaussians=False, level=0, external_splats=None
    ):
        means = []
        scales = []
        rotations = []
        opacities = []
        shs = []

        for i in range(0, self.MAX_LEVEL):
            _means = self.get_xyz(level=i)
            _scales = self.get_scaling(level=i)
            _rotations = self.get_rotation(level=i)
            _opacities = self.get_opacity(level=i)
            _shs = self.get_features(level=i)

            means.append(_means)
            scales.append(_scales)
            rotations.append(_rotations)
            opacities.append(_opacities)
            shs.append(_shs)

        means = torch.cat(means, dim=0)
        scales = torch.cat(scales, dim=0)
        rotations = torch.cat(rotations, dim=0)
        opacities = torch.cat(opacities, dim=0)
        shs = torch.cat(shs, dim=0)

        external_splats = self._validated_external_splats(
            external_splats, means.device, means.dtype
        )
        if external_splats is not None:
            means = torch.cat((means, external_splats["means"]), dim=0)
            scales = torch.cat((scales, external_splats["scales"]), dim=0)
            rotations = torch.cat((rotations, external_splats["quats"]), dim=0)
            opacities = torch.cat((opacities, external_splats["opacities"]), dim=0)
            shs = torch.cat((shs, external_splats["shs"]), dim=0)

        if detach_gaussians:
            means = means.detach()
            scales = scales.detach()
            rotations = rotations.detach()
            opacities = opacities.detach()
            shs = shs.detach()

        poses = torch.stack([cam.get_pose() for cam in cams], dim=0)
        Ks = torch.stack([cam.get_int_mat(level) for cam in cams], dim=0)

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=shs,
            viewmats=poses,  # we don't need to inverse the matrix here because the pose is already world2cam
            Ks=Ks,  # [N, 3, 3]
            width=cams[0].get_width(level),
            height=cams[0].get_height(level),
            near_plane=cams[0].near,
            far_plane=cams[0].far,
            radius_clip=self.radius_clip,
            render_mode=self.render_mode,
            sh_degree=self.active_sh_degree,
            distloss=True,
        )

        if normals_from_depth.ndim == 3:
            normals_from_depth = normals_from_depth.unsqueeze(0)

        colors = render_colors
        if colors.shape[3] == 4:
            colors, depths = colors[..., 0:3], colors[..., 3:4]
        elif colors.shape[3] == 3:
            depths = None
        else:
            assert False, "render_colors should be 3 or 4 channel"

        if random_bg:
            bgc = torch.rand((colors.shape[0], 1, 1, 3)).float().to(colors.device)
            colors = colors + bgc * (1 - render_alphas)

        vig_img = self.get_vignette_img(level)
        if vig_img is not None:
            colors = colors * vig_img[0]

        exposure_gain = (
            torch.from_numpy(
                np.array([cam.exposure_gain for cam in cams]).reshape(-1, 1, 1, 1)
            )
            .float()
            .to(self.device)
        )
        colors = colors * exposure_gain / self.scene_exposure_gain

        if depths is not None:
            render_pkg = {
                "render": colors,
                "depth": depths,
                "normal": render_normals,
                "normal_from_depth": normals_from_depth,
                "distortion": render_distort,
                "opacity": render_alphas,
            }
        else:
            render_pkg = {
                "render": colors,
                "normal": render_normals,
                "normal_from_depth": normals_from_depth,
                "distortion": render_distort,
                "opacity": render_alphas,
            }

        return render_pkg

    def render_batch(
        self,
        cams,
        random_bg=False,
        detach_gaussians=False,
        level=0,
        external_splats=None,
        appearance_probe=False,
        appearance_sh_degree_override=None,
    ):
        if self.gaussian_type == "2dgs":
            if appearance_sh_degree_override is not None:
                raise ValueError("Appearance SH override requires 3DGS")
            return self.render_batch_2dgs(
                cams,
                random_bg=random_bg,
                detach_gaussians=detach_gaussians,
                level=level,
                external_splats=external_splats,
            )
        elif self.gaussian_type == "3dgs":
            return self.render_batch_3dgs(
                cams,
                random_bg=random_bg,
                detach_gaussians=detach_gaussians,
                level=level,
                external_splats=external_splats,
                appearance_probe=appearance_probe,
                appearance_sh_degree_override=appearance_sh_degree_override,
            )
        else:
            raise NotImplementedError

    def _streaming_appearance_render_degrees(self, level):
        config = self.streaming_appearance_lod_config
        if not (
            config["enabled"]
            and config["compute_routing"]
            and self.gaussian_type == "3dgs"
            and self.active_sh_degree == config["target_degree"]
            and self.streaming_appearance_lod_stats["evidence_updates"]
            >= config["compute_routing_warmup_evidence_updates"]
        ):
            return None
        if config["birth_degree"] != 2 or config["target_degree"] != 3:
            raise ValueError("Compute-routed TGBR currently supports SH2 -> SH3")
        degrees = [
            self.gaussian_groups[group_id].non_trainable_params[
                "appearance_sh_degree"
            ]
            for gaussian_level in range(self.MAX_LEVEL)
            for group_id in self.active_gaussian_groups[gaussian_level]
        ]
        if not degrees:
            return torch.empty(0, device=self.device, dtype=torch.uint8)
        return torch.cat(degrees, dim=0)

    @torch.no_grad()
    def _record_streaming_compute_routing(self, projection_info):
        route = projection_info.get("heterogeneous_sh")
        if route is None:
            return
        stats = self.streaming_compute_routing_stats
        stats["render_calls"] += 1
        stats["rasterization_calls"] += 1
        for key in (
            "packed_rows",
            "base_rows",
            "promoted_target_rows",
            "probe_rows",
            "target_rows",
            "skipped_target_band_rows",
        ):
            stats[key] = stats[key] + route[key]

    def _managed_active_group_ids(self):
        """Return active groups owned by their local optimizer exactly once."""
        seen = set()
        result = []
        for level in range(self.MAX_LEVEL):
            for group_id in self.active_gaussian_groups[level]:
                if group_id in seen or group_id in self.progressive_group_ids:
                    continue
                if self.gaussian_groups[group_id].splats is None:
                    continue
                seen.add(group_id)
                result.append(group_id)
        return result

    def reset_optimizer(self, config, BS):
        for i in self._managed_active_group_ids():
            self.gaussian_groups[i].reset_optimizer(config, BS)
            self._apply_frozen_group_learning_rates(i)
        self._reset_progressive_optimizers(config, BS)

    def _apply_frozen_group_learning_rates(self, group_id):
        group = self.gaussian_groups[group_id]
        frozen = []
        if group_id in self.frozen_position_group_ids:
            frozen.append("means")
        if group_id in self.frozen_geometry_group_ids:
            frozen.extend(("means", "scales", "quats"))
        for name in set(frozen):
            if name in group.optimizers:
                group.update_opt_lr(0.0, name)

    def freeze_group_positions(self, group_id):
        """Keep certified world positions fixed while appearance/footprint adapt."""
        if group_id not in self.valid_groups:
            raise ValueError("Cannot freeze an inactive Gaussian group")
        self.frozen_position_group_ids.add(int(group_id))
        self._apply_frozen_group_learning_rates(int(group_id))

    def bound_group_scale_expansion(self, group_id, max_expansion):
        """Bound all rows in one isolated newborn group to their birth footprints."""

        if group_id not in self.valid_groups:
            raise ValueError("Cannot bound an inactive Gaussian group")
        factor = float(max_expansion)
        if factor < 1.0:
            raise ValueError("Scale expansion bound must be at least one")
        group = self.gaussian_groups[int(group_id)]
        group.non_trainable_params["max_scale_expansions"].fill_(factor)
        group.constrain_scale_expansion()

    def isolate_optimization_to_group(self, group_id):
        """Temporarily disable gradients for every managed group except one."""

        group_id = int(group_id)
        if group_id not in self.valid_groups:
            raise ValueError("Cannot isolate an inactive Gaussian group")
        disabled = []
        for candidate in self._managed_active_group_ids():
            if candidate != group_id and self.remove_optimization(candidate):
                disabled.append(candidate)
        self.add_optimization(group_id)
        return disabled

    def restore_group_optimization(self, group_ids):
        for group_id in group_ids:
            self.add_optimization(int(group_id))

    def get_group_num_gaussians(self, group_id):
        group_id = int(group_id)
        if group_id not in self.valid_groups:
            return 0
        return self.gaussian_groups[group_id].get_num

    def bound_group_positions(self, group_id, max_displacement):
        """Allow local RGB fitting without losing a certified canonical identity."""
        if group_id not in self.valid_groups:
            raise ValueError("Cannot bound an inactive Gaussian group")
        self.gaussian_groups[int(group_id)].set_mean_trust_region(max_displacement)

    def freeze_group_geometry(self, group_id):
        """Keep a detail group's sparse-track geometry fixed during RGB fitting."""
        if group_id not in self.valid_groups:
            raise ValueError("Cannot freeze an inactive Gaussian group")
        self.frozen_geometry_group_ids.add(int(group_id))
        self._apply_frozen_group_learning_rates(int(group_id))

    def set_parameter_optimizer_lr(self, name, learning_rate):
        """Set one parameter block LR across active groups for staged refinement."""

        value = float(learning_rate)
        for group_id in self._managed_active_group_ids():
            optimizer = self.gaussian_groups[group_id].optimizers.get(name)
            if optimizer is not None:
                optimizer.param_groups[0]["lr"] = value
        optimizer = self.progressive_optimizers.get(name)
        if optimizer is not None:
            optimizer.param_groups[0]["lr"] = value

    @torch.no_grad()
    def configure_sh_degree_masks(self, degrees, zero_inactive=True):
        """Assign a per-Gaussian SH ceiling in render order."""

        degrees = torch.as_tensor(degrees, device=self.device, dtype=torch.uint8)
        if degrees.ndim != 1 or degrees.numel() != self.get_num_gaussians:
            raise ValueError("SH degree assignments must match the active Gaussian count")
        if torch.any(degrees > self.max_sh_degree):
            raise ValueError("SH degree assignment exceeds Model.sh_degree")

        masks = {}
        seen = set()
        cursor = 0
        coefficient_count = (self.max_sh_degree + 1) ** 2 - 1
        for level in range(self.MAX_LEVEL):
            for group_id in self.active_gaussian_groups[level]:
                if group_id in seen:
                    continue
                seen.add(group_id)
                group = self.gaussian_groups[group_id]
                count = group.get_num
                group_degrees = degrees[cursor : cursor + count]
                cursor += count
                mask = torch.zeros(
                    (count, coefficient_count, 1),
                    device=group.splats["shN"].device,
                    dtype=group.splats["shN"].dtype,
                )
                for degree in range(1, self.max_sh_degree + 1):
                    begin, end = sh_band_bounds(degree)
                    if begin >= coefficient_count:
                        break
                    mask[:, begin : min(end, coefficient_count)] = (
                        group_degrees >= degree
                    ).reshape(-1, 1, 1)
                masks[int(group_id)] = mask
                if zero_inactive:
                    group.splats["shN"].mul_(mask)
        if cursor != degrees.numel():
            raise RuntimeError("SH degree assignments did not consume render-order rows")
        self.sh_degree_masks = masks

    def _streaming_appearance_group_records(self):
        records = []
        seen = set()
        cursor = 0
        for level in range(self.MAX_LEVEL):
            for group_id in self.active_gaussian_groups[level]:
                if group_id in seen:
                    continue
                seen.add(group_id)
                group = self.gaussian_groups[group_id]
                if group.splats is None:
                    continue
                count = group.get_num
                records.append((int(group_id), group, cursor, cursor + count))
                cursor += count
        return records

    @torch.no_grad()
    def observe_streaming_appearance_lod(
        self, projection_info, cameras, current_camera_index
    ):
        config = self.streaming_appearance_lod_config
        if not config["enabled"]:
            return None
        if projection_info is None or not cameras:
            return None
        gaussian_ids = projection_info.get("gaussian_ids")
        camera_ids = projection_info.get("camera_ids")
        radii = projection_info.get("radii")
        depths = projection_info.get("depths")
        if gaussian_ids is None or camera_ids is None or radii is None:
            raise ValueError(
                "Packed projection info is required for StreamingAppearanceLOD"
            )
        gaussian_ids = gaussian_ids.reshape(-1).long()
        camera_ids = camera_ids.reshape(-1).long()
        if radii.ndim > 1:
            radii = radii.amax(dim=-1)
        radii = radii.reshape(-1).float()
        records = self._streaming_appearance_group_records()
        gaussian_count = records[-1][3] if records else 0
        gradient_mode = config["selection_mode"] in {
            "gradient",
            "gradient_agreement",
            "gradient_shuffled",
        }
        valid = (
            (gaussian_ids >= 0)
            & (gaussian_ids < gaussian_count)
            & torch.isfinite(radii)
            & (radii > 0)
        )
        if not gradient_mode:
            valid &= camera_ids == int(current_camera_index)
        if depths is not None:
            depths = depths.reshape(-1)
            valid &= torch.isfinite(depths) & (depths > 0)
        if not torch.any(valid):
            return None

        gaussian_ids = gaussian_ids[valid]
        radii = radii[valid]
        unique_ids, inverse = torch.unique(gaussian_ids, return_inverse=True)
        unique_radii = None
        camera_center = None
        if not gradient_mode:
            unique_radii = torch.zeros(
                unique_ids.numel(), device=radii.device, dtype=radii.dtype
            )
            unique_radii.scatter_reduce_(
                0, inverse, radii, reduce="amax", include_self=False
            )
            pose = cameras[int(current_camera_index)].get_pose().detach().to(radii)
            camera_center = -pose[:3, :3].T @ pose[:3, 3]

        observed_rows = 0
        gradient_signal_rows = 0
        gradient_positive_rows = 0
        gradient_signal_sum = 0.0
        gradient_signal_max = 0.0
        band_begin, band_end = sh_band_bounds(config["target_degree"])
        decay = float(config["utility_ema_decay"])
        if gradient_mode:
            band_width = (band_end - band_begin) * 3
            gradient_ema_dtype = (
                torch.float16
                if config["gradient_ema_dtype"] == "float16"
                else torch.float32
            )
            for _, group, _, _ in records:
                if "appearance_band_gradient_ema" not in group.non_trainable_params:
                    group.non_trainable_params[
                        "appearance_band_gradient_ema"
                    ] = torch.zeros(
                        (group.get_num, band_width),
                        device=self.device,
                        dtype=gradient_ema_dtype,
                    )
        for _, group, begin, end in records:
            selected = (unique_ids >= begin) & (unique_ids < end)
            if not torch.any(selected):
                continue
            local_ids = unique_ids[selected] - begin
            if gradient_mode:
                gradient = group.splats["shN"].grad
                if gradient is None or gradient.shape[1] < band_end:
                    continue
                signal = gradient[local_ids, band_begin:band_end].float()
                signal_vector = signal.reshape(signal.shape[0], -1)
                signal = torch.sum(signal.square(), dim=(1, 2))
                signal = torch.where(
                    torch.isfinite(signal), signal, torch.zeros_like(signal)
                )
                gradient_signal_rows += int(signal.numel())
                gradient_positive_rows += int((signal > 0).sum().item())
                gradient_signal_sum += float(signal.sum().item())
                if signal.numel():
                    gradient_signal_max = max(
                        gradient_signal_max, float(signal.max().item())
                    )
                utility = group.non_trainable_params[
                    "appearance_band_utility_ema"
                ]
                utility[local_ids] = (
                    decay * utility[local_ids] + (1.0 - decay) * signal
                )
                gradient_ema = group.non_trainable_params.get(
                    "appearance_band_gradient_ema"
                )
                if gradient_ema.shape[1] != signal_vector.shape[1]:
                    raise RuntimeError("Streaming SH gradient band width changed")
                scaled_signal_vector = signal_vector * float(
                    config["gradient_ema_scale"]
                )
                gradient_ema[local_ids] = (
                    decay * gradient_ema[local_ids]
                    + (1.0 - decay) * scaled_signal_vector
                ).to(dtype=gradient_ema.dtype)
                group.non_trainable_params["appearance_view_count"].index_add_(
                    0,
                    local_ids,
                    torch.ones_like(local_ids, dtype=torch.int32),
                )
            else:
                local_radii = unique_radii[selected]
                directions = torch.nn.functional.normalize(
                    camera_center.reshape(1, 3) - group.get_xyz[local_ids],
                    dim=-1,
                    eps=1.0e-8,
                )
                group.non_trainable_params["appearance_view_count"].index_add_(
                    0,
                    local_ids,
                    torch.ones_like(local_ids, dtype=torch.int32),
                )
                group.non_trainable_params["appearance_radius_sum"].index_add_(
                    0, local_ids, local_radii
                )
                group.non_trainable_params["appearance_direction_sum"].index_add_(
                    0, local_ids, directions
                )
            observed_rows += int(local_ids.numel())

        stats = self.streaming_appearance_lod_stats
        stats["evidence_updates"] += 1
        stats["observed_rows"] += observed_rows
        stats["gradient_signal_rows"] += gradient_signal_rows
        stats["gradient_positive_rows"] += gradient_positive_rows
        stats["gradient_signal_sum"] += gradient_signal_sum
        stats["gradient_signal_max"] = max(
            float(stats["gradient_signal_max"]), gradient_signal_max
        )
        if stats["evidence_updates"] % int(config["promotion_interval"]) == 0:
            degrees = torch.cat(
                [
                    group.non_trainable_params["appearance_sh_degree"]
                    for _, group, _, _ in records
                ]
            )
            counts = torch.cat(
                [
                    group.non_trainable_params["appearance_view_count"]
                    for _, group, _, _ in records
                ]
            )
            if config["selection_mode"] == "gradient_agreement":
                gradient_ema = torch.cat(
                    [
                        group.non_trainable_params[
                            "appearance_band_gradient_ema"
                        ]
                        for _, group, _, _ in records
                    ]
                )
                degrees, selected = select_gradient_agreement_promotions(
                    degrees,
                    gradient_ema,
                    counts,
                    config,
                )
            elif gradient_mode:
                utility = torch.cat(
                    [
                        group.non_trainable_params[
                            "appearance_band_utility_ema"
                        ]
                        for _, group, _, _ in records
                    ]
                )
                degrees, selected = select_gradient_promotions(
                    degrees,
                    utility,
                    counts,
                    config,
                    update_index=stats["promotion_updates"],
                )
            else:
                radius_sum = torch.cat(
                    [
                        group.non_trainable_params["appearance_radius_sum"]
                        for _, group, _, _ in records
                    ]
                )
                direction_sum = torch.cat(
                    [
                        group.non_trainable_params["appearance_direction_sum"]
                        for _, group, _, _ in records
                    ]
                )
                degrees, selected = select_monotonic_promotions(
                    degrees,
                    counts,
                    radius_sum,
                    direction_sum,
                    config,
                    update_index=stats["promotion_updates"],
                )
            cursor = 0
            for _, group, _, _ in records:
                count = group.get_num
                group.non_trainable_params["appearance_sh_degree"].copy_(
                    degrees[cursor : cursor + count]
                )
                cursor += count
            stats["promotion_updates"] += 1
            stats["promoted_rows"] += int(selected.numel())
            stats["current_target_fraction"] = (
                float((degrees >= int(config["target_degree"])).sum().item())
                / float(degrees.numel())
                if degrees.numel()
                else 0.0
            )
        return observed_rows

    def _mask_streaming_appearance_tensor(self, group, tensor):
        if not self.streaming_appearance_lod_config["enabled"]:
            return
        degrees = group.non_trainable_params["appearance_sh_degree"]
        degrees.clamp_(min=int(self.streaming_appearance_lod_config["birth_degree"]))
        coefficient_count = tensor.shape[1]
        for degree in range(1, self.max_sh_degree + 1):
            begin, end = sh_band_bounds(degree)
            if begin >= coefficient_count:
                break
            tensor[degrees < degree, begin : min(end, coefficient_count)] = 0

    def mask_sh_degree_gradients(self):
        """Prevent unavailable SH bands from receiving optimizer updates."""

        for group_id, mask in self.sh_degree_masks.items():
            gradient = self.gaussian_groups[group_id].splats["shN"].grad
            if gradient is not None:
                gradient.mul_(mask)
        streaming_config = getattr(
            self, "streaming_appearance_lod_config", {"enabled": False}
        )
        if streaming_config["enabled"]:
            for _, group, _, _ in self._streaming_appearance_group_records():
                gradient = group.splats["shN"].grad
                if gradient is not None:
                    self._mask_streaming_appearance_tensor(group, gradient)

    @torch.no_grad()
    def streaming_high_band_gradient_ratio(self):
        """Return per-coefficient target-band energy over lower-band energy."""

        config = self.streaming_appearance_lod_config
        if not config["enabled"]:
            return float("nan")
        band_begin, band_end = sh_band_bounds(config["target_degree"])
        lower_band_energy = torch.zeros((), device=self.device, dtype=torch.float64)
        high_band_energy = torch.zeros((), device=self.device, dtype=torch.float64)
        lower_band_values = 0
        high_band_values = 0
        observed = False
        for _, group, _, _ in self._streaming_appearance_group_records():
            gradient = group.splats["shN"].grad
            if gradient is None:
                continue
            observed = True
            gradient = gradient.detach()
            if gradient.shape[1] >= band_end:
                lower = gradient[:, :band_begin]
                lower_norm = torch.linalg.vector_norm(lower.float()).double()
                lower_band_energy.add_(lower_norm.square())
                lower_band_values += int(lower.numel())
                high = gradient[:, band_begin:band_end]
                high_norm = torch.linalg.vector_norm(
                    high.float()
                ).double()
                high_band_energy.add_(high_norm.square())
                high_band_values += int(high.numel())
        if not observed:
            return float("nan")
        if lower_band_values <= 0 or high_band_values <= 0:
            return float("nan")
        lower_mean_energy = float(lower_band_energy.item()) / float(
            lower_band_values
        )
        high_mean_energy = float(high_band_energy.item()) / float(high_band_values)
        if high_mean_energy <= 0.0:
            return 0.0
        if lower_mean_energy <= 0.0:
            return float("inf")
        return max(0.0, high_mean_energy / lower_mean_energy)

    @torch.no_grad()
    def constrain_sh_degree_masks(self):
        """Keep inactive SH bands exactly zero, including after Adam momentum."""

        for group_id, mask in self.sh_degree_masks.items():
            self.gaussian_groups[group_id].splats["shN"].mul_(mask)
        streaming_config = getattr(
            self, "streaming_appearance_lod_config", {"enabled": False}
        )
        if streaming_config["enabled"]:
            for _, group, _, _ in self._streaming_appearance_group_records():
                self._mask_streaming_appearance_tensor(group, group.splats["shN"])

    def step_all_lr(self):
        for i in self._managed_active_group_ids():
            self.gaussian_groups[i].step_lr()
            self._apply_frozen_group_learning_rates(i)
        if self.progressive_scheduler_args:
            self.progressive_lr_step += 1
            for name, scheduler in self.progressive_scheduler_args.items():
                if name in self.progressive_optimizers:
                    scene_scale = self.scene_scale if name != "quats" else 1.0
                    self.progressive_optimizers[name].param_groups[0]["lr"] = (
                        scheduler(self.progressive_lr_step)
                        * scene_scale
                        * math.sqrt(self.BS)
                    )

    def get_avg_pos_lr(self):
        avg_pos_lr = 0
        managed_group_ids = self._managed_active_group_ids()
        for i in managed_group_ids:
            avg_pos_lr += self.gaussian_groups[i].get_pos_lr()
        for i in self.progressive_group_ids:
            if "means" in self.progressive_optimizers:
                avg_pos_lr += self.progressive_optimizers["means"].param_groups[0]["lr"]
        denominator = len(managed_group_ids) + len(self.progressive_group_ids)
        return avg_pos_lr / max(1, denominator)

    def update(
        self,
        replay_steps=0,
        gradient_decay=1.0,
        learning_rate_scale=1.0,
    ):
        replay_steps = int(replay_steps)
        gradient_decay = float(gradient_decay)
        if replay_steps < 0:
            raise ValueError("Optimizer replay steps must be non-negative")
        if not math.isfinite(gradient_decay) or not 0.0 <= gradient_decay <= 1.0:
            raise ValueError("Optimizer replay gradient decay must be in [0, 1]")
        learning_rate_scale = float(learning_rate_scale)
        if not math.isfinite(learning_rate_scale) or learning_rate_scale <= 0.0:
            raise ValueError("Optimizer learning-rate scale must be finite and positive")

        active_groups = [
            self.gaussian_groups[index]
            for index in self._managed_active_group_ids()
            if self.gaussian_groups[index].is_optimize
        ]
        learning_rates = []
        if learning_rate_scale != 1.0:
            for group in active_groups:
                for optimizer in group.optimizers.values():
                    for parameter_group in optimizer.param_groups:
                        learning_rates.append(
                            (parameter_group, float(parameter_group["lr"]))
                        )
                        parameter_group["lr"] *= learning_rate_scale
        total_steps = replay_steps + 1
        post_step_records = self._frontview_post_step_projection
        try:
            for replay_index in range(total_steps):
                for group in active_groups:
                    record = post_step_records.get(id(group))
                    previous_means = (
                        group.splats["means"][record[0]].detach().clone()
                        if record is not None
                        else None
                    )
                    group.optimizer_step()
                    if record is not None:
                        indices, reference_rays, radial_scales = record
                        with torch.no_grad():
                            group.splats["means"][indices] = project_raywise_update(
                                previous_means,
                                group.splats["means"][indices],
                                reference_rays,
                                radial_scales,
                            )
                        group.constrain_mean_displacement()
                        stats = self.frontview_observability_stats
                        stats["post_step_projection_calls"] += 1
                        stats["post_step_projected_rows"] += int(indices.numel())
                for optimizer in self.progressive_optimizers.values():
                    optimizer.step()
                self.constrain_sh_degree_masks()
                if replay_index + 1 < total_steps:
                    for group in active_groups:
                        group.scale_optimizer_gradients(gradient_decay)
                    for optimizer in self.progressive_optimizers.values():
                        for parameter_group in optimizer.param_groups:
                            for parameter in parameter_group["params"]:
                                if parameter.grad is not None:
                                    parameter.grad.mul_(gradient_decay)
        finally:
            self._frontview_post_step_projection = {}
            for parameter_group, learning_rate in learning_rates:
                parameter_group["lr"] = learning_rate

        for group in active_groups:
            group.zero_optimizer_gradients()
        for optimizer in self.progressive_optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def precondition_frontview_mean_gradients(
        self, cameras, config, update_evidence=True
    ):
        """Update causal evidence and control the weak birth-ray gauge."""

        if not bool(config.get("enabled", False)) or not cameras:
            self._frontview_post_step_projection = {}
            return None
        records = []
        anchors = []
        reference_rays = []
        information = []
        gaussian_scales = []
        reference_ranges = []
        birth_log_depth_stds = []
        gradients = []
        for group_id in self._managed_active_group_ids():
            group = self.gaussian_groups[group_id]
            gradient = group.splats["means"].grad
            if gradient is None:
                continue
            mask = group.non_trainable_params["directional_observability_mask"]
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            records.append((group, indices, int(indices.numel())))
            if update_evidence:
                anchors.append(group.non_trainable_params["mean_anchors"][indices])
            reference_rays.append(
                group.non_trainable_params["reference_rays"][indices]
            )
            reference_ranges.append(
                torch.linalg.vector_norm(
                    group.non_trainable_params["mean_anchors"][indices]
                    - group.non_trainable_params["reference_camera_centers"][indices],
                    dim=1,
                )
            )
            gaussian_scales.append(group.get_scaling[indices])
            information.append(
                group.non_trainable_params["max_parallax_sin2"][indices]
            )
            birth_log_depth_stds.append(
                group.non_trainable_params["birth_log_depth_stds"][indices]
            )
            gradients.append(gradient[indices])
        if not records:
            self._frontview_post_step_projection = {}
            return None

        reference_rays = torch.cat(reference_rays, dim=0)
        max_parallax_sin2 = torch.cat(information, dim=0)
        gaussian_scales = torch.cat(gaussian_scales, dim=0)
        reference_ranges = torch.cat(reference_ranges, dim=0)
        birth_log_depth_stds = torch.cat(birth_log_depth_stds, dim=0)
        gradients = torch.cat(gradients, dim=0)
        if update_evidence:
            anchors = torch.cat(anchors, dim=0)
            margin = float(config["visibility_margin_px"])
            for camera in cameras:
                pose = camera.get_pose().detach().to(
                    device=anchors.device, dtype=anchors.dtype
                )
                intrinsics = camera.get_int_mat(0).detach().to(
                    device=anchors.device, dtype=anchors.dtype
                )
                rotation = pose[:3, :3]
                translation = pose[:3, 3]
                camera_points = anchors @ rotation.T + translation
                depth = camera_points[:, 2]
                projected = camera_points @ intrinsics.T
                u = projected[:, 0] / torch.clamp(depth, min=1.0e-8)
                v = projected[:, 1] / torch.clamp(depth, min=1.0e-8)
                visible = (
                    (depth > float(camera.near))
                    & (depth < float(camera.far))
                    & (u >= -margin)
                    & (u < float(camera.get_width(0)) + margin)
                    & (v >= -margin)
                    & (v < float(camera.get_height(0)) + margin)
                )
                camera_center = -(rotation.T @ translation)
                current_rays = torch.nn.functional.normalize(
                    anchors - camera_center.reshape(1, 3), dim=1, eps=1.0e-8
                )
                cosine = torch.sum(reference_rays * current_rays, dim=1)
                parallax_sin2 = torch.clamp(1.0 - cosine.square(), 0.0, 1.0)
                max_parallax_sin2 = torch.maximum(
                    max_parallax_sin2,
                    torch.where(
                        visible, parallax_sin2, torch.zeros_like(parallax_sin2)
                    ),
                )

        if config.get("learning_scale_mode") == "posterior_information":
            radial_scales = posterior_information_scale(
                max_parallax_sin2,
                gaussian_scales,
                reference_ranges,
                birth_log_depth_stds,
            )
        elif config.get("learning_scale_mode") == "resolution_information":
            radial_scales = resolution_information_scale(
                max_parallax_sin2,
                gaussian_scales,
                reference_ranges,
            )
        else:
            radial_scales = parallax_learning_scale(
                max_parallax_sin2,
                float(config["min_ray_lr_scale"]),
                float(config["unlock_parallax_deg"]),
            )
        if bool(config.get("shuffle_evidence", False)) and radial_scales.numel() > 1:
            generator = torch.Generator(device=radial_scales.device)
            generator.manual_seed(
                int(config.get("shuffle_seed", 42))
                + int(self.frontview_observability_stats["calls"])
            )
            radial_scales = radial_scales[
                torch.randperm(
                    radial_scales.numel(),
                    generator=generator,
                    device=radial_scales.device,
                )
            ]
        optimization_mode = config.get(
            "optimization_mode", "gradient_preconditioner"
        )
        adjusted = None
        if optimization_mode == "gradient_preconditioner":
            adjusted = precondition_raywise_gradient(
                gradients, reference_rays, radial_scales
            )

        cursor = 0
        post_step_records = {}
        for group, indices, count in records:
            end = cursor + count
            group.non_trainable_params["max_parallax_sin2"][indices] = (
                max_parallax_sin2[cursor:end]
            )
            if optimization_mode == "gradient_preconditioner":
                group.splats["means"].grad[indices] = adjusted[cursor:end]
            else:
                post_step_records[id(group)] = (
                    indices,
                    reference_rays[cursor:end],
                    radial_scales[cursor:end],
                )
            cursor = end
        self._frontview_post_step_projection = post_step_records

        rows = int(radial_scales.numel())
        mean_scale = float(radial_scales.mean().item())
        unlocked_fraction = float((radial_scales >= 0.999).float().mean().item())
        stats = self.frontview_observability_stats
        stats["learning_scale_mode"] = config.get(
            "learning_scale_mode", "fixed_angle"
        )
        stats["optimization_mode"] = optimization_mode
        stats["responsibility_scope"] = config.get(
            "responsibility_scope", "all_depthcov"
        )
        stats["calls"] += 1
        stats["rows"] += rows
        stats["radial_scale_sum"] += mean_scale * rows
        stats["last_rows"] = rows
        stats["last_mean_radial_scale"] = mean_scale
        stats["last_unlocked_fraction"] = unlocked_fraction
        stats["evidence_updates"] += int(update_evidence)
        return {
            "rows": rows,
            "mean_radial_scale": mean_scale,
            "unlocked_fraction": unlocked_fraction,
        }

    @torch.no_grad()
    def update_dynamic_footprint_trust(self, cameras):
        """Release only certificate-owned scale caps as evidence accumulates."""

        config = self.frontview_far_field_config
        if not bool(config.get("footprint_trust_dynamic_update", False)) or not cameras:
            return 0

        records = []
        anchors = []
        reference_centers = []
        reference_rays = []
        max_parallax = []
        target_scales = []
        log_depth_stds = []
        evidence_pending = []
        ownership = []
        for group_id in self._managed_active_group_ids():
            group = self.gaussian_groups[group_id]
            params = group.non_trainable_params
            certificate_rows = torch.isfinite(params["footprint_target_scales"])
            indices = torch.nonzero(certificate_rows, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            records.append((group, indices, int(indices.numel())))
            anchors.append(params["mean_anchors"][indices])
            reference_centers.append(params["reference_camera_centers"][indices])
            reference_rays.append(params["reference_rays"][indices])
            max_parallax.append(params["max_parallax_sin2"][indices])
            target_scales.append(params["footprint_target_scales"][indices])
            log_depth_stds.append(params["birth_log_depth_stds"][indices])
            evidence_pending.append(
                params["footprint_evidence_pending_mask"][indices]
            )
            ownership.append(params["footprint_trust_mask"][indices])
        if not records:
            return 0

        anchors = torch.cat(anchors, dim=0)
        reference_centers = torch.cat(reference_centers, dim=0)
        reference_rays = torch.cat(reference_rays, dim=0)
        max_parallax_sin2 = torch.cat(max_parallax, dim=0)
        target_scales = torch.cat(target_scales, dim=0)
        birth_log_depth_stds = torch.cat(log_depth_stds, dim=0)
        evidence_pending = torch.cat(evidence_pending, dim=0)
        ownership = torch.cat(ownership, dim=0)
        margin = float(config.get("visibility_margin_px", 16.0))
        for camera in cameras:
            pose = camera.get_pose().detach().to(
                device=anchors.device, dtype=anchors.dtype
            )
            intrinsics = camera.get_int_mat(0).detach().to(
                device=anchors.device, dtype=anchors.dtype
            )
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            camera_points = anchors @ rotation.T + translation
            depth = camera_points[:, 2]
            projected = camera_points @ intrinsics.T
            u = projected[:, 0] / torch.clamp(depth, min=1.0e-8)
            v = projected[:, 1] / torch.clamp(depth, min=1.0e-8)
            visible = (
                (depth > float(camera.near))
                & (depth < float(camera.far))
                & (u >= -margin)
                & (u < float(camera.get_width(0)) + margin)
                & (v >= -margin)
                & (v < float(camera.get_height(0)) + margin)
            )
            camera_center = -(rotation.T @ translation)
            current_rays = torch.nn.functional.normalize(
                anchors - camera_center.reshape(1, 3), dim=1, eps=1.0e-8
            )
            cosine = torch.sum(reference_rays * current_rays, dim=1)
            observed = torch.clamp(1.0 - cosine.square(), 0.0, 1.0)
            max_parallax_sin2 = torch.maximum(
                max_parallax_sin2,
                torch.where(visible, observed, torch.zeros_like(observed)),
            )

        reference_ranges = torch.linalg.vector_norm(
            anchors - reference_centers, dim=1
        )
        release_evidence = max_parallax_sin2
        shuffled = bool(config.get("footprint_trust_dynamic_shuffle", False))
        shuffle_mode = config.get(
            "footprint_trust_dynamic_shuffle_mode", "evidence"
        )
        stats = self.frontview_far_field_stats
        shuffle_seed = int(config.get("shuffle_seed", 42)) + int(
            stats["footprint_trust_dynamic_calls"]
        )
        if shuffled and shuffle_mode == "evidence" and release_evidence.numel() > 1:
            release_evidence = shuffle_within_log_depth_regimes(
                release_evidence,
                reference_ranges,
                shuffle_seed,
            )
        newly_resolved = evidence_pending & resolved_footprint_mask(
            max_parallax_sin2,
            target_scales,
            reference_ranges,
            birth_log_depth_stds,
        )
        if shuffled and shuffle_mode == "certificate" and newly_resolved.numel() > 1:
            released = matched_events_within_log_depth_regimes(
                newly_resolved,
                ownership,
                reference_ranges,
                shuffle_seed,
            )
        else:
            released = resolved_footprint_mask(
                release_evidence,
                target_scales,
                reference_ranges,
                birth_log_depth_stds,
            )

        cursor = 0
        released_rows = 0
        for group, indices, count in records:
            end = cursor + count
            group.non_trainable_params["max_parallax_sin2"][indices] = (
                max_parallax_sin2[cursor:end]
            )
            group.non_trainable_params["footprint_evidence_pending_mask"][
                indices[newly_resolved[cursor:end]]
            ] = False
            local_release = released[cursor:end]
            if torch.any(local_release):
                current_limits = group.non_trainable_params[
                    "max_scale_expansions"
                ][indices]
                release_limits = group.non_trainable_params[
                    "footprint_release_scale_expansions"
                ][indices]
                updated_limits, updated_ownership = release_owned_scale_caps(
                    current_limits,
                    release_limits,
                    ownership[cursor:end],
                    local_release,
                )
                group.non_trainable_params["max_scale_expansions"][indices] = (
                    updated_limits
                )
                group.non_trainable_params["footprint_trust_mask"][indices] = (
                    updated_ownership
                )
                released_rows += int(
                    torch.count_nonzero(
                        ownership[cursor:end] & local_release
                    ).item()
                )
            cursor = end

        stats["footprint_trust_dynamic_calls"] += 1
        stats["footprint_trust_dynamic_rows"] += int(released.numel())
        stats["footprint_trust_dynamic_released_rows"] += released_rows
        stats["footprint_trust_dynamic_shuffled_calls"] += int(shuffled)
        return released_rows

    def frontview_observability_summary(self):
        stats = dict(self.frontview_observability_stats)
        rows = int(stats["rows"])
        stats["mean_radial_scale"] = (
            float(stats["radial_scale_sum"]) / rows if rows else None
        )
        return stats

    @torch.no_grad()
    def _frontview_reprojection_scores(
        self,
        camera,
        reference_cameras,
        pixels,
        depths,
        depth_confidence,
        level,
    ):
        count = int(pixels.shape[0])
        scores = torch.zeros(count, device=pixels.device, dtype=torch.float32)
        support_sum = torch.zeros_like(scores)
        valid_any = torch.zeros(count, device=pixels.device, dtype=torch.bool)
        if count == 0 or not reference_cameras:
            return scores, valid_any, support_sum

        world_points = unproject_pts_tensor(
            pixels,
            depths,
            camera.get_int_mat(level),
            camera.get_raw_pose().detach(),
        )
        current_pose = camera.get_raw_pose().detach().to(world_points)
        current_center = -current_pose[:3, :3].T @ current_pose[:3, 3]
        current_rays = torch.nn.functional.normalize(
            world_points - current_center.reshape(1, 3), dim=1, eps=1.0e-8
        )

        def sample_image(image, uv):
            height, width = image.shape[:2]
            grid = torch.stack(
                (2.0 * uv[:, 0] / width - 1.0, 2.0 * uv[:, 1] / height - 1.0),
                dim=1,
            ).reshape(1, -1, 1, 2)
            sampled = torch.nn.functional.grid_sample(
                image.permute(2, 0, 1).unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            return sampled[0, :, :, 0].T

        current_image = camera.get_gt_image(level).to(world_points)
        current_image = current_image / max(float(camera.exposure_gain), 1.0e-8)
        current_color = sample_image(current_image, pixels)
        sigma = float(self.frontview_sampling_config["photo_sigma"])
        parallax_reference = math.sin(
            math.radians(
                float(self.frontview_sampling_config["parallax_reference_deg"])
            )
        )
        parallax_floor = float(self.frontview_sampling_config["parallax_floor"])

        for reference in reference_cameras:
            pose = reference.get_raw_pose().detach().to(world_points)
            camera_points = world_points @ pose[:3, :3].T + pose[:3, 3]
            z = camera_points[:, 2]
            projected = camera_points @ reference.get_int_mat(level).to(world_points).T
            uv = projected[:, :2] / torch.clamp(z.reshape(-1, 1), min=1.0e-8)
            width = reference.get_width(level)
            height = reference.get_height(level)
            valid = (
                (z > float(reference.near))
                & (z < float(reference.far))
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] < float(width))
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] < float(height))
            )
            reference_image = reference.get_gt_image(level).to(world_points)
            reference_image = reference_image / max(
                float(reference.exposure_gain), 1.0e-8
            )
            reference_color = sample_image(reference_image, uv)
            photo_error = torch.mean(torch.abs(current_color - reference_color), dim=1)
            consistency = torch.exp(-photo_error / sigma)
            photo_score = (
                1.0 - consistency
                if self.frontview_sampling_config["photo_mode"] == "disocclusion"
                else consistency
            )
            reference_center = -pose[:3, :3].T @ pose[:3, 3]
            reference_rays = torch.nn.functional.normalize(
                world_points - reference_center.reshape(1, 3), dim=1, eps=1.0e-8
            )
            cosine = torch.sum(current_rays * reference_rays, dim=1)
            parallax = torch.sqrt(torch.clamp(1.0 - cosine.square(), 0.0, 1.0))
            parallax_score = torch.clamp(
                parallax / max(parallax_reference, 1.0e-8), 0.0, 1.0
            )
            score = photo_score * (
                parallax_floor + (1.0 - parallax_floor) * parallax_score
            )
            scores = torch.maximum(scores, torch.where(valid, score, scores))
            support_sum += torch.where(valid, score, torch.zeros_like(score))
            valid_any |= valid

        scores *= torch.clamp(depth_confidence, 0.0, 1.0).pow(
            float(self.frontview_sampling_config["confidence_power"])
        )
        support_density = support_sum / float(len(reference_cameras))
        support_density *= torch.clamp(depth_confidence, 0.0, 1.0).pow(
            float(self.frontview_sampling_config["confidence_power"])
        )
        return scores, valid_any, support_density

    @torch.no_grad()
    def _frontview_recovery_multiview_depths(
        self,
        camera,
        reference_cameras,
        pixels,
        fallback_depths,
        depth_confidence,
        far_depth_m,
        level,
    ):
        """Select one causal inverse-depth hypothesis for each fallback ray."""

        config = self.frontview_coverage_recovery_config
        rows = int(pixels.shape[0])
        if rows == 0 or not reference_cameras:
            return fallback_depths, torch.zeros_like(fallback_depths), 0
        hypotheses_np = self.frontview_coverage_depth_prior.hypotheses(
            int(camera.cam_idx),
            int(config["multiview_depth_hypotheses"]),
            far_depth_m=far_depth_m,
        )
        if len(hypotheses_np) == 0:
            return fallback_depths, torch.zeros_like(fallback_depths), 0
        hypotheses = torch.as_tensor(
            hypotheses_np, device=pixels.device, dtype=fallback_depths.dtype
        )
        hypothesis_count = int(hypotheses.numel())
        tiled_pixels = pixels.repeat(hypothesis_count, 1)
        tiled_depths = hypotheses.reshape(-1, 1).expand(-1, rows).reshape(-1)
        tiled_confidence = depth_confidence.repeat(hypothesis_count)
        _, valid, support_density = self._frontview_reprojection_scores(
            camera,
            reference_cameras,
            tiled_pixels,
            tiled_depths,
            tiled_confidence,
            level,
        )
        scores = support_density.reshape(hypothesis_count, rows)
        valid = valid.reshape(hypothesis_count, rows)
        if config["multiview_depth_mode"] == "posterior_inverse_depth":
            scores = torch.clamp(
                scores
                / torch.clamp(
                    depth_confidence.reshape(1, rows),
                    min=torch.finfo(scores.dtype).eps,
                ),
                0.0,
                1.0,
            )
        supported = valid.any(dim=0)
        if config["multiview_depth_mode"] == "posterior_inverse_depth":
            selected, evidence, supported = posterior_inverse_depth_fusion(
                scores,
                valid,
                hypotheses,
                fallback_depths,
            )
        else:
            evidence, best_indices = torch.max(scores, dim=0)
            selected = hypotheses[best_indices]
        if bool(config["shuffle_multiview_depth"]) and bool(supported.any().item()):
            supported_rows = torch.nonzero(supported, as_tuple=False).flatten()
            generator = torch.Generator(device=pixels.device)
            generator.manual_seed(
                int(config["multiview_depth_seed"]) + int(camera.cam_idx)
            )
            permutation = torch.randperm(
                supported_rows.numel(), generator=generator, device=pixels.device
            )
            selected_shuffled = selected.clone()
            evidence_shuffled = evidence.clone()
            selected_shuffled[supported_rows] = selected[supported_rows[permutation]]
            evidence_shuffled[supported_rows] = evidence[supported_rows[permutation]]
            selected = selected_shuffled
            evidence = evidence_shuffled
        result = torch.where(supported, selected, fallback_depths)
        return result, torch.where(supported, evidence, 0.0), hypothesis_count

    def frontview_sampling_summary(self):
        stats = dict(self.frontview_sampling_stats)
        selected = int(stats["selected_rows"])
        stats["mean_selected_score"] = (
            float(stats["score_sum"]) / selected if selected else None
        )
        adaptive_calls = int(stats["adaptive_calls"])
        stats["mean_adaptive_iterations"] = (
            float(stats["adaptive_iterations"]) / adaptive_calls
            if adaptive_calls
            else None
        )
        stats["mean_adaptive_objective"] = (
            float(stats["adaptive_objective_sum"]) / adaptive_calls
            if adaptive_calls
            else None
        )
        stats["mean_adaptive_boundaries_m"] = (
            [
                float(value) / adaptive_calls
                for value in stats["adaptive_boundary_sum_m"]
            ]
            if adaptive_calls
            else []
        )
        coverage_calls = int(stats["adaptive_coverage_calls"])
        stats["mean_adaptive_coverage_cell_pixels"] = (
            float(stats["adaptive_coverage_cell_pixel_sum"]) / coverage_calls
            if coverage_calls
            else None
        )
        density_calls = int(stats["rate_distortion_calls"])
        stats["mean_rate_distortion_cell_pixels"] = (
            float(stats["rate_distortion_cell_pixel_sum"]) / density_calls
            if density_calls
            else None
        )
        return stats

    def frontview_inverse_depth_certificate_summary(self):
        stats = dict(self.frontview_inverse_depth_certificate_stats)
        rows = int(stats["rows"])
        certified = int(stats["certified_rows"])
        stats["enabled"] = bool(
            self.frontview_inverse_depth_certificate_config["enabled"]
        )
        stats["uncertified_policy"] = self.frontview_inverse_depth_certificate_config[
            "uncertified_policy"
        ]
        stats["mean_information_gain"] = (
            float(stats["information_gain_sum"]) / rows if rows else None
        )
        stats["mean_absolute_log_depth_shift"] = (
            float(stats["absolute_log_depth_shift_sum"]) / certified
            if certified
            else None
        )
        stats["mean_posterior_log_std"] = (
            float(stats["posterior_log_std_sum"]) / certified
            if certified
            else None
        )
        stats["mean_valid_views"] = (
            float(stats["valid_view_sum"]) / rows if rows else None
        )
        stats["mean_baseline_information"] = (
            float(stats["baseline_information_sum"]) / rows if rows else None
        )
        return stats

    def frontview_depth_transport_summary(self):
        stats = dict(self.frontview_depth_transport_stats)
        valid = int(stats["valid_calibration_anchors"])
        corrected = int(stats["corrected_rows"])
        stats["mean_absolute_log_residual"] = (
            float(stats["absolute_log_residual_sum"]) / valid if valid else None
        )
        stats["mean_absolute_log_correction"] = (
            float(stats["absolute_log_correction_sum"]) / corrected
            if corrected
            else None
        )
        return stats

    def frontview_coverage_recovery_depth_summary(self):
        stats = dict(self.frontview_coverage_depth_stats)
        stats.update(self.frontview_coverage_depth_prior.summary())
        stats["depth_fallback_enabled"] = bool(
            self.frontview_coverage_recovery_config["depth_fallback_enabled"]
        )
        supported = int(stats["multiview_depth_supported_rows"])
        calls = int(stats["multiview_depth_calls"])
        stats["mean_multiview_depth_score"] = (
            float(stats["multiview_depth_score_sum"]) / supported
            if supported
            else None
        )
        stats["mean_multiview_depth_concentration"] = (
            float(stats["multiview_depth_concentration_sum"]) / supported
            if supported
            else None
        )
        stats["multiview_depth_mode"] = self.frontview_coverage_recovery_config[
            "multiview_depth_mode"
        ]
        stats["mean_multiview_selected_depth_m"] = (
            float(stats["multiview_depth_selected_sum_m"]) / supported
            if supported
            else None
        )
        stats["mean_multiview_hypothesis_count"] = (
            float(stats["multiview_depth_hypotheses_sum"]) / calls
            if calls
            else None
        )
        return stats

    def causal_metric_birth_summary(self):
        return self.causal_metric_birth.summary()

    def observe_causal_landmarks(self, camera):
        return self.causal_landmark_memory.observe(camera)

    def causal_landmark_memory_summary(self):
        return self.causal_landmark_memory.summary()

    def causal_dual_responsibility_summary(self):
        result = dict(self.causal_dual_responsibility_stats)
        total = (
            int(result["metric_rows"])
            + int(result["proxy_rows"])
            + int(result["partial_metric_rows"])
        )
        result.update(
            enabled=bool(self.causal_dual_responsibility_config["enabled"]),
            finite_depth_certificate_enabled=bool(
                self.causal_dual_responsibility_config[
                    "finite_depth_certificate_enabled"
                ]
            ),
            finite_depth_certificate_scope=self.causal_dual_responsibility_config[
                "finite_depth_certificate_scope"
            ],
            total_classified_rows=total,
            proxy_fraction=(
                float(result["proxy_rows"]) / float(total) if total else 0.0
            ),
            mean_metric_confidence=(
                float(result["metric_confidence_sum"]) / float(total)
                if total
                else 0.0
            ),
        )
        certificate_rows = int(result["finite_certificate_rows"])
        result["mean_finite_certificate"] = (
            float(result["finite_certificate_value_sum"])
            / float(certificate_rows)
            if certificate_rows
            else None
        )
        result["mean_finite_observability"] = (
            float(result["finite_certificate_observability_sum"])
            / float(certificate_rows)
            if certificate_rows
            else None
        )
        result["mean_finite_support"] = (
            float(result["finite_certificate_support_sum"])
            / float(certificate_rows)
            if certificate_rows
            else None
        )
        return result

    def frontview_birth_summary(self):
        stats = dict(self.frontview_birth_stats)
        stats.update(self.frontview_track_ledger.summary())
        return stats

    def frontview_far_field_summary(self):
        stats = dict(self.frontview_far_field_stats)
        causal_rows = int(stats["causal_route_rows"])
        stats["routing_mode"] = self.frontview_far_field_config["routing_mode"]
        stats["projective_nms_mode"] = self.frontview_far_field_config[
            "projective_nms_mode"
        ]
        stats["map_redundancy_gate"] = bool(
            self.frontview_far_field_config["map_redundancy_gate"]
        )
        stats["map_redundancy_evidence"] = self.frontview_far_field_config[
            "map_redundancy_evidence"
        ]
        stats["posterior_budget_refill"] = bool(
            self.frontview_far_field_config["posterior_budget_refill"]
        )
        stats["shuffle_refill_evidence"] = bool(
            self.frontview_far_field_config["shuffle_refill_evidence"]
        )
        photometric_calls = int(stats["map_gate_photometric_calls"])
        stats["mean_map_gate_residual_scale"] = (
            float(stats["map_gate_residual_scale_sum"]) / photometric_calls
            if photometric_calls
            else None
        )
        stats["projective_covariance_mode"] = self.frontview_far_field_config[
            "projective_covariance_mode"
        ]
        stats["fallback_support_mode"] = self.frontview_far_field_config[
            "fallback_support_mode"
        ]
        covariance_rows = int(stats["projective_covariance_rows"])
        stats["mean_projective_radial_factor"] = (
            float(stats["projective_radial_factor_sum"]) / covariance_rows
            if covariance_rows
            else None
        )
        structure_rows = int(stats["fallback_structure_rows"])
        stats["mean_fallback_anisotropy"] = (
            float(stats["fallback_anisotropy_sum"]) / structure_rows
            if structure_rows
            else None
        )
        information_rows = int(stats["fallback_information_rows"])
        stats["mean_fallback_information_radius_factor"] = (
            float(stats["fallback_information_radius_factor_sum"])
            / information_rows
            if information_rows
            else None
        )
        stats["mean_fallback_information_density"] = (
            float(stats["fallback_information_density_sum"])
            / information_rows
            if information_rows
            else None
        )
        stats["mean_causal_parallax_pixels"] = (
            float(stats["causal_parallax_pixel_sum"]) / causal_rows
            if causal_rows
            else None
        )
        stats["mean_projected_support_pixels"] = (
            float(stats["causal_support_pixel_sum"]) / causal_rows
            if causal_rows
            else None
        )
        stats["mean_log_depth_std"] = (
            float(stats["causal_log_depth_std_sum"]) / causal_rows
            if causal_rows
            else None
        )
        footprint_rows = int(stats["footprint_trust_rows"])
        footprint_bounded_rows = int(stats["footprint_trust_bounded_rows"])
        footprint_calls = int(stats["footprint_trust_calls"])
        stats["footprint_trust_mode"] = self.frontview_far_field_config[
            "footprint_trust_mode"
        ]
        stats["footprint_trust_scope"] = self.frontview_far_field_config[
            "footprint_trust_scope"
        ]
        stats["footprint_trust_dynamic_shuffle_mode"] = (
            self.frontview_far_field_config[
                "footprint_trust_dynamic_shuffle_mode"
            ]
        )
        stats["mean_footprint_trust_information"] = (
            float(stats["footprint_trust_information_sum"]) / footprint_rows
            if footprint_rows
            else None
        )
        stats["mean_footprint_scale_limit"] = (
            float(stats["footprint_trust_limit_sum"]) / footprint_bounded_rows
            if footprint_bounded_rows
            else None
        )
        stats["footprint_trust_bounded_fraction"] = (
            float(footprint_bounded_rows) / footprint_rows
            if footprint_rows
            else None
        )
        stats["mean_footprint_trust_cell_pixels"] = (
            float(stats["footprint_trust_cell_pixel_sum"]) / footprint_calls
            if footprint_calls
            else None
        )
        residual_rd_rows = int(stats["footprint_trust_residual_rd_rows"])
        stats["mean_footprint_trust_residual_rd_radius"] = (
            float(stats["footprint_trust_residual_rd_radius_sum"])
            / residual_rd_rows
            if residual_rd_rows
            else None
        )
        adaptive_calls = int(stats["adaptive_route_calls"])
        stats["mean_adaptive_route_boundaries_m"] = (
            [
                float(value) / adaptive_calls
                for value in stats["adaptive_route_boundary_sum_m"]
            ]
            if adaptive_calls
            else None
        )
        stats["mean_adaptive_route_objective"] = (
            float(stats["adaptive_route_objective_sum"]) / adaptive_calls
            if adaptive_calls
            else None
        )
        stats["mean_adaptive_route_iterations"] = (
            float(stats["adaptive_route_iterations"]) / adaptive_calls
            if adaptive_calls
            else None
        )
        budget_calls = int(stats["budget_nms_calls"])
        stats["mean_budget_nms_cell_pixels"] = (
            float(stats["budget_nms_cell_pixel_sum"]) / budget_calls
            if budget_calls
            else None
        )
        stats["mean_budget_nms_log_depth_width"] = (
            float(stats["budget_nms_log_depth_width_sum"]) / budget_calls
            if budget_calls
            else None
        )
        stats["responsibility_basis"] = self.frontview_far_field_config[
            "responsibility_basis"
        ]
        stats["responsibility_shuffle_mode"] = self.frontview_far_field_config[
            "responsibility_shuffle_mode"
        ]
        stats["preserve_sparse_track_geometry"] = bool(
            self.frequency_sampling_config.get(
                "preserve_sparse_track_geometry", False
            )
        )
        stats["propagate_raster_sparse_identity"] = bool(
            self.frequency_sampling_config.get(
                "propagate_raster_sparse_identity", False
            )
        )
        stats["ray_atlas"] = self.frontview_ray_atlas.summary()
        return stats

    def observe_frontview_directional_layer(self, camera, render_pkg=None):
        uncertainty_mass = (
            render_pkg.get("uncertainty_mass") if render_pkg is not None else None
        )
        return self.frontview_directional_layer.observe(
            camera, uncertainty_mass=uncertainty_mass
        )

    def activate_frontview_directional_layer(self, enabled=True):
        return self.frontview_directional_layer.activate(enabled)

    def save_frontview_directional_layer(self, path):
        self.frontview_directional_layer.save(path)

    def load_frontview_directional_layer(self, path, config_overrides=None):
        self.frontview_directional_layer.load(path)
        if config_overrides:
            config = dict(self.frontview_directional_layer.config)
            config.update(config_overrides)
            self.frontview_directional_layer.config = (
                validate_front_view_directional_layer_config(config)
            )

    def frontview_directional_layer_summary(self):
        return self.frontview_directional_layer.summary()

    def frontview_scale_cover_summary(self):
        return self.frontview_scale_cover.summary()

    def streaming_appearance_lod_summary(self):
        result = dict(self.streaming_appearance_lod_stats)
        signal_rows = int(result.get("gradient_signal_rows", 0))
        result["gradient_signal_mean"] = (
            float(result.get("gradient_signal_sum", 0.0)) / signal_rows
            if signal_rows
            else None
        )
        records = self._streaming_appearance_group_records()
        if records:
            degrees = torch.cat(
                [
                    group.non_trainable_params["appearance_sh_degree"]
                    for _, group, _, _ in records
                ]
            )
            counts = torch.cat(
                [
                    group.non_trainable_params["appearance_view_count"]
                    for _, group, _, _ in records
                ]
            )
        else:
            degrees = torch.empty(0, device=self.device, dtype=torch.uint8)
            counts = torch.empty(0, device=self.device, dtype=torch.int32)
        birth_degree = int(self.streaming_appearance_lod_config["birth_degree"])
        target_degree = int(self.streaming_appearance_lod_config["target_degree"])
        degrees = torch.clamp(degrees, min=birth_degree)
        degree_counts = {
            "sh{}".format(degree): int((degrees == degree).sum().item())
            for degree in range(target_degree)
        }
        degree_counts["sh{}".format(target_degree)] = int(
            (degrees >= target_degree).sum().item()
        )
        result.update(
            {
                "enabled": bool(self.streaming_appearance_lod_config["enabled"]),
                "selection_mode": self.streaming_appearance_lod_config[
                    "selection_mode"
                ],
                "birth_degree": birth_degree,
                "target_degree": target_degree,
                "max_target_fraction": float(
                    self.streaming_appearance_lod_config["max_target_fraction"]
                ),
                "degree_counts": degree_counts,
                "observed_gaussians": int((counts > 0).sum().item()),
            }
        )
        routing = {}
        for key, value in self.streaming_compute_routing_stats.items():
            routing[key] = int(value.item()) if torch.is_tensor(value) else int(value)
        packed_rows = routing["packed_rows"]
        base_terms = (birth_degree + 1) ** 2
        target_terms = (target_degree + 1) ** 2
        evaluated_terms = (
            routing["base_rows"] * base_terms
            + routing["target_rows"] * target_terms
        )
        dense_terms = packed_rows * target_terms
        routing.update(
            {
                "enabled": bool(
                    self.streaming_appearance_lod_config["compute_routing"]
                ),
                "evaluated_sh_basis_terms": evaluated_terms,
                "dense_sh_basis_terms": dense_terms,
                "sh_basis_term_reduction_fraction": (
                    1.0 - float(evaluated_terms) / float(dense_terms)
                    if dense_terms
                    else 0.0
                ),
                "target_band_row_reduction_fraction": (
                    float(routing["skipped_target_band_rows"]) / packed_rows
                    if packed_rows
                    else 0.0
                ),
            }
        )
        routing["route_active_sh_basis_term_reduction_fraction"] = routing[
            "sh_basis_term_reduction_fraction"
        ]
        routing["route_active_target_band_row_reduction_fraction"] = routing[
            "target_band_row_reduction_fraction"
        ]
        shn_parameter_bytes = 0
        shn_optimizer_state_bytes = 0
        evidence_state_bytes = 0
        for _, group, _, _ in records:
            shn = group.splats["shN"]
            shn_parameter_bytes += shn.numel() * shn.element_size()
            optimizer = group.optimizers.get("shN") if group.optimizers else None
            if optimizer is not None:
                state = optimizer.state.get(shn, {})
                for state_name in ("exp_avg", "exp_avg_sq"):
                    tensor = state.get(state_name)
                    if torch.is_tensor(tensor):
                        shn_optimizer_state_bytes += (
                            tensor.numel() * tensor.element_size()
                        )
            tensor = group.non_trainable_params.get(
                "appearance_band_gradient_ema"
            )
            if torch.is_tensor(tensor):
                evidence_state_bytes += tensor.numel() * tensor.element_size()
        routing.update(
            {
                "shn_parameter_bytes": shn_parameter_bytes,
                "shn_optimizer_state_bytes": shn_optimizer_state_bytes,
                "gradient_evidence_state_bytes": evidence_state_bytes,
                "gradient_ema_dtype": self.streaming_appearance_lod_config[
                    "gradient_ema_dtype"
                ],
                "gradient_ema_scale": self.streaming_appearance_lod_config[
                    "gradient_ema_scale"
                ],
                "warmup_evidence_updates": self.streaming_appearance_lod_config[
                    "compute_routing_warmup_evidence_updates"
                ],
            }
        )
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            routing["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(
                self.device
            )
            routing["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(
                self.device
            )
        result["compute_routing"] = routing
        if self.tgbr_sparse_model_stats is not None:
            result["sparse_model"] = dict(self.tgbr_sparse_model_stats)
        return result

    def frontview_sparse_scale_map_summary(self):
        return self.frontview_sparse_scale_map.summary()

    @torch.no_grad()
    def split_stable_detail_gaussians(
        self, global_indices, tangent_x, tangent_y, config
    ):
        """Split selected render-order rows while preserving Gaussian-group ownership."""
        selected = torch.as_tensor(global_indices, device=self.device, dtype=torch.long)
        tangent_x = torch.as_tensor(tangent_x, device=self.device, dtype=torch.float32)
        tangent_y = torch.as_tensor(tangent_y, device=self.device, dtype=torch.float32)
        if selected.numel() == 0:
            return 0, 0
        if tangent_x.shape != (len(selected), 3) or tangent_y.shape != (len(selected), 3):
            raise ValueError("Stable detail tangents must match selected Gaussian rows")

        parents = 0
        cursor = 0
        for level in range(self.MAX_LEVEL):
            for group_id in list(self.active_gaussian_groups[level]):
                group = self.gaussian_groups[group_id]
                count = group.get_num
                in_group = (selected >= cursor) & (selected < cursor + count)
                if torch.any(in_group) and group_id not in self.progressive_group_ids:
                    if (
                        self.worldtest_certificate_authority is not None
                        and not self.worldtest_group_certificates.get(group_id)
                    ):
                        self.worldtest_certificate_authority.bypass_count += 1
                        raise RuntimeError(
                            "Stable detail child split has no certified parent provenance"
                        )
                    local = selected[in_group] - cursor
                    params = {
                        name: value.detach().clone()
                        for name, value in group.splats.items()
                    }
                    split = split_gaussian_parameters(
                        params,
                        local,
                        tangent_x[in_group],
                        tangent_y[in_group],
                        scale_ratio=float(config.get("child_scale_ratio", 0.55)),
                        offset_fraction=float(config.get("child_offset_fraction", 0.35)),
                    )
                    old_limits = group.non_trainable_params[
                        "max_scale_expansions"
                    ]
                    old_anchors = group.non_trainable_params["mean_anchors"]
                    old_displacements = group.non_trainable_params[
                        "max_mean_displacements"
                    ]
                    keep = torch.ones(
                        (count,), device=old_limits.device, dtype=torch.bool
                    )
                    keep[local] = False
                    child_limits = torch.full(
                        (4 * len(local),),
                        float("inf"),
                        device=old_limits.device,
                        dtype=old_limits.dtype,
                    )
                    split["max_scale_expansions"] = torch.cat(
                        (old_limits[keep], child_limits), dim=0
                    )
                    for name, value in group.non_trainable_params.items():
                        if name in (
                            "init_scales",
                            "max_scale_expansions",
                            "mean_anchors",
                            "max_mean_displacements",
                        ):
                            continue
                        child_shape = (4 * len(local),) + tuple(value.shape[1:])
                        if value.dtype == torch.bool:
                            children = torch.zeros(
                                child_shape, device=value.device, dtype=value.dtype
                            )
                        elif name in (
                            "footprint_target_scales",
                            "footprint_release_scale_expansions",
                        ):
                            children = torch.full(
                                child_shape,
                                float("inf"),
                                device=value.device,
                                dtype=value.dtype,
                            )
                        elif name in ("track_ids", "gaussian_uids", "birth_frame_ids"):
                            children = torch.full(
                                child_shape,
                                -1,
                                device=value.device,
                                dtype=value.dtype,
                            )
                        else:
                            children = torch.zeros(
                                child_shape, device=value.device, dtype=value.dtype
                            )
                        split[name] = torch.cat((value[keep], children), dim=0)
                    child_anchors = split["means"][-4 * len(local) :].detach().clone()
                    child_displacements = old_displacements[local][None].expand(
                        4, -1
                    ).reshape(-1)
                    split["mean_anchors"] = torch.cat(
                        (old_anchors[keep], child_anchors), dim=0
                    )
                    split["max_mean_displacements"] = torch.cat(
                        (old_displacements[keep], child_displacements), dim=0
                    )
                    group.replace_gaussians(split)
                    parents += int(len(local))
                cursor += count
        return parents, parents * 3

    def _progressive_learning_rates(self, config=None, scene_scale=None):
        config = self.init_gaussian_config if config is None else config
        scene_scale = self.scene_scale if scene_scale is None else scene_scale
        learning_rates = {
            "means": config["means_lr_init"] * scene_scale,
            "scales": config["scales_lr_init"],
            "quats": config.get("quats_lr_init", config.get("quats_lr")),
            "opacities": config["opacities_lr"],
            "sh0": config["sh_lr"],
            "shN": config["sh_lr"] / 20,
        }
        return {
            name: value * self.progressive_lr_multipliers[name]
            for name, value in learning_rates.items()
        }

    def _register_progressive_group(self, group_idx):
        group = self.gaussian_groups[group_idx]
        learning_rates = self._progressive_learning_rates()
        for name, parameter in group.splats.items():
            if name not in self.progressive_optimizers:
                self.progressive_optimizers[name] = torch.optim.Adam(
                    [
                        {
                            "params": [parameter],
                            "lr": learning_rates[name] * math.sqrt(self.BS),
                            "name": name,
                        }
                    ],
                    eps=1e-15 / math.sqrt(self.BS),
                    betas=(1 - self.BS * (1 - 0.9), 1 - self.BS * (1 - 0.999)),
                    foreach=True,
                )
                self.progressive_optimizer_group_ids[name] = [group_idx]
                self.progressive_optimizer_indices[name] = {group_idx: 0}
            else:
                self.progressive_optimizers[name].param_groups[0]["params"].append(
                    parameter
                )
                group_ids = self.progressive_optimizer_group_ids[name]
                self.progressive_optimizer_indices[name][group_idx] = len(group_ids)
                group_ids.append(group_idx)

    def _unregister_progressive_group(self, group_idx):
        group = self.gaussian_groups[group_idx]
        for name, parameter in group.splats.items():
            optimizer = self.progressive_optimizers.get(name)
            if optimizer is None:
                continue
            params = optimizer.param_groups[0]["params"]
            group_ids = self.progressive_optimizer_group_ids[name]
            indices = self.progressive_optimizer_indices[name]
            index = indices.pop(group_idx)
            last_index = len(params) - 1
            if index != last_index:
                params[index] = params[last_index]
                moved_group_id = group_ids[last_index]
                group_ids[index] = moved_group_id
                indices[moved_group_id] = index
            params.pop()
            group_ids.pop()
            optimizer.state.pop(parameter, None)

    def _reset_progressive_optimizers(self, config, BS):
        self.progressive_optimizers = {}
        self.progressive_optimizer_group_ids = {}
        self.progressive_optimizer_indices = {}
        self.progressive_lr_step = 0
        optimized_group_ids = [
            group_idx
            for group_idx in self.progressive_group_ids
            if self.gaussian_groups[group_idx].is_optimize
        ]
        if not optimized_group_ids:
            return
        learning_rates = self._progressive_learning_rates(config, self.scene_scale)
        for name, learning_rate in learning_rates.items():
            params = [
                self.gaussian_groups[group_idx].splats[name]
                for group_idx in optimized_group_ids
            ]
            self.progressive_optimizers[name] = torch.optim.Adam(
                [{"params": params, "lr": learning_rate * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                foreach=True,
            )
            group_ids = list(optimized_group_ids)
            self.progressive_optimizer_group_ids[name] = group_ids
            self.progressive_optimizer_indices[name] = {
                group_id: index for index, group_id in enumerate(group_ids)
            }
        self.progressive_scheduler_args = {
            "means": get_expon_lr_func(
                config["means_lr_init"], config["means_lr_final"], max_steps=config["max_steps"]
            ),
            "scales": get_expon_lr_func(
                config["scales_lr_init"], config["scales_lr_final"], max_steps=config["max_steps"]
            ),
        }

    def add_progressive_group(self, params, optimize=True, level=0):
        """Create one independently removable group for a progressive root."""
        group = Gaussians(
            BS=self.BS,
            scene_scale=self.scene_scale,
            init_config=self.init_gaussian_config,
            max_sh_degree=self.max_sh_degree,
            managed_optimizers=False,
        )
        group.to_device(self.device)
        group.replace_gaussians(params)
        if not optimize:
            group.disable_grad()
        group_idx = len(self.gaussian_groups)
        self.gaussian_groups.append(group)
        self.active_gaussian_groups[level].append(group_idx)
        self.valid_groups.append(group_idx)
        self.progressive_group_ids.add(group_idx)
        self._register_progressive_group(group_idx)
        return group_idx

    def update_progressive_group(self, group_idx, params):
        if group_idx not in self.progressive_group_ids:
            raise ValueError("Group {} is not progressive-owned".format(group_idx))
        self.gaussian_groups[group_idx].update_gaussians_data(params)

    def export_progressive_group(self, group_idx):
        if group_idx not in self.progressive_group_ids:
            raise ValueError("Group {} is not progressive-owned".format(group_idx))
        return {
            name: value.detach().clone()
            for name, value in self.gaussian_groups[group_idx].splats.items()
        }

    def remove_progressive_group(self, group_idx, level=0):
        if group_idx not in self.progressive_group_ids:
            raise ValueError("Group {} is not progressive-owned".format(group_idx))
        self._unregister_progressive_group(group_idx)
        self.remove_group(group_idx, level)
        self.progressive_group_ids.remove(group_idx)

    @torch.no_grad()
    def merge_progressive_groups_into_baseline(self, level=0):
        """Move all progressive rows into one managed baseline group."""
        source_group_ids = sorted(self.progressive_group_ids)
        if not source_group_ids:
            return 0, 0

        target_candidates = [
            group_idx
            for group_idx in self.active_gaussian_groups[level]
            if group_idx not in self.progressive_group_ids
            and self.gaussian_groups[group_idx].splats is not None
        ]
        if not target_candidates:
            raise RuntimeError("No baseline Gaussian group is available for compaction")
        target_group_idx = self.current_gaussian_group.get(level)
        if target_group_idx not in target_candidates:
            target_group_idx = target_candidates[0]

        merged = {
            name: torch.cat(
                [self.gaussian_groups[group_idx].splats[name].detach() for group_idx in source_group_ids],
                dim=0,
            )
            for name in ("means", "scales", "quats", "opacities", "sh0", "shN")
        }
        merged["init_scales"] = torch.exp(merged["scales"]).mean(dim=1)
        merged["max_scale_expansions"] = torch.full(
            (merged["means"].shape[0],),
            float("inf"),
            device=merged["means"].device,
            dtype=torch.float32,
        )
        merged_count = int(merged["means"].shape[0])
        self.gaussian_groups[target_group_idx].extend_gaussians(merged)

        for group_idx in source_group_ids:
            self.remove_progressive_group(group_idx, level=level)
        return len(source_group_ids), merged_count

    def export_raw_splats(self, exclude_group_ids=None):
        """Return raw model-domain splats on CPU, optionally excluding owned groups."""
        excluded = set(exclude_group_ids or ())
        group_ids = [
            group_idx for group_idx in self.valid_groups
            if group_idx not in excluded and self.gaussian_groups[group_idx].splats is not None
        ]
        if not group_ids:
            return {}
        return {
            name: torch.cat(
                [self.gaussian_groups[group_idx].splats[name].detach().cpu() for group_idx in group_ids],
                dim=0,
            )
            for name in ("means", "scales", "quats", "opacities", "sh0", "shN")
        }

    def create_new_group(self, level=0):
        if self.gaussian_pos_schedule_steps > 0:  # update position lr rate
            raise NotImplementedError

        self.gaussian_groups.append(
            Gaussians(
                BS=self.BS,
                scene_scale=self.scene_scale,
                init_config=self.init_gaussian_config,
                max_sh_degree=self.max_sh_degree,
            )
        )
        self.gaussian_groups[-1].to_device(self.device)
        self.current_gaussian_group[level] = len(self.gaussian_groups) - 1
        current_group_id = self.current_gaussian_group[level]
        self.active_gaussian_groups[level].append(current_group_id)
        self.valid_groups.append(current_group_id)
        Log(
            "Creating new group: active groups: {}, totol groups: {}".format(
                len(self.active_gaussian_groups), len(self.gaussian_groups)
            ),
            tag="GaussianModel",
        )

    @torch.no_grad()
    def add_track_detail_gaussians(
        self,
        world_points,
        colors,
        log_scales,
        initial_opacity,
        max_scale_expansion,
        group_id=None,
        freeze_geometry=True,
        level=0,
        admission_certificate=None,
    ):
        """Append sparse-track appearance carriers without claiming new geometry."""
        certificate = self._require_worldtest_certificate(
            admission_certificate, path="track_or_surface_or_flow_detail"
        )
        count = int(np.asarray(world_points).shape[0])
        if count == 0:
            return group_id, 0
        if group_id is None:
            group = Gaussians(
                BS=self.BS,
                scene_scale=self.scene_scale,
                init_config=self.init_gaussian_config,
                max_sh_degree=self.max_sh_degree,
            )
            group.to_device(self.device)
            group_id = len(self.gaussian_groups)
            self.gaussian_groups.append(group)
            self.active_gaussian_groups[level].append(group_id)
            self.valid_groups.append(group_id)
        elif group_id not in self.valid_groups:
            raise ValueError("Track detail target group is not active")

        expansion = np.full(
            (count,), float(max_scale_expansion), dtype=np.float32
        )
        self.gaussian_groups[group_id].extend_gaussians_from_color_points(
            np.asarray(world_points, dtype=np.float32),
            np.asarray(colors, dtype=np.float32),
            np.asarray(log_scales, dtype=np.float32),
            initial_opacity=float(initial_opacity),
            max_scale_expansion=expansion,
        )
        if freeze_geometry:
            self.freeze_group_geometry(group_id)
        if certificate is not None:
            self.worldtest_group_certificates.setdefault(int(group_id), set()).add(
                certificate.certificate_id
            )
        return int(group_id), count

    @torch.no_grad()
    def reassign_track_detail_responsibility(
        self, world_points, colors, log_scales, scores, config, level=0
    ):
        """Bind repeated-track detail evidence to existing Gaussian rows."""
        from scipy.spatial import cKDTree

        points = np.asarray(world_points, dtype=np.float32)
        if len(points) == 0:
            return {"tracks": 0, "matched": 0, "distance_mean": 0.0}
        group_ids = []
        local_rows = []
        means = []
        seen = set()
        for group_id in self.active_gaussian_groups[level]:
            if group_id in seen or group_id in self.progressive_group_ids:
                continue
            seen.add(group_id)
            group = self.gaussian_groups[group_id]
            if group.splats is None:
                continue
            count = group.get_num
            means.append(group.get_xyz.detach().cpu().numpy())
            group_ids.append(np.full((count,), group_id, dtype=np.int64))
            local_rows.append(np.arange(count, dtype=np.int64))
        if not means:
            return {"tracks": len(points), "matched": 0, "distance_mean": 0.0}

        all_means = np.concatenate(means, axis=0)
        all_group_ids = np.concatenate(group_ids)
        all_local_rows = np.concatenate(local_rows)
        distances, nearest = cKDTree(all_means).query(points, k=1, workers=-1)
        valid = np.isfinite(distances) & (
            distances <= float(config["reassign_max_distance"])
        )
        candidates = np.flatnonzero(valid)
        order = candidates[
            np.argsort(np.asarray(scores, dtype=np.float32)[candidates])[::-1]
        ]
        selected_tracks = []
        selected_rows = []
        claimed = set()
        for track_index in order:
            global_row = int(nearest[track_index])
            if global_row in claimed:
                continue
            claimed.add(global_row)
            selected_tracks.append(int(track_index))
            selected_rows.append(global_row)
        if not selected_tracks:
            return {"tracks": len(points), "matched": 0, "distance_mean": 0.0}

        selected_tracks = np.asarray(selected_tracks, dtype=np.int64)
        selected_rows = np.asarray(selected_rows, dtype=np.int64)
        color_blend = float(config["reassign_color_blend"])
        mean_blend = float(config["reassign_mean_blend"])
        opacity_floor = float(config["reassign_opacity_floor"])
        scale_multiplier = float(config["reassign_scale_multiplier"])
        target_sh0 = torch.from_numpy(
            rgb_to_sh_np(np.asarray(colors, dtype=np.float32)[selected_tracks])
        ).to(self.device, dtype=torch.float32).reshape(-1, 1, 3)
        target_means = torch.from_numpy(points[selected_tracks]).to(
            self.device, dtype=torch.float32
        )
        target_log_scales = torch.from_numpy(
            np.asarray(log_scales, dtype=np.float32)[selected_tracks]
        ).to(self.device, dtype=torch.float32)
        target_log_scales = target_log_scales + math.log(scale_multiplier)

        for group_id in np.unique(all_group_ids[selected_rows]):
            mask = all_group_ids[selected_rows] == group_id
            tracks_for_group = torch.from_numpy(np.flatnonzero(mask)).to(
                self.device, dtype=torch.long
            )
            rows = torch.from_numpy(all_local_rows[selected_rows[mask]]).to(
                self.device, dtype=torch.long
            )
            group = self.gaussian_groups[int(group_id)]
            desired_scale = target_log_scales[tracks_for_group]
            group.splats["scales"][rows] = torch.minimum(
                group.splats["scales"][rows], desired_scale.expand(-1, 3)
            )
            group.splats["means"][rows] = (
                (1.0 - mean_blend) * group.splats["means"][rows]
                + mean_blend * target_means[tracks_for_group]
            )
            group.splats["sh0"][rows] = (
                (1.0 - color_blend) * group.splats["sh0"][rows]
                + color_blend * target_sh0[tracks_for_group]
            )
            current_opacity = group.get_opacity[rows]
            opacity = torch.maximum(
                current_opacity,
                torch.full_like(current_opacity, opacity_floor),
            )
            group.splats["opacities"][rows] = group.inverse_opacity_activation(
                torch.clamp(opacity, min=1.0e-4, max=1.0 - 1.0e-4)
            )
            group.non_trainable_params["init_scales"][rows] = torch.exp(
                desired_scale.reshape(-1)
            )
            group.non_trainable_params["max_scale_expansions"][rows] = 1.0

        selected_distances = distances[selected_tracks]
        return {
            "tracks": int(len(points)),
            "matched": int(len(selected_tracks)),
            "distance_mean": float(np.mean(selected_distances)),
            "distance_p95": float(np.percentile(selected_distances, 95)),
        }

    def deactivate_gaussian_group(self, group_idx, level):
        if group_idx in self.active_gaussian_groups[level]:
            self.active_gaussian_groups[level].remove(group_idx)
            self.gaussian_groups[group_idx].to_device("cpu")
            return True
        else:
            return False

    def activate_gaussian_group(self, group_idx, level):
        if group_idx not in self.active_gaussian_groups[level]:
            self.active_gaussian_groups[level].append(group_idx)
            self.gaussian_groups[group_idx].to_device(self.device)
            return True
        else:
            return False

    def remove_group(self, group_idx, level):
        if group_idx in self.active_gaussian_groups[level]:
            self.active_gaussian_groups[level].remove(group_idx)

        if group_idx in self.valid_groups:
            self.valid_groups.remove(group_idx)
            self.gaussian_groups[group_idx].clean()

    def export_group(self, group_idx):
        if group_idx not in self.valid_groups:
            raise ValueError("Gaussian group {} is not active".format(group_idx))
        return {
            name: value.detach().clone()
            for name, value in self.gaussian_groups[group_idx].splats.items()
        }

    def restore_gaussian_group(
        self, params, level=0, optimize=True, admission_certificate=None
    ):
        """Restore an archived group with fresh zero-initialized Adam state."""
        certificate = self._require_worldtest_certificate(
            admission_certificate, path="archive_restore"
        )
        group = Gaussians(
            BS=self.BS,
            scene_scale=self.scene_scale,
            init_config=self.init_gaussian_config,
            max_sh_degree=self.max_sh_degree,
        )
        group.to_device(self.device)
        group.replace_gaussians(params)
        if not optimize:
            group.disable_grad()
        group_idx = len(self.gaussian_groups)
        self.gaussian_groups.append(group)
        self.active_gaussian_groups[level].append(group_idx)
        self.valid_groups.append(group_idx)
        if certificate is not None:
            self.worldtest_group_certificates[group_idx] = {
                certificate.certificate_id
            }
        return group_idx

    def remove_optimization(self, group_idx):
        if self.gaussian_groups[group_idx].is_optimize:
            self.gaussian_groups[group_idx].disable_grad()
            return True
        else:
            return False

    def add_optimization(self, group_idx):
        if not self.gaussian_groups[group_idx].is_optimize:
            self.gaussian_groups[group_idx].enable_grad()
            return True
        else:
            return False

    @torch.no_grad()
    def propose_new_gaussians(
        self,
        cam,
        create_new_group=False,
        render_pkg=None,
        level=0,
        reference_cameras=None,
        causal_reference_cameras=None,
        coverage_recovery=False,
        coverage_recovery_translation_m=None,
        coverage_recovery_budget=None,
    ):
        """Run the unmodified host proposal generator without permanent writes.

        HashBlock is queried to retain MODP's coverage filter, but occupancy is
        only written by :meth:`commit_proposals`.
        """
        # t1 = torch.cuda.Event(enable_timing=True)
        # t2 = torch.cuda.Event(enable_timing=True)

        cur_view_scale_size = None

        if self.densification_mode == "adaptive_semi-dense_extra-pts":
            extra_pts_num = int(
                self.extra_pts_num // (2**level) ** 1.5
            )  # theoretically it should be 2.0, use 1.5 to generate more points
            if coverage_recovery_budget is not None:
                extra_pts_num = min(extra_pts_num, int(coverage_recovery_budget))
                if extra_pts_num < 1:
                    raise ValueError("Coverage recovery budget must be positive")
            candidate_pool_num = extra_pts_num
            if self._frontview_residual_cover_enabled():
                candidate_pool_num *= int(
                    self.frontview_residual_cover_config["pool_multiplier"]
                )
            elif self._frontview_birth_enabled():
                candidate_pool_num *= int(
                    self.frontview_birth_config["pool_multiplier"]
                )
            elif bool(self.frontview_sampling_config.get("enabled", False)):
                candidate_pool_num *= int(
                    self.frontview_sampling_config["pool_multiplier"]
                )

            sparse_depth = cam.get_sparse_depth(level)
            device = sparse_depth.device
            height = cam.get_height(level)
            w = cam.get_width(level)
            preserve_sparse_tracks = bool(
                self.frequency_sampling_config.get(
                    "preserve_sparse_track_geometry", False
                )
            )
            exact_sparse_world = None
            proposal_track_ids = None
            if preserve_sparse_tracks:
                sparse_rows = cam.get_color_pts_depth()
                sparse_world = (
                    np.asarray(sparse_rows[:, :3], dtype=np.float32)
                    if len(sparse_rows)
                    else np.empty((0, 3), dtype=np.float32)
                )
                observations = zbuffer_sparse_tracks(
                    sparse_world,
                    cam.get_raw_pose().detach().cpu().numpy(),
                    cam.get_int_mat(level).detach().cpu().numpy(),
                    w,
                    height,
                )
                pts_2d = torch.from_numpy(observations.uv).to(device=device)
                pts_depth = torch.from_numpy(observations.depths).to(device=device)
                exact_sparse_world = torch.from_numpy(observations.world_points).to(
                    device=device
                )
                camera_track_ids = np.asarray(cam.get_point_ids(), dtype=np.int64)
                proposal_track_ids = torch.from_numpy(
                    camera_track_ids[observations.source_indices]
                ).to(device=device, dtype=torch.long)
                sparse_depth = torch.zeros(
                    (height, w), device=device, dtype=pts_depth.dtype
                )
                if len(observations.pixel_indices):
                    pixels = torch.from_numpy(observations.pixel_indices).to(
                        device=device, dtype=torch.long
                    )
                    sparse_depth.reshape(-1)[pixels] = pts_depth
                sparse_pts_mask = sparse_depth.reshape(-1) > 0
            else:
                flatten_depth = sparse_depth.reshape(-1)
                sparse_pts_mask = flatten_depth > 0
                pts_2d_flat = torch.arange(len(flatten_depth), device=device)[
                    sparse_pts_mask
                ]
                pts_2d = torch.stack(
                    [pts_2d_flat % w + 0.5, pts_2d_flat // w + 0.5], axis=-1
                ).float()
                pts_depth = flatten_depth[sparse_pts_mask]
                if self._frontview_track_fusion_enabled() or bool(
                    self.frequency_sampling_config.get(
                        "propagate_raster_sparse_identity", False
                    )
                ):
                    proposal_track_ids = cam.get_sparse_point_ids(level).to(
                        device=device, dtype=torch.long
                    )
                    if len(proposal_track_ids) != len(pts_depth):
                        raise RuntimeError(
                            "Sparse track identities must align with sparse depth"
                        )
                else:
                    proposal_track_ids = torch.full(
                        (len(pts_depth),), -1, device=device, dtype=torch.long
                    )
            flatten_index = torch.arange(height * w, device=device)
            proposal_sparse_valid = torch.ones(
                (len(pts_depth),), device=device, dtype=torch.bool
            )
            proposal_depth_confidence = torch.ones(
                (len(pts_depth),), device=device, dtype=torch.float32
            )
            proposal_multiview_support = torch.zeros(
                (len(pts_depth),), device=device, dtype=torch.float32
            )
            proposal_budget_primary = torch.ones(
                (len(pts_depth),), device=device, dtype=torch.bool
            )
            proposal_tracked_metric = torch.zeros(
                (len(pts_depth),), device=device, dtype=torch.bool
            )
            fusion_pts_2d = pts_2d
            fusion_pts_depth = pts_depth
            fusion_track_ids = proposal_track_ids

            if len(pts_2d) > 500:
                if (
                    bool(self.frontview_sampling_config.get("enabled", False))
                    and self.frontview_sampling_config.get("anchor_selection_mode")
                    == "projective_coverage"
                ):
                    random_indexes = projective_coverage_indices(
                        pts_2d,
                        pts_depth,
                        torch.ones_like(pts_depth),
                        500,
                        self.frontview_sampling_config["depth_edges_m"],
                        self.frontview_sampling_config["depth_fractions"],
                        image_width=w,
                        cell_px=int(
                            self.frontview_sampling_config["anchor_cell_px"]
                        ),
                        shuffle=bool(
                            self.frontview_sampling_config[
                                "shuffle_anchor_coverage"
                            ]
                        ),
                        seed=int(self.frontview_sampling_config["shuffle_seed"])
                        + int(cam.cam_idx),
                    )
                elif len(pts_2d) > 3000:
                    random_indexes = torch.randint(len(pts_2d), (500,)).to(device)
                else:
                    random_indexes = torch.randperm(len(pts_2d))[:500].to(device)
                selected_pts_2d = pts_2d[random_indexes]
                selected_pts_depth = pts_depth[random_indexes]
            else:
                selected_pts_2d = pts_2d
                selected_pts_depth = pts_depth

            calibration_pts_2d = selected_pts_2d[:0]
            calibration_pts_depth = selected_pts_depth[:0]
            transport_enabled = bool(
                self.frontview_depth_transport_config.get("enabled", False)
            )
            if transport_enabled:
                transport_config = self.frontview_depth_transport_config
                training_indices, calibration_indices = split_depth_anchors(
                    len(selected_pts_2d),
                    transport_config["calibration_fraction"],
                    transport_config["min_training_anchors"],
                    transport_config["min_calibration_anchors"],
                    seed=int(transport_config["split_seed"]) + int(cam.cam_idx),
                    device=device,
                )
                if len(calibration_indices) > 0:
                    calibration_pts_2d = selected_pts_2d[calibration_indices]
                    calibration_pts_depth = selected_pts_depth[calibration_indices]
                    selected_pts_2d = selected_pts_2d[training_indices]
                    selected_pts_depth = selected_pts_depth[training_indices]

            landmark_conditioning_mode = self.causal_landmark_memory_config[
                "conditioning_mode"
            ]
            landmark_batch = None
            if (
                self.causal_landmark_memory.enabled
                and landmark_conditioning_mode in ("all_queries", "admitted_mean")
            ):
                occupied_pixels = torch.nonzero(
                    sparse_pts_mask, as_tuple=False
                ).flatten()
                remaining_conditioning_budget = max(0, 500 - len(selected_pts_2d))
                landmark_batch = self.causal_landmark_memory.project(
                    cam,
                    level,
                    exclude_ids=cam.get_point_ids(),
                    occupied_pixel_indices=occupied_pixels.detach().cpu().numpy(),
                    maximum_points=remaining_conditioning_budget,
                )
                if (
                    len(landmark_batch)
                    and landmark_conditioning_mode == "all_queries"
                ):
                    selected_pts_2d = torch.cat(
                        (
                            selected_pts_2d,
                            torch.from_numpy(landmark_batch.uv).to(
                                device=device, dtype=selected_pts_2d.dtype
                            ),
                        ),
                        dim=0,
                    )
                    selected_pts_depth = torch.cat(
                        (
                            selected_pts_depth,
                            torch.from_numpy(landmark_batch.depths).to(
                                device=device, dtype=selected_pts_depth.dtype
                            ),
                        ),
                        dim=0,
                    )

            frequency_importance = None
            footprint_evidence = None
            admission_evidence = None
            depth_prior_fallback_pixel_indices = None
            diff = None
            if render_pkg is not None:
                diff = render_pkg["diff"].reshape(-1)
                valid_extra = torch.logical_and(
                    (diff >= self.err_threshold), torch.logical_not(sparse_pts_mask)
                )

                valid_semi_pts = diff[sparse_pts_mask] >= self.semi_dense_err_threshold
                pts_2d = pts_2d[valid_semi_pts]
                pts_depth = pts_depth[valid_semi_pts]  # remove low error regions
                if exact_sparse_world is not None:
                    exact_sparse_world = exact_sparse_world[valid_semi_pts]
                proposal_sparse_valid = proposal_sparse_valid[valid_semi_pts]
                proposal_depth_confidence = proposal_depth_confidence[valid_semi_pts]
                proposal_multiview_support = proposal_multiview_support[
                    valid_semi_pts
                ]
                proposal_budget_primary = proposal_budget_primary[valid_semi_pts]
                proposal_tracked_metric = proposal_tracked_metric[valid_semi_pts]
                proposal_track_ids = proposal_track_ids[valid_semi_pts]

                extra_pts_2d = flatten_index[valid_extra]
                if len(extra_pts_2d) > 0 and extra_pts_num > 0:
                    if (
                        coverage_recovery
                        and self.frontview_coverage_recovery_config[
                            "depth_fallback_enabled"
                        ]
                    ):
                        recovery_config = self.frontview_coverage_recovery_config
                        extra_pts_2d = residual_grid_indices(
                            valid_extra.view(height, w),
                            diff.view(height, w),
                            recovery_config["depth_fallback_cell_px"],
                            candidate_pool_num,
                        )
                    elif bool(self.frequency_sampling_config.get("enabled", False)):
                        (
                            frequency_importance,
                            footprint_evidence,
                            admission_evidence,
                        ) = frequency_evidence_map(
                            cam.get_gt_image(level),
                            render_pkg["diff"],
                            render_pkg.get("opacity"),
                            {
                                **self.frequency_sampling_config,
                                "residual_threshold": self.err_threshold,
                            },
                        )
                        extra_pts_2d = sample_frequency_balanced_indices(
                            valid_extra.view(cam.get_height(level), w),
                            frequency_importance,
                            extra_pts_num,
                        )
                    else:
                        extra_pts_2d_index = torch.randint(
                            len(extra_pts_2d),
                            (min(candidate_pool_num, len(extra_pts_2d)),),
                        ).to(device)
                        extra_pts_2d = extra_pts_2d[extra_pts_2d_index]
                    extra_pts_2d = torch.stack(
                        [extra_pts_2d % w + 0.5, extra_pts_2d // w + 0.5], axis=-1
                    ).float()

                else:
                    extra_pts_2d = None
            else:
                # extra_pts_2d = np.random.choice(flatten_index, extra_pts_num, replace=False)
                extra_pts_2d_index = torch.randperm(len(flatten_index))[
                    :candidate_pool_num
                ].to(device)
                extra_pts_2d = flatten_index[extra_pts_2d_index]
                extra_pts_2d = torch.stack(
                    [extra_pts_2d % w + 0.5, extra_pts_2d // w + 0.5], axis=-1
                ).float()

            if extra_pts_2d is not None:
                # est_depth, std_mask = self.depth_cov_estimator.query(cam.get_gt_image(level), selected_pts_depth, selected_pts_2d, extra_pts_2d)
                candidate_count = len(extra_pts_2d)
                query_pts_2d = extra_pts_2d
                original_posterior_depth_all = None
                if len(calibration_pts_2d) > 0:
                    query_pts_2d = torch.cat(
                        (extra_pts_2d, calibration_pts_2d), dim=0
                    )
                if (
                    landmark_conditioning_mode == "admitted_mean"
                    and landmark_batch is not None
                    and len(landmark_batch)
                ):
                    depth_context = self.depth_cov_estimator.prepare_tensor_context(
                        cam.get_gt_image(level)
                    )
                    est_depth_all, std_mask_all, depth_std_all = (
                        self.depth_cov_estimator.query_prepared_tensor(
                            depth_context,
                            selected_pts_depth,
                            selected_pts_2d,
                            query_pts_2d,
                        )
                    )
                    conditioned_uv = torch.cat(
                        (
                            selected_pts_2d,
                            torch.from_numpy(landmark_batch.uv).to(
                                device=device, dtype=selected_pts_2d.dtype
                            ),
                        ),
                        dim=0,
                    )
                    conditioned_depth = torch.cat(
                        (
                            selected_pts_depth,
                            torch.from_numpy(landmark_batch.depths).to(
                                device=device, dtype=selected_pts_depth.dtype
                            ),
                        ),
                        dim=0,
                    )
                    conditioned_estimate, _, conditioned_std = (
                        self.depth_cov_estimator.query_prepared_tensor(
                            depth_context,
                            conditioned_depth,
                            conditioned_uv,
                            query_pts_2d,
                        )
                    )
                    admitted = torch.zeros_like(std_mask_all)
                    admitted[:candidate_count] = std_mask_all[:candidate_count]
                    transport_weights = torch.ones_like(est_depth_all)
                    transported_estimate = conditioned_estimate
                    if (
                        self.causal_landmark_memory_config["transport_rule"]
                        == "variance_gain"
                    ):
                        transported_estimate, transport_weights = (
                            information_gain_transport(
                                est_depth_all,
                                conditioned_estimate,
                                depth_std_all,
                                conditioned_std,
                            )
                        )
                    self.causal_landmark_memory.record_admitted_mean(
                        est_depth_all[admitted].detach().cpu().numpy(),
                        transported_estimate[admitted].detach().cpu().numpy(),
                        len(landmark_batch),
                        transport_weights[admitted].detach().cpu().numpy(),
                        depth_std_all[admitted].detach().cpu().numpy(),
                        conditioned_std[admitted].detach().cpu().numpy(),
                    )
                    if self.causal_landmark_memory.uses_original_posterior_responsibility:
                        original_posterior_depth_all = est_depth_all.clone()
                    est_depth_all[admitted] = transported_estimate[admitted]
                    if self.causal_landmark_memory_config[
                        "propagate_conditioned_uncertainty"
                    ]:
                        depth_std_all[admitted] = conditioned_std[admitted]
                else:
                    est_depth_all, std_mask_all, depth_std_all = (
                        self.depth_cov_estimator.query_tensor(
                            cam.get_gt_image(level),
                            selected_pts_depth,
                            selected_pts_2d,
                            query_pts_2d,
                            return_std=True,
                        )
                    )
                est_depth = est_depth_all[:candidate_count]
                responsibility_est_depth = (
                    est_depth.clone()
                    if original_posterior_depth_all is None
                    else original_posterior_depth_all[:candidate_count]
                )
                std_mask = std_mask_all[:candidate_count]
                original_responsibility_mask = (
                    torch.zeros_like(std_mask)
                    if original_posterior_depth_all is None
                    else std_mask.clone()
                )
                depth_std = depth_std_all[:candidate_count]
                calibration_pred_depth = est_depth_all[candidate_count:]
                calibration_valid = std_mask_all[candidate_count:]
                causal_metric_batch = None
                causal_posterior_prior_maps = None

                if (
                    self.causal_landmark_memory.enabled
                    and landmark_conditioning_mode == "fallback_repair"
                    and coverage_recovery
                    and int(std_mask.sum().item())
                    < int(
                        self.frontview_coverage_recovery_config[
                            "depth_fallback_min_valid"
                        ]
                    )
                ):
                    valid_before_count = int(std_mask.sum().item())
                    invalid_before_count = int((~std_mask).sum().item())
                    occupied_pixels = torch.nonzero(
                        sparse_pts_mask, as_tuple=False
                    ).flatten()
                    landmark_batch = self.causal_landmark_memory.project(
                        cam,
                        level,
                        exclude_ids=cam.get_point_ids(),
                        occupied_pixel_indices=(
                            occupied_pixels.detach().cpu().numpy()
                        ),
                        maximum_points=max(0, 500 - len(selected_pts_2d)),
                    )
                    if len(landmark_batch):
                        repaired_uv = torch.cat(
                            (
                                selected_pts_2d,
                                torch.from_numpy(landmark_batch.uv).to(
                                    device=device,
                                    dtype=selected_pts_2d.dtype,
                                ),
                            ),
                            dim=0,
                        )
                        repaired_depth = torch.cat(
                            (
                                selected_pts_depth,
                                torch.from_numpy(landmark_batch.depths).to(
                                    device=device,
                                    dtype=selected_pts_depth.dtype,
                                ),
                            ),
                            dim=0,
                        )
                        invalid_before = ~std_mask
                        repaired_estimate, repaired_valid, repaired_std = (
                            self.depth_cov_estimator.query_tensor(
                                cam.get_gt_image(level),
                                repaired_depth,
                                repaired_uv,
                                extra_pts_2d,
                                return_std=True,
                            )
                        )
                        newly_valid = invalid_before & repaired_valid
                        est_depth[newly_valid] = repaired_estimate[newly_valid]
                        depth_std[newly_valid] = repaired_std[newly_valid]
                        std_mask[newly_valid] = True
                        newly_valid_count = int(newly_valid.sum().item())
                    else:
                        newly_valid_count = 0
                    self.causal_landmark_memory.record_repair(
                        valid_before=valid_before_count,
                        invalid_before=invalid_before_count,
                        newly_valid=newly_valid_count,
                        conditioning_landmarks=len(landmark_batch),
                    )

                if (
                    coverage_recovery
                    and self.frontview_coverage_recovery_config[
                        "depth_fallback_enabled"
                    ]
                ):
                    sparse_prior_depth = self.frontview_coverage_depth_prior.estimate(
                        int(cam.cam_idx)
                    )
                    valid_before = int(std_mask.sum().item())
                    fallback_candidate_mask = torch.zeros_like(std_mask)

                    motion_floor = None
                    prior_depth = sparse_prior_depth
                    if (
                        prior_depth is not None
                        and coverage_recovery_translation_m is not None
                        and self.frontview_coverage_recovery_config[
                            "depth_fallback_motion_floor_enabled"
                        ]
                    ):
                        motion_floor = motion_conditioned_depth_floor(
                            coverage_recovery_translation_m,
                            0.5 * (cam.get_fx(level) + cam.get_fy(level)),
                            self.frontview_coverage_recovery_config[
                                "depth_fallback_max_projected_drift_px"
                            ],
                            self.frontview_coverage_recovery_config[
                                "depth_fallback_motion_floor_max_m"
                            ],
                        )
                        prior_depth = max(float(prior_depth), float(motion_floor))
                        stats = self.frontview_coverage_depth_stats
                        stats["depth_fallback_motion_floor_calls"] += 1
                        stats["depth_fallback_motion_floor_sum_m"] += float(
                            motion_floor
                        )
                        stats["last_depth_fallback_motion_floor_m"] = float(
                            motion_floor
                        )
                    causal_metric_triggered = bool(
                        self.causal_metric_birth.enabled
                        and valid_before
                        < int(
                            self.frontview_coverage_recovery_config[
                                "depth_fallback_min_valid"
                            ]
                        )
                    )
                    causal_replaces_fallback = bool(
                        causal_metric_triggered
                        and causal_birth_replaces_depth_fallback(
                            self.causal_metric_birth_config,
                            dual_responsibility_enabled=(
                                self.causal_dual_responsibility_config["enabled"]
                            ),
                        )
                    )
                    if causal_metric_triggered:
                        causal_metric_batch = self.causal_metric_birth.certify(
                            cam,
                            list(
                                causal_reference_cameras
                                if causal_reference_cameras is not None
                                else (reference_cameras or ())
                            ),
                            level,
                            valid_extra.view(height, w),
                            diff.view(height, w),
                            float(self.depth_cov_estimator.std_valid_threshold),
                            int(extra_pts_num),
                            query_uv=extra_pts_2d[~std_mask],
                        )
                        if (
                            self.causal_metric_birth_config["birth_mode"]
                            == "tracked_features"
                            and self.causal_metric_birth_config["posterior_action"]
                            != "fuse"
                        ):
                            causal_metric_batch = None
                        if (
                            self.causal_metric_birth_config["birth_mode"]
                            == "depthcov_recondition"
                        ):
                            reconditioned_uv = selected_pts_2d
                            reconditioned_depth = selected_pts_depth
                            if len(causal_metric_batch):
                                reconditioned_uv = torch.cat(
                                    (
                                        reconditioned_uv,
                                        torch.from_numpy(causal_metric_batch.uv).to(
                                            device=device,
                                            dtype=reconditioned_uv.dtype,
                                        ),
                                    ),
                                    dim=0,
                                )
                                reconditioned_depth = torch.cat(
                                    (
                                        reconditioned_depth,
                                        torch.from_numpy(
                                            causal_metric_batch.depths
                                        ).to(
                                            device=device,
                                            dtype=reconditioned_depth.dtype,
                                        ),
                                    ),
                                    dim=0,
                                )
                            (
                                reconditioned_estimate,
                                reconditioned_valid,
                                reconditioned_std,
                            ) = self.depth_cov_estimator.query_tensor(
                                cam.get_gt_image(level),
                                reconditioned_depth,
                                reconditioned_uv,
                                query_pts_2d,
                                return_std=True,
                            )
                            est_depth_all = reconditioned_estimate
                            std_mask_all = reconditioned_valid
                            depth_std_all = reconditioned_std
                            est_depth = est_depth_all[:candidate_count]
                            std_mask = std_mask_all[:candidate_count]
                            depth_std = depth_std_all[:candidate_count]
                            calibration_pred_depth = est_depth_all[candidate_count:]
                            calibration_valid = std_mask_all[candidate_count:]
                            self.causal_metric_birth.record_reconditioned_validity(
                                valid_before, int(std_mask.sum().item())
                            )
                            causal_metric_batch = None
                        if (
                            self.causal_metric_birth_config["birth_mode"]
                            == "cross_fitted_gauge"
                            and causal_metric_batch is not None
                        ):
                            gauge_applied_rows = 0
                            fallback_hypotheses = [
                                value
                                for value in (sparse_prior_depth, motion_floor)
                                if value is not None
                            ]
                            if len(causal_metric_batch) >= 4 and fallback_hypotheses:
                                track_uv = torch.from_numpy(
                                    causal_metric_batch.uv
                                ).to(device=device, dtype=selected_pts_2d.dtype)
                                (
                                    track_field_depth,
                                    _,
                                    track_field_std,
                                ) = self.depth_cov_estimator.query_tensor(
                                    cam.get_gt_image(level),
                                    selected_pts_depth,
                                    selected_pts_2d,
                                    track_uv,
                                    return_std=True,
                                )
                                gauge = cross_fitted_track_depth_gauge(
                                    causal_metric_batch.uv,
                                    track_field_depth.detach().cpu().numpy(),
                                    track_field_std.detach().cpu().numpy(),
                                    causal_metric_batch.depths,
                                    causal_metric_batch.log_depth_stds,
                                    fallback_hypotheses,
                                    fallback_log_std=float(
                                        self.depth_cov_estimator.std_valid_threshold
                                    ),
                                    shuffle_binding=bool(
                                        self.causal_metric_birth_config[
                                            "shuffle_field_binding"
                                        ]
                                    ),
                                    seed=(
                                        int(
                                            self.causal_metric_birth_config[
                                                "shuffle_seed"
                                            ]
                                        )
                                        + int(cam.cam_idx)
                                    ),
                                )
                            else:
                                gauge = cross_fitted_track_depth_gauge(
                                    np.empty((0, 2), dtype=np.float32),
                                    np.empty((0,), dtype=np.float32),
                                    np.empty((0,), dtype=np.float32),
                                    np.empty((0,), dtype=np.float32),
                                    np.empty((0,), dtype=np.float32),
                                    fallback_hypotheses or [1.0],
                                    fallback_log_std=float(
                                        self.depth_cov_estimator.std_valid_threshold
                                    ),
                                )
                            apply_gauge = (
                                self.causal_metric_birth_config["posterior_action"]
                                == "fuse"
                            )
                            if apply_gauge and gauge.accepted_field:
                                finite_field = (
                                    torch.isfinite(est_depth)
                                    & torch.isfinite(depth_std)
                                    & (est_depth > 0.0)
                                )
                                predictive_std = torch.sqrt(
                                    torch.clamp(
                                        depth_std.square()
                                        + float(gauge.log_scale_variance),
                                        min=0.0,
                                    )
                                )
                                field_rows = finite_field
                                est_depth[field_rows] *= math.exp(gauge.log_scale)
                                depth_std[field_rows] = predictive_std[field_rows]
                                std_mask[field_rows] = True
                                gauge_applied_rows = int(field_rows.sum().item())
                                prior_depth = None
                            elif apply_gauge and gauge.selected_model == "fallback":
                                prior_depth = float(gauge.selected_fallback_depth)
                            elif apply_gauge:
                                prior_depth = None
                            self.causal_metric_birth.record_depth_gauge(
                                gauge,
                                applied_rows=gauge_applied_rows,
                                frame_id=int(cam.cam_idx),
                            )
                            causal_metric_batch = None
                        if (
                            self.causal_metric_birth_config["birth_mode"]
                            == "posterior_proxy"
                            and causal_metric_batch is not None
                        ):
                            query_pixels = (
                                extra_pts_2d[:, 1].int() * w
                                + extra_pts_2d[:, 0].int()
                            )
                            prior_depth_map = torch.full(
                                (height * w,),
                                float("nan"),
                                device=device,
                                dtype=est_depth.dtype,
                            )
                            prior_std_map = torch.full(
                                (height * w,),
                                float(self.depth_cov_estimator.std_valid_threshold),
                                device=device,
                                dtype=depth_std.dtype,
                            )
                            prior_valid_map = torch.zeros(
                                (height * w,), device=device, dtype=torch.bool
                            )
                            prior_depth_map[query_pixels] = est_depth
                            prior_std_map[query_pixels] = depth_std
                            prior_valid_map[query_pixels] = std_mask
                            causal_posterior_prior_maps = (
                                prior_depth_map,
                                prior_std_map,
                                prior_valid_map,
                            )
                    if (
                        prior_depth is not None
                        and valid_before
                        < int(
                        self.frontview_coverage_recovery_config[
                            "depth_fallback_min_valid"
                        ]
                        )
                        and not causal_replaces_fallback
                    ):
                        fallback = ~std_mask
                        fallback_candidate_mask = fallback.clone()
                        fallback_uv = extra_pts_2d[fallback]
                        depth_prior_fallback_pixel_indices = (
                            fallback_uv[:, 1].int() * w + fallback_uv[:, 0].int()
                        )
                    map_rows = 0
                    fallback_kwargs = {
                        "prior_depth_m": prior_depth,
                        "std_valid_threshold": self.depth_cov_estimator.std_valid_threshold,
                        "min_valid": self.frontview_coverage_recovery_config[
                            "depth_fallback_min_valid"
                        ],
                        "confidence": self.frontview_coverage_recovery_config[
                            "depth_fallback_confidence"
                        ],
                    }
                    if causal_replaces_fallback:
                        fallback_rows = 0
                        map_rows = 0
                    elif (
                        self.frontview_coverage_recovery_config[
                            "depth_fallback_map_enabled"
                        ]
                        and render_pkg.get("depth") is not None
                        and render_pkg.get("opacity") is not None
                    ):
                        extra_pixel_indices = (
                            extra_pts_2d[:, 1].int() * w
                            + extra_pts_2d[:, 0].int()
                        )
                        (
                            est_depth,
                            std_mask,
                            depth_std,
                            fallback_rows,
                            map_rows,
                        ) = apply_visible_surface_depth_fallback(
                            est_depth,
                            std_mask,
                            depth_std,
                            visible_depth=self._geometry_depth(render_pkg).reshape(-1)[
                                extra_pixel_indices
                            ],
                            visible_opacity=self._geometry_opacity(render_pkg).reshape(-1)[
                                extra_pixel_indices
                            ],
                            front_ratio=self.frontview_coverage_recovery_config[
                                "depth_fallback_map_front_ratio"
                            ],
                            min_opacity=self.frontview_coverage_recovery_config[
                                "depth_fallback_map_min_opacity"
                            ],
                            min_prior_ratio=self.frontview_coverage_recovery_config[
                                "depth_fallback_map_min_prior_ratio"
                            ],
                            max_prior_ratio=self.frontview_coverage_recovery_config[
                                "depth_fallback_map_max_prior_ratio"
                            ],
                            **fallback_kwargs,
                        )
                    else:
                        (
                            est_depth,
                            std_mask,
                            depth_std,
                            fallback_rows,
                        ) = apply_sparse_depth_prior_fallback(
                            est_depth,
                            std_mask,
                            depth_std,
                            **fallback_kwargs,
                        )
                    stats = self.frontview_coverage_depth_stats
                    if fallback_rows:
                        stats["depth_fallback_calls"] += 1
                        stats["depth_fallback_rows"] += int(fallback_rows)
                        stats["depth_fallback_map_rows"] += int(map_rows)
                        stats["depth_fallback_prior_rows"] += int(
                            fallback_rows - map_rows
                        )
                        stats["last_depth_fallback_frame"] = int(cam.cam_idx)
                        if self.frontview_coverage_recovery_config[
                            "multiview_depth_enabled"
                        ]:
                            fallback_indices = torch.nonzero(
                                fallback_candidate_mask, as_tuple=False
                            ).flatten()
                            selected_depths, support_scores, hypothesis_count = (
                                self._frontview_recovery_multiview_depths(
                                    cam,
                                    list(reference_cameras or ()),
                                    extra_pts_2d[fallback_indices],
                                    est_depth[fallback_indices],
                                    torch.full_like(
                                        est_depth[fallback_indices],
                                        float(
                                            self.frontview_coverage_recovery_config[
                                                "depth_fallback_confidence"
                                            ]
                                        ),
                                    ),
                                    prior_depth,
                                    level,
                                )
                            )
                            est_depth[fallback_indices] = selected_depths
                            supported = support_scores > 0.0
                            if self.frontview_coverage_recovery_config[
                                "multiview_depth_mode"
                            ] == "posterior_inverse_depth":
                                depth_std[fallback_indices] = (
                                    self.depth_cov_estimator.std_valid_threshold
                                    * (1.0 - support_scores)
                                )
                            stats["multiview_depth_calls"] += 1
                            stats["multiview_depth_rows"] += int(
                                fallback_indices.numel()
                            )
                            stats["multiview_depth_supported_rows"] += int(
                                supported.sum().item()
                            )
                            stats["multiview_depth_concentrated_rows"] += int(
                                (support_scores >= 0.5).sum().item()
                            )
                            stats["multiview_depth_score_sum"] += float(
                                support_scores.sum().item()
                            )
                            if self.frontview_coverage_recovery_config[
                                "multiview_depth_mode"
                            ] == "posterior_inverse_depth":
                                stats["multiview_depth_concentration_sum"] += float(
                                    support_scores.sum().item()
                                )
                            stats["multiview_depth_selected_sum_m"] += float(
                                selected_depths[supported].sum().item()
                            )
                            stats["multiview_depth_hypotheses_sum"] += int(
                                hypothesis_count
                            )
                            stats["multiview_depth_shuffled_calls"] += int(
                                self.frontview_coverage_recovery_config[
                                    "shuffle_multiview_depth"
                                ]
                            )
                    elif prior_depth is None and valid_before < int(
                        self.frontview_coverage_recovery_config[
                            "depth_fallback_min_valid"
                        ]
                    ):
                        stats["depth_fallback_skipped_no_prior"] += 1

                # Log("Valid additional pts: {}".format(np.sum(std_mask)), tag="GaussianModel")

                if torch.sum(std_mask) > 0 or (
                    causal_metric_batch is not None and len(causal_metric_batch) > 0
                ):
                    responsibility_est_depth = torch.where(
                        original_responsibility_mask,
                        responsibility_est_depth,
                        est_depth,
                    )
                    extra_pts_depth = est_depth[std_mask]
                    extra_responsibility_depth = responsibility_est_depth[std_mask]
                    extra_pts_2d = extra_pts_2d[std_mask]
                    extra_depth_confidence = torch.clamp(
                        1.0
                        - depth_std[std_mask]
                        / max(self.depth_cov_estimator.std_valid_threshold, 1.0e-8),
                        min=0.0,
                        max=1.0,
                    )
                    extra_budget_primary = torch.ones(
                        len(extra_pts_depth), device=device, dtype=torch.bool
                    )
                    if transport_enabled:
                        stats = self.frontview_depth_transport_stats
                        stats["calls"] += 1
                        stats["training_anchors"] += len(selected_pts_2d)
                        stats["calibration_anchors"] += len(calibration_pts_2d)
                        calibration_valid &= (
                            torch.isfinite(calibration_pred_depth)
                            & torch.isfinite(calibration_pts_depth)
                            & (calibration_pred_depth > 0)
                            & (calibration_pts_depth > 0)
                        )
                        valid_calibration_count = int(calibration_valid.sum().item())
                        stats["valid_calibration_anchors"] += valid_calibration_count
                        minimum = int(
                            self.frontview_depth_transport_config[
                                "min_calibration_anchors"
                            ]
                        )
                        if valid_calibration_count >= minimum:
                            valid_pred = calibration_pred_depth[calibration_valid]
                            valid_true = calibration_pts_depth[calibration_valid]
                            stats["calibrated_calls"] += 1
                            stats["absolute_log_residual_sum"] += float(
                                torch.sum(
                                    torch.abs(torch.log(valid_true) - torch.log(valid_pred))
                                ).item()
                            )
                            if self.frontview_depth_transport_config[
                                "apply_correction"
                            ]:
                                extra_pts_depth, corrections = (
                                    transport_candidate_depths(
                                        extra_pts_2d,
                                        extra_pts_depth,
                                        calibration_pts_2d[calibration_valid],
                                        valid_pred,
                                        valid_true,
                                        cam.get_gt_image(level),
                                        neighbors=self.frontview_depth_transport_config[
                                            "neighbors"
                                        ],
                                        clip_quantiles=self.frontview_depth_transport_config[
                                            "clip_quantiles"
                                        ],
                                        shuffle_residual_locations=self.frontview_depth_transport_config[
                                            "shuffle_residual_locations"
                                        ],
                                        seed=int(
                                            self.frontview_depth_transport_config[
                                                "split_seed"
                                            ]
                                        )
                                        + int(cam.cam_idx),
                                    )
                                )
                                stats["corrected_rows"] += len(corrections)
                                stats["absolute_log_correction_sum"] += float(
                                    torch.sum(torch.abs(corrections)).item()
                                )
                        else:
                            stats["skipped_calls"] += 1
                    if self._frontview_birth_enabled():
                        if bool(
                            self.frontview_birth_config[
                                "temporal_map_competition"
                            ]
                        ):
                            temporal_reject, temporal_stats = (
                                self._temporal_birth_rejections(
                                    cam,
                                    reference_cameras,
                                    extra_pts_2d,
                                    extra_pts_depth,
                                    level,
                                )
                            )
                            temporal_keep = ~temporal_reject
                            extra_pts_depth = extra_pts_depth[temporal_keep]
                            extra_responsibility_depth = extra_responsibility_depth[
                                temporal_keep
                            ]
                            extra_pts_2d = extra_pts_2d[temporal_keep]
                            extra_depth_confidence = extra_depth_confidence[
                                temporal_keep
                            ]
                            temporal_accumulator = self.frontview_birth_stats
                            temporal_accumulator["temporal_calls"] += 1
                            for key in (
                                "tested_rows",
                                "free_space_rows",
                                "duplicate_rows",
                                "rejected_rows",
                            ):
                                temporal_accumulator["temporal_" + key] += int(
                                    temporal_stats[key]
                                )
                        pixel_indices = (
                            extra_pts_2d[:, 1].int() * w + extra_pts_2d[:, 0].int()
                        )
                        if render_pkg is None:
                            birth_residual = torch.ones_like(extra_pts_depth)
                            birth_map_depth = torch.full_like(extra_pts_depth, -1.0)
                            birth_map_opacity = torch.zeros_like(extra_pts_depth)
                        else:
                            birth_residual = render_pkg["diff"].reshape(-1)[
                                pixel_indices
                            ]
                            birth_map_depth = self._geometry_depth(render_pkg).reshape(-1)[
                                pixel_indices
                            ]
                            opacity = self._geometry_opacity(render_pkg)
                            birth_map_opacity = (
                                torch.zeros_like(extra_pts_depth)
                                if opacity is None
                                else opacity.reshape(-1)[pixel_indices]
                            )
                        atlas_occupied, atlas_projected = (
                            self._current_projective_map_occupancy(
                                cam, extra_pts_2d, extra_pts_depth, level
                            )
                        )
                        anchor_occupied = None
                        if bool(
                            self.frontview_birth_config[
                                "sparse_anchor_competition"
                            ]
                        ):
                            anchor_occupied = multi_layer_projective_occupancy(
                                extra_pts_2d,
                                extra_pts_depth,
                                pts_2d,
                                pts_depth,
                                self.frontview_birth_config,
                            )
                        selection, birth_stats = layered_projective_birth_indices(
                            extra_pts_2d,
                            extra_pts_depth,
                            extra_depth_confidence,
                            birth_residual,
                            birth_map_depth,
                            birth_map_opacity,
                            extra_pts_num,
                            self.frontview_birth_config,
                            seed=int(self.frontview_birth_config["selection_seed"])
                            + int(cam.cam_idx),
                            map_occupied=atlas_occupied,
                            anchor_occupied=anchor_occupied,
                        )
                        extra_pts_depth = extra_pts_depth[selection]
                        extra_responsibility_depth = extra_responsibility_depth[
                            selection
                        ]
                        extra_pts_2d = extra_pts_2d[selection]
                        extra_depth_confidence = extra_depth_confidence[selection]
                        extra_budget_primary = extra_budget_primary[selection]
                        stats = self.frontview_birth_stats
                        stats["calls"] += 1
                        stats["pool_rows"] += int(birth_stats["pool"])
                        stats["selected_rows"] += int(birth_stats["selected"])
                        stats["priority_selected_rows"] += int(
                            birth_stats["priority_selected"]
                        )
                        stats["coverage_selected_rows"] += int(
                            birth_stats["coverage_selected"]
                        )
                        stats["fallback_selected_rows"] += int(
                            birth_stats["fallback_selected"]
                        )
                        stats["map_rejected_rows"] += int(
                            birth_stats["map_rejected"]
                        )
                        stats["atlas_projected_rows"] += int(atlas_projected)
                        stats["atlas_rejected_rows"] += int(
                            birth_stats.get("atlas_rejected", 0)
                        )
                        stats["anchor_rejected_rows"] += int(
                            birth_stats.get("anchor_rejected", 0)
                        )
                        stats["cell_rejected_rows"] += int(
                            birth_stats["cell_rejected"]
                        )
                        stats["pool_layer_counts"] = [
                            old + new
                            for old, new in zip(
                                stats["pool_layer_counts"],
                                birth_stats.get("pool_layer_counts", [0, 0, 0]),
                            )
                        ]
                        stats["selected_layer_counts"] = [
                            old + new
                            for old, new in zip(
                                stats["selected_layer_counts"],
                                birth_stats.get("selected_layer_counts", [0, 0, 0]),
                            )
                        ]
                        stats["last_layer_edges_m"] = birth_stats.get(
                            "layer_edges", []
                        )
                    elif (
                        bool(self.frontview_sampling_config.get("enabled", False))
                        and len(extra_pts_depth) > extra_pts_num
                    ):
                        selection_mode = self.frontview_sampling_config[
                            "selection_mode"
                        ]
                        scores = torch.zeros_like(extra_depth_confidence)
                        valid_reference = torch.zeros_like(
                            extra_depth_confidence, dtype=torch.bool
                        )
                        selection_seed = int(
                            self.frontview_sampling_config["shuffle_seed"]
                        ) + int(cam.cam_idx)
                        if selection_mode == "depth_stratified":
                            selection = depth_stratified_indices(
                                extra_pts_depth,
                                extra_pts_num,
                                self.frontview_sampling_config["depth_edges_m"],
                                self.frontview_sampling_config["depth_fractions"],
                                seed=selection_seed,
                                shuffle_depth_bands=bool(
                                    self.frontview_sampling_config[
                                        "shuffle_depth_bands"
                                    ]
                                ),
                            )
                            adaptive_metadata = None
                        elif selection_mode.startswith("adaptive_log_depth_"):
                            if diff is None:
                                residuals = torch.ones_like(extra_depth_confidence)
                            else:
                                candidate_pixels = (
                                    extra_pts_2d[:, 1].int() * w
                                    + extra_pts_2d[:, 0].int()
                                )
                                residuals = diff[candidate_pixels]
                            scores = residuals * torch.clamp(
                                extra_depth_confidence, 0.0, 1.0
                            )
                            rate_distortion_metadata = None
                            density_weights = None
                            if selection_mode in (
                                "adaptive_log_depth_rate_distortion",
                                "adaptive_log_depth_rate_distortion_shuffled",
                            ):
                                (
                                    density_weights,
                                    rate_distortion_metadata,
                                ) = rate_distortion_density_weights(
                                    cam.get_gt_image(level),
                                    extra_pts_2d,
                                    extra_depth_confidence,
                                    image_size=(w, cam.get_height(level)),
                                    budget=extra_pts_num,
                                    pool_multiplier=int(
                                        self.frontview_sampling_config[
                                            "pool_multiplier"
                                        ]
                                    ),
                                )
                            selection, adaptive_metadata = (
                                adaptive_log_depth_indices(
                                    extra_responsibility_depth,
                                    extra_depth_confidence,
                                    residuals,
                                    extra_pts_num,
                                    uv=extra_pts_2d,
                                    image_size=(w, cam.get_height(level)),
                                    pool_multiplier=int(
                                        self.frontview_sampling_config[
                                            "pool_multiplier"
                                        ]
                                    ),
                                    weighted=(
                                        selection_mode
                                        == "adaptive_log_depth_importance"
                                    ),
                                    shuffle_regimes=selection_mode in (
                                        "adaptive_log_depth_shuffled",
                                        "adaptive_log_depth_coverage_shuffled",
                                    ),
                                    coverage_priority=(
                                        "confidence"
                                        if selection_mode
                                        in (
                                            "adaptive_log_depth_coverage",
                                            "adaptive_log_depth_coverage_shuffled",
                                        )
                                        else (
                                            "residual_confidence"
                                            if selection_mode
                                            == "adaptive_log_depth_residual_coverage"
                                            else None
                                        )
                                    ),
                                    density_weights=density_weights,
                                    shuffle_density=(
                                        selection_mode
                                        == "adaptive_log_depth_rate_distortion_shuffled"
                                    ),
                                    seed=selection_seed,
                                )
                            )
                        elif selection_mode == "projective_coverage":
                            selection = projective_coverage_indices(
                                extra_pts_2d,
                                extra_responsibility_depth,
                                extra_depth_confidence,
                                extra_pts_num,
                                self.frontview_sampling_config["depth_edges_m"],
                                self.frontview_sampling_config["depth_fractions"],
                                image_width=w,
                                cell_px=int(
                                    self.frontview_sampling_config[
                                        "projective_cell_px"
                                    ]
                                ),
                                shuffle=bool(
                                    self.frontview_sampling_config[
                                        "shuffle_projective_coverage"
                                    ]
                                ),
                                seed=selection_seed,
                            )
                            adaptive_metadata = None
                        elif selection_mode == "residual_importance":
                            if diff is None:
                                residuals = torch.ones_like(extra_depth_confidence)
                            else:
                                candidate_pixels = (
                                    extra_pts_2d[:, 1].int() * w
                                    + extra_pts_2d[:, 0].int()
                                )
                                residuals = diff[candidate_pixels]
                            scores = residuals * torch.clamp(
                                extra_depth_confidence, 0.0, 1.0
                            )
                            selection = residual_importance_indices(
                                residuals,
                                extra_depth_confidence,
                                extra_pts_num,
                                seed=selection_seed,
                            )
                            adaptive_metadata = None
                        else:
                            if selection_mode == "evidence_balanced":
                                scores, valid_reference, _ = (
                                    self._frontview_reprojection_scores(
                                        cam,
                                        list(reference_cameras or ()),
                                        extra_pts_2d,
                                        extra_pts_depth,
                                        extra_depth_confidence,
                                        level,
                                    )
                                )
                            selection = evidence_balanced_indices(
                                scores,
                                extra_pts_num,
                                (
                                    float(
                                        self.frontview_sampling_config[
                                            "evidence_fraction"
                                        ]
                                    )
                                    if selection_mode == "evidence_balanced"
                                    else 0.0
                                ),
                                shuffle_evidence=bool(
                                    self.frontview_sampling_config[
                                        "shuffle_evidence"
                                    ]
                                ),
                                seed=selection_seed,
                            )
                            adaptive_metadata = None
                        stats = self.frontview_sampling_stats
                        stats["calls"] += 1
                        stats["pool_rows"] += int(scores.numel())
                        stats["selected_rows"] += int(selection.numel())
                        stats["valid_reference_rows"] += int(
                            valid_reference.sum().item()
                        )
                        stats["score_sum"] += float(scores[selection].sum().item())
                        stats["last_mean_score"] = float(
                            scores[selection].mean().item()
                        )
                        if adaptive_metadata is not None:
                            pool_counts = adaptive_metadata["pool_counts"]
                            selected_counts = adaptive_metadata["selected_counts"]
                            stats["adaptive_calls"] += 1
                            stats["adaptive_iterations"] += int(
                                adaptive_metadata["iterations"]
                            )
                            stats["adaptive_objective_sum"] += float(
                                adaptive_metadata["objective"]
                            )
                            stats["adaptive_boundary_sum_m"] = [
                                old + new
                                for old, new in zip(
                                    stats["adaptive_boundary_sum_m"],
                                    adaptive_metadata["boundaries_m"],
                                )
                            ]
                            stats["last_adaptive_boundaries_m"] = (
                                adaptive_metadata["boundaries_m"]
                            )
                            stats["adaptive_assigned_pool_counts"] = [
                                old + new
                                for old, new in zip(
                                    stats["adaptive_assigned_pool_counts"],
                                    adaptive_metadata["assigned_pool_counts"],
                                )
                            ]
                            stats["adaptive_quotas"] = [
                                old + new
                                for old, new in zip(
                                    stats["adaptive_quotas"],
                                    adaptive_metadata["quotas"],
                                )
                            ]
                            if adaptive_metadata["coverage_cell_px"] is not None:
                                stats["adaptive_coverage_calls"] += 1
                                stats[
                                    "adaptive_coverage_cell_pixel_sum"
                                ] += float(adaptive_metadata["coverage_cell_px"])
                                stats["adaptive_coverage_representatives"] = [
                                    old + new
                                    for old, new in zip(
                                        stats[
                                            "adaptive_coverage_representatives"
                                        ],
                                        adaptive_metadata[
                                            "coverage_representatives"
                                        ],
                                    )
                                ]
                            if rate_distortion_metadata is not None:
                                stats["rate_distortion_calls"] += 1
                                stats["rate_distortion_cell_pixel_sum"] += float(
                                    rate_distortion_metadata["cell_px"]
                                )
                                density_min = rate_distortion_metadata["min"]
                                density_max = rate_distortion_metadata["max"]
                                stats["rate_distortion_density_min"] = (
                                    density_min
                                    if stats["rate_distortion_density_min"] is None
                                    else min(
                                        stats["rate_distortion_density_min"],
                                        density_min,
                                    )
                                )
                                stats["rate_distortion_density_max"] = (
                                    density_max
                                    if stats["rate_distortion_density_max"] is None
                                    else max(
                                        stats["rate_distortion_density_max"],
                                        density_max,
                                    )
                                )
                                stats["rate_distortion_shuffled_calls"] += int(
                                    adaptive_metadata["density_shuffled"]
                                )
                        else:
                            edge0, edge1 = (
                                float(value)
                                for value in self.frontview_sampling_config[
                                    "depth_edges_m"
                                ]
                            )
                            pool_counts = (
                                int((extra_responsibility_depth < edge0).sum().item()),
                                int(
                                    (
                                        (extra_responsibility_depth >= edge0)
                                        & (extra_responsibility_depth < edge1)
                                    ).sum().item()
                                ),
                                int((extra_responsibility_depth >= edge1).sum().item()),
                            )
                            selected_depth = extra_responsibility_depth[selection]
                            selected_counts = (
                                int((selected_depth < edge0).sum().item()),
                                int(
                                    (
                                        (selected_depth >= edge0)
                                        & (selected_depth < edge1)
                                    ).sum().item()
                                ),
                                int((selected_depth >= edge1).sum().item()),
                            )
                        stats["pool_depth_counts"] = [
                            old + new
                            for old, new in zip(
                                stats["pool_depth_counts"], pool_counts
                            )
                        ]
                        stats["selected_depth_counts"] = [
                            old + new
                            for old, new in zip(
                                stats["selected_depth_counts"], selected_counts
                            )
                        ]
                        refill_depthcov_pool = (
                            self._frontview_identity_lod_enabled()
                            and self.frontview_identity_lod_config[
                                "refill_depthcov_budget"
                            ]
                        ) or (
                            self._frontview_residual_cover_enabled()
                            and self.frontview_residual_cover_config[
                                "refill_depthcov_pool"
                            ]
                        ) or (
                            self._frontview_far_field_enabled()
                            and self.frontview_far_field_config[
                                "posterior_budget_refill"
                            ]
                        )
                        if refill_depthcov_pool:
                            primary = selection
                            is_primary = torch.zeros(
                                len(extra_pts_depth),
                                device=selection.device,
                                dtype=torch.bool,
                            )
                            is_primary[primary] = True
                            reserve = torch.nonzero(
                                ~is_primary, as_tuple=False
                            ).reshape(-1)
                            selection = torch.cat((primary, reserve), dim=0)
                            extra_budget_primary = torch.cat(
                                (
                                    torch.ones_like(primary, dtype=torch.bool),
                                    torch.zeros_like(reserve, dtype=torch.bool),
                                )
                            )
                            if self.frontview_far_field_config[
                                "posterior_budget_refill"
                            ]:
                                stats["posterior_refill_pool_rows"] += int(
                                    len(selection)
                                )
                                stats["posterior_refill_primary_rows"] += int(
                                    len(primary)
                                )
                        else:
                            extra_budget_primary = torch.ones_like(
                                selection, dtype=torch.bool
                            )
                        extra_pts_depth = extra_pts_depth[selection]
                        extra_responsibility_depth = extra_responsibility_depth[
                            selection
                        ]
                        extra_pts_2d = extra_pts_2d[selection]
                        extra_depth_confidence = extra_depth_confidence[selection]

                    extra_multiview_support = torch.zeros_like(extra_depth_confidence)
                    if (
                        float(
                            self.frontview_scale_cover_config.get(
                                "quota_multiview_support_weight", 0.0
                            )
                        )
                        > 0.0
                    ):
                        _, _, extra_multiview_support = (
                            self._frontview_reprojection_scores(
                                cam,
                                list(reference_cameras or ()),
                                extra_pts_2d,
                                extra_pts_depth,
                                extra_depth_confidence,
                                level,
                            )
                        )

                    if (
                        causal_metric_batch is not None
                        and causal_posterior_prior_maps is not None
                        and self.causal_metric_birth_config["birth_mode"]
                        == "posterior_proxy"
                    ):
                        selected_pixels = (
                            extra_pts_2d[:, 1].int() * w
                            + extra_pts_2d[:, 0].int()
                        )
                        prior_depth_map, prior_std_map, prior_valid_map = (
                            causal_posterior_prior_maps
                        )
                        causal_posterior = fuse_candidate_log_depth_posteriors(
                            extra_pts_2d,
                            prior_depth_map[selected_pixels],
                            prior_std_map[selected_pixels],
                            prior_valid_map[selected_pixels],
                            causal_metric_batch,
                            image_size=(w, height),
                            birth_budget=int(extra_pts_num),
                            config=self.causal_metric_birth_config,
                            seed=(
                                int(self.causal_metric_birth_config["shuffle_seed"])
                                + int(cam.cam_idx)
                            ),
                        )
                        certified = causal_posterior.certified
                        apply_posterior = (
                            self.causal_metric_birth_config["posterior_action"]
                            == "fuse"
                        )
                        if apply_posterior and bool(certified.any().item()):
                            extra_pts_depth[certified] = causal_posterior.depths[
                                certified
                            ].to(extra_pts_depth)
                            extra_responsibility_depth[certified] = (
                                causal_posterior.depths[certified].to(
                                    extra_responsibility_depth
                                )
                            )
                            posterior_confidence = torch.clamp(
                                1.0
                                - causal_posterior.log_depth_stds[certified]
                                / max(
                                    float(
                                        self.depth_cov_estimator.std_valid_threshold
                                    ),
                                    1.0e-8,
                                ),
                                0.0,
                                1.0,
                            )
                            extra_depth_confidence[certified] = posterior_confidence
                            extra_multiview_support[certified] = torch.maximum(
                                extra_multiview_support[certified],
                                causal_posterior.information_gain[certified].to(
                                    extra_multiview_support
                                ),
                            )
                            if depth_prior_fallback_pixel_indices is not None:
                                certified_pixels = selected_pixels[certified]
                                depth_prior_fallback_pixel_indices = (
                                    depth_prior_fallback_pixel_indices[
                                        ~torch.isin(
                                            depth_prior_fallback_pixel_indices,
                                            certified_pixels,
                                        )
                                    ]
                                )
                        self.causal_metric_birth.record_posterior(causal_posterior)
                        causal_metric_batch = None

                    extra_metric_identity = torch.zeros(
                        len(extra_pts_depth), device=device, dtype=torch.bool
                    )
                    extra_exact_world = torch.full(
                        (len(extra_pts_depth), 3),
                        float("nan"),
                        device=device,
                        dtype=torch.float32,
                    )
                    if causal_metric_batch is not None and len(causal_metric_batch):
                        footprint_reanchor = (
                            self.causal_metric_birth_config["birth_mode"]
                            == "footprint_reanchor"
                        )
                        dual_replace = self.causal_dual_responsibility_config[
                            "enabled"
                        ]
                        replacement_positions = None
                        assigned_tracks = None
                        assigned_distances = None
                        if dual_replace or footprint_reanchor:
                            selected_pixels = (
                                extra_pts_2d[:, 1].int() * w
                                + extra_pts_2d[:, 0].int()
                            )
                            fallback_positions = torch.nonzero(
                                torch.isin(
                                    selected_pixels,
                                    depth_prior_fallback_pixel_indices,
                                ),
                                as_tuple=False,
                            ).reshape(-1)
                            tracked_count = min(
                                int(len(causal_metric_batch)),
                                int(fallback_positions.numel()),
                            )
                            if tracked_count:
                                tracked_uv_for_assignment = torch.from_numpy(
                                    causal_metric_batch.uv[:tracked_count]
                                ).to(device=device, dtype=torch.float32)
                            if footprint_reanchor and tracked_count:
                                support_radius = budgeted_fallback_radius(
                                    (cam.get_width(level), cam.get_height(level)),
                                    int(self.extra_pts_num),
                                )
                                (
                                    replacement_positions,
                                    assigned_tracks,
                                    assigned_distances,
                                ) = bind_tracks_to_proxy_slots(
                                    extra_pts_2d,
                                    fallback_positions,
                                    tracked_uv_for_assignment,
                                    support_radius_px=support_radius,
                                )
                                tracked_count = int(len(replacement_positions))
                                self.causal_metric_birth.stats[
                                    "reanchor_proxy_rows"
                                ] += int(fallback_positions.numel())
                            elif tracked_count:
                                replacement_positions = (
                                    nearest_unique_replacement_positions(
                                        extra_pts_2d,
                                        fallback_positions,
                                        tracked_uv_for_assignment,
                                    )
                                )
                        else:
                            remaining_budget = max(
                                0, int(extra_pts_num) - int(len(extra_pts_depth))
                            )
                            tracked_count = min(
                                int(len(causal_metric_batch)), remaining_budget
                            )
                        if tracked_count:
                            tracked_uv = torch.from_numpy(
                                causal_metric_batch.uv[:tracked_count]
                            ).to(device=device, dtype=torch.float32)
                            tracked_depth = torch.from_numpy(
                                causal_metric_batch.depths[:tracked_count]
                            ).to(device=device, dtype=extra_pts_depth.dtype)
                            tracked_world = torch.from_numpy(
                                causal_metric_batch.world_points[:tracked_count]
                            ).to(device=device, dtype=torch.float32)
                            tracked_log_std = torch.from_numpy(
                                causal_metric_batch.log_depth_stds[:tracked_count]
                            ).to(device=device, dtype=torch.float32)
                            tracked_information = torch.from_numpy(
                                causal_metric_batch.information_gains[:tracked_count]
                            ).to(device=device, dtype=torch.float32)
                            if assigned_tracks is not None:
                                tracked_uv = torch.from_numpy(
                                    causal_metric_batch.uv
                                ).to(device=device, dtype=torch.float32)[
                                    assigned_tracks
                                ]
                                tracked_depth = torch.from_numpy(
                                    causal_metric_batch.depths
                                ).to(device=device, dtype=extra_pts_depth.dtype)[
                                    assigned_tracks
                                ]
                                tracked_world = torch.from_numpy(
                                    causal_metric_batch.world_points
                                ).to(device=device, dtype=torch.float32)[
                                    assigned_tracks
                                ]
                                tracked_log_std = torch.from_numpy(
                                    causal_metric_batch.log_depth_stds
                                ).to(device=device, dtype=torch.float32)[
                                    assigned_tracks
                                ]
                                tracked_information = torch.from_numpy(
                                    causal_metric_batch.information_gains
                                ).to(device=device, dtype=torch.float32)[
                                    assigned_tracks
                                ]
                            tracked_confidence = torch.clamp(
                                1.0
                                - tracked_log_std
                                / max(
                                    float(
                                        self.depth_cov_estimator.std_valid_threshold
                                    ),
                                    1.0e-8,
                                ),
                                0.0,
                                1.0,
                            )
                            if footprint_reanchor:
                                old_depth = extra_pts_depth[
                                    replacement_positions
                                ].clone()
                                extra_pts_depth[replacement_positions] = tracked_depth
                                self.causal_metric_birth.record_reanchoring(
                                    0,
                                    old_depth,
                                    tracked_depth,
                                    assigned_distances,
                                )
                            elif dual_replace:
                                extra_pts_2d[replacement_positions] = tracked_uv
                                extra_pts_depth[replacement_positions] = tracked_depth
                                extra_responsibility_depth[
                                    replacement_positions
                                ] = tracked_depth
                                extra_depth_confidence[replacement_positions] = (
                                    tracked_confidence
                                )
                                extra_multiview_support[replacement_positions] = (
                                    tracked_information
                                )
                                extra_budget_primary[replacement_positions] = True
                                extra_metric_identity[replacement_positions] = True
                                extra_exact_world[replacement_positions] = tracked_world
                                self.causal_dual_responsibility_stats.setdefault(
                                    "tracked_proxy_replacements", 0
                                )
                                self.causal_dual_responsibility_stats[
                                    "tracked_proxy_replacements"
                                ] += int(tracked_count)
                            else:
                                extra_pts_2d = torch.cat(
                                    (extra_pts_2d, tracked_uv), dim=0
                                )
                                extra_pts_depth = torch.cat(
                                    (extra_pts_depth, tracked_depth), dim=0
                                )
                                extra_responsibility_depth = torch.cat(
                                    (extra_responsibility_depth, tracked_depth), dim=0
                                )
                                extra_depth_confidence = torch.cat(
                                    (extra_depth_confidence, tracked_confidence), dim=0
                                )
                                extra_multiview_support = torch.cat(
                                    (extra_multiview_support, tracked_information), dim=0
                                )
                                extra_budget_primary = torch.cat(
                                    (
                                        extra_budget_primary,
                                        torch.ones(
                                            tracked_count,
                                            device=device,
                                            dtype=torch.bool,
                                        ),
                                    ),
                                    dim=0,
                                )
                                extra_metric_identity = torch.cat(
                                    (
                                        extra_metric_identity,
                                        torch.ones(
                                            tracked_count,
                                            device=device,
                                            dtype=torch.bool,
                                        ),
                                    ),
                                    dim=0,
                                )
                                extra_exact_world = torch.cat(
                                    (extra_exact_world, tracked_world), dim=0
                                )

                    # Uncomment this part if need to visualize feature points
                    # extra_pts_idx = extra_pts_2d[:, 1].int() * cam.get_width(level) + extra_pts_2d[:, 0].int()
                    # vis_feature_mask = sparse_depth.detach().cpu().numpy().reshape(-1)
                    # vis_feature_mask[vis_feature_mask > 0] = 1.0
                    # vis_feature_mask[extra_pts_idx.detach().cpu().numpy()] = 0.5
                    # cam.feature_mask = vis_feature_mask.reshape(sparse_depth.shape[0], sparse_depth.shape[1], 1)

                    pts_2d = torch.cat(
                        [pts_2d, extra_pts_2d], dim=0
                    )  # concat semi-dense pts with extra pts
                    pts_depth = torch.cat([pts_depth, extra_pts_depth], dim=0)
                    responsibility_depth = torch.cat(
                        [pts_depth[: len(pts_depth) - len(extra_pts_depth)],
                         extra_responsibility_depth],
                        dim=0,
                    )
                    proposal_sparse_valid = torch.cat(
                        [
                            proposal_sparse_valid,
                            extra_metric_identity,
                        ],
                        dim=0,
                    )
                    proposal_depth_confidence = torch.cat(
                        [proposal_depth_confidence, extra_depth_confidence], dim=0
                    )
                    proposal_multiview_support = torch.cat(
                        [proposal_multiview_support, extra_multiview_support], dim=0
                    )
                    proposal_budget_primary = torch.cat(
                        [proposal_budget_primary, extra_budget_primary], dim=0
                    )
                    proposal_tracked_metric = torch.cat(
                        [proposal_tracked_metric, extra_metric_identity], dim=0
                    )
                    proposal_track_ids = torch.cat(
                        [
                            proposal_track_ids,
                            torch.full(
                                (len(extra_pts_depth),),
                                -1,
                                device=device,
                                dtype=torch.long,
                            ),
                        ],
                        dim=0,
                    )

            # pts_3d = unproject_pts(pts_2d, pts_depth, cam.get_int_mat(level).cpu().numpy(), cam.get_raw_pose().detach().cpu().numpy())
            sparse_count = len(pts_depth) - (
                len(extra_pts_depth)
                if extra_pts_2d is not None and "extra_pts_depth" in locals()
                else 0
            )
            pts_3d = unproject_pts_tensor(
                pts_2d, pts_depth, cam.get_int_mat(level), cam.get_raw_pose().detach()
            )
            if "responsibility_depth" not in locals():
                responsibility_depth = pts_depth.clone()
            responsibility_pts_3d = unproject_pts_tensor(
                pts_2d,
                responsibility_depth,
                cam.get_int_mat(level),
                cam.get_raw_pose().detach(),
            )
            if exact_sparse_world is not None and sparse_count > 0:
                pts_3d[:sparse_count] = exact_sparse_world
                responsibility_pts_3d[:sparse_count] = exact_sparse_world
                self.frontview_far_field_stats["exact_sparse_world_rows"] += int(
                    sparse_count
                )
            if (
                "extra_metric_identity" in locals()
                and len(extra_metric_identity)
                and bool(extra_metric_identity.any().item())
            ):
                dense_world = pts_3d[sparse_count:]
                dense_world[extra_metric_identity] = extra_exact_world[
                    extra_metric_identity
                ].to(device=dense_world.device, dtype=dense_world.dtype)
                dense_responsibility_world = responsibility_pts_3d[sparse_count:]
                dense_responsibility_world[extra_metric_identity] = extra_exact_world[
                    extra_metric_identity
                ].to(
                    device=dense_responsibility_world.device,
                    dtype=dense_responsibility_world.dtype,
                )

            color_img = cam.get_gt_image(level)

            vig_img = self.get_vignette_img(level)

            if vig_img is not None:
                color_for_pts = (
                    color_img.to(self.device)
                    * self.scene_exposure_gain
                    / cam.exposure_gain
                    / vig_img
                )
                invalid_mask = vig_img == 0
                color_for_pts[invalid_mask] = 0.0
                color_for_pts = color_for_pts[0]
            else:
                color_for_pts = (
                    color_img.to(self.device)
                    * self.scene_exposure_gain
                    / cam.exposure_gain
                )

            # color_for_pts = cv2.blur(color_for_pts, (3, 3))
            color_for_pts = color_for_pts.reshape(-1, 3)

            pts_2d_index = (
                pts_2d[:, 1].int() * cam.get_width(level) + pts_2d[:, 0].int()
            )

            pts_color = color_for_pts[pts_2d_index]
            fusion_pts_2d_index = (
                fusion_pts_2d[:, 1].int() * cam.get_width(level)
                + fusion_pts_2d[:, 0].int()
            )
            fusion_colors = color_for_pts[fusion_pts_2d_index]

            if render_pkg is None:
                residual_scores = torch.zeros_like(pts_depth)
                coverage_scores = torch.ones_like(pts_depth)
                proposal_stable_depth = torch.full_like(pts_depth, float("nan"))
            else:
                residual_scores = render_pkg["diff"].reshape(-1)[pts_2d_index]
                proposal_stable_depth = self._geometry_depth(render_pkg).reshape(-1)[
                    pts_2d_index
                ]
                opacity = self._geometry_opacity(render_pkg)
                if opacity is None:
                    coverage_scores = torch.ones_like(pts_depth)
                else:
                    coverage_scores = 1.0 - opacity.reshape(-1)[pts_2d_index]

            init_scale = (
                torch.log(
                    0.5 * pts_depth / ((cam.get_fx(level) + cam.get_fy(level)) / 2.0)
                ).reshape(-1, 1)
                + self.init_scale_offset
            )
            responsibility_init_scale = (
                torch.log(
                    0.5
                    * responsibility_depth
                    / ((cam.get_fx(level) + cam.get_fy(level)) / 2.0)
                ).reshape(-1, 1)
                + self.init_scale_offset
            )
            tracked_support_pixels = None
            if (
                self.causal_metric_birth.enabled
                and self.causal_metric_birth_config["support_mode"] != "point"
                and bool(proposal_tracked_metric.any().item())
            ):
                tracked_support_pixels = budgeted_fallback_radius(
                    (cam.get_width(level), cam.get_height(level)),
                    int(self.extra_pts_num),
                )
                tracked_rows = proposal_tracked_metric
                init_scale[tracked_rows, 0] = torch.log(
                    pts_depth[tracked_rows]
                    * float(tracked_support_pixels)
                    / (0.5 * (cam.get_fx(level) + cam.get_fy(level)))
                )
                responsibility_init_scale[tracked_rows, 0] = torch.log(
                    responsibility_depth[tracked_rows]
                    * float(tracked_support_pixels)
                    / (0.5 * (cam.get_fx(level) + cam.get_fy(level)))
                )
            fallback_scale_rows = torch.zeros_like(pts_depth, dtype=torch.bool)
            if depth_prior_fallback_pixel_indices is not None:
                all_pixel_indices = (
                    pts_2d[:, 1].int() * cam.get_width(level) + pts_2d[:, 0].int()
                )
                fallback_scale_rows = torch.isin(
                    all_pixel_indices, depth_prior_fallback_pixel_indices
                )
                fallback_scale_rows &= ~proposal_tracked_metric
                init_scale[fallback_scale_rows] += math.log(
                    float(
                        self.frontview_coverage_recovery_config[
                            "depth_fallback_scale_multiplier"
                        ]
                    )
                )
                responsibility_init_scale[fallback_scale_rows] += math.log(
                    float(
                        self.frontview_coverage_recovery_config[
                            "depth_fallback_scale_multiplier"
                        ]
                    )
                )
                self.frontview_coverage_depth_stats[
                    "depth_fallback_scaled_rows"
                ] += int(fallback_scale_rows.sum().item())
            fallback_support_mode = self.frontview_far_field_config[
                "fallback_support_mode"
            ]
            if fallback_support_mode != "legacy" and bool(
                fallback_scale_rows.any().item()
            ):
                support_pixels = budgeted_fallback_radius(
                    (cam.get_width(level), cam.get_height(level)),
                    int(self.extra_pts_num),
                )
                fallback_depth = pts_depth[fallback_scale_rows]
                replacement = torch.log(
                    fallback_depth
                    * float(support_pixels)
                    / (0.5 * (cam.get_fx(level) + cam.get_fy(level)))
                )
                if fallback_support_mode.startswith("budget_information"):
                    radius_factors, density = (
                        budget_normalized_information_radii(
                            cam.get_gt_image(level),
                            pts_2d.detach().cpu().numpy(),
                            fallback_scale_rows.detach().cpu().numpy(),
                            support_pixels,
                            shuffle=fallback_support_mode.endswith("_shuffled"),
                            seed=int(self.frontview_far_field_config["shuffle_seed"])
                            + int(cam.cam_idx),
                        )
                    )
                    row_factors = torch.as_tensor(
                        radius_factors[fallback_scale_rows.detach().cpu().numpy()],
                        device=replacement.device,
                        dtype=replacement.dtype,
                    )
                    replacement += torch.log(row_factors)
                    values = radius_factors[
                        fallback_scale_rows.detach().cpu().numpy()
                    ]
                    density_values = density[
                        fallback_scale_rows.detach().cpu().numpy()
                    ]
                    stats = self.frontview_far_field_stats
                    stats["fallback_information_rows"] += int(len(values))
                    stats["fallback_information_radius_factor_sum"] += float(
                        np.sum(values)
                    )
                    stats["fallback_information_density_sum"] += float(
                        np.sum(density_values)
                    )
                    minimum = float(np.min(values))
                    maximum = float(np.max(values))
                    old_min = stats["fallback_information_radius_factor_min"]
                    old_max = stats["fallback_information_radius_factor_max"]
                    stats["fallback_information_radius_factor_min"] = (
                        minimum if old_min is None else min(old_min, minimum)
                    )
                    stats["fallback_information_radius_factor_max"] = (
                        maximum if old_max is None else max(old_max, maximum)
                    )
                    stats["fallback_information_shuffled_calls"] += int(
                        fallback_support_mode.endswith("_shuffled")
                    )
                init_scale[fallback_scale_rows, 0] = replacement
                responsibility_replacement = torch.log(
                    responsibility_depth[fallback_scale_rows]
                    * float(support_pixels)
                    / (0.5 * (cam.get_fx(level) + cam.get_fy(level)))
                )
                if fallback_support_mode.startswith("budget_information"):
                    responsibility_replacement += torch.log(row_factors)
                responsibility_init_scale[
                    fallback_scale_rows, 0
                ] = responsibility_replacement
            proposal_frequency_scores = torch.zeros_like(pts_depth)
            if admission_evidence is not None:
                proposal_frequency_scores = admission_evidence.reshape(-1)[
                    pts_2d_index
                ]
            if footprint_evidence is not None:
                footprint_log_offset = frequency_footprint_log_offset(
                    footprint_evidence.reshape(-1)[pts_2d_index],
                    self.frequency_sampling_config.get(
                        "max_footprint_shrink_fraction", 0.0
                    ),
                ).reshape(-1, 1)
                init_scale = init_scale + footprint_log_offset
                responsibility_init_scale = (
                    responsibility_init_scale + footprint_log_offset
                )
            cur_view_scale_size = cam.get_view_size(level)

            proposal_cidec_certified = torch.zeros_like(
                proposal_sparse_valid, dtype=torch.bool
            )
            proposal_cidec_conflicted = torch.zeros_like(
                proposal_sparse_valid, dtype=torch.bool
            )
            proposal_cidec_reject = torch.zeros_like(
                proposal_sparse_valid, dtype=torch.bool
            )
            proposal_cidec_log_stds = float(
                self.depth_cov_estimator.std_valid_threshold
            ) * (1.0 - torch.clamp(proposal_depth_confidence, 0.0, 1.0))
            if self.frontview_inverse_depth_certificate_config["enabled"]:
                dense_rows = torch.nonzero(
                    ~proposal_sparse_valid, as_tuple=False
                ).flatten()
                if dense_rows.numel():
                    threshold = float(self.depth_cov_estimator.std_valid_threshold)
                    prior_log_stds = threshold * (
                        1.0
                        - torch.clamp(
                            proposal_depth_confidence[dense_rows], 0.0, 1.0
                        )
                    )
                    prior_depths = pts_depth[dense_rows].clone()
                    certificate = causal_inverse_depth_posterior(
                        cam,
                        list(reference_cameras or ()),
                        pts_2d[dense_rows],
                        prior_depths,
                        prior_log_stds,
                        level,
                        self.frontview_inverse_depth_certificate_config,
                    )
                    certified = certificate["certified"]
                    conflicted = certificate["conflicted"]
                    proposal_cidec_certified[dense_rows] = certified
                    proposal_cidec_conflicted[dense_rows] = conflicted
                    proposal_cidec_log_stds[dense_rows] = certificate[
                        "posterior_log_stds"
                    ]
                    policy = self.frontview_inverse_depth_certificate_config[
                        "uncertified_policy"
                    ]
                    if policy == "reject":
                        proposal_cidec_reject[dense_rows] = ~certified
                    elif policy == "projective_reject_conflict":
                        proposal_cidec_reject[dense_rows] = conflicted & ~certified

                    corrected_depths = certificate["depths"]
                    pts_depth[dense_rows] = corrected_depths
                    depth_ratio = corrected_depths / torch.clamp(
                        prior_depths, min=1.0e-8
                    )
                    init_scale[dense_rows] += torch.log(depth_ratio).reshape(-1, 1)
                    pts_3d[dense_rows] = unproject_pts_tensor(
                        pts_2d[dense_rows],
                        corrected_depths,
                        cam.get_int_mat(level),
                        cam.get_raw_pose().detach(),
                    )
                    posterior_confidence = torch.clamp(
                        1.0
                        - certificate["posterior_log_stds"]
                        / max(threshold, 1.0e-8),
                        0.0,
                        1.0,
                    )
                    proposal_depth_confidence[dense_rows] = torch.where(
                        certified,
                        posterior_confidence,
                        proposal_depth_confidence[dense_rows],
                    )
                    proposal_multiview_support[dense_rows] = torch.maximum(
                        proposal_multiview_support[dense_rows],
                        certificate["information_gain"],
                    )
                    stats = self.frontview_inverse_depth_certificate_stats
                    stats["calls"] += 1
                    stats["rows"] += int(dense_rows.numel())
                    stats["certified_rows"] += int(certified.sum().item())
                    stats["conflicted_rows"] += int(conflicted.sum().item())
                    stats["rejected_conflicted_rows"] += int(
                        proposal_cidec_reject[dense_rows].sum().item()
                    )
                    stats["information_gain_sum"] += float(
                        certificate["information_gain"].sum().item()
                    )
                    stats["absolute_log_depth_shift_sum"] += float(
                        torch.abs(certificate["posterior_shift"][certified]).sum().item()
                    )
                    stats["posterior_log_std_sum"] += float(
                        certificate["posterior_log_stds"][certified].sum().item()
                    )
                    stats["valid_view_sum"] += int(
                        certificate["valid_views"].sum().item()
                    )
                    stats["baseline_information_sum"] += float(
                        certificate["baseline_information"].sum().item()
                    )
                    stats["shuffle_calls"] += int(
                        self.frontview_inverse_depth_certificate_config[
                            "shuffle_evidence"
                        ]
                    )

            use_original_responsibility = (
                self.causal_landmark_memory.uses_original_posterior_responsibility
            )
            if not use_original_responsibility:
                responsibility_depth = pts_depth
                responsibility_pts_3d = pts_3d
                responsibility_init_scale = init_scale
            pts_3d = pts_3d.cpu().numpy()
            responsibility_pts_3d = responsibility_pts_3d.cpu().numpy()
            pts_color = pts_color.cpu().numpy()
            init_scale = init_scale.cpu().numpy()
            responsibility_init_scale = responsibility_init_scale.cpu().numpy()
            pts_2d = pts_2d.detach().cpu().numpy().astype(np.float32, copy=False)
            pts_depth = pts_depth.detach().cpu().numpy().astype(np.float32, copy=False)
            responsibility_depth = (
                responsibility_depth.detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            residual_scores = (
                residual_scores.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            coverage_scores = (
                coverage_scores.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            proposal_sparse_valid = proposal_sparse_valid.detach().cpu().numpy()
            proposal_depth_confidence = (
                proposal_depth_confidence.detach().cpu().numpy().astype(np.float32)
            )
            proposal_multiview_support = (
                proposal_multiview_support.detach().cpu().numpy().astype(np.float32)
            )
            proposal_stable_depth = (
                proposal_stable_depth.detach().cpu().numpy().astype(np.float32)
            )
            proposal_frequency_scores = (
                proposal_frequency_scores.detach().cpu().numpy().astype(np.float32)
            )
            proposal_track_ids = proposal_track_ids.detach().cpu().numpy().astype(
                np.int64
            )
            proposal_budget_primary = (
                proposal_budget_primary.detach().cpu().numpy().astype(np.bool_)
            )
            proposal_tracked_metric = (
                proposal_tracked_metric.detach().cpu().numpy().astype(np.bool_)
            )
            proposal_depth_fallback = (
                fallback_scale_rows.detach().cpu().numpy().astype(np.bool_)
            )
            proposal_cidec_certified = (
                proposal_cidec_certified.detach().cpu().numpy().astype(np.bool_)
            )
            proposal_cidec_conflicted = (
                proposal_cidec_conflicted.detach().cpu().numpy().astype(np.bool_)
            )
            proposal_cidec_reject = (
                proposal_cidec_reject.detach().cpu().numpy().astype(np.bool_)
            )
            proposal_cidec_log_stds = (
                proposal_cidec_log_stds.detach().cpu().numpy().astype(np.float32)
            )
            fusion_track_ids = (
                fusion_track_ids.detach().cpu().numpy().astype(np.int64)
            )
            fusion_colors = fusion_colors.detach().cpu().numpy()
            fusion_pts_depth = (
                fusion_pts_depth.detach().cpu().numpy().astype(np.float32, copy=False)
            )

        else:
            raise NotImplementedError

        cur_view_scale_size *= self.camera_scale_rescalar
        proposal_cover_sizes = self.frontview_scale_cover.candidate_target_sizes(
            pts_depth,
            (cam.get_fx(level) + cam.get_fy(level)) / 2.0,
            self.camera_scale_rescalar,
            cur_view_scale_size,
            proposal_sparse_valid,
            int(cam.cam_idx),
            gaussian_scales=np.exp(init_scale[:, 0]),
        )
        proposal_responsibility_cover_sizes = proposal_cover_sizes
        if use_original_responsibility:
            proposal_responsibility_cover_sizes = self.frontview_scale_cover.candidate_target_sizes(
                responsibility_depth,
                (cam.get_fx(level) + cam.get_fy(level)) / 2.0,
                self.camera_scale_rescalar,
                cur_view_scale_size,
                proposal_sparse_valid,
                int(cam.cam_idx),
                gaussian_scales=np.exp(responsibility_init_scale[:, 0]),
            )
        camera_pose = cam.get_pose().detach()
        birth_camera_center = (
            -camera_pose[:3, :3].T @ camera_pose[:3, 3]
        ).cpu().numpy()
        proposal_view_directions = (
            self.frontview_scale_cover.candidate_view_directions(
                pts_3d,
                birth_camera_center,
                pts_depth,
                proposal_sparse_valid,
                int(cam.cam_idx),
            )
        )
        proposal_responsibility_view_directions = proposal_view_directions
        if use_original_responsibility:
            proposal_responsibility_view_directions = self.frontview_scale_cover.candidate_view_directions(
                responsibility_pts_3d,
                birth_camera_center,
                responsibility_depth,
                proposal_sparse_valid,
                int(cam.cam_idx),
            )
        self.causal_landmark_memory.record_responsibility_coordinates(
            pts_depth, responsibility_depth
        )

        self.fuse_frontview_track_appearance(
            fusion_track_ids,
            fusion_colors,
            fusion_pts_depth,
            int(cam.cam_idx),
        )

        before_filter_num = pts_3d.shape[0]
        posterior_refill_requested = 0
        proposal_far_field = np.zeros((before_filter_num,), dtype=np.bool_)
        focal_pixels = 0.5 * (cam.get_fx(level) + cam.get_fy(level))
        proposal_projected_radii = projected_gaussian_radii(
            init_scale, pts_depth, focal_pixels
        )
        proposal_responsibility_projected_radii = proposal_projected_radii
        if use_original_responsibility:
            proposal_responsibility_projected_radii = projected_gaussian_radii(
                responsibility_init_scale, responsibility_depth, focal_pixels
            )
        proposal_log_depth_stds = proposal_cidec_log_stds
        proposal_parallax_pixels = np.zeros((before_filter_num,), dtype=np.float32)
        proposal_parallax_support = np.zeros((before_filter_num,), dtype=np.int32)
        proposal_responsibility_parallax_pixels = proposal_parallax_pixels
        proposal_responsibility_parallax_support = proposal_parallax_support
        proposal_map_redundant = np.zeros((before_filter_num,), dtype=np.bool_)
        proposal_map_log_odds = np.full(
            (before_filter_num,), -np.inf, dtype=np.float64
        )
        proposal_radial_scale_factors = np.ones(
            (before_filter_num,), dtype=np.float32
        )
        proposal_covariance_scale_factors = np.ones(
            (before_filter_num, 3), dtype=np.float32
        )
        proposal_initial_quaternions = np.zeros(
            (before_filter_num, 4), dtype=np.float32
        )
        proposal_initial_quaternions[:, 0] = 1.0
        proposal_scale_expansion_limits = np.full(
            (before_filter_num,), np.inf, dtype=np.float32
        )
        proposal_footprint_target_scales = np.full(
            (before_filter_num,), np.inf, dtype=np.float32
        )
        if (
            self.causal_metric_birth.enabled
            and self.causal_metric_birth_config["support_mode"]
            == "budget_structure"
            and np.any(proposal_tracked_metric)
        ):
            (
                tracked_factors,
                tracked_quaternions,
                _,
            ) = structure_aligned_covariances(
                cam.get_gt_image(level),
                pts_2d,
                proposal_view_directions,
                cam.get_raw_pose().detach().cpu().numpy()[:3, :3],
                proposal_tracked_metric,
                support_radius_pixels=float(tracked_support_pixels),
                certificate_strength=np.clip(
                    proposal_depth_confidence, 0.0, 1.0
                ),
                shuffle=False,
                seed=int(self.causal_metric_birth_config["shuffle_seed"])
                + int(cam.cam_idx),
            )
            # Tangential support follows the per-frame birth budget. Radial
            # sigma follows propagated metric depth uncertainty: sigma_z=z*sigma_logz.
            radial_ratio = (
                proposal_cidec_log_stds
                * float(0.5 * (cam.get_fx(level) + cam.get_fy(level)))
                / max(float(tracked_support_pixels), 1.0e-8)
            )
            tracked_factors[proposal_tracked_metric, 2] = np.maximum(
                radial_ratio[proposal_tracked_metric],
                np.finfo(np.float32).tiny,
            )
            proposal_covariance_scale_factors[proposal_tracked_metric] = (
                tracked_factors[proposal_tracked_metric]
            )
            proposal_initial_quaternions[proposal_tracked_metric] = (
                tracked_quaternions[proposal_tracked_metric]
            )
        if (
            self._frontview_far_field_enabled()
            and self.frontview_far_field_config["routing_mode"]
            in ("causal_observability", "adaptive_observability")
        ):
            references = list(reference_cameras or ())
            proposal_parallax_pixels, proposal_parallax_support = (
                visible_parallax_pixels(
                    pts_3d,
                    cam.get_raw_pose().detach().cpu().numpy(),
                    [
                        reference.get_raw_pose().detach().cpu().numpy()
                        for reference in references
                    ],
                    [
                        reference.get_int_mat(level).detach().cpu().numpy()
                        for reference in references
                    ],
                    [
                        (reference.get_width(level), reference.get_height(level))
                        for reference in references
                    ],
                    focal_pixels,
                )
            )
            if not use_original_responsibility:
                proposal_responsibility_parallax_pixels = (
                    proposal_parallax_pixels
                )
                proposal_responsibility_parallax_support = (
                    proposal_parallax_support
                )
            else:
                (
                    proposal_responsibility_parallax_pixels,
                    proposal_responsibility_parallax_support,
                ) = visible_parallax_pixels(
                    responsibility_pts_3d,
                    cam.get_raw_pose().detach().cpu().numpy(),
                    [
                        reference.get_raw_pose().detach().cpu().numpy()
                        for reference in references
                    ],
                    [
                        reference.get_int_mat(level).detach().cpu().numpy()
                        for reference in references
                    ],
                    [
                        (reference.get_width(level), reference.get_height(level))
                        for reference in references
                    ],
                    focal_pixels,
                )
            footprint_mode = self.frontview_far_field_config[
                "footprint_trust_mode"
            ]
            if footprint_mode != "disabled":
                footprint_scope = self.frontview_far_field_config[
                    "footprint_trust_scope"
                ]
                footprint_eligible = ~proposal_sparse_valid
                if footprint_scope == "projective_responsibility":
                    footprint_eligible = far_field_responsibility_mask(
                        responsibility_depth,
                        proposal_sparse_valid,
                        proposal_track_ids,
                        self.frontview_far_field_config,
                        parallax_pixels=proposal_responsibility_parallax_pixels,
                        projected_radii=proposal_responsibility_projected_radii,
                        log_depth_stds=proposal_log_depth_stds,
                    )
                footprint_projective_owner = None
                if footprint_mode.startswith((
                    "certificate_owner_area",
                    "certificate_residual_rd",
                )):
                    footprint_projective_owner = far_field_responsibility_mask(
                        responsibility_depth,
                        proposal_sparse_valid,
                        proposal_track_ids,
                        self.frontview_far_field_config,
                        parallax_pixels=proposal_responsibility_parallax_pixels,
                        projected_radii=proposal_responsibility_projected_radii,
                        log_depth_stds=proposal_log_depth_stds,
                    )
                footprint_radius_factors = None
                if footprint_mode.startswith("certificate_residual_rd"):
                    if render_pkg is None:
                        footprint_radius_factors = np.ones(
                            (before_filter_num,), dtype=np.float32
                        )
                    else:
                        radius_factors = residual_rate_distortion_radius_factors(
                            render_pkg["diff"],
                            torch.as_tensor(
                                pts_2d, device=self.device, dtype=torch.float32
                            ),
                            torch.as_tensor(
                                footprint_projective_owner,
                                device=self.device,
                                dtype=torch.bool,
                            ),
                            image_size=(
                                cam.get_width(level),
                                cam.get_height(level),
                            ),
                            budget=int(self.extra_pts_num),
                            pool_multiplier=int(
                                self.frontview_sampling_config.get(
                                    "pool_multiplier", 1
                                )
                            ),
                            visibility=(
                                render_pkg.get("opacity")
                                if "_visible" in footprint_mode
                                else None
                            ),
                            detail_protection="_detail" in footprint_mode,
                        )
                        footprint_radius_factors = (
                            radius_factors.detach().cpu().numpy().astype(np.float32)
                        )
                (
                    proposal_scale_expansion_limits,
                    footprint_information,
                    footprint_metadata,
                ) = observability_footprint_trust_limits(
                    proposal_parallax_pixels,
                    proposal_projected_radii,
                    proposal_log_depth_stds,
                    pts_depth,
                    footprint_eligible,
                    (cam.get_width(level), cam.get_height(level)),
                    int(self.extra_pts_num),
                    int(self.frontview_sampling_config.get("pool_multiplier", 1)),
                    mode=footprint_mode,
                    projective_owner=(
                        footprint_projective_owner
                        if footprint_mode.startswith("certificate_owner_area")
                        else None
                    ),
                    responsibility_radius_factors=footprint_radius_factors,
                    seed=int(self.frontview_far_field_config["shuffle_seed"])
                    + int(cam.cam_idx),
                )
                eligible_rows = footprint_eligible
                bounded_rows = np.isfinite(proposal_scale_expansion_limits)
                stats = self.frontview_far_field_stats
                stats["footprint_trust_calls"] += 1
                stats["footprint_trust_rows"] += int(np.sum(eligible_rows))
                stats["footprint_trust_bounded_rows"] += int(
                    np.sum(bounded_rows)
                )
                stats["footprint_trust_information_sum"] += float(
                    np.sum(footprint_information[eligible_rows])
                )
                stats["footprint_trust_limit_sum"] += float(
                    np.sum(
                        proposal_scale_expansion_limits[bounded_rows],
                        dtype=np.float64,
                    )
                )
                if footprint_metadata["cell_px"] is not None:
                    stats["footprint_trust_cell_pixel_sum"] += float(
                        footprint_metadata["cell_px"]
                    )
                stats["footprint_trust_shuffled_calls"] += int(
                    footprint_metadata["shuffled"]
                )
                stats["footprint_trust_projective_scope_calls"] += int(
                    footprint_scope == "projective_responsibility"
                )
                stats["footprint_trust_owner_area_calls"] += int(
                    footprint_projective_owner is not None
                )
                stats["footprint_trust_owner_area_rows"] += int(
                    footprint_metadata["projective_owner_rows"]
                )
                residual_rd_rows = (
                    0
                    if footprint_projective_owner is None
                    or not footprint_mode.startswith("certificate_residual_rd")
                    else int(np.count_nonzero(footprint_projective_owner))
                )
                stats["footprint_trust_residual_rd_calls"] += int(
                    footprint_mode.startswith("certificate_residual_rd")
                )
                stats["footprint_trust_residual_rd_rows"] += residual_rd_rows
                if residual_rd_rows:
                    stats["footprint_trust_residual_rd_radius_sum"] += float(
                        np.sum(
                            footprint_radius_factors[footprint_projective_owner],
                            dtype=np.float64,
                        )
                    )
                bounded = np.isfinite(proposal_scale_expansion_limits)
                proposal_footprint_target_scales[bounded] = (
                    np.exp(init_scale[bounded, 0])
                    * proposal_scale_expansion_limits[bounded]
                ).astype(np.float32, copy=False)
        responsibility_parent_uids = np.full(
            (before_filter_num,), -1, dtype=np.int64
        )
        responsibility_levels = np.zeros((before_filter_num,), dtype=np.int16)
        responsibility_sectors = np.full(
            (before_filter_num,), -1, dtype=np.int16
        )
        if self._frontview_residual_cover_enabled():
            target_image = color_for_pts.reshape(
                cam.get_height(level), cam.get_width(level), 3
            )
            rendered_image = (
                torch.zeros_like(target_image)
                if render_pkg is None
                else geometry_decision_render(render_pkg)
            )
            map_depth = (
                torch.full(
                    target_image.shape[:2],
                    -1.0,
                    device=target_image.device,
                    dtype=target_image.dtype,
                )
                if render_pkg is None
                else self._geometry_depth(render_pkg)
            )
            map_opacity = (
                torch.zeros(
                    target_image.shape[:2],
                    device=target_image.device,
                    dtype=target_image.dtype,
                )
                if render_pkg is None
                else self._geometry_opacity(render_pkg)
            )
            keep_indices = self.frontview_residual_cover.filter_candidates(
                frame_id=int(cam.cam_idx),
                uv=pts_2d,
                depths=pts_depth,
                world_points=pts_3d,
                log_scales=init_scale,
                colors=pts_color,
                residual_scores=residual_scores,
                depth_confidences=proposal_depth_confidence,
                sparse_valid=proposal_sparse_valid,
                track_ids=proposal_track_ids,
                rendered=rendered_image,
                target=target_image,
                focal_px=0.5 * (cam.get_fx(level) + cam.get_fy(level)),
                depthcov_budget=extra_pts_num,
                map_depth=map_depth,
                map_opacity=map_opacity,
                global_means=(
                    self.get_xyz().detach().cpu().numpy()
                    if (
                        self.frontview_residual_cover_config["use_covariance_lod"]
                        or self.frontview_residual_cover_config[
                            "covariance_competition_enabled"
                        ]
                    )
                    else None
                ),
                global_scales=(
                    self.get_scaling().detach().cpu().numpy()
                    if (
                        self.frontview_residual_cover_config["use_covariance_lod"]
                        or self.frontview_residual_cover_config[
                            "covariance_competition_enabled"
                        ]
                    )
                    else None
                ),
                global_quaternions=(
                    self.get_rotation().detach().cpu().numpy()
                    if (
                        self.frontview_residual_cover_config["use_covariance_lod"]
                        or self.frontview_residual_cover_config[
                            "covariance_competition_enabled"
                        ]
                    )
                    else None
                ),
                global_opacities=(
                    self.get_opacity().detach().cpu().numpy()
                    if (
                        self.frontview_residual_cover_config["use_covariance_lod"]
                        or self.frontview_residual_cover_config[
                            "covariance_competition_enabled"
                        ]
                    )
                    else None
                ),
            )
            occupied_mask = np.ones((before_filter_num,), dtype=np.bool_)
            occupied_mask[keep_indices] = False
        elif self._frontview_identity_lod_enabled():
            (
                keep_indices,
                accepted_parents,
                accepted_levels,
                accepted_sectors,
            ) = self.frontview_identity_lod.filter_candidates(
                frame_id=int(cam.cam_idx),
                uv=pts_2d,
                depths=pts_depth,
                residual_scores=residual_scores,
                depth_confidences=proposal_depth_confidence,
                sparse_valid=proposal_sparse_valid,
                track_ids=proposal_track_ids,
                projection_info=(
                    None if render_pkg is None else render_pkg.get("projection_info")
                ),
                global_uids=self.get_gaussian_uids(),
                depthcov_budget=extra_pts_num,
                global_means=self.get_xyz(),
                global_scales=self.get_scaling(),
                world_points=pts_3d,
                log_scales=init_scale,
            )
            occupied_mask = np.ones((before_filter_num,), dtype=np.bool_)
            occupied_mask[keep_indices] = False
            responsibility_parent_uids[keep_indices] = accepted_parents
            responsibility_levels[keep_indices] = accepted_levels
            responsibility_sectors[keep_indices] = accepted_sectors
        elif self._frontview_birth_enabled():
            keep_indices = self.frontview_track_ledger.new_indices(
                proposal_track_ids, pts_depth, at_commit=False
            )
            occupied_mask = np.ones((before_filter_num,), dtype=np.bool_)
            occupied_mask[keep_indices] = False
            near_hash = np.zeros((before_filter_num,), dtype=np.bool_)
            if bool(self.frontview_birth_config["near_hash_competition"]):
                near_hash = (
                    (proposal_track_ids < 0)
                    & (pts_depth < float(self.frontview_birth_config["near_hash_depth_m"]))
                    & ~occupied_mask
                )
                near_positions = np.flatnonzero(near_hash)
                if len(near_positions):
                    occupied_mask[near_positions] |= self.hash_block.getOccupy(
                        pts_3d[near_positions],
                        pts_color[near_positions],
                        cur_view_scale_size,
                    )
                self.frontview_birth_stats["near_hash_query_rows"] += int(
                    len(near_positions)
                )
                self.frontview_birth_stats["far_hash_bypass_rows"] += int(
                    np.sum((proposal_track_ids < 0) & ~near_hash)
                )
        elif (
            self._frontview_far_field_enabled()
            or self._frontview_scale_cover_enabled()
        ):
            if self._frontview_far_field_enabled():
                proposal_far_field, adaptive_route_metadata = (
                    far_field_responsibility_mask(
                        responsibility_depth,
                        proposal_sparse_valid,
                        proposal_track_ids,
                        self.frontview_far_field_config,
                        parallax_pixels=proposal_responsibility_parallax_pixels,
                        projected_radii=proposal_responsibility_projected_radii,
                        log_depth_stds=proposal_log_depth_stds,
                        return_metadata=True,
                    )
                )
                if self.frontview_inverse_depth_certificate_config["enabled"]:
                    dense = ~proposal_sparse_valid
                    policy = self.frontview_inverse_depth_certificate_config[
                        "uncertified_policy"
                    ]
                    proposal_far_field[dense & proposal_cidec_certified] = False
                    if policy in ("projective", "projective_reject_conflict"):
                        proposal_far_field[dense & ~proposal_cidec_certified] = True
                    self.frontview_inverse_depth_certificate_stats[
                        "projective_uncertified_rows"
                    ] += int(np.sum(dense & ~proposal_cidec_certified))
                if (
                    self.frontview_far_field_config["routing_mode"]
                    in ("causal_observability", "adaptive_observability")
                ):
                    dense = ~proposal_sparse_valid
                    stats = self.frontview_far_field_stats
                    stats["causal_route_calls"] += 1
                    stats["causal_route_rows"] += int(before_filter_num)
                    stats["causal_visible_rows"] += int(
                        np.sum(proposal_responsibility_parallax_support > 0)
                    )
                    stats["causal_projective_rows"] += int(
                        np.sum(proposal_far_field)
                    )
                    stats["causal_metric_depthcov_rows"] += int(
                        np.sum(dense & ~proposal_far_field)
                    )
                    stats["causal_parallax_pixel_sum"] += float(
                        np.sum(proposal_responsibility_parallax_pixels)
                    )
                    stats["causal_support_pixel_sum"] += float(
                        np.sum(proposal_responsibility_projected_radii)
                    )
                    stats["causal_log_depth_std_sum"] += float(
                        np.sum(proposal_log_depth_stds)
                    )
                    if adaptive_route_metadata is not None:
                        boundaries = adaptive_route_metadata["boundaries_m"]
                        stats["adaptive_route_calls"] += 1
                        stats["adaptive_route_far_rows"] += int(
                            adaptive_route_metadata["far_rows"]
                        )
                        stats["adaptive_route_iterations"] += int(
                            adaptive_route_metadata["iterations"]
                        )
                        stats["adaptive_route_objective_sum"] += float(
                            adaptive_route_metadata["objective"]
                        )
                        stats["adaptive_route_boundary_sum_m"] = [
                            old + new
                            for old, new in zip(
                                stats["adaptive_route_boundary_sum_m"], boundaries
                            )
                        ]
                        stats["adaptive_route_regime_counts"] = [
                            old + new
                            for old, new in zip(
                                stats["adaptive_route_regime_counts"],
                                adaptive_route_metadata["regime_counts"],
                            )
                        ]
                        stats["last_adaptive_route_boundaries_m"] = boundaries
                    proposal_radial_scale_factors = (
                        projective_radial_scale_factors(
                            proposal_parallax_pixels,
                            proposal_projected_radii,
                            proposal_log_depth_stds,
                            proposal_far_field,
                            mode=self.frontview_far_field_config[
                                "projective_covariance_mode"
                            ],
                            seed=int(self.frontview_far_field_config["shuffle_seed"])
                            + int(cam.cam_idx),
                        )
                    )
                support_mode = self.frontview_far_field_config[
                    "fallback_support_mode"
                ]
                fallback_projective = proposal_depth_fallback & proposal_far_field
                if support_mode != "legacy" and np.any(fallback_projective):
                    stats = self.frontview_far_field_stats
                    stats["fallback_support_rows"] += int(
                        np.sum(fallback_projective)
                    )
                    if "structure" in support_mode:
                        certificate_mode = support_mode.startswith(
                            "budget_certificate_structure"
                        )
                        certificate_strength = None
                        support_radius_pixels = None
                        if certificate_mode:
                            parallax_margin = np.divide(
                                proposal_parallax_pixels,
                                proposal_projected_radii,
                                out=np.zeros_like(proposal_parallax_pixels),
                                where=proposal_projected_radii > 0.0,
                            )
                            precision_denom = (
                                proposal_parallax_pixels * proposal_log_depth_stds
                            )
                            precision_margin = np.divide(
                                proposal_projected_radii,
                                precision_denom,
                                out=np.ones_like(proposal_projected_radii),
                                where=precision_denom > 0.0,
                            )
                            certificate_information = np.clip(
                                np.minimum(parallax_margin, precision_margin),
                                0.0,
                                1.0,
                            )
                            certificate_strength = 1.0 - certificate_information
                            support_radius_pixels = budgeted_fallback_radius(
                                (cam.get_width(level), cam.get_height(level)),
                                int(self.extra_pts_num),
                            )
                        (
                            proposal_covariance_scale_factors,
                            proposal_initial_quaternions,
                            anisotropy,
                        ) = structure_aligned_covariances(
                            cam.get_gt_image(level),
                            pts_2d,
                            proposal_view_directions,
                            cam.get_raw_pose()[:3, :3].detach().cpu().numpy(),
                            fallback_projective,
                            support_radius_pixels=support_radius_pixels,
                            certificate_strength=certificate_strength,
                            shuffle=support_mode.endswith("_shuffled"),
                            seed=int(self.frontview_far_field_config["shuffle_seed"])
                            + int(cam.cam_idx),
                        )
                        values = anisotropy[fallback_projective]
                        stats["fallback_structure_rows"] += int(len(values))
                        stats["fallback_anisotropy_sum"] += float(np.sum(values))
                        maximum = float(np.max(values))
                        stats["fallback_anisotropy_max"] = (
                            maximum
                            if stats["fallback_anisotropy_max"] is None
                            else max(stats["fallback_anisotropy_max"], maximum)
                        )
                if (
                    self.frontview_far_field_config["responsibility_basis"]
                    == "persistent_identity"
                ):
                    has_identity = proposal_track_ids >= 0
                    self.frontview_far_field_stats["identity_route_calls"] += 1
                    self.frontview_far_field_stats["identity_route_rows"] += int(
                        before_filter_num
                    )
                    self.frontview_far_field_stats[
                        "persistent_identity_rows"
                    ] += int(np.sum(has_identity))
                    self.frontview_far_field_stats[
                        "sparse_missing_identity_rows"
                    ] += int(np.sum(proposal_sparse_valid & ~has_identity))
                    self.frontview_far_field_stats[
                        "identity_far_field_rows"
                    ] += int(np.sum(proposal_far_field))
            if self._frontview_far_field_enabled() and bool(
                self.frontview_far_field_config["shuffle_responsibility"]
            ):
                far_count = int(np.sum(proposal_far_field))
                proposal_far_field = matched_responsibility_shuffle(
                    proposal_far_field,
                    ~proposal_sparse_valid,
                    responsibility_depth,
                    int(self.frontview_far_field_config["shuffle_seed"])
                    + int(cam.cam_idx),
                    mode=self.frontview_far_field_config[
                        "responsibility_shuffle_mode"
                    ],
                )
                self.frontview_far_field_stats["responsibility_shuffle_calls"] += 1
                self.frontview_far_field_stats["responsibility_shuffled_rows"] += (
                    far_count
                )
            if (
                self._frontview_far_field_enabled()
                and self.frontview_far_field_config["map_redundancy_gate"]
            ):
                map_evidence = self.frontview_far_field_config[
                    "map_redundancy_evidence"
                ]
                map_residuals = None
                map_residual_scale = None
                photometric_available = (
                    map_evidence != "geometry" and render_pkg is not None
                )
                if photometric_available:
                    rendered_diff = render_pkg["diff"].reshape(-1)
                    rendered_opacity = render_pkg["opacity"].reshape(-1)
                    weights = torch.clamp(rendered_opacity, 0.0, 1.0)
                    map_residual_scale = float(
                        torch.sqrt(
                            torch.sum(weights * rendered_diff.square())
                            / torch.clamp_min(
                                torch.sum(weights),
                                torch.finfo(rendered_diff.dtype).eps,
                            )
                        ).item()
                    )
                    map_residual_scale = max(
                        map_residual_scale, np.finfo(np.float32).eps
                    )
                    map_residuals = residual_scores.copy()
                    if map_evidence == "photometric_shuffled":
                        rows = np.flatnonzero(proposal_far_field)
                        if len(rows) > 1:
                            rng = np.random.default_rng(
                                int(self.frontview_far_field_config["shuffle_seed"])
                                + int(cam.cam_idx)
                            )
                            map_residuals[rows] = map_residuals[
                                rows[rng.permutation(len(rows))]
                            ]
                proposal_map_log_odds = projective_map_posterior_log_odds(
                    pts_depth,
                    proposal_stable_depth,
                    1.0 - coverage_scores,
                    proposal_parallax_pixels,
                    proposal_projected_radii,
                    residuals=map_residuals,
                    residual_scale=map_residual_scale,
                )
                proposal_map_redundant = (
                    proposal_map_log_odds >= 0.0
                ) & proposal_far_field
                posterior_refill_requested = int(
                    np.sum(
                        proposal_map_redundant
                        & proposal_budget_primary
                        & ~proposal_sparse_valid
                    )
                )
                stats = self.frontview_far_field_stats
                stats["map_gate_calls"] += 1
                stats["map_gate_rows"] += int(np.sum(proposal_far_field))
                stats["map_gate_rejected_rows"] += int(
                    np.sum(proposal_map_redundant)
                )
                if photometric_available:
                    stats["map_gate_photometric_calls"] += 1
                    stats["map_gate_residual_scale_sum"] += float(
                        map_residual_scale
                    )
                    stats["map_gate_shuffled_calls"] += int(
                        map_evidence == "photometric_shuffled"
                    )
            if self._frontview_scale_cover_enabled():
                if self.frontview_scale_cover.projected_handoff_enabled:
                    intrinsics = cam.get_int_mat(level)
                    self.frontview_scale_cover.activate_projected_handoff(
                        cam.get_raw_pose().detach().cpu().numpy(),
                        0.5 * (cam.get_fx(level) + cam.get_fy(level)),
                        cam.get_width(level),
                        cam.get_height(level),
                        float(intrinsics[0, 2].item()),
                        float(intrinsics[1, 2].item()),
                        cam.near,
                        cam.far,
                        int(cam.cam_idx),
                    )
                scale_cover_source_ranks = (
                    self.frontview_scale_cover.candidate_source_ranks(
                        proposal_sparse_valid,
                        responsibility_depth,
                        int(cam.cam_idx),
                    )
                )
                appearance_eligible = (
                    self.frontview_scale_cover.appearance_certificates(
                        residual_scores,
                        proposal_depth_confidence,
                        proposal_sparse_valid,
                        responsibility_depth,
                        int(cam.cam_idx),
                    )
                )
                occupied_mask, scale_parent_uids = (
                    self.frontview_scale_cover.occupied_with_parents(
                        responsibility_pts_3d,
                        proposal_responsibility_cover_sizes,
                        pts_color,
                        scale_cover_source_ranks,
                        appearance_eligible,
                        view_directions=proposal_responsibility_view_directions,
                        residual_scores=residual_scores,
                        depth_confidences=proposal_depth_confidence,
                        sparse_valid=proposal_sparse_valid,
                        depths=responsibility_depth,
                        frame_id=int(cam.cam_idx),
                    )
                )
                occupied_mask = (
                    self.frontview_scale_cover.apply_sparse_track_identity(
                        occupied_mask,
                        proposal_track_ids,
                        proposal_sparse_valid,
                        responsibility_depth,
                        int(cam.cam_idx),
                    )
                )
                scale_parent_uids = self.frontview_scale_cover.shuffle_parents(
                    scale_parent_uids,
                    responsibility_depth,
                    proposal_sparse_valid,
                    int(cam.cam_idx),
                )
                responsibility_parent_uids[:] = scale_parent_uids
                occupied_mask = self.frontview_scale_cover.route_evidence_quota(
                    occupied_mask,
                    responsibility_depth,
                    proposal_sparse_valid,
                    residual_scores,
                    coverage_scores,
                    proposal_depth_confidence,
                    int(cam.cam_idx),
                    eligible=~proposal_far_field,
                    uv=pts_2d,
                    multiview_support_scores=proposal_multiview_support,
                )
                occupied_mask = self.frontview_scale_cover.shuffle(
                    occupied_mask,
                    responsibility_depth,
                    proposal_sparse_valid,
                    int(cam.cam_idx),
                    eligible=~proposal_far_field,
                )
            elif self._frontview_sparse_scale_map_enabled():
                occupied_mask = self.frontview_sparse_scale_map.occupied(
                    pts_3d, cur_view_scale_size
                )
            else:
                occupied_mask = self.hash_block.getOccupy(
                    pts_3d, pts_color, cur_view_scale_size
                )
                self.frontview_far_field_stats["hash_query_rows"] += int(
                    before_filter_num
                )
            occupied_mask[proposal_far_field] = proposal_map_redundant[
                proposal_far_field
            ]
            if (
                self.frontview_far_field_config["unobservable_birth_policy"]
                == "reject"
            ):
                occupied_mask[proposal_far_field] = True
                self.frontview_far_field_stats["unobservable_rejected_rows"] += int(
                    np.sum(proposal_far_field)
                )
            occupied_mask |= proposal_cidec_reject
            self.frontview_far_field_stats["host_rows"] += int(before_filter_num)
            self.frontview_far_field_stats["hash_bypass_rows"] += int(
                np.sum(proposal_far_field)
            )
        else:
            occupied_mask = self.hash_block.getOccupy(
                pts_3d, pts_color, cur_view_scale_size
            )
        pts_3d = pts_3d[~occupied_mask]
        responsibility_pts_3d = responsibility_pts_3d[~occupied_mask]
        pts_color = pts_color[~occupied_mask]
        pts_2d = pts_2d[~occupied_mask]
        pts_depth = pts_depth[~occupied_mask]
        responsibility_depth = responsibility_depth[~occupied_mask]
        residual_scores = residual_scores[~occupied_mask]
        coverage_scores = coverage_scores[~occupied_mask]
        proposal_sparse_valid = proposal_sparse_valid[~occupied_mask]
        proposal_depth_confidence = proposal_depth_confidence[~occupied_mask]
        proposal_multiview_support = proposal_multiview_support[~occupied_mask]
        proposal_stable_depth = proposal_stable_depth[~occupied_mask]
        proposal_frequency_scores = proposal_frequency_scores[~occupied_mask]
        proposal_track_ids = proposal_track_ids[~occupied_mask]
        proposal_budget_primary = proposal_budget_primary[~occupied_mask]
        proposal_map_log_odds = proposal_map_log_odds[~occupied_mask]
        proposal_cover_sizes = proposal_cover_sizes[~occupied_mask]
        proposal_responsibility_cover_sizes = (
            proposal_responsibility_cover_sizes[~occupied_mask]
        )
        proposal_view_directions = proposal_view_directions[~occupied_mask]
        proposal_responsibility_view_directions = (
            proposal_responsibility_view_directions[~occupied_mask]
        )
        proposal_far_field = proposal_far_field[~occupied_mask]
        proposal_projected_radii = proposal_projected_radii[~occupied_mask]
        proposal_responsibility_projected_radii = (
            proposal_responsibility_projected_radii[~occupied_mask]
        )
        proposal_log_depth_stds = proposal_log_depth_stds[~occupied_mask]
        proposal_parallax_pixels = proposal_parallax_pixels[~occupied_mask]
        proposal_radial_scale_factors = proposal_radial_scale_factors[
            ~occupied_mask
        ]
        proposal_covariance_scale_factors = proposal_covariance_scale_factors[
            ~occupied_mask
        ]
        proposal_initial_quaternions = proposal_initial_quaternions[
            ~occupied_mask
        ]
        proposal_scale_expansion_limits = proposal_scale_expansion_limits[
            ~occupied_mask
        ]
        proposal_footprint_target_scales = proposal_footprint_target_scales[
            ~occupied_mask
        ]
        proposal_depth_fallback = proposal_depth_fallback[~occupied_mask]
        proposal_tracked_metric = proposal_tracked_metric[~occupied_mask]
        responsibility_parent_uids = responsibility_parent_uids[~occupied_mask]
        responsibility_levels = responsibility_levels[~occupied_mask]
        responsibility_sectors = responsibility_sectors[~occupied_mask]

        if (
            self._frontview_residual_cover_enabled()
            or self._frontview_identity_lod_enabled()
        ):
            no_conflict_index = np.ones((len(pts_3d),), dtype=np.bool_)
        elif self._frontview_birth_enabled():
            no_conflict_index = np.ones((len(pts_3d),), dtype=np.bool_)
            if bool(self.frontview_birth_config["near_hash_competition"]):
                near_hash = (
                    (proposal_track_ids < 0)
                    & (pts_depth < float(self.frontview_birth_config["near_hash_depth_m"]))
                )
                near_positions = np.flatnonzero(near_hash)
                if len(near_positions):
                    near_keep = self.hash_block.get_no_conflict_index(
                        pts_3d[near_positions], cur_view_scale_size
                    )
                    no_conflict_index[near_positions] = near_keep
        elif (
            self._frontview_far_field_enabled()
            or self._frontview_scale_cover_enabled()
        ):
            far_field = proposal_far_field
            no_conflict_index = np.ones((len(pts_3d),), dtype=np.bool_)
            near_positions = np.flatnonzero(~far_field)
            if len(near_positions) and not (
                self._frontview_scale_cover_enabled()
                or self._frontview_sparse_scale_map_enabled()
            ):
                no_conflict_index[near_positions] = (
                    self.hash_block.get_no_conflict_index(
                        pts_3d[near_positions], cur_view_scale_size
                    )
                )
            far_positions = np.flatnonzero(far_field)
            if len(far_positions):
                far_keep = projective_survivor_mask(
                    pts_2d[far_positions],
                    responsibility_depth[far_positions],
                    residual_scores[far_positions],
                    self.frontview_far_field_config,
                    projected_radii=proposal_responsibility_projected_radii[
                        far_positions
                    ],
                    log_depth_stds=proposal_log_depth_stds[far_positions],
                    image_size=(cam.get_width(level), cam.get_height(level)),
                    birth_budget=int(self.extra_pts_num),
                    pool_multiplier=int(
                        self.frontview_sampling_config.get("pool_multiplier", 1)
                    ),
                    max_log_depth_std=float(
                        self.depth_cov_estimator.std_valid_threshold
                    ),
                    primary_mask=(
                        proposal_budget_primary[far_positions]
                        if self.frontview_far_field_config[
                            "posterior_budget_refill"
                        ]
                        else None
                    ),
                )
                no_conflict_index[far_positions] = far_keep
                self.frontview_far_field_stats["projective_rejected_rows"] += int(
                    np.sum(~far_keep)
                )
                if (
                    self.frontview_far_field_config["projective_nms_mode"]
                    == "gaussian_support"
                ):
                    stats = self.frontview_far_field_stats
                    stats["adaptive_nms_calls"] += 1
                    stats["adaptive_nms_rows"] += int(len(far_positions))
                    stats["adaptive_nms_rejected_rows"] += int(np.sum(~far_keep))
                elif (
                    self.frontview_far_field_config["projective_nms_mode"]
                    == "budget_cells"
                ):
                    cell_px, log_depth_width = budget_cell_parameters(
                        (cam.get_width(level), cam.get_height(level)),
                        int(self.extra_pts_num),
                        int(self.frontview_sampling_config.get("pool_multiplier", 1)),
                        float(self.depth_cov_estimator.std_valid_threshold),
                    )
                    stats = self.frontview_far_field_stats
                    stats["budget_nms_calls"] += 1
                    stats["budget_nms_cell_pixel_sum"] += float(cell_px)
                    stats["budget_nms_log_depth_width_sum"] += float(
                        log_depth_width
                    )
        else:
            no_conflict_index = self.hash_block.get_no_conflict_index(
                pts_3d, cur_view_scale_size
            )
        pts_3d = pts_3d[no_conflict_index]
        responsibility_pts_3d = responsibility_pts_3d[no_conflict_index]
        pts_color = pts_color[no_conflict_index]
        pts_2d = pts_2d[no_conflict_index]
        pts_depth = pts_depth[no_conflict_index]
        responsibility_depth = responsibility_depth[no_conflict_index]
        residual_scores = residual_scores[no_conflict_index]
        coverage_scores = coverage_scores[no_conflict_index]
        proposal_sparse_valid = proposal_sparse_valid[no_conflict_index]
        proposal_depth_confidence = proposal_depth_confidence[no_conflict_index]
        proposal_multiview_support = proposal_multiview_support[no_conflict_index]
        proposal_stable_depth = proposal_stable_depth[no_conflict_index]
        proposal_frequency_scores = proposal_frequency_scores[no_conflict_index]
        proposal_track_ids = proposal_track_ids[no_conflict_index]
        proposal_budget_primary = proposal_budget_primary[no_conflict_index]
        proposal_map_log_odds = proposal_map_log_odds[no_conflict_index]
        proposal_cover_sizes = proposal_cover_sizes[no_conflict_index]
        proposal_responsibility_cover_sizes = (
            proposal_responsibility_cover_sizes[no_conflict_index]
        )
        proposal_view_directions = proposal_view_directions[no_conflict_index]
        proposal_responsibility_view_directions = (
            proposal_responsibility_view_directions[no_conflict_index]
        )
        proposal_far_field = proposal_far_field[no_conflict_index]
        proposal_projected_radii = proposal_projected_radii[no_conflict_index]
        proposal_responsibility_projected_radii = (
            proposal_responsibility_projected_radii[no_conflict_index]
        )
        proposal_log_depth_stds = proposal_log_depth_stds[no_conflict_index]
        proposal_parallax_pixels = proposal_parallax_pixels[no_conflict_index]
        proposal_radial_scale_factors = proposal_radial_scale_factors[
            no_conflict_index
        ]
        proposal_covariance_scale_factors = proposal_covariance_scale_factors[
            no_conflict_index
        ]
        proposal_initial_quaternions = proposal_initial_quaternions[
            no_conflict_index
        ]
        proposal_scale_expansion_limits = proposal_scale_expansion_limits[
            no_conflict_index
        ]
        proposal_footprint_target_scales = proposal_footprint_target_scales[
            no_conflict_index
        ]
        proposal_depth_fallback = proposal_depth_fallback[no_conflict_index]
        proposal_tracked_metric = proposal_tracked_metric[no_conflict_index]
        responsibility_parent_uids = responsibility_parent_uids[no_conflict_index]
        responsibility_levels = responsibility_levels[no_conflict_index]
        responsibility_sectors = responsibility_sectors[no_conflict_index]

        if (
            self._frontview_far_field_enabled()
            and self.frontview_far_field_config["posterior_budget_refill"]
        ):
            keep_refill, refill_stats = posterior_budget_refill_mask(
                proposal_budget_primary,
                proposal_sparse_valid,
                proposal_map_log_odds,
                residual_scores,
                posterior_refill_requested,
                reserve_eligible=proposal_far_field,
                shuffle_evidence=self.frontview_far_field_config[
                    "shuffle_refill_evidence"
                ],
                seed=int(self.frontview_far_field_config["shuffle_seed"])
                + int(cam.cam_idx),
            )
            stats = self.frontview_far_field_stats
            stats["posterior_refill_calls"] += 1
            stats["posterior_refill_requested_rows"] += int(
                refill_stats["requested"]
            )
            stats["posterior_refill_reserve_rows"] += int(refill_stats["reserves"])
            stats["posterior_refill_selected_rows"] += int(refill_stats["selected"])
            stats["posterior_refill_shuffled_calls"] += int(
                self.frontview_far_field_config["shuffle_refill_evidence"]
            )
            pts_3d = pts_3d[keep_refill]
            responsibility_pts_3d = responsibility_pts_3d[keep_refill]
            pts_color = pts_color[keep_refill]
            pts_2d = pts_2d[keep_refill]
            pts_depth = pts_depth[keep_refill]
            responsibility_depth = responsibility_depth[keep_refill]
            residual_scores = residual_scores[keep_refill]
            coverage_scores = coverage_scores[keep_refill]
            proposal_sparse_valid = proposal_sparse_valid[keep_refill]
            proposal_depth_confidence = proposal_depth_confidence[keep_refill]
            proposal_multiview_support = proposal_multiview_support[keep_refill]
            proposal_stable_depth = proposal_stable_depth[keep_refill]
            proposal_frequency_scores = proposal_frequency_scores[keep_refill]
            proposal_track_ids = proposal_track_ids[keep_refill]
            proposal_budget_primary = proposal_budget_primary[keep_refill]
            proposal_cover_sizes = proposal_cover_sizes[keep_refill]
            proposal_responsibility_cover_sizes = (
                proposal_responsibility_cover_sizes[keep_refill]
            )
            proposal_view_directions = proposal_view_directions[keep_refill]
            proposal_responsibility_view_directions = (
                proposal_responsibility_view_directions[keep_refill]
            )
            proposal_far_field = proposal_far_field[keep_refill]
            proposal_projected_radii = proposal_projected_radii[keep_refill]
            proposal_responsibility_projected_radii = (
                proposal_responsibility_projected_radii[keep_refill]
            )
            proposal_log_depth_stds = proposal_log_depth_stds[keep_refill]
            proposal_parallax_pixels = proposal_parallax_pixels[keep_refill]
            proposal_radial_scale_factors = proposal_radial_scale_factors[keep_refill]
            proposal_covariance_scale_factors = proposal_covariance_scale_factors[
                keep_refill
            ]
            proposal_initial_quaternions = proposal_initial_quaternions[keep_refill]
            proposal_scale_expansion_limits = proposal_scale_expansion_limits[
                keep_refill
            ]
            proposal_footprint_target_scales = proposal_footprint_target_scales[
                keep_refill
            ]
            proposal_depth_fallback = proposal_depth_fallback[keep_refill]
            proposal_tracked_metric = proposal_tracked_metric[keep_refill]
            responsibility_parent_uids = responsibility_parent_uids[keep_refill]
            responsibility_levels = responsibility_levels[keep_refill]
            responsibility_sectors = responsibility_sectors[keep_refill]

        if isinstance(init_scale, np.ndarray):
            init_scale = init_scale[~occupied_mask]
            responsibility_init_scale = responsibility_init_scale[~occupied_mask]
            init_scale = init_scale[no_conflict_index]
            responsibility_init_scale = responsibility_init_scale[no_conflict_index]
            if (
                self._frontview_far_field_enabled()
                and self.frontview_far_field_config["posterior_budget_refill"]
            ):
                init_scale = init_scale[keep_refill]
                responsibility_init_scale = responsibility_init_scale[keep_refill]

        proposal_metric_certificates = np.ones(
            (len(pts_depth),), dtype=np.float32
        )
        if (
            self.causal_dual_responsibility_config[
                "finite_depth_certificate_enabled"
            ]
            and len(pts_depth)
        ):
            certificate_result = causal_finite_depth_certificates(
                cam,
                list(reference_cameras or ()),
                torch.as_tensor(pts_2d, device=self.device, dtype=torch.float32),
                torch.as_tensor(pts_depth, device=self.device, dtype=torch.float32),
                level,
                pixel_sigma=self.causal_dual_responsibility_config[
                    "finite_depth_pixel_sigma"
                ],
            )
            proposal_metric_certificates = (
                certificate_result["certificate"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            certificate_stats = self.causal_dual_responsibility_stats
            certificate_stats["finite_certificate_calls"] += 1
            certificate_stats["finite_certificate_rows"] += int(len(pts_depth))
            certificate_stats["finite_certificate_valid_rows"] += int(
                torch.count_nonzero(certificate_result["valid_views"] > 0).item()
            )
            certificate_stats["finite_certificate_observability_sum"] += float(
                certificate_result["observability"].sum().item()
            )
            certificate_stats["finite_certificate_support_sum"] += float(
                certificate_result["finite_support"].sum().item()
            )
            certificate_stats["finite_certificate_value_sum"] += float(
                certificate_result["certificate"].sum().item()
            )

        Log(
            "Proposing new gaussians (before/after host filter): {} / {}".format(
                before_filter_num, pts_3d.shape[0]
            ),
            tag="GaussianModel",
        )

        half_patch = 4.0 * (2**level)
        patch_bboxes = np.stack(
            [
                pts_2d[:, 0] - half_patch,
                pts_2d[:, 1] - half_patch,
                pts_2d[:, 0] + half_patch,
                pts_2d[:, 1] + half_patch,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        return GaussianProposalBatch(
            source_frame_id=int(cam.cam_idx),
            level=int(level),
            uv=pts_2d,
            patch_bboxes=patch_bboxes,
            depths=pts_depth,
            inverse_depths=1.0 / np.maximum(pts_depth, 1.0e-8),
            world_points=pts_3d.astype(np.float32, copy=False),
            log_scales=init_scale.astype(np.float32, copy=False),
            colors=pts_color.astype(np.float32, copy=False),
            residual_scores=residual_scores,
            coverage_scores=coverage_scores,
            sparse_depth_valid=proposal_sparse_valid,
            cover_sizes=proposal_cover_sizes,
            view_directions=proposal_view_directions,
            responsibility_depths=responsibility_depth,
            responsibility_world_points=responsibility_pts_3d.astype(
                np.float32, copy=False
            ),
            responsibility_log_scales=responsibility_init_scale.astype(
                np.float32, copy=False
            ),
            responsibility_cover_sizes=proposal_responsibility_cover_sizes,
            responsibility_view_directions=(
                proposal_responsibility_view_directions
            ),
            radial_scale_factors=proposal_radial_scale_factors,
            covariance_scale_factors=proposal_covariance_scale_factors,
            scale_expansion_limits=proposal_scale_expansion_limits,
            footprint_target_scales=proposal_footprint_target_scales,
            initial_max_parallax_sin2=np.clip(
                proposal_parallax_pixels / max(float(focal_pixels), 1.0e-8),
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
            ** 2,
            initial_quaternions=proposal_initial_quaternions,
            depth_fallback=proposal_depth_fallback,
            stable_depths=proposal_stable_depth,
            depth_confidences=proposal_depth_confidence,
            metric_certificates=proposal_metric_certificates,
            multiview_support_scores=proposal_multiview_support,
            frequency_scores=proposal_frequency_scores,
            track_ids=proposal_track_ids,
            budget_primary=proposal_budget_primary,
            responsibility_parent_uids=responsibility_parent_uids,
            responsibility_levels=responsibility_levels,
            responsibility_sectors=responsibility_sectors,
            source_kinds=np.where(
                proposal_tracked_metric,
                "tracked_metric",
                np.where(
                    proposal_sparse_valid,
                    "sparse",
                    np.where(proposal_far_field, "depthcov_far", "depthcov"),
                ),
            ).astype("U32"),
            view_scale_size=float(cur_view_scale_size),
            create_new_group=bool(create_new_group),
            metadata={
                "densification_mode": self.densification_mode,
                "before_host_filter": int(before_filter_num),
                "scale_cover_epoch": (
                    int(self.frontview_scale_cover.admission_epoch)
                    if self._frontview_scale_cover_enabled()
                    else None
                ),
                "birth_camera_center": birth_camera_center,
                "birth_focal_pixels": float(focal_pixels),
                "depthcov_log_std_threshold": float(
                    self.depth_cov_estimator.std_valid_threshold
                ),
                "birth_image_size": (
                    int(cam.get_width(level)),
                    int(cam.get_height(level)),
                ),
                "birth_candidate_budget": int(self.extra_pts_num),
                "birth_pool_multiplier": int(
                    self.frontview_sampling_config.get("pool_multiplier", 1)
                ),
                "frequency_sampling_enabled": bool(
                    self.frequency_sampling_config.get("enabled", False)
                ),
                "frequency_score_mean": float(
                    np.mean(proposal_frequency_scores)
                    if len(proposal_frequency_scores)
                    else 0.0
                ),
                "frequency_score_p95": float(
                    np.percentile(proposal_frequency_scores, 95)
                    if len(proposal_frequency_scores)
                    else 0.0
                ),
            },
        )

    @torch.no_grad()
    def commit_proposals(
        self,
        proposals,
        indices=None,
        world_points=None,
        colors=None,
        log_scales=None,
        initial_opacity=0.5,
        force_new_group=False,
        target_group_id=None,
        admission_certificate=None,
    ):
        """Permanently insert selected proposals and initialize optimizer rows."""
        selected = proposals.select(indices)
        certificate = self._require_worldtest_certificate(
            admission_certificate, proposals=selected, path="commit_proposals"
        )
        if world_points is not None:
            selected.world_points = np.asarray(world_points, dtype=np.float32)
            selected.metadata["scale_cover_recheck_required"] = True
        if colors is not None:
            selected.colors = np.asarray(colors, dtype=np.float32)
            selected.metadata["scale_cover_recheck_required"] = True
        if log_scales is not None:
            selected.log_scales = np.asarray(log_scales, dtype=np.float32)
        if not (
            len(selected.world_points)
            == len(selected.colors)
            == len(selected.log_scales)
        ):
            raise ValueError("Refined commit arrays must match the selected proposal count")

        certificate_ids = (
            [certificate.certificate_id] * len(selected)
            if certificate is not None
            else None
        )
        return self._commit_selected_proposals(
            selected,
            initial_opacity=initial_opacity,
            force_new_group=force_new_group,
            target_group_id=target_group_id,
            certificate_ids_by_row=certificate_ids,
            proposed_count=len(proposals),
            result_source_frame_id=proposals.source_frame_id,
        )

    @torch.no_grad()
    def commit_certified_proposals(
        self,
        proposal_batches,
        admission_certificates,
        initial_opacity=0.5,
        force_new_group=False,
        target_group_id=None,
    ):
        """Validate per-track certificates, then mutate the map once for the frame."""
        batches = [batch.select(None) for batch in proposal_batches if len(batch) > 0]
        certificates = list(admission_certificates)
        if len(batches) != len(proposal_batches) or len(batches) != len(certificates):
            raise ValueError("Certified proposal batches and certificates must match")
        if not batches:
            raise ValueError("At least one certified proposal batch is required")

        certificate_ids = []
        for batch, certificate in zip(batches, certificates):
            validated = self._require_worldtest_certificate(
                certificate, proposals=batch, path="commit_certified_proposals"
            )
            if validated is None:
                raise RuntimeError("Certified batch commit requires an active authority")
            certificate_ids.extend([validated.certificate_id] * len(batch))

        combined = GaussianProposalBatch.concatenate(
            batches, source_frame_id=max(batch.source_frame_id for batch in batches)
        )
        return self._commit_selected_proposals(
            combined,
            initial_opacity=initial_opacity,
            force_new_group=force_new_group,
            target_group_id=target_group_id,
            certificate_ids_by_row=certificate_ids,
            sequential_hash_semantics=True,
        )

    def _commit_selected_proposals(
        self,
        selected,
        *,
        initial_opacity,
        force_new_group,
        target_group_id,
        certificate_ids_by_row=None,
        proposed_count=None,
        result_source_frame_id=None,
        sequential_hash_semantics=False,
    ):
        selected_count = len(selected)
        selected_opacity = initial_opacity
        if not np.isscalar(initial_opacity):
            selected_opacity = np.asarray(initial_opacity, dtype=np.float32).reshape(-1)
            if selected_opacity.shape != (selected_count,):
                raise ValueError(
                    "Per-Gaussian initial opacity must match selected proposals"
                )
        if certificate_ids_by_row is not None and len(certificate_ids_by_row) != selected_count:
            raise ValueError("Per-row certificate IDs must match selected proposals")

        committed_uids = None
        ray_atlas_commit_keys = []
        if self._frontview_residual_cover_enabled():
            kept_positions = self.frontview_residual_cover.prepare_commit(selected)
        elif self._frontview_identity_lod_enabled():
            kept_positions, committed_uids = self.frontview_identity_lod.prepare_commit(
                selected
            )
        elif self._frontview_birth_enabled():
            kept_positions = self.frontview_track_ledger.new_indices(
                selected.track_ids, selected.depths, at_commit=True
            )
            if bool(self.frontview_birth_config["near_hash_competition"]):
                staged = selected.select(kept_positions)
                near_hash = (
                    (staged.track_ids < 0)
                    & (
                        staged.depths
                        < float(self.frontview_birth_config["near_hash_depth_m"])
                    )
                )
                local_keep = np.ones((len(staged),), dtype=np.bool_)
                near_positions = np.flatnonzero(near_hash)
                if len(near_positions):
                    occupied = self.hash_block.getOccupy(
                        staged.world_points[near_positions],
                        staged.colors[near_positions],
                        staged.view_scale_size,
                    )
                    available_positions = near_positions[~occupied]
                    local_keep[near_positions] = False
                    if len(available_positions):
                        no_conflict = self.hash_block.get_no_conflict_index(
                            staged.world_points[available_positions],
                            staged.view_scale_size,
                        )
                        local_keep[available_positions[no_conflict]] = True
                kept_positions = kept_positions[np.flatnonzero(local_keep)]
        elif (
            self._frontview_far_field_enabled()
            or self._frontview_scale_cover_enabled()
        ):
            far_field = selected.source_kinds == "depthcov_far"
            if self._frontview_scale_cover_enabled():
                reuse_same_epoch = bool(
                    self.frontview_scale_cover_config[
                        "reuse_same_epoch_admission"
                    ]
                    and not selected.metadata.get(
                        "scale_cover_recheck_required", False
                    )
                    and selected.metadata.get("scale_cover_epoch")
                    == self.frontview_scale_cover.admission_epoch
                )
                if reuse_same_epoch:
                    occupied = np.zeros((selected_count,), dtype=np.bool_)
                    self.frontview_scale_cover.stats[
                        "same_epoch_commit_reuses"
                    ] += 1
                else:
                    self.frontview_scale_cover.stats[
                        "stale_commit_requeries"
                    ] += 1
                    scale_cover_source_ranks = (
                        self.frontview_scale_cover.candidate_source_ranks(
                            selected.sparse_depth_valid,
                            selected.responsibility_depths,
                            selected.source_frame_id,
                        )
                    )
                    appearance_eligible = (
                        self.frontview_scale_cover.appearance_certificates(
                            selected.residual_scores,
                            selected.depth_confidences,
                            selected.sparse_depth_valid,
                            selected.responsibility_depths,
                            selected.source_frame_id,
                        )
                    )
                    occupied = self.frontview_scale_cover.occupied(
                        selected.responsibility_world_points,
                        selected.responsibility_cover_sizes,
                        selected.colors,
                        scale_cover_source_ranks,
                        appearance_eligible,
                        view_directions=selected.responsibility_view_directions,
                        residual_scores=selected.residual_scores,
                        depth_confidences=selected.depth_confidences,
                        sparse_valid=selected.sparse_depth_valid,
                        depths=selected.responsibility_depths,
                        frame_id=selected.source_frame_id,
                        directional_preselected=True,
                    )
                    occupied = (
                        self.frontview_scale_cover.apply_sparse_track_identity(
                            occupied,
                            selected.track_ids,
                            selected.sparse_depth_valid,
                            selected.responsibility_depths,
                            selected.source_frame_id,
                        )
                    )
                    occupied = self.frontview_scale_cover.route_evidence_quota(
                        occupied,
                        selected.responsibility_depths,
                        selected.sparse_depth_valid,
                        selected.residual_scores,
                        selected.coverage_scores,
                        selected.depth_confidences,
                        selected.source_frame_id,
                        eligible=~far_field,
                        uv=selected.uv,
                        multiview_support_scores=selected.multiview_support_scores,
                    )
                    occupied = self.frontview_scale_cover.shuffle(
                        occupied,
                        selected.responsibility_depths,
                        selected.sparse_depth_valid,
                        selected.source_frame_id,
                        eligible=~far_field,
                    )
            elif self._frontview_sparse_scale_map_enabled():
                occupied = self.frontview_sparse_scale_map.occupied(
                    selected.world_points, selected.view_scale_size
                )
            else:
                occupied = self.hash_block.getOccupy(
                    selected.world_points,
                    selected.colors,
                    selected.view_scale_size,
                )
                self.frontview_far_field_stats["hash_query_rows"] += int(
                    selected_count
                )
            occupied[far_field] = False
            local_keep = ~occupied
            near_positions = np.flatnonzero(local_keep & ~far_field)
            if len(near_positions) and not (
                self._frontview_scale_cover_enabled()
                or self._frontview_sparse_scale_map_enabled()
            ):
                near_keep = self.hash_block.get_no_conflict_index(
                    selected.world_points[near_positions],
                    selected.view_scale_size,
                )
                local_keep[near_positions] = near_keep
            far_positions = np.flatnonzero(local_keep & far_field)
            if len(far_positions):
                projected_radii = projected_gaussian_radii(
                    selected.responsibility_log_scales[far_positions],
                    selected.responsibility_depths[far_positions],
                    selected.metadata.get("birth_focal_pixels", 1.0),
                )
                log_depth_stds = float(
                    selected.metadata.get(
                        "depthcov_log_std_threshold",
                        self.depth_cov_estimator.std_valid_threshold,
                    )
                ) * (
                    1.0
                    - np.clip(
                        selected.depth_confidences[far_positions], 0.0, 1.0
                    )
                )
                far_keep = projective_survivor_mask(
                    selected.uv[far_positions],
                    selected.responsibility_depths[far_positions],
                    selected.residual_scores[far_positions],
                    self.frontview_far_field_config,
                    projected_radii=projected_radii,
                    log_depth_stds=log_depth_stds,
                    image_size=selected.metadata.get("birth_image_size"),
                    birth_budget=selected.metadata.get("birth_candidate_budget"),
                    pool_multiplier=selected.metadata.get("birth_pool_multiplier"),
                    max_log_depth_std=selected.metadata.get(
                        "depthcov_log_std_threshold"
                    ),
                    primary_mask=(
                        selected.budget_primary[far_positions]
                        if self.frontview_far_field_config[
                            "posterior_budget_refill"
                        ]
                        else None
                    ),
                )
                local_keep[far_positions] = far_keep
            kept_positions = np.flatnonzero(local_keep)
            ray_atlas = getattr(self, "frontview_ray_atlas", None)
            if (
                ray_atlas is not None
                and ray_atlas.enabled
                and len(kept_positions)
            ):
                atlas_positions = kept_positions[far_field[kept_positions]]
                if len(atlas_positions):
                    atlas_projected_radii = projected_gaussian_radii(
                        selected.responsibility_log_scales[atlas_positions],
                        selected.responsibility_depths[atlas_positions],
                        selected.metadata["birth_focal_pixels"],
                    )
                    atlas_log_depth_stds = float(
                        selected.metadata["depthcov_log_std_threshold"]
                    ) * (
                        1.0
                        - np.clip(
                            selected.depth_confidences[atlas_positions],
                            0.0,
                            1.0,
                        )
                    )
                    atlas_keep, atlas_keys = ray_atlas.admit(
                        selected.responsibility_view_directions[atlas_positions],
                        selected.responsibility_depths[atlas_positions],
                        image_size=selected.metadata["birth_image_size"],
                        birth_budget=selected.metadata["birth_candidate_budget"],
                        pool_multiplier=selected.metadata["birth_pool_multiplier"],
                        focal_pixels=selected.metadata["birth_focal_pixels"],
                        max_log_depth_std=selected.metadata[
                            "depthcov_log_std_threshold"
                        ],
                        frame_id=selected.source_frame_id,
                        world_points=selected.responsibility_world_points[
                            atlas_positions
                        ],
                        camera_center=selected.metadata["birth_camera_center"],
                        evidence_scores=(
                            selected.residual_scores[atlas_positions]
                            * np.clip(
                                selected.depth_confidences[atlas_positions],
                                0.0,
                                1.0,
                            )
                        ),
                        projected_radii=atlas_projected_radii,
                        log_depth_stds=atlas_log_depth_stds,
                    )
                    retained_atlas_positions = atlas_positions[atlas_keep]
                    keep_mask = np.ones((len(selected),), dtype=np.bool_)
                    keep_mask[atlas_positions[~atlas_keep]] = False
                    kept_positions = kept_positions[keep_mask[kept_positions]]
                    ray_atlas_commit_keys = [
                        key for key, keep in zip(atlas_keys, atlas_keep) if keep
                    ]
            if self._frontview_far_field_enabled():
                self.frontview_far_field_stats["commit_rows"] += int(selected_count)
        else:
            # Re-check delayed batches against permanent MODP occupancy.
            occupied = self.hash_block.getOccupy(
                selected.world_points,
                selected.colors,
                selected.view_scale_size,
            )
            keep = ~occupied
            if sequential_hash_semantics:
                no_conflict = self.hash_block.get_sequential_no_conflict_index(
                    selected.world_points[keep],
                    selected.colors[keep],
                    selected.view_scale_size,
                )
            else:
                no_conflict = self.hash_block.get_no_conflict_index(
                    selected.world_points[keep], selected.view_scale_size
                )
            kept_positions = np.flatnonzero(keep)[no_conflict]
        committed = selected.select(kept_positions)
        if (
            self._frontview_scale_cover_enabled()
            and self.frontview_scale_cover.tracks_uids
        ):
            committed_uids = self.frontview_scale_cover.allocate_uids(
                len(committed)
            )
        committed_opacity = (
            selected_opacity
            if np.isscalar(selected_opacity)
            else selected_opacity[kept_positions]
        )

        group_id = None
        if len(committed) > 0:
            current_group_id = (
                self.current_gaussian_group[committed.level]
                if target_group_id is None
                else int(target_group_id)
            )
            if current_group_id not in self.valid_groups:
                raise ValueError("Commit target group is not active")
            if (
                target_group_id is None
                and (committed.create_new_group or force_new_group)
                and self.gaussian_groups[current_group_id].get_num > 0
            ):
                self.create_new_group(level=committed.level)
                current_group_id = self.current_gaussian_group[committed.level]
            if (
                not self._frontview_residual_cover_enabled()
                and not self._frontview_identity_lod_enabled()
                and not self._frontview_birth_enabled()
                and not self._frontview_far_field_enabled()
                and not self._frontview_scale_cover_enabled()
                and not self._frontview_sparse_scale_map_enabled()
            ):
                self.hash_block.setOccupy(
                    committed.world_points,
                    committed.colors,
                    committed.view_scale_size,
                )
            elif (
                self._frontview_far_field_enabled()
                or self._frontview_scale_cover_enabled()
                or self._frontview_sparse_scale_map_enabled()
            ):
                hash_rows = (
                    committed.source_kinds != "depthcov_far"
                    if self._frontview_far_field_enabled()
                    else np.ones((len(committed),), dtype=np.bool_)
                )
                if np.any(hash_rows):
                    if self._frontview_scale_cover_enabled():
                        self.frontview_scale_cover.register(
                            committed.responsibility_world_points[hash_rows],
                            committed.responsibility_cover_sizes[hash_rows],
                            committed.colors[hash_rows],
                            uids=(
                                committed_uids[hash_rows]
                                if committed_uids is not None
                                else None
                            ),
                            parent_uids=committed.responsibility_parent_uids[
                                hash_rows
                            ],
                            source_ranks=np.where(
                                committed.source_kinds[hash_rows] == "sparse",
                                2,
                                1,
                            ).astype(np.int8),
                            view_directions=(
                                committed.responsibility_view_directions[hash_rows]
                            ),
                        )
                    elif self._frontview_sparse_scale_map_enabled():
                        self.frontview_sparse_scale_map.register(
                            committed.world_points[hash_rows],
                            committed.view_scale_size,
                        )
                    else:
                        self.hash_block.setOccupy(
                            committed.world_points[hash_rows],
                            committed.colors[hash_rows],
                            committed.view_scale_size,
                        )
                        self.frontview_far_field_stats["hash_set_rows"] += int(
                            np.sum(hash_rows)
                        )
                if (
                    self._frontview_scale_cover_enabled()
                    and committed_uids is not None
                    and np.any(~hash_rows)
                ):
                    if self.frontview_scale_cover.projected_handoff_enabled:
                        self.frontview_scale_cover.stage_projected_handoff(
                            committed.responsibility_world_points[~hash_rows],
                            committed.responsibility_cover_sizes[~hash_rows],
                            committed.colors[~hash_rows],
                            uids=committed_uids[~hash_rows],
                            source_ranks=np.ones(
                                int(np.sum(~hash_rows)), dtype=np.int8
                            ),
                            view_directions=(
                                committed.responsibility_view_directions[~hash_rows]
                            ),
                        )
                    else:
                        self.frontview_scale_cover.register_uids(
                            committed_uids[~hash_rows]
                        )
                if self._frontview_scale_cover_enabled():
                    landmark_memory = getattr(self, "causal_landmark_memory", None)
                    if landmark_memory is not None:
                        landmark_memory.record_responsibility_registration(
                            committed.depths[hash_rows],
                            committed.responsibility_depths[hash_rows],
                        )
                    self.frontview_scale_cover.register_sparse_tracks(
                        committed.track_ids,
                        committed.sparse_depth_valid,
                    )
            elif self._frontview_birth_enabled() and bool(
                self.frontview_birth_config["near_hash_competition"]
            ):
                hash_rows = (committed.source_kinds == "sparse") | (
                    committed.depths
                    < float(self.frontview_birth_config["near_hash_depth_m"])
                )
                if np.any(hash_rows):
                    self.hash_block.setOccupy(
                        committed.world_points[hash_rows],
                        committed.colors[hash_rows],
                        committed.view_scale_size,
                    )
                    self.frontview_birth_stats["near_hash_set_rows"] += int(
                        np.sum(hash_rows)
                    )
            scale_expansion_limits = None
            frequency_config = getattr(self, "frequency_sampling_config", {})
            control = frequency_config.get(
                "conditional_scale_control", {}
            )
            if isinstance(control, dict) and bool(control.get("enabled", False)):
                scale_expansion_limits = conditional_scale_expansion_limits(
                    committed.frequency_scores, frequency_config
                )
            footprint_release_limits = (
                np.full((len(committed),), np.inf, dtype=np.float32)
                if scale_expansion_limits is None
                else np.asarray(scale_expansion_limits, dtype=np.float32).copy()
            )
            proposal_limits = np.asarray(
                committed.scale_expansion_limits, dtype=np.float32
            )
            if np.any(np.isfinite(proposal_limits)):
                scale_expansion_limits = (
                    proposal_limits
                    if scale_expansion_limits is None
                    else np.minimum(scale_expansion_limits, proposal_limits)
                )
            if self._frontview_birth_enabled():
                layered_limits = layered_scale_expansion_limits(
                    committed.depths,
                    committed.source_kinds,
                    self.frontview_birth_config,
                )
                scale_expansion_limits = (
                    layered_limits
                    if scale_expansion_limits is None
                    else np.minimum(scale_expansion_limits, layered_limits)
                )
                self.frontview_birth_stats["scale_capped_rows"] += int(
                    np.isfinite(layered_limits).sum()
                )
                footprint_release_limits = np.minimum(
                    footprint_release_limits, layered_limits
                )
            target_group = self.gaussian_groups[current_group_id]
            first_new_row = target_group.get_num
            metric_confidences = None
            uncertainty_confidences = None
            dual_responsibility_config = getattr(
                self,
                "causal_dual_responsibility_config",
                {"enabled": False},
            )
            if dual_responsibility_config["enabled"]:
                uncertainty_confidences = proposal_metric_confidences(
                    committed.source_kinds,
                    committed.depth_confidences,
                    committed.depth_fallback,
                    dual_responsibility_config,
                )
                metric_confidences = uncertainty_confidences.copy()
                if dual_responsibility_config[
                    "finite_depth_certificate_enabled"
                ]:
                    certificate = committed.metric_certificates
                    if dual_responsibility_config[
                        "finite_depth_certificate_scope"
                    ] == "depthcov":
                        certificate = np.where(
                            np.char.startswith(
                                committed.source_kinds, "depthcov"
                            ),
                            certificate,
                            1.0,
                        )
                    else:
                        certificate = np.where(
                            committed.source_kinds == "tracked_metric",
                            1.0,
                            certificate,
                        )
                    metric_confidences *= certificate.astype(
                        np.float32, copy=False
                    )
                dual_stats = self.causal_dual_responsibility_stats
                dual_stats["metric_rows"] += int(
                    np.sum(metric_confidences >= 1.0)
                )
                dual_stats["proxy_rows"] += int(
                    np.sum(metric_confidences <= 0.0)
                )
                dual_stats["partial_metric_rows"] += int(
                    np.sum(
                        (metric_confidences > 0.0)
                        & (metric_confidences < 1.0)
                    )
                )
                dual_stats["metric_confidence_sum"] += float(
                    np.sum(metric_confidences)
                )
            committed_scales = committed.log_scales
            committed_quaternions = None
            covariance_mode = getattr(
                self, "frontview_far_field_config", {}
            ).get(
                "projective_covariance_mode", "isotropic"
            )
            fallback_support_mode = getattr(
                self, "frontview_far_field_config", {}
            ).get("fallback_support_mode", "legacy")
            covariance_rows = committed.source_kinds == "depthcov_far"
            fallback_structure_rows = (
                covariance_rows
                & committed.depth_fallback
                & bool(fallback_support_mode.startswith("budget_structure"))
            )
            causal_metric_birth_config = getattr(
                self,
                "causal_metric_birth_config",
                {"support_mode": "point"},
            )
            tracked_structure_rows = (
                (committed.source_kinds == "tracked_metric")
                & (
                    causal_metric_birth_config["support_mode"]
                    == "budget_structure"
                )
            )
            if (
                (covariance_mode != "isotropic" and np.any(covariance_rows))
                or np.any(fallback_structure_rows)
                or np.any(tracked_structure_rows)
            ):
                base_scales = np.asarray(committed.log_scales, dtype=np.float32)
                committed_scales = (
                    np.repeat(base_scales, 3, axis=1)
                    if base_scales.shape[1] == 1
                    else base_scales.copy()
                )
                committed_scales[covariance_rows, 2] += np.log(
                    committed.radial_scale_factors[covariance_rows]
                )
                committed_quaternions = np.zeros(
                    (len(committed), 4), dtype=np.float32
                )
                committed_quaternions[:, 0] = 1.0
                if np.any(fallback_structure_rows):
                    committed_scales[fallback_structure_rows] += np.log(
                        committed.covariance_scale_factors[
                            fallback_structure_rows
                        ]
                    )
                    committed_quaternions[fallback_structure_rows] = (
                        committed.initial_quaternions[fallback_structure_rows]
                    )
                if np.any(tracked_structure_rows):
                    committed_scales[tracked_structure_rows] += np.log(
                        committed.covariance_scale_factors[
                            tracked_structure_rows
                        ]
                    )
                    committed_quaternions[tracked_structure_rows] = (
                        committed.initial_quaternions[tracked_structure_rows]
                    )
                if covariance_mode != "isotropic" and np.any(covariance_rows):
                    committed_quaternions[covariance_rows] = ray_aligned_quaternions(
                        committed.view_directions[covariance_rows]
                    )
                    factors = committed.radial_scale_factors[covariance_rows]
                    stats = self.frontview_far_field_stats
                    stats["projective_covariance_rows"] += int(len(factors))
                    stats["projective_radial_factor_sum"] += float(np.sum(factors))
                    factor_min = float(np.min(factors))
                    factor_max = float(np.max(factors))
                    stats["projective_radial_factor_min"] = (
                        factor_min
                        if stats["projective_radial_factor_min"] is None
                        else min(stats["projective_radial_factor_min"], factor_min)
                    )
                    stats["projective_radial_factor_max"] = (
                        factor_max
                        if stats["projective_radial_factor_max"] is None
                        else max(stats["projective_radial_factor_max"], factor_max)
                    )
            target_group.extend_gaussians_from_color_points(
                committed.world_points,
                committed.colors,
                committed_scales,
                initial_opacity=committed_opacity,
                max_scale_expansion=scale_expansion_limits,
                footprint_trust_mask=(
                    np.isfinite(committed.scale_expansion_limits)
                    if getattr(self, "frontview_far_field_config", {}).get(
                        "footprint_trust_dynamic_update", False
                    )
                    else None
                ),
                footprint_target_scales=committed.footprint_target_scales,
                footprint_release_scale_expansions=footprint_release_limits,
                directional_observability_mask=(
                    (
                        committed.source_kinds == "depthcov_far"
                        if self.frontview_observability_config.get(
                            "responsibility_scope", "all_depthcov"
                        )
                        == "projective_only"
                        else np.char.startswith(
                            committed.source_kinds, "depthcov"
                        )
                    )
                    if "birth_camera_center" in committed.metadata
                    else None
                ),
                directional_log_depth_stds=(
                    float(
                        committed.metadata.get(
                            "depthcov_log_std_threshold",
                            getattr(
                                getattr(self, "depth_cov_estimator", None),
                                "std_valid_threshold",
                                0.0,
                            ),
                        )
                    )
                    * (1.0 - np.clip(committed.depth_confidences, 0.0, 1.0))
                ).astype(np.float32, copy=False),
                directional_initial_max_parallax_sin2=(
                    committed.initial_max_parallax_sin2
                ),
                reference_camera_center=committed.metadata.get(
                    "birth_camera_center"
                ),
                track_ids=committed.track_ids,
                gaussian_uids=committed_uids,
                birth_frame_ids=np.full(
                    (len(committed),),
                    int(committed.source_frame_id),
                    dtype=np.int64,
                ),
                metric_confidences=metric_confidences,
                uncertainty_confidences=uncertainty_confidences,
                initial_quaternions=committed_quaternions,
            )
            if ray_atlas_commit_keys:
                self.frontview_ray_atlas.register(
                    ray_atlas_commit_keys,
                    committed.source_frame_id,
                )
            if self._frontview_track_fusion_enabled() and not getattr(
                self, "frontview_track_lookup_dirty", True
            ):
                for offset, track_id in enumerate(committed.track_ids.tolist()):
                    if int(track_id) >= 0:
                        self.frontview_track_lookup.setdefault(
                            int(track_id), []
                        ).append((int(current_group_id), first_new_row + offset))
            if self._frontview_identity_lod_enabled():
                self.frontview_identity_lod.mark_committed(
                    committed_uids, committed
                )
            if self._frontview_residual_cover_enabled():
                self.frontview_residual_cover.mark_committed(committed)
            if self._frontview_birth_enabled():
                self.frontview_track_ledger.mark_committed(
                    committed.track_ids, committed.depths
                )
            group_id = int(current_group_id)
            if certificate_ids_by_row is not None:
                committed_certificate_ids = {
                    certificate_ids_by_row[int(position)] for position in kept_positions
                }
                self.worldtest_group_certificates.setdefault(group_id, set()).update(
                    committed_certificate_ids
                )

        Log(
            "Committing new gaussians (selected/committed): {} / {}".format(
                selected_count, len(committed)
            ),
            tag="GaussianModel",
        )
        return CommitResult(
            source_frame_id=(
                selected.source_frame_id
                if result_source_frame_id is None
                else int(result_source_frame_id)
            ),
            proposed=(
                selected_count if proposed_count is None else int(proposed_count)
            ),
            selected=selected_count,
            committed=len(committed),
            group_id=group_id,
            committed_indices=kept_positions,
        )

    @torch.no_grad()
    def add_new_gaussians(
        self,
        cam,
        create_new_group=False,
        render_pkg=None,
        level=0,
        reference_cameras=None,
        causal_reference_cameras=None,
        coverage_recovery=False,
        coverage_recovery_translation_m=None,
        coverage_recovery_budget=None,
    ):
        """Baseline-compatible immediate proposal and commit path."""
        proposal_kwargs = {
            "create_new_group": create_new_group,
            "render_pkg": render_pkg,
            "level": level,
        }
        if reference_cameras is not None:
            proposal_kwargs["reference_cameras"] = reference_cameras
        if causal_reference_cameras is not None:
            proposal_kwargs["causal_reference_cameras"] = causal_reference_cameras
        if coverage_recovery:
            proposal_kwargs["coverage_recovery"] = True
            proposal_kwargs["coverage_recovery_translation_m"] = (
                coverage_recovery_translation_m
            )
            proposal_kwargs["coverage_recovery_budget"] = coverage_recovery_budget
        proposals = self.propose_new_gaussians(cam, **proposal_kwargs)
        initial_opacity = 0.5
        if self._frontview_birth_enabled() and bool(
            self.frontview_birth_config["responsibility_opacity"]
        ):
            initial_opacity = responsibility_initial_opacities(
                proposals.source_kinds,
                proposals.residual_scores,
                proposals.coverage_scores,
                proposals.depth_confidences,
                initial_opacity,
                self.frontview_birth_config,
            )
            depthcov = proposals.source_kinds == "depthcov"
            self.frontview_birth_stats["responsibility_opacity_rows"] += int(
                np.sum(depthcov)
            )
            self.frontview_birth_stats["responsibility_opacity_sum"] += float(
                np.sum(initial_opacity[depthcov])
            )
        return self.commit_proposals(proposals, initial_opacity=initial_opacity)

    @torch.no_grad()
    def propose_new_gaussians_pts_only(self, cam):
        """Create sparse host proposals without occupying the permanent map."""
        color_pts_depth = cam.get_color_pts_depth()

        if len(color_pts_depth) == 0:
            empty = np.empty((0,), dtype=np.float32)
            return GaussianProposalBatch(
                source_frame_id=int(cam.cam_idx),
                level=0,
                uv=np.empty((0, 2), dtype=np.float32),
                patch_bboxes=np.empty((0, 4), dtype=np.float32),
                depths=empty,
                inverse_depths=empty,
                world_points=np.empty((0, 3), dtype=np.float32),
                log_scales=np.empty((0, 1), dtype=np.float32),
                colors=np.empty((0, 3), dtype=np.float32),
                residual_scores=empty,
                coverage_scores=empty,
                sparse_depth_valid=np.empty((0,), dtype=np.bool_),
                view_scale_size=float(cam.get_view_size(0) * self.camera_scale_rescalar),
                metadata={
                    "densification_mode": "sparse_points_only",
                    "scale_cover_epoch": (
                        int(self.frontview_scale_cover.admission_epoch)
                        if self._frontview_scale_cover_enabled()
                        else None
                    ),
                },
            )

        assert color_pts_depth.shape[1] == 7

        pts_3d = color_pts_depth[:, :3]
        point_ids = np.asarray(cam.get_point_ids(), dtype=np.int64)
        if point_ids.shape != (len(pts_3d),):
            raise ValueError("Camera sparse point IDs are not aligned with sparse rows")
        pts_color = color_pts_depth[:, 3:6]
        depth = color_pts_depth[:, 6]

        ones = np.ones((len(pts_3d), 1), dtype=np.float32)
        camera_points = np.matmul(
            np.concatenate((pts_3d, ones), axis=1),
            cam.get_raw_pose().detach().cpu().numpy().T,
        )[:, :3]
        screen = np.matmul(camera_points, cam.get_int_mat(0).detach().cpu().numpy().T)
        uv = screen[:, :2] / np.maximum(screen[:, 2:3], 1.0e-8)

        proposal_frequency_scores = np.zeros((len(uv),), dtype=np.float32)
        if bool(self.frequency_sampling_config.get("enabled", False)):
            image = cam.get_gt_image(0)
            height, width = image.shape[:2]
            _, _, admission_evidence = frequency_evidence_map(
                image,
                image.new_full((height, width), self.err_threshold),
                None,
                {
                    **self.frequency_sampling_config,
                    "residual_threshold": self.err_threshold,
                },
            )
            uv_tensor = torch.as_tensor(uv, device=image.device)
            x = torch.floor(uv_tensor[:, 0]).long()
            y = torch.floor(uv_tensor[:, 1]).long()
            valid_uv = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            scores = torch.zeros((len(uv),), device=image.device)
            scores[valid_uv] = admission_evidence[y[valid_uv], x[valid_uv]]
            proposal_frequency_scores = scores.cpu().numpy().astype(np.float32)

        init_scale = (
            np.log(0.5 * depth / ((cam.get_fx(0) + cam.get_fy(0)) / 2.0)).reshape(-1, 1)
            + self.init_scale_offset
        )
        cur_view_scale_size = cam.get_view_size(0)

        cur_view_scale_size *= self.camera_scale_rescalar
        camera_pose = cam.get_pose().detach()
        birth_camera_center = (
            -camera_pose[:3, :3].T @ camera_pose[:3, 3]
        ).cpu().numpy()
        proposal_view_directions = (
            self.frontview_scale_cover.candidate_view_directions(
                pts_3d,
                birth_camera_center,
                depth,
                np.ones((len(pts_3d),), dtype=np.bool_),
                int(cam.cam_idx),
            )
        )

        before_filter_num = pts_3d.shape[0]
        responsibility_parent_uids = np.full(
            (before_filter_num,), -1, dtype=np.int64
        )
        responsibility_levels = np.zeros((before_filter_num,), dtype=np.int16)
        responsibility_sectors = np.full(
            (before_filter_num,), -1, dtype=np.int16
        )
        if self._frontview_residual_cover_enabled():
            image = cam.get_gt_image(0)
            keep_indices = self.frontview_residual_cover.filter_candidates(
                frame_id=int(cam.cam_idx),
                uv=uv,
                depths=depth,
                world_points=pts_3d,
                log_scales=init_scale,
                colors=pts_color,
                residual_scores=np.zeros((before_filter_num,), dtype=np.float32),
                depth_confidences=np.ones((before_filter_num,), dtype=np.float32),
                sparse_valid=np.ones((before_filter_num,), dtype=np.bool_),
                track_ids=point_ids,
                rendered=torch.zeros_like(image),
                target=image,
                focal_px=0.5 * (cam.get_fx(0) + cam.get_fy(0)),
                depthcov_budget=0,
            )
            occupied_mask = np.ones((before_filter_num,), dtype=np.bool_)
            occupied_mask[keep_indices] = False
            no_conflict_index = np.ones((len(keep_indices),), dtype=np.bool_)
        elif self._frontview_identity_lod_enabled():
            keep_indices, parents, levels, sectors = (
                self.frontview_identity_lod.filter_candidates(
                    frame_id=int(cam.cam_idx),
                    uv=uv,
                    depths=depth,
                    residual_scores=np.zeros((before_filter_num,), dtype=np.float32),
                    depth_confidences=np.ones((before_filter_num,), dtype=np.float32),
                    sparse_valid=np.ones((before_filter_num,), dtype=np.bool_),
                    track_ids=point_ids,
                    projection_info=None,
                    global_uids=self.get_gaussian_uids(),
                    depthcov_budget=0,
                )
            )
            occupied_mask = np.ones((before_filter_num,), dtype=np.bool_)
            occupied_mask[keep_indices] = False
            responsibility_parent_uids[keep_indices] = parents
            responsibility_levels[keep_indices] = levels
            responsibility_sectors[keep_indices] = sectors
            no_conflict_index = np.ones((len(keep_indices),), dtype=np.bool_)
        elif self._frontview_birth_enabled():
            keep_indices = self.frontview_track_ledger.new_indices(
                point_ids, at_commit=False
            )
            occupied_mask = np.ones((before_filter_num,), dtype=np.bool_)
            occupied_mask[keep_indices] = False
            no_conflict_index = np.ones((len(keep_indices),), dtype=np.bool_)
        elif self._frontview_scale_cover_enabled():
            scale_cover_source_ranks = (
                self.frontview_scale_cover.candidate_source_ranks(
                    np.ones((before_filter_num,), dtype=np.bool_),
                    depth,
                    int(cam.cam_idx),
                )
            )
            appearance_eligible = self.frontview_scale_cover.appearance_certificates(
                np.zeros((before_filter_num,), dtype=np.float32),
                np.ones((before_filter_num,), dtype=np.float32),
                np.ones((before_filter_num,), dtype=np.bool_),
                depth,
                int(cam.cam_idx),
            )
            occupied_mask, scale_parent_uids = (
                self.frontview_scale_cover.occupied_with_parents(
                    pts_3d,
                    cur_view_scale_size,
                    pts_color,
                    scale_cover_source_ranks,
                    appearance_eligible,
                    view_directions=proposal_view_directions,
                    residual_scores=np.zeros((before_filter_num,), dtype=np.float32),
                    depth_confidences=np.ones((before_filter_num,), dtype=np.float32),
                    sparse_valid=np.ones((before_filter_num,), dtype=np.bool_),
                    depths=depth,
                    frame_id=int(cam.cam_idx),
                )
            )
            scale_parent_uids = self.frontview_scale_cover.shuffle_parents(
                scale_parent_uids,
                depth,
                np.ones((before_filter_num,), dtype=np.bool_),
                int(cam.cam_idx),
            )
            responsibility_parent_uids[:] = scale_parent_uids
            occupied_mask = self.frontview_scale_cover.shuffle(
                occupied_mask,
                depth,
                np.ones((before_filter_num,), dtype=np.bool_),
                int(cam.cam_idx),
            )
        elif self._frontview_sparse_scale_map_enabled():
            occupied_mask = self.frontview_sparse_scale_map.occupied(
                pts_3d, cur_view_scale_size
            )
        else:
            occupied_mask = self.hash_block.getOccupy(
                pts_3d, pts_color, cur_view_scale_size
            )
        pts_3d = pts_3d[~occupied_mask]
        pts_color = pts_color[~occupied_mask]
        uv = uv[~occupied_mask]
        depth = depth[~occupied_mask]
        proposal_frequency_scores = proposal_frequency_scores[~occupied_mask]
        point_ids = point_ids[~occupied_mask]
        proposal_view_directions = proposal_view_directions[~occupied_mask]
        responsibility_parent_uids = responsibility_parent_uids[~occupied_mask]
        responsibility_levels = responsibility_levels[~occupied_mask]
        responsibility_sectors = responsibility_sectors[~occupied_mask]

        if (
            self._frontview_scale_cover_enabled()
            or self._frontview_sparse_scale_map_enabled()
        ):
            no_conflict_index = np.ones((len(pts_3d),), dtype=np.bool_)
        elif (
            not self._frontview_residual_cover_enabled()
            and not self._frontview_identity_lod_enabled()
            and not self._frontview_birth_enabled()
        ):
            no_conflict_index = self.hash_block.get_no_conflict_index(
                pts_3d, cur_view_scale_size
            )
        pts_3d = pts_3d[no_conflict_index]
        pts_color = pts_color[no_conflict_index]
        uv = uv[no_conflict_index]
        depth = depth[no_conflict_index]
        proposal_frequency_scores = proposal_frequency_scores[no_conflict_index]
        point_ids = point_ids[no_conflict_index]
        proposal_view_directions = proposal_view_directions[no_conflict_index]
        responsibility_parent_uids = responsibility_parent_uids[no_conflict_index]
        responsibility_levels = responsibility_levels[no_conflict_index]
        responsibility_sectors = responsibility_sectors[no_conflict_index]

        if isinstance(init_scale, np.ndarray):
            init_scale = init_scale[~occupied_mask]
            init_scale = init_scale[no_conflict_index]

        Log(
            "Proposing sparse gaussians (before/after host filter): {} / {}".format(
                before_filter_num, pts_3d.shape[0]
            ),
            tag="GaussianModel",
        )

        half_patch = 4.0
        patch_bboxes = np.stack(
            (
                uv[:, 0] - half_patch,
                uv[:, 1] - half_patch,
                uv[:, 0] + half_patch,
                uv[:, 1] + half_patch,
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        return GaussianProposalBatch(
            source_frame_id=int(cam.cam_idx),
            level=0,
            uv=uv.astype(np.float32, copy=False),
            patch_bboxes=patch_bboxes,
            depths=depth.astype(np.float32, copy=False),
            inverse_depths=1.0 / np.maximum(depth, 1.0e-8),
            world_points=pts_3d.astype(np.float32, copy=False),
            log_scales=init_scale.astype(np.float32, copy=False),
            colors=pts_color.astype(np.float32, copy=False),
            residual_scores=np.zeros((len(pts_3d),), dtype=np.float32),
            coverage_scores=np.ones((len(pts_3d),), dtype=np.float32),
            sparse_depth_valid=np.ones((len(pts_3d),), dtype=np.bool_),
            view_directions=proposal_view_directions,
            frequency_scores=proposal_frequency_scores,
            track_ids=point_ids,
            source_kinds=np.full((len(pts_3d),), "sparse", dtype="U32"),
            responsibility_parent_uids=responsibility_parent_uids,
            responsibility_levels=responsibility_levels,
            responsibility_sectors=responsibility_sectors,
            view_scale_size=float(cur_view_scale_size),
            metadata={
                "densification_mode": "sparse_points_only",
                "before_host_filter": int(before_filter_num),
                "scale_cover_epoch": (
                    int(self.frontview_scale_cover.admission_epoch)
                    if self._frontview_scale_cover_enabled()
                    else None
                ),
                "birth_camera_center": birth_camera_center,
            },
        )

    @torch.no_grad()
    def add_new_gaussians_pts_only(self, cam):
        """Baseline-compatible immediate sparse proposal and commit path."""
        return self.commit_proposals(self.propose_new_gaussians_pts_only(cam))

    # will merge group_idx_from to group_idx_to, and remove group_idx_from
    # by default it will merge to the first group
    # also change the pointer of current_gaussian_group if needed
    def merge_gaussian_group(self, group_idx_from, group_idx_to=0):
        raise NotImplementedError

    @torch.no_grad()
    def _retire_frontview_capacity(self, current_frame_id, processed_frames):
        config = self.frontview_residual_cover_config
        start_frame = int(config["retirement_start_frame"])
        if int(processed_frames) < start_frame:
            self.frontview_residual_cover.record_retirement(
                capacity=self.get_num_gaussians,
                eligible=0,
                retired=0,
            )
            return [], []
        capacity = int(
            round(
                int(config["retirement_capacity_base"])
                + float(config["retirement_capacity_per_frame"])
                * (int(processed_frames) - start_frame)
            )
        )
        excess = max(0, self.get_num_gaussians - capacity)
        candidates = []
        minimum_age = int(config["retirement_min_age_frames"])
        expansion_weight = float(config["retirement_expansion_weight"])
        for level in range(self.MAX_LEVEL):
            for group_id in self.active_gaussian_groups[level]:
                if group_id in self.progressive_group_ids:
                    continue
                group = self.gaussian_groups[group_id]
                tracks = group.non_trainable_params["track_ids"]
                births = group.non_trainable_params["birth_frame_ids"]
                eligible = (
                    (tracks < 0)
                    & (births >= 0)
                    & ((int(current_frame_id) - births) >= minimum_age)
                )
                rows = torch.nonzero(eligible, as_tuple=False).reshape(-1)
                if rows.numel() == 0:
                    continue
                opacity = group.get_opacity[rows].reshape(-1)
                gradient_utility = group.non_trainable_params[
                    "gradient_utility_ema"
                ][rows]
                expansion = group.get_scaling[rows].mean(dim=1) / torch.clamp(
                    group.get_init_scales[rows], min=1.0e-8
                )
                score_mode = config["retirement_score_mode"]
                if score_mode == "gradient":
                    base_scores = gradient_utility
                elif score_mode == "gradient_opacity":
                    base_scores = gradient_utility * opacity
                else:
                    base_scores = opacity
                scores = base_scores / torch.clamp(
                    expansion, min=1.0
                ).pow(expansion_weight)
                candidates.append(
                    (
                        int(group_id),
                        rows.detach().cpu().numpy(),
                        scores.detach().cpu().numpy(),
                    )
                )
        eligible_count = sum(len(rows) for _, rows, _ in candidates)
        retire_count = min(excess, eligible_count)
        removed_track_ids = []
        removed_gaussian_uids = []
        if retire_count > 0:
            group_ids = np.concatenate(
                [np.full(len(rows), group_id, dtype=np.int64) for group_id, rows, _ in candidates]
            )
            rows = np.concatenate([rows for _, rows, _ in candidates])
            scores = np.concatenate([scores for _, _, scores in candidates])
            if bool(config["shuffle_retirement"]):
                rng = np.random.default_rng(
                    int(config["shuffle_seed"]) + int(current_frame_id)
                )
                selected = rng.choice(
                    eligible_count, size=retire_count, replace=False
                )
            else:
                selected = np.argpartition(scores, retire_count - 1)[:retire_count]
            selected_group_ids = group_ids[selected]
            selected_rows = rows[selected]
            for group_id in np.unique(selected_group_ids):
                group = self.gaussian_groups[int(group_id)]
                keep = np.ones((group.get_num,), dtype=np.bool_)
                keep[selected_rows[selected_group_ids == group_id]] = False
                removed_track_ids.append(group.prune_with_mask(keep))
                removed_gaussian_uids.append(
                    getattr(
                        group,
                        "last_pruned_gaussian_uids",
                        np.empty((0,), dtype=np.int64),
                    )
                )
        self.frontview_residual_cover.record_retirement(
            capacity=capacity,
            eligible=eligible_count,
            retired=retire_count,
        )
        return removed_track_ids, removed_gaussian_uids

    @staticmethod
    def _pruning_tensor_summary(values, mask):
        values = values.reshape(-1)[mask]
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return {"count": 0, "mean": None, "q10": None, "q50": None, "q90": None}
        quantiles = torch.quantile(
            values.float(),
            torch.tensor((0.1, 0.5, 0.9), device=values.device),
        )
        return {
            "count": int(values.numel()),
            "mean": float(values.float().mean().item()),
            "q10": float(quantiles[0].item()),
            "q50": float(quantiles[1].item()),
            "q90": float(quantiles[2].item()),
        }

    @torch.no_grad()
    def _audit_opacity_pruning(self, camera, current_frame_id, processed_frames):
        if not getattr(self, "opacity_pruning_audit_enabled", False) or camera is None:
            return
        pose = camera.get_pose().detach().to(self.device, dtype=torch.float32)
        call = {
            "frame_id": int(current_frame_id),
            "processed_frames": int(processed_frames),
            "threshold": float(self.opacity_prune_threshold),
            "groups": [],
        }
        for level in range(self.MAX_LEVEL):
            for group_id in self.active_gaussian_groups[level]:
                if group_id in self.progressive_group_ids:
                    continue
                group = self.gaussian_groups[group_id]
                count = int(group.get_num)
                if count == 0:
                    continue
                params = group.non_trainable_params
                opacity = group.get_opacity.reshape(-1)
                removed = opacity <= float(self.opacity_prune_threshold)
                means = group.get_xyz
                camera_points = means @ pose[:3, :3].T + pose[:3, 3]
                depth = camera_points[:, 2]
                projection = camera_points @ camera.get_int_mat(0).detach().to(
                    means.device, dtype=means.dtype
                ).T
                u = projection[:, 0] / torch.clamp(depth, min=1.0e-8)
                v = projection[:, 1] / torch.clamp(depth, min=1.0e-8)
                visible = (
                    (depth > float(camera.near))
                    & (depth < float(camera.far))
                    & (u >= 0.0)
                    & (u < float(camera.get_width(0)))
                    & (v >= 0.0)
                    & (v < float(camera.get_height(0)))
                )
                birth = params["birth_frame_ids"]
                age = torch.clamp(
                    torch.full_like(birth, int(current_frame_id)) - birth,
                    min=0,
                ).float()
                scale = group.get_scaling.mean(dim=1)
                expansion = scale / torch.clamp(group.get_init_scales, min=1.0e-8)
                row = {
                    "group_id": int(group_id),
                    "level": int(level),
                    "before": count,
                    "removed": int(removed.sum().item()),
                    "removed_visible": int((removed & visible).sum().item()),
                    "removed_persistent_identity": int(
                        (removed & (params["track_ids"] >= 0)).sum().item()
                    ),
                    "removed_depthcov_owned": int(
                        (removed & params["directional_observability_mask"]).sum().item()
                    ),
                }
                for name, values in (
                    ("opacity", opacity),
                    ("camera_depth_m", depth),
                    ("age_frames", age),
                    ("world_scale_m", scale),
                    ("scale_expansion", expansion),
                    ("metric_confidence", params["metric_confidences"]),
                    ("parallax_sin2", params["max_parallax_sin2"]),
                    ("birth_log_depth_std", params["birth_log_depth_stds"]),
                    ("gradient_utility", params["gradient_utility_ema"]),
                    ("appearance_views", params["appearance_view_count"].float()),
                ):
                    row["removed_" + name] = self._pruning_tensor_summary(values, removed)
                    row["retained_" + name] = self._pruning_tensor_summary(values, ~removed)
                call["groups"].append(row)
        call["before"] = int(sum(row["before"] for row in call["groups"]))
        call["removed"] = int(sum(row["removed"] for row in call["groups"]))
        call["removed_visible"] = int(
            sum(row["removed_visible"] for row in call["groups"])
        )
        self.opacity_pruning_audit_calls.append(call)

    def opacity_pruning_audit_summary(self):
        return {
            "enabled": self.opacity_pruning_audit_enabled,
            "calls": list(self.opacity_pruning_audit_calls),
            "total_removed": int(
                sum(call["removed"] for call in self.opacity_pruning_audit_calls)
            ),
            "total_removed_visible": int(
                sum(call["removed_visible"] for call in self.opacity_pruning_audit_calls)
            ),
        }

    def prune_w_opacity(
        self, camera=None, current_frame_id=None, processed_frames=None
    ):
        if self.opacity_prune_threshold > 0:
            if current_frame_id is not None and processed_frames is not None:
                self._audit_opacity_pruning(
                    camera, int(current_frame_id), int(processed_frames)
                )
            before_num = self.get_num_gaussians
            removed_track_ids = []
            removed_gaussian_uids = []
            for j in range(self.MAX_LEVEL):
                for i in self.active_gaussian_groups[j]:
                    if i in self.progressive_group_ids:
                        continue
                    removed_track_ids.append(
                        self.gaussian_groups[i].prune_w_opacity(
                            self.opacity_prune_threshold
                        )
                    )
                    removed_gaussian_uids.append(
                        getattr(
                            self.gaussian_groups[i],
                            "last_pruned_gaussian_uids",
                            np.empty((0,), dtype=np.int64),
                        )
                    )
            if (
                self._frontview_residual_cover_enabled()
                and self.frontview_residual_cover_config["retirement_enabled"]
                and current_frame_id is not None
                and processed_frames is not None
            ):
                retirement_tracks, retirement_uids = (
                    self._retire_frontview_capacity(
                        int(current_frame_id), int(processed_frames)
                    )
                )
                removed_track_ids.extend(retirement_tracks)
                removed_gaussian_uids.extend(retirement_uids)
            if self._frontview_birth_enabled() and removed_track_ids:
                self.frontview_track_ledger.release(
                    np.concatenate(removed_track_ids, axis=0)
                )
            if self._frontview_residual_cover_enabled() and removed_track_ids:
                self.frontview_residual_cover.release(
                    np.concatenate(removed_track_ids, axis=0)
                )
            if (
                self._frontview_scale_cover_enabled()
                and removed_track_ids
            ):
                self.frontview_scale_cover.release_sparse_tracks(
                    np.concatenate(removed_track_ids, axis=0)
                )
            if self._frontview_identity_lod_enabled() and removed_gaussian_uids:
                self.frontview_identity_lod.release(
                    np.concatenate(removed_gaussian_uids, axis=0)
                )
            if (
                self._frontview_scale_cover_enabled()
                and self.frontview_scale_cover.tracks_uids
                and removed_gaussian_uids
            ):
                self.frontview_scale_cover.release(
                    np.concatenate(removed_gaussian_uids, axis=0)
                )
            after_num = self.get_num_gaussians
            if self._frontview_track_fusion_enabled() and after_num != before_num:
                self.frontview_track_lookup_dirty = True
            Log(
                "Pruning done: before/after: {}/{}".format(before_num, after_num),
                tag="GaussianModel",
            )

    @torch.no_grad()
    def save_as_ply(self, path):
        xyz = (
            torch.cat(
                [self.gaussian_groups[i].splats["means"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        normals = np.zeros_like(xyz)

        f_dc = (
            torch.cat(
                [self.gaussian_groups[i].splats["sh0"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            torch.cat(
                [self.gaussian_groups[i].splats["shN"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        opacities = (
            torch.cat(
                [
                    self.gaussian_groups[i].splats["opacities"]
                    for i in self.valid_groups
                ],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )
        uncertainty_confidences = (
            torch.cat(
                [
                    self.gaussian_groups[i].non_trainable_params[
                        "uncertainty_confidences"
                    ]
                    for i in self.valid_groups
                ],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )
        scales = (
            torch.cat(
                [self.gaussian_groups[i].splats["scales"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        rotations = (
            torch.cat(
                [self.gaussian_groups[i].splats["quats"] for i in self.valid_groups],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        metric_confidences = (
            torch.cat(
                [
                    self.gaussian_groups[i].non_trainable_params[
                        "metric_confidences"
                    ]
                    for i in self.valid_groups
                ],
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )

        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        for i in range(f_rest.shape[1]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(scales.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(rotations.shape[1]):
            l.append("rot_{}".format(i))
        l.append("metric_confidence")
        l.append("uncertainty_confidence")

        dtype_full = [(attribute, "f4") for attribute in l]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                opacities,
                scales,
                rotations,
                metric_confidences,
                uncertainty_confidences,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    @torch.no_grad()
    def save_as_tgbr_sparse_ply(self, path):
        """Save SH2 densely and only allocated SH3 bands in a packed bank."""

        config = self.streaming_appearance_lod_config
        if not config["enabled"]:
            raise RuntimeError("TGBR sparse export requires StreamingAppearanceLOD")
        base_degree = int(config["birth_degree"])
        target_degree = int(config["target_degree"])
        if self.max_sh_degree != target_degree:
            raise RuntimeError(
                "TGBR sparse export requires Model.sh_degree == target_degree"
            )

        params = {
            name: torch.cat(
                [
                    self.gaussian_groups[index].splats[name]
                    for index in self.valid_groups
                ],
                dim=0,
            )
            for name in ("means", "sh0", "shN", "opacities", "scales", "quats")
        }
        degrees = torch.cat(
            [
                self.gaussian_groups[index].non_trainable_params[
                    "appearance_sh_degree"
                ]
                for index in self.valid_groups
            ],
            dim=0,
        )
        metric_confidences = torch.cat(
            [
                self.gaussian_groups[index].non_trainable_params[
                    "metric_confidences"
                ]
                for index in self.valid_groups
            ],
            dim=0,
        )
        uncertainty_confidences = torch.cat(
            [
                self.gaussian_groups[index].non_trainable_params[
                    "uncertainty_confidences"
                ]
                for index in self.valid_groups
            ],
            dim=0,
        )
        active_mask = degrees >= target_degree
        stats = write_tgbr_sparse_ply(
            path,
            means=params["means"].detach().cpu().numpy(),
            sh0=params["sh0"].detach().cpu().numpy(),
            shN=params["shN"].detach().cpu().numpy(),
            opacities=params["opacities"].detach().cpu().numpy(),
            scales=params["scales"].detach().cpu().numpy(),
            quats=params["quats"].detach().cpu().numpy(),
            active_mask=active_mask.detach().cpu().numpy(),
            base_degree=base_degree,
            target_degree=target_degree,
            metric_confidences=metric_confidences.detach().cpu().numpy(),
            uncertainty_confidences=(
                uncertainty_confidences.detach().cpu().numpy()
            ),
        )
        self.tgbr_sparse_model_stats = stats
        return stats

    @torch.no_grad()
    def save_raw_splats_as_ply(self, params, path):
        """Write model-domain parameter tensors using the repository PLY schema."""
        xyz = params["means"].detach().float().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            params["sh0"].detach().float().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        )
        f_rest = (
            params["shN"].detach().float().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        )
        opacities = params["opacities"].detach().float().cpu().numpy().reshape(-1, 1)
        scales = params["scales"].detach().float().cpu().numpy()
        rotations = params["quats"].detach().float().cpu().numpy()
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += ["f_dc_{}".format(i) for i in range(f_dc.shape[1])]
        names += ["f_rest_{}".format(i) for i in range(f_rest.shape[1])]
        names += ["opacity"]
        names += ["scale_{}".format(i) for i in range(scales.shape[1])]
        names += ["rot_{}".format(i) for i in range(rotations.shape[1])]
        elements = np.empty(xyz.shape[0], dtype=[(name, "f4") for name in names])
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scales, rotations), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    @torch.no_grad()
    def load_from_ply(self, path):
        assert len(self.gaussian_groups) <= 4
        assert self.gaussian_groups[0].get_num <= 1

        self.gaussian_groups[0].load_from_ply(path, self.max_sh_degree)
        for group in self.gaussian_groups[1:]:
            group.replace_gaussians(
                {name: value.detach()[:0] for name, value in group.splats.items()}
            )
        Log(
            "Loaded gaussians from ply",
            self.gaussian_groups[0].get_num,
            tag="GaussianModel",
        )
