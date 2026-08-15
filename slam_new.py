# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from os.path import join as pjoin

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from utils_new.eval_utils import eval_gaussians
from utils_new.background_model import fit_and_save_sky_background
from utils_new.logging_utils import Log
from utils_new.scene_mapper import SceneMapper
from utils_new.tool_utils import load_config, mkdir_p


class SLAM:
    def __init__(self, configs):
        self.scene_mapper = SceneMapper(configs)
        self.configs = configs

    def run(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        Log("Start reconstruction", tag="SLAM")
        scene_mapper_meta, optimization_infos = self.scene_mapper.run()
        background_config = self.configs.get("BackgroundModel", {})
        if bool(background_config.get("enabled", False)):
            _, background_path = fit_and_save_sky_background(
                self.configs,
                self.configs["Results"]["save_dir"],
                options=background_config,
            )
            Log("Saved far-field background: " + str(background_path), tag="SLAM")
        Log("End reconstruction", tag="SLAM")

        end.record()

        torch.cuda.synchronize()
        online_recon_time = start.elapsed_time(end) / 1000.0  # in seconds
        Log("Total reconstruction time: ", online_recon_time, "s", tag="SLAM")
        Log("Total Frames: ", scene_mapper_meta["num_processed_frames"], tag="SLAM")
        Log("Total Gaussians: ", scene_mapper_meta["num_gaussians"], tag="SLAM")
        Log("Total keyframes: ", scene_mapper_meta["num_keyframes"], tag="SLAM")

        info = {}
        info["online_recon_time"] = online_recon_time
        info["num_processed_frames"] = scene_mapper_meta["num_processed_frames"]
        info["num_gaussians"] = scene_mapper_meta["num_gaussians"]
        info["num_keyframes"] = scene_mapper_meta["num_keyframes"]
        info["kf_ids"] = scene_mapper_meta["kf_ids"]
        for timing_key in (
            "online_mapping_seconds",
            "online_stage_export_seconds",
            "post_refinement_seconds",
        ):
            if timing_key in scene_mapper_meta:
                info[timing_key] = scene_mapper_meta[timing_key]
        if "progressive_runtime" in scene_mapper_meta:
            info["progressive_runtime"] = scene_mapper_meta["progressive_runtime"]
        if "aerocommit_summary" in scene_mapper_meta:
            info["aerocommit_summary"] = scene_mapper_meta["aerocommit_summary"]
            info["aerocommit_runtime"] = scene_mapper_meta.get(
                "aerocommit_runtime", {}
            )
            info["num_gaussians_full"] = scene_mapper_meta["num_gaussians_full"]
        if "worldtest_summary" in scene_mapper_meta:
            info["worldtest_summary"] = scene_mapper_meta["worldtest_summary"]
        if "frontview_observability" in scene_mapper_meta:
            info["frontview_observability"] = scene_mapper_meta[
                "frontview_observability"
            ]
        if "frontview_coverage_recovery" in scene_mapper_meta:
            info["frontview_coverage_recovery"] = scene_mapper_meta[
                "frontview_coverage_recovery"
            ]
        if "causal_metric_birth" in scene_mapper_meta:
            info["causal_metric_birth"] = scene_mapper_meta[
                "causal_metric_birth"
            ]
        if "causal_persistent_landmark_memory" in scene_mapper_meta:
            info["causal_persistent_landmark_memory"] = scene_mapper_meta[
                "causal_persistent_landmark_memory"
            ]
        if "causal_dual_responsibility" in scene_mapper_meta:
            info["causal_dual_responsibility"] = scene_mapper_meta[
                "causal_dual_responsibility"
            ]
        if "causal_depth_audit" in scene_mapper_meta:
            info["causal_depth_audit"] = scene_mapper_meta["causal_depth_audit"]
        if "frontview_directional_layer" in scene_mapper_meta:
            info["frontview_directional_layer"] = scene_mapper_meta[
                "frontview_directional_layer"
            ]
        if "frontview_sampling" in scene_mapper_meta:
            info["frontview_sampling"] = scene_mapper_meta["frontview_sampling"]
        if "frontview_depth_transport" in scene_mapper_meta:
            info["frontview_depth_transport"] = scene_mapper_meta[
                "frontview_depth_transport"
            ]
        if "frontview_birth" in scene_mapper_meta:
            info["frontview_birth"] = scene_mapper_meta["frontview_birth"]
        if "frontview_far_field" in scene_mapper_meta:
            info["frontview_far_field"] = scene_mapper_meta["frontview_far_field"]
        if "frontview_identity_lod" in scene_mapper_meta:
            info["frontview_identity_lod"] = scene_mapper_meta[
                "frontview_identity_lod"
            ]
        if "frontview_residual_cover" in scene_mapper_meta:
            info["frontview_residual_cover"] = scene_mapper_meta[
                "frontview_residual_cover"
            ]
        if "frontview_scale_cover" in scene_mapper_meta:
            info["frontview_scale_cover"] = scene_mapper_meta[
                "frontview_scale_cover"
            ]
        if "streaming_appearance_lod" in scene_mapper_meta:
            info["streaming_appearance_lod"] = scene_mapper_meta[
                "streaming_appearance_lod"
            ]
        if "tgbr_frequency_schedule" in scene_mapper_meta:
            info["tgbr_frequency_schedule"] = scene_mapper_meta[
                "tgbr_frequency_schedule"
            ]
        if "tgbr_directional_step_budget" in scene_mapper_meta:
            info["tgbr_directional_step_budget"] = scene_mapper_meta[
                "tgbr_directional_step_budget"
            ]
        if "tgbr_directional_view_budget" in scene_mapper_meta:
            info["tgbr_directional_view_budget"] = scene_mapper_meta[
                "tgbr_directional_view_budget"
            ]
        if "tgbr_optimization_budget" in scene_mapper_meta:
            info["tgbr_optimization_budget"] = scene_mapper_meta[
                "tgbr_optimization_budget"
            ]
        if "tgbr_exact_replay" in scene_mapper_meta:
            info["tgbr_exact_replay"] = scene_mapper_meta[
                "tgbr_exact_replay"
            ]
        if "tgbr_spectral_replay" in scene_mapper_meta:
            info["tgbr_spectral_replay"] = scene_mapper_meta[
                "tgbr_spectral_replay"
            ]
        if "tgbr_spectral_residency" in scene_mapper_meta:
            info["tgbr_spectral_residency"] = scene_mapper_meta[
                "tgbr_spectral_residency"
            ]
        if "tgbr_replay_residency" in scene_mapper_meta:
            info["tgbr_replay_residency"] = scene_mapper_meta[
                "tgbr_replay_residency"
            ]
        if "cuda_memory_profile" in scene_mapper_meta:
            info["cuda_memory_profile"] = scene_mapper_meta[
                "cuda_memory_profile"
            ]
        if "frontview_track_fusion" in scene_mapper_meta:
            info["frontview_track_fusion"] = scene_mapper_meta[
                "frontview_track_fusion"
            ]
        if "frontview_sparse_scale_map" in scene_mapper_meta:
            info["frontview_sparse_scale_map"] = scene_mapper_meta[
                "frontview_sparse_scale_map"
            ]

        torch.cuda.empty_cache()

        if self.configs["Results"].get("skip_eval", False):
            info["eval_time"] = 0.0
            info["eval_res"] = {}
        else:
            # Evaluate the reconstructed gaussians
            start.record()
            eval_res = eval_gaussians(
                self.scene_mapper.gaussians, optimization_infos, self.configs
            )
            end.record()
            torch.cuda.synchronize()
            eval_time = start.elapsed_time(end) / 1000.0  # in seconds

            info["eval_time"] = eval_time
            info["eval_res"] = eval_res

        if "frontview_directional_layer" in info:
            info["frontview_directional_layer"] = (
                self.scene_mapper.gaussians.frontview_directional_layer_summary()
            )

        json.dump(
            info,
            open(
                pjoin(self.configs["Results"]["save_dir"], "results.json"),
                "w",
                encoding="utf-8",
            ),
            indent=4,
        )


def set_default_values(configs):
    if "scene_exposure_gain" not in configs["Mapper"]:
        configs["Mapper"]["scene_exposure_gain"] = 20.0


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducible Python/NumPy/Torch seed.",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        help="Optional per-process PyTorch CPU thread limit.",
    )

    args = parser.parse_args(sys.argv[1:])

    if args.cpu_threads is not None:
        if args.cpu_threads <= 0:
            parser.error("--cpu_threads must be positive")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    config["Results"]["random_seed"] = args.seed
    save_dir = None

    mkdir_p(config["Results"]["save_dir"])
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dataset_name = (
        config["Dataset"]["name"]
        if config["Mapper"]["use_dataset"]
        else config["Streamer"]["name"]
    )
    save_dir = os.path.join(
        config["Results"]["save_dir"],
        dataset_name,
        current_datetime + "_" + args.exp_name,
    )
    config["Results"]["save_dir"] = save_dir
    mkdir_p(save_dir)
    with open(os.path.join(save_dir, "config.yaml"), "w") as file:
        documents = yaml.dump(config, file)

    Log("saving results in " + save_dir, tag="SLAM")

    set_default_values(config)

    slam = SLAM(config)

    slam.run()

    # All done
    Log("Done.", tag="SLAM")
