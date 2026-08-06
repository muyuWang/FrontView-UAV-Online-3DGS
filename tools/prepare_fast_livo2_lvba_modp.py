#!/usr/bin/env python3
"""Prepare one FAST-LIVO2 bag as Global-LVBA and MODP input.

The script intentionally reuses two existing Conda environments:

* ``ros_noetic_fastlivo`` only runs ROS and FAST-LIVO2.
* the caller's environment (normally ``worldvln``) performs all conversion.

It never creates or modifies a Conda environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


WORKSPACE = Path("/home/wmy/workspace_vla")
ONLINE_REPO = WORKSPACE / "Online-3DGS-Monocular"
MODP_REPO = WORKSPACE / "modp_raw"
FAST_LIVO_REPO = WORKSPACE / "FAST-LIVO2"
FAST_LIVO_WS = WORKSPACE / "ros_noetic_fastlivo"
GLOBAL_LVBA_REPO = WORKSPACE / "Global-LVBA"
FAST_LIVO_DATA = Path("/data_0/wmy/workspace_vla/uavdata/fast-livo2")
LVBA_ROOT = ONLINE_REPO / "data/LVBA"
MODP_DATA_ROOT = ONLINE_REPO / "data/Online3DGS_LVBA"
MODP_LOG_ROOT = MODP_REPO / "Logs_lvba_baseline"
ROS_ENV = "ros_noetic_fastlivo"

ORIGINAL_WIDTH = 1280
ORIGINAL_HEIGHT = 1024
SCALE = 0.5
WIDTH = int(ORIGINAL_WIDTH * SCALE)
HEIGHT = int(ORIGINAL_HEIGHT * SCALE)
FX = 1293.56944 * SCALE
FY = 1293.3155 * SCALE
CX = 626.91359 * SCALE
CY = 522.799224 * SCALE
DISTORTION = np.array([-0.076160, 0.123001, -0.00113, 0.000251, 0.0])
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

# FAST-LIVO2 Avia calibration. R_IL/t_IL map LiDAR points into the IMU body.
T_IL = np.array([0.04165, 0.02326, -0.0284], dtype=np.float64)
R_IL = np.eye(3, dtype=np.float64)
R_CL = np.array(
    [
        [0.00610193, -0.999863, -0.0154172],
        [-0.00615449, 0.0153796, -0.999863],
        [0.999962, 0.00619598, -0.0060598],
    ],
    dtype=np.float64,
)
T_CL = np.array([0.0194384, 0.104689, -0.0251952], dtype=np.float64)
R_LI = R_IL.T
T_LI = -R_LI @ T_IL
R_CI = R_CL @ R_LI
T_CI = R_CL @ T_LI + T_CL


def run(command: list[str], *, cwd: Path | None = None, check: bool = True):
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def ros_command(command: str) -> list[str]:
    setup = FAST_LIVO_WS / "devel/setup.bash"
    return [
        "conda",
        "run",
        "-n",
        ROS_ENV,
        "bash",
        "-lc",
        f"source {setup} && exec {command}",
    ]


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_for_ros_master(timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ros_command("rosparam list"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("ROS master did not become available")


def count_outputs(image_dir: Path, pcd_dir: Path) -> tuple[int, int, int, int]:
    image_count = len(list(image_dir.glob("*.png")))
    pcd_count = len(list(pcd_dir.glob("*.pcd")))
    image_pose = image_dir / "image_poses.txt"
    lidar_pose = pcd_dir / "lidar_poses.txt"
    image_pose_count = sum(1 for line in image_pose.open() if line.strip()) if image_pose.exists() else 0
    lidar_pose_count = sum(1 for line in lidar_pose.open() if line.strip()) if lidar_pose.exists() else 0
    return image_count, pcd_count, image_pose_count, lidar_pose_count


def make_fast_livo_launch(scene: str, path: Path, tracking_mode: str) -> None:
    avia = FAST_LIVO_REPO / "config/avia.yaml"
    camera = FAST_LIVO_REPO / "config/camera_pinhole.yaml"
    content = f"""<launch>
  <rosparam command=\"load\" file=\"{escape(str(avia))}\" />
  <param name=\"common/img_en\" type=\"int\" value=\"{'0' if tracking_mode == 'lio' else '1'}\" />
  <param name=\"pcd_save/pcd_save_en\" value=\"true\" />
  <param name=\"pcd_save/type\" value=\"1\" />
  <param name=\"pcd_save/interval\" value=\"1\" />
  <param name=\"pcd_save/colmap_output_en\" value=\"false\" />
  <param name=\"image_save/img_save_en\" value=\"{'false' if tracking_mode == 'lio' else 'true'}\" />
  <param name=\"image_save/interval\" value=\"1\" />
  <param name=\"evo/seq_name\" value=\"{escape(scene)}\" />
  <node pkg=\"fast_livo\" type=\"fastlivo_mapping\" name=\"laserMapping\" output=\"screen\">
    <rosparam file=\"{escape(str(camera))}\" />
  </node>
</launch>
"""
    path.write_text(content, encoding="utf-8")


def clear_fast_livo_log(image_dir: Path, pcd_dir: Path) -> None:
    generated = [*image_dir.glob("*.png"), *pcd_dir.glob("*.pcd")]
    generated.extend([image_dir / "image_poses.txt", pcd_dir / "lidar_poses.txt"])
    existing = [path for path in generated if path.exists()]
    if existing:
        raise RuntimeError(
            "FAST-LIVO2 Log contains generated output. Move/remove it before another run: "
            + ", ".join(str(path) for path in existing[:5])
        )


def run_fast_livo2(scene: str, bag: Path, lvba_scene: Path, tracking_mode: str) -> dict:
    executable = FAST_LIVO_WS / "devel/lib/fast_livo/fastlivo_mapping"
    if not executable.exists():
        raise FileNotFoundError(f"Existing FAST-LIVO2 executable not found: {executable}")
    if not bag.exists():
        raise FileNotFoundError(f"Bag not found: {bag}")

    image_dir = FAST_LIVO_REPO / "Log/image"
    pcd_dir = FAST_LIVO_REPO / "Log/pcd"
    image_dir.mkdir(parents=True, exist_ok=True)
    pcd_dir.mkdir(parents=True, exist_ok=True)
    clear_fast_livo_log(image_dir, pcd_dir)

    log_dir = lvba_scene / "processing_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    launch_log = (log_dir / "fast_livo2.log").open("w", encoding="utf-8")
    core_log = (log_dir / "roscore.log").open("w", encoding="utf-8")
    bag_log = (log_dir / "rosbag_play.log").open("w", encoding="utf-8")
    roscore = launch = bag_process = None
    started = time.time()

    with tempfile.TemporaryDirectory(prefix=f"fast_livo2_{scene}_") as temp:
        launch_path = Path(temp) / f"{scene}.launch"
        make_fast_livo_launch(scene, launch_path, tracking_mode)
        try:
            roscore = subprocess.Popen(
                ros_command("roscore"),
                stdout=core_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_ros_master()
            launch = subprocess.Popen(
                ros_command(f"roslaunch {launch_path}"),
                stdout=launch_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(6)
            if launch.poll() is not None:
                raise RuntimeError(f"FAST-LIVO2 exited early; inspect {launch_log.name}")

            bag_process = subprocess.Popen(
                ros_command(f"rosbag play --delay=2 {bag}"),
                stdout=bag_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while bag_process.poll() is None:
                if launch.poll() is not None:
                    stop_process(bag_process)
                    raise RuntimeError(
                        f"FAST-LIVO2 exited while the bag was playing; inspect {launch_log.name}"
                    )
                time.sleep(1)
            bag_return = bag_process.returncode
            if bag_return != 0:
                raise RuntimeError(f"rosbag play failed with code {bag_return}; inspect {bag_log.name}")

            # The subscriber queues can still contain work after rosbag exits. Wait until
            # the output counts have remained unchanged for 30 seconds.
            last = None
            stable_since = time.monotonic()
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                counts = count_outputs(image_dir, pcd_dir)
                print(f"FAST-LIVO2 output counts image/pcd/image_pose/lidar_pose={counts}", flush=True)
                changed = counts != last
                if changed:
                    last = counts
                    stable_since = time.monotonic()
                required_counts = counts[1:4:2] if tracking_mode == "lio" else counts
                if not changed and min(required_counts) > 0 and time.monotonic() - stable_since >= 30:
                    break
                if launch.poll() is not None:
                    raise RuntimeError(f"FAST-LIVO2 exited before outputs settled; inspect {launch_log.name}")
                time.sleep(10)
            else:
                raise TimeoutError("FAST-LIVO2 outputs did not settle within 30 minutes")
        finally:
            stop_process(bag_process)
            stop_process(launch)
            stop_process(roscore)
            launch_log.close()
            core_log.close()
            bag_log.close()

    counts = count_outputs(image_dir, pcd_dir)
    if counts[1] != counts[3]:
        raise RuntimeError(f"FAST-LIVO2 output/pose count mismatch: {counts}")
    if tracking_mode == "livo" and counts[0] != counts[2]:
        raise RuntimeError(f"FAST-LIVO2 image/pose count mismatch: {counts}")
    metadata = {
        "scene": scene,
        "bag": str(bag),
        "runtime_seconds": time.time() - started,
        "image_count": counts[0],
        "pcd_count": counts[1],
        "image_pose_count": counts[2],
        "lidar_pose_count": counts[3],
        "fast_livo_repo": str(FAST_LIVO_REPO),
        "fast_livo_executable": str(executable),
        "ros_environment": ROS_ENV,
        "tracking_mode": tracking_mode,
        "calibration": "FAST-LIVO2 config/avia.yaml + config/camera_pinhole.yaml",
        "pcd_frame": "IMU body frame (FAST-LIVO2 pcd_save/type=1)",
    }
    return metadata


def replace_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        contents = [item for item in path.iterdir() if item.name != ".gitkeep"]
        if contents and not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        for item in contents:
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        path.mkdir(parents=True)


def extract_bag_images_and_interpolate_poses(
    bag: Path,
    target_image: Path,
    lidar_pose_path: Path,
    image_time_offset: float = 0.1,
) -> dict:
    from rosbags.highlevel import AnyReader

    lidar_poses = load_tum(lidar_pose_path)
    lidar_ts = lidar_poses[:, 0]
    positions = lidar_poses[:, 1:4]
    rotations = Rotation.from_quat(lidar_poses[:, 4:8])
    slerp = Slerp(lidar_ts, rotations)
    pose_rows = []
    dropped = 0

    with AnyReader([bag]) as reader:
        connections = [
            connection for connection in reader.connections
            if connection.topic == "/left_camera/image"
        ]
        if len(connections) != 1:
            raise ValueError(f"Expected one /left_camera/image connection, got {len(connections)}")
        for connection, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            timestamp = (
                float(message.header.stamp.sec)
                + float(message.header.stamp.nanosec) * 1e-9
                + image_time_offset
            )
            if timestamp < lidar_ts[0] or timestamp > lidar_ts[-1]:
                dropped += 1
                continue

            height, width, step = int(message.height), int(message.width), int(message.step)
            rows = np.asarray(message.data, dtype=np.uint8).reshape(height, step)
            encoding = str(message.encoding).lower()
            if encoding not in {"rgb8", "bgr8"}:
                raise ValueError(f"Unsupported image encoding: {message.encoding}")
            image = rows[:, : width * 3].reshape(height, width, 3)
            if encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            filename = f"{timestamp:.6f}.png"
            if not cv2.imwrite(str(target_image / filename), image):
                raise RuntimeError(f"Could not write image: {target_image / filename}")

            position = np.array([
                np.interp(timestamp, lidar_ts, positions[:, axis]) for axis in range(3)
            ])
            quaternion = slerp([timestamp]).as_quat()[0]
            pose_rows.append([timestamp, *position.tolist(), *quaternion.tolist()])

    if not pose_rows:
        raise RuntimeError("No bag images overlap the FAST-LIVO2 LiDAR trajectory")
    np.savetxt(target_image / "image_poses.txt", np.asarray(pose_rows), fmt="%.6f")
    return {
        "extracted_image_count": len(pose_rows),
        "dropped_images_outside_lidar_pose_range": dropped,
        "image_time_offset_seconds": image_time_offset,
        "image_pose_method": "linear position interpolation plus quaternion SLERP of FAST-LIVO2 LIO poses",
    }


def stage_global_lvba(scene: str, lvba_scene: Path, metadata: dict, overwrite: bool) -> None:
    source_image = FAST_LIVO_REPO / "Log/image"
    source_pcd = FAST_LIVO_REPO / "Log/pcd"
    target_image = lvba_scene / "all_image"
    target_pcd = lvba_scene / "all_pcd_body"
    replace_directory(target_image, overwrite)
    replace_directory(target_pcd, overwrite)

    for source in sorted(source_pcd.glob("*.pcd")):
        shutil.move(str(source), target_pcd / source.name)
    shutil.move(str(source_pcd / "lidar_poses.txt"), target_pcd / "lidar_poses.txt")

    if metadata["tracking_mode"] == "livo":
        for source in sorted(source_image.glob("*.png")):
            shutil.move(str(source), target_image / source.name)
        shutil.move(str(source_image / "image_poses.txt"), target_image / "image_poses.txt")
    else:
        image_metadata = extract_bag_images_and_interpolate_poses(
            Path(metadata["bag"]), target_image, target_pcd / "lidar_poses.txt"
        )
        metadata.update(image_metadata)
        metadata["image_count"] = image_metadata["extracted_image_count"]
        metadata["image_pose_count"] = image_metadata["extracted_image_count"]

    (lvba_scene / "fast_livo2_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_global_lvba_config(scene, lvba_scene)

    dataset_dir = GLOBAL_LVBA_REPO / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    link = dataset_dir / scene
    if link.is_symlink() and link.resolve() == lvba_scene.resolve():
        return
    if link.exists() or link.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Global-LVBA dataset link already exists: {link}")
        if link.is_dir() and not link.is_symlink():
            raise RuntimeError(f"Refusing to replace a real directory: {link}")
        link.unlink()
    link.symlink_to(lvba_scene.resolve(), target_is_directory=True)


def write_global_lvba_config(scene: str, lvba_scene: Path, stride: int = 5) -> None:
    content = f"""cam_model:
  cam_width: {ORIGINAL_WIDTH}
  cam_height: {ORIGINAL_HEIGHT}
  scale: {SCALE}
  cam_fx: {FX / SCALE}
  cam_fy: {FY / SCALE}
  cam_cx: {CX / SCALE}
  cam_cy: {CY / SCALE}
  cam_d0: {DISTORTION[0]}
  cam_d1: {DISTORTION[1]}
  cam_d2: {DISTORTION[2]}
  cam_d3: {DISTORTION[3]}

extrin_calib:
  extrinsic_T: {T_IL.tolist()}
  extrinsic_R: {R_IL.reshape(-1).tolist()}
  Rcl: {R_CL.reshape(-1).tolist()}
  Pcl: {T_CL.tolist()}

data_config:
  data_path: "dataset/{scene}/"
  colmap_db_path: "Colmap/colmap_sub{stride}.db"
  image_sample_step: {stride}
  enable_lidar_ba: false
  enable_visual_ba: true

window_ba:
  enable: true
  size: 20
  anchor_leaf_size: 0.01
  use_window_ba_rel: true

BALM_stage1:
  enable: true
  root_voxel_size: 1.0
  eigen_ratio_array: [0.2, 0.2, 0.2, 0.2]

BALM_stage2:
  root_voxel_size: 0.5
  eigen_ratio_array: [0.08, 0.08, 0.08, 0.08]

track_fusion:
  min_view_angle: 8.0
  reproj_mean_thr: 3.0

colmap_output:
  enable: false
  filter_size_points3D: 0.01
"""
    (lvba_scene / "global_lvba_config.yaml").write_text(content, encoding="utf-8")


def build_colmap_database(lvba_scene: Path, stride: int, overwrite: bool) -> Path:
    import pycolmap

    image_dir = lvba_scene / "all_image"
    images = sorted(image_dir.glob("*.png"), key=lambda path: float(path.stem))
    names = [path.name for path in images[::stride]]
    database = lvba_scene / "Colmap" / f"colmap_sub{stride}.db"
    database.parent.mkdir(exist_ok=True)
    if database.exists():
        if not overwrite:
            raise FileExistsError(f"COLMAP database exists: {database}")
        database.unlink()

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    reader.camera_params = ",".join(
        str(value) for value in [FX, FY, CX, CY, *DISTORTION[:4]]
    )
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = 8192
    print(f"Extracting COLMAP features for {len(names)} images", flush=True)
    pycolmap.extract_features(
        database,
        image_dir,
        image_names=names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.auto,
    )
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 10
    pairing.loop_detection = False
    pycolmap.match_sequential(database, pairing_options=pairing, device=pycolmap.Device.auto)
    return database


def load_tum(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float64)
    poses = np.atleast_2d(poses)
    if poses.shape[1] != 8:
        raise ValueError(f"Expected Nx8 TUM poses: {path}")
    return poses


def pose_matrix(row: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    matrix[:3, 3] = row[1:4]
    return matrix


def camera_world_from_imu_pose(imu_world: np.ndarray) -> np.ndarray:
    camera_imu = np.eye(4, dtype=np.float64)
    camera_imu[:3, :3] = R_CI
    camera_imu[:3, 3] = T_CI
    return camera_imu @ np.linalg.inv(imu_world)


def read_binary_pcd_xyz(path: Path) -> np.ndarray:
    header = []
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD has no DATA line: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded.startswith("DATA"):
                offset = stream.tell()
                break
    entries = {line.split()[0]: line.split()[1:] for line in header if line.split()}
    if entries["DATA"][0] != "binary":
        raise ValueError(f"Only binary PCD is supported: {path}")
    fields = entries["FIELDS"]
    sizes = [int(value) for value in entries["SIZE"]]
    types = entries["TYPE"]
    counts = [int(value) for value in entries.get("COUNT", ["1"] * len(fields))]
    points = int(entries["POINTS"][0])
    type_map = {
        ("F", 4): "<f4", ("F", 8): "<f8", ("I", 1): "i1", ("I", 2): "<i2",
        ("I", 4): "<i4", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
    }
    dtype = np.dtype([
        (name, type_map[(kind, size)]) if count == 1 else (name, type_map[(kind, size)], (count,))
        for name, size, kind, count in zip(fields, sizes, types, counts)
    ])
    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(points,))
    return np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)


def sorted_timestamp_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(directory.glob(f"*{suffix}"), key=lambda path: float(path.stem))


def validate_stream(files: list[Path], poses: np.ndarray, label: str) -> np.ndarray:
    if len(files) != len(poses):
        raise ValueError(f"{label} count mismatch: files={len(files)}, poses={len(poses)}")
    timestamps = np.array([float(path.stem) for path in files])
    error = np.abs(timestamps - poses[:, 0])
    if error.max(initial=0.0) > 1e-5:
        raise ValueError(f"{label} filename/pose timestamp mismatch: max={error.max():.6f}s")
    return timestamps


def nearest_indices(source: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source, queries).clip(0, len(source) - 1)
    left = (right - 1).clip(0, len(source) - 1)
    choose_left = np.abs(source[left] - queries) <= np.abs(source[right] - queries)
    indices = np.where(choose_left, left, right)
    return indices, np.abs(source[indices] - queries)


def modp_config(dataset: Path, scene: str, end_cutoff: int) -> str:
    base = MODP_REPO / "configs/aria/orb_tracking/aria_base.yaml"
    name = f"LVBA-{scene}-baseline" if end_cutoff == 0 else f"LVBA-{scene}-smoke"
    section = f"""  name: "{name}"
  type: "aria"
  data_source: "orb"
  dataset_path: "{dataset}"
  num_threads: 0
  begin_cutoff: 0
  end_cutoff: {end_cutoff}
  stride: 1
  max_pts_num: -1
  vignette: False
  use_vignette_type: "post-render"
  Calibration:
    fx: {FX:.6f}
    fy: {FY:.6f}
    cx: {CX:.6f}
    cy: {CY:.6f}
    width: {WIDTH}
    height: {HEIGHT}
    near: 0.10
    far: 120.0
"""
    return f"""inherit_from: "{base}"

Dataset:
{section}
Testset:
{section}
Results:
  save_dir: "{MODP_LOG_ROOT}"
  save_gt: True
  save_exr: False
  save_mesh: False
"""


def convert_to_modp(
    scene: str,
    source: Path,
    output: Path,
    max_points: int,
    overwrite: bool,
) -> dict:
    image_dir = source / "all_image"
    pcd_dir = source / "all_pcd_body"
    image_poses = load_tum(image_dir / "image_poses.txt")
    lidar_poses = load_tum(pcd_dir / "lidar_poses.txt")
    images = sorted_timestamp_files(image_dir, ".png")
    pcds = sorted_timestamp_files(pcd_dir, ".pcd")
    image_ts = validate_stream(images, image_poses, "image")
    pcd_ts = validate_stream(pcds, lidar_poses, "PCD")
    lidar_indices, time_errors = nearest_indices(pcd_ts, image_ts)
    if time_errors.max() > 0.2:
        raise ValueError(f"Image/LiDAR synchronization exceeds 0.2 s: {time_errors.max():.6f}s")

    rectified = output / "rectified"
    point_dir = output / "orb_point_clouds"
    output.mkdir(parents=True, exist_ok=True)
    replace_directory(rectified, overwrite)
    replace_directory(point_dir, overwrite)

    original_t_cw = [camera_world_from_imu_pose(pose_matrix(row)) for row in image_poses]
    world_to_normalized = original_t_cw[0]
    normalized_to_world = np.linalg.inv(world_to_normalized)
    cameras = []
    input_counts, visible_counts, saved_counts, depths = [], [], [], []
    rng = np.random.default_rng(20260721)

    for index, (image_path, lidar_index) in enumerate(zip(images, lidar_indices)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if (image.shape[1], image.shape[0]) != (WIDTH, HEIGHT):
            raise ValueError(f"Unexpected image size {image.shape[1]}x{image.shape[0]}: {image_path}")
        undistorted = cv2.undistort(image, K, DISTORTION, None, K)
        target_name = f"aria_{index:05d}.png"
        if not cv2.imwrite(str(rectified / target_name), undistorted):
            raise RuntimeError(f"Failed to write rectified image {target_name}")

        points_body = read_binary_pcd_xyz(pcds[int(lidar_index)])
        t_wi = pose_matrix(lidar_poses[int(lidar_index)])
        points_world = (t_wi[:3, :3] @ points_body.T).T + t_wi[:3, 3]
        points_camera = (original_t_cw[index][:3, :3] @ points_world.T).T + original_t_cw[index][:3, 3]
        z = points_camera[:, 2]
        u = FX * points_camera[:, 0] / z + CX
        v = FY * points_camera[:, 1] / z + CY
        mask = (
            np.isfinite(points_camera).all(axis=1) & (z >= 0.1) & (z <= 120.0)
            & (u >= 0.0) & (u < WIDTH) & (v >= 0.0) & (v < HEIGHT)
        )
        selected = np.flatnonzero(mask)
        if len(selected) > max_points:
            selected = rng.choice(selected, max_points, replace=False)
        normalized = (
            world_to_normalized[:3, :3] @ points_world[selected].T
        ).T + world_to_normalized[:3, 3]
        np.save(point_dir / f"point_cloud_{index}.npy", normalized.astype(np.float32))

        t_cw_normalized = original_t_cw[index] @ normalized_to_world
        cameras.append({
            "T_camera_world": t_cw_normalized.tolist(),
            "image": target_name,
            "timestamp": float(image_ts[index]),
            "frame_index": index,
            "lidar_timestamp": float(pcd_ts[int(lidar_index)]),
            "intrinsic": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
            "width": WIDTH,
            "height": HEIGHT,
            "focal": 0.5 * (FX + FY),
        })
        input_counts.append(len(points_body))
        visible_counts.append(int(mask.sum()))
        saved_counts.append(len(selected))
        depths.extend(z[selected].tolist())
        if index % 100 == 0 or index + 1 == len(images):
            print(
                f"[{index + 1}/{len(images)}] input={len(points_body)} "
                f"visible={int(mask.sum())} saved={len(selected)} dt={time_errors[index]:.6f}s",
                flush=True,
            )

    payload = json.dumps({"cameras": cameras}, indent=2) + "\n"
    (output / "trajectory_orb.json").write_text(payload, encoding="utf-8")
    (output / "trajectory.json").write_text(payload, encoding="utf-8")
    stats = {
        "scene": scene,
        "source": str(source),
        "method": "FAST-LIVO2 poses with globally transformed LiDAR scans",
        "pose_source_kind": "fast_livo2",
        "frame_count": len(cameras),
        "pose_source": (
            "FAST-LIVO2 causal LiDAR-inertial odometry with image-pose interpolation"
            if (source / "fast_livo2_metadata.json").exists()
            and json.loads((source / "fast_livo2_metadata.json").read_text())["tracking_mode"] == "lio"
            else "FAST-LIVO2 causal LiDAR-inertial-visual odometry"
        ),
        "images": "OpenCV-undistorted FAST-LIVO2 images (not symbolic links)",
        "pose_convention": "world-to-camera, normalized to camera 0",
        "point_coordinate_system": "normalized world",
        "sparse_world_geometry": "persistent",
        "world_frame": "camera-0-normalized FAST-LIVO2 world",
        "depth_source": "globally transformed LiDAR point",
        "image_lidar_timestamp_error_seconds": {
            "max": float(time_errors.max()), "mean": float(time_errors.mean()),
            "p95": float(np.percentile(time_errors, 95)),
        },
        "point_counts": {
            "input_min": int(np.min(input_counts)), "input_mean": float(np.mean(input_counts)),
            "input_max": int(np.max(input_counts)), "visible_min": int(np.min(visible_counts)),
            "visible_mean": float(np.mean(visible_counts)), "visible_max": int(np.max(visible_counts)),
            "saved_min": int(np.min(saved_counts)), "saved_mean": float(np.mean(saved_counts)),
            "saved_max": int(np.max(saved_counts)),
        },
        "saved_depth_m": {
            "min": float(np.min(depths)), "median": float(np.median(depths)),
            "p95": float(np.percentile(depths, 95)), "max": float(np.max(depths)),
        },
        "max_points_per_frame": max_points,
        "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "width": WIDTH, "height": HEIGHT},
        "distortion_removed": DISTORTION.tolist(),
        "extrinsics": {"R_CI": R_CI.tolist(), "t_CI": T_CI.tolist()},
    }
    (output / "conversion_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    smoke_frames = min(60, len(cameras))
    (output / "config_modp_smoke.yaml").write_text(
        modp_config(output, scene, len(cameras) - smoke_frames), encoding="utf-8"
    )
    (output / "config_modp_full.yaml").write_text(modp_config(output, scene, 0), encoding="utf-8")
    (output / "README.md").write_text(
        f"# {scene}: FAST-LIVO2 to MODP\n\n"
        "Poses and body-frame point clouds come from a causal FAST-LIVO2 pass over the source bag. "
        "Images are undistorted into the pinhole model used by MODP. Each frame uses the nearest "
        "timestamped LiDAR scan, transformed into the camera-0-normalized world frame.\n\n"
        "Prepare another scene with the existing environments:\n\n"
        "```bash\n"
        "conda run --no-capture-output -n worldvln python "
        f"{ONLINE_REPO}/tools/prepare_fast_livo2_lvba_modp.py <SCENE>\n"
        "```\n\n"
        "The script uses `worldvln` for bag image extraction, COLMAP feature generation, and MODP "
        f"conversion. It only delegates FAST-LIVO2 itself to the existing `{ROS_ENV}` environment; "
        "it never creates a Conda environment.\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="Bag basename and output scene name, e.g. Red_Sculpture")
    parser.add_argument(
        "--steps",
        default="fastlivo,lvba,colmap,modp",
        help="Comma-separated steps: fastlivo,lvba,colmap,modp",
    )
    parser.add_argument("--bag", type=Path, help="Override the default <scene>.bag path")
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--colmap-stride", type=int, default=5)
    parser.add_argument(
        "--tracking-mode",
        choices=("lio", "livo"),
        default="lio",
        help="Use robust LiDAR-inertial tracking by default; livo also updates poses visually",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = set(steps) - {"fastlivo", "lvba", "colmap", "modp"}
    if unknown:
        raise ValueError(f"Unknown steps: {sorted(unknown)}")
    if "fastlivo" in steps and "lvba" not in steps:
        raise ValueError("The fastlivo step must be followed by lvba so fixed Log output is staged safely")

    scene = args.scene
    bag = args.bag or FAST_LIVO_DATA / f"{scene}.bag"
    lvba_scene = LVBA_ROOT / scene
    modp_scene = MODP_DATA_ROOT / scene
    lvba_scene.mkdir(parents=True, exist_ok=True)
    metadata = None

    if "fastlivo" in steps:
        metadata = run_fast_livo2(
            scene, bag.resolve(), lvba_scene, tracking_mode=args.tracking_mode
        )
    if "lvba" in steps:
        if metadata is None:
            raise ValueError("lvba staging requires fastlivo in the same invocation")
        stage_global_lvba(scene, lvba_scene, metadata, args.overwrite)
    if "colmap" in steps:
        database = build_colmap_database(lvba_scene, args.colmap_stride, args.overwrite)
        print(f"COLMAP database: {database}")
    if "modp" in steps:
        stats = convert_to_modp(scene, lvba_scene, modp_scene, args.max_points, args.overwrite)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
