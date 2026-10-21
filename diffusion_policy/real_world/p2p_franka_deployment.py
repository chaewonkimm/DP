"""Safe deployment helpers for the p2p Diffusion Policy checkpoints.

The policy is task agnostic here: square peg emits a relative 9D TCP pose and
valve emits the same pose followed by a binary gripper command.  Hardware
imports are intentionally lazy so recorded-data checks work in the ``robodiff``
environment even when ROS 2 is not sourced.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import socketserver
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import dill
import hydra
import numpy as np
import requests
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.p2p_remote_desired_bridge_core import (
    select_action_chunk, validate_chunk_window,
)
from diffusion_policy.real_world.p2p_gripper_actions import (
    GripperActionSettings, quantize_gripper_command,
)
from diffusion_policy.real_world.p2p_ft_bias import FTBiasSettings
from diffusion_policy.dataset.p2p_image_dataset import (
    absolute_pose9d_to_relative,
    relative_action_to_absolute,
    _ortho6d_to_rotation_matrix,
)


EXECUTE_CONFIRMATION = "I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA"


def _jsonable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return _jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_policy_from_checkpoint(
        checkpoint_path: str,
        device: str,
        weights: str = "ema_model",
        num_inference_steps: Optional[int] = 8,
        ) -> Tuple[torch.nn.Module, DictConfig]:
    """Instantiate only the policy and restore its checkpoint state.

    The checkpoint also contains the optimizer and non-EMA model.  PyTorch has
    to deserialize the full payload, but only the requested policy state is
    copied into the returned module.
    """
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    with checkpoint.open("rb") as stream:
        payload = torch.load(stream, pickle_module=dill, map_location="cpu")
    if "cfg" not in payload or "state_dicts" not in payload:
        raise ValueError(f"Not a DP workspace checkpoint: {checkpoint}")
    if weights not in payload["state_dicts"]:
        available = sorted(payload["state_dicts"].keys())
        raise KeyError(f"Checkpoint has no {weights!r}; available: {available}")

    train_cfg = payload["cfg"]
    policy = hydra.utils.instantiate(train_cfg.policy)
    policy.load_state_dict(payload["state_dicts"][weights], strict=True)
    del payload

    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)
    policy.eval().to(torch.device(device))
    return policy, train_cfg


def _resize_rgb_sequence(images: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Resize/crop THWC frames like RDP's real inference utility."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected THWC RGB sequence, got {images.shape}")
    output_h, output_w = output_shape
    result = []
    for image in images:
        input_h, input_w = image.shape[:2]
        scale = max(output_w / input_w, output_h / input_h)
        resized_w = int(math.ceil(input_w * scale))
        resized_h = int(math.ceil(input_h * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
        x0 = (resized_w - output_w) // 2
        y0 = (resized_h - output_h) // 2
        result.append(resized[y0:y0 + output_h, x0:x0 + output_w])
    return np.stack(result)


def prepare_policy_observation(
        env_observation: Dict[str, np.ndarray],
        shape_meta: DictConfig,
        ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """Apply the exact modalities and relative-pose convention used in training."""
    expected = shape_meta["obs"]
    missing = [key for key in expected if key not in env_observation]
    if missing:
        raise KeyError(f"Live observation is missing keys: {missing}")

    absolute_pose = np.asarray(
        env_observation["left_robot_tcp_pose"], dtype=np.float32)
    if absolute_pose.ndim != 2 or absolute_pose.shape[1] < 9:
        raise ValueError(
            "left_robot_tcp_pose must have shape (T, >=9), got "
            f"{absolute_pose.shape}")
    absolute_pose = absolute_pose[:, :9]
    base_pose = absolute_pose[-1].copy()

    result: Dict[str, torch.Tensor] = {}
    for key, attributes in expected.items():
        modality = attributes.get("type", "low_dim")
        shape = tuple(int(item) for item in attributes["shape"])
        value = np.asarray(env_observation[key])
        if modality == "rgb":
            channels, height, width = shape
            if channels != 3:
                raise ValueError(f"Only 3-channel RGB is supported, got {shape} for {key}")
            value = _resize_rgb_sequence(value, (height, width))
            value = np.moveaxis(value, -1, 1).astype(np.float32) / 255.0
        elif modality == "low_dim":
            value = value[..., :shape[0]].astype(np.float32)
            if key == "left_robot_tcp_pose":
                value = absolute_pose9d_to_relative(value, base_pose)
        else:
            raise ValueError(f"Unsupported modality {modality!r} for {key}")
        expected_sample_shape = shape
        if tuple(value.shape[1:]) != expected_sample_shape:
            raise ValueError(
                f"{key} has per-frame shape {value.shape[1:]}, expected "
                f"{expected_sample_shape}")
        result[key] = torch.from_numpy(np.ascontiguousarray(value))[None]
    return result, base_pose


def observation_key_sets(shape_meta: DictConfig) -> Tuple[tuple[str, ...], tuple[str, ...]]:
    """Return checkpoint-declared RGB and low-dimensional observation keys."""
    rgb_keys = []
    lowdim_keys = []
    for key, attributes in shape_meta["obs"].items():
        modality = str(attributes.get("type", "low_dim"))
        if modality == "rgb":
            rgb_keys.append(str(key))
        elif modality == "low_dim":
            lowdim_keys.append(str(key))
        else:
            raise ValueError(f"Unsupported observation modality {modality!r} for {key}")
    return tuple(rgb_keys), tuple(lowdim_keys)


def validate_checkpoint_deployment_contract(
        train_cfg: DictConfig,
        deploy_cfg: DictConfig,
        ) -> None:
    """Reject a wrong checkpoint/config pairing before any ROS process starts."""
    actual_task = str(train_cfg.task.name)
    expected_task = deploy_cfg.task.get("expected_checkpoint_task_name")
    if expected_task is not None and actual_task != str(expected_task):
        raise RuntimeError(
            f"Checkpoint task {actual_task!r} does not match deployment task "
            f"{str(expected_task)!r}")

    actual_keys = tuple(str(key) for key in train_cfg.shape_meta.obs.keys())
    expected_keys = deploy_cfg.task.get("expected_observation_keys")
    if expected_keys is not None and set(actual_keys) != {
            str(key) for key in expected_keys}:
        raise RuntimeError(
            f"Checkpoint observations {sorted(actual_keys)} do not match deployment "
            f"observations {sorted(str(key) for key in expected_keys)}")

    action_dim = int(train_cfg.shape_meta.action.shape[0])
    if action_dim != int(deploy_cfg.task.action_dim):
        raise RuntimeError(
            f"Checkpoint action_dim={action_dim} does not match deployment "
            f"action_dim={int(deploy_cfg.task.action_dim)}")

    rgb_keys, lowdim_keys = observation_key_sets(train_cfg.shape_meta)
    configured_rgb = tuple(str(key) for key in deploy_cfg.task.get(
        "rgb_obs_keys", rgb_keys))
    if set(rgb_keys) != set(configured_rgb):
        raise RuntimeError(
            f"Checkpoint RGB keys {sorted(rgb_keys)} do not match deployment "
            f"RGB keys {sorted(configured_rgb)}")

    if "left_robot_tcp_pose" not in lowdim_keys:
        raise RuntimeError(
            "Relative-action rollout requires left_robot_tcp_pose in the checkpoint")
    if "left_goal_pose_rel" in lowdim_keys:
        goal = deploy_cfg.task.get("goal_pose_world_9d")
        if goal is None:
            raise RuntimeError(
                "Goal-conditioned checkpoint requires task.goal_pose_world_9d")
        goal_array = np.asarray(goal, dtype=np.float64)
        if goal_array.shape != (9,) or not np.isfinite(goal_array).all():
            raise RuntimeError("task.goal_pose_world_9d must contain 9 finite values")
        # Also validates that the two rotation-6D columns are non-degenerate.
        goal_rotation = _ortho6d_to_rotation_matrix(goal_array[None, 3:9])
        if not np.isfinite(goal_rotation).all():
            raise RuntimeError(
                "task.goal_pose_world_9d contains a degenerate rotation-6D value")

    require_filter = bool(deploy_cfg.task.get("require_wrench_filter", False))
    filter_enabled = bool(deploy_cfg.live.remote_desired.get(
        "wrench_filter", {}).get("enabled", False))
    uses_wrench = "left_robot_tcp_wrench" in lowdim_keys
    if require_filter and not uses_wrench:
        raise RuntimeError(
            "task.require_wrench_filter=true but checkpoint has no wrench input")
    if require_filter and not filter_enabled:
        raise RuntimeError(
            "This checkpoint requires live.remote_desired.wrench_filter.enabled=true")


def validate_inference_seed(seed: Optional[int]) -> Optional[int]:
    if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
            or not 0 <= seed < 2**63):
        raise ValueError("inference_seed must be null or an integer in [0, 2**63 - 1]")
    return seed


@torch.no_grad()
def predict_relative_actions(
        policy: torch.nn.Module,
        observation: Dict[str, torch.Tensor],
        inference_seed: Optional[int] = None,
        ) -> np.ndarray:
    """An explicit seed restarts a local RNG for EACH chunk, not each rollout."""
    inference_seed = validate_inference_seed(inference_seed)
    device = policy.device
    tensor_observation = dict_apply(
        observation, lambda tensor: tensor.to(device, non_blocking=True))
    if inference_seed is None:
        prediction = policy.predict_action(tensor_observation)["action"]
    else:
        # Do not reset the global RNG or reuse an advancing generator. This
        # repeats the initial noise (and any stochastic scheduler draws) on
        # every inference with the same shape/device/dtype and seed.
        generator = torch.Generator(device=device).manual_seed(inference_seed)
        prediction = policy.predict_action(
            tensor_observation, generator=generator)["action"]
    actions = prediction[0].detach().cpu().numpy().astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] not in (9, 10):
        raise ValueError(f"Policy returned an unsupported action shape: {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned NaN or infinite action values")
    return actions


def pose9d_to_pose7d(pose: np.ndarray) -> np.ndarray:
    """Convert xyz + rotation-6D to xyz + quaternion(wxyz)."""
    pose = np.asarray(pose, dtype=np.float64).reshape(9)
    matrix = _ortho6d_to_rotation_matrix(pose[None, 3:9])[0]
    quaternion_xyzw = Rotation.from_matrix(matrix).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return np.concatenate([pose[:3], quaternion_wxyz])


def pose7d_to_pose9d(pose: np.ndarray) -> np.ndarray:
    """Convert xyz + quaternion(wxyz) from the Franka server to pose9D."""
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    quaternion_xyzw = pose[[4, 5, 6, 3]]
    matrix = Rotation.from_quat(quaternion_xyzw).as_matrix()
    rotation6d = matrix[:, :2].T.reshape(6)
    return np.concatenate([pose[:3], rotation6d]).astype(np.float32)


def pose7d_xyzw_to_pose9d(pose: np.ndarray) -> np.ndarray:
    """Convert ROS xyz+quaternion(xyzw) to xyz+rotation-6D."""
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    matrix = Rotation.from_quat(pose[3:7]).as_matrix()
    rotation6d = matrix[:, :2].T.reshape(6)
    return np.concatenate([pose[:3], rotation6d]).astype(np.float32)


def pose9d_to_pose7d_xyzw(pose: np.ndarray) -> np.ndarray:
    """Convert xyz+rotation-6D to the xyzw quaternion order used by ROS."""
    pose = np.asarray(pose, dtype=np.float64).reshape(9)
    matrix = _ortho6d_to_rotation_matrix(pose[None, 3:9])[0]
    return np.concatenate([pose[:3], Rotation.from_matrix(matrix).as_quat()])


def rotation_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    rotation_a = _ortho6d_to_rotation_matrix(
        np.asarray(pose_a, dtype=np.float64).reshape(1, -1)[:, 3:9])[0]
    rotation_b = _ortho6d_to_rotation_matrix(
        np.asarray(pose_b, dtype=np.float64).reshape(1, -1)[:, 3:9])[0]
    return float(Rotation.from_matrix(rotation_a.T @ rotation_b).magnitude())


def validate_execute_configuration(cfg: DictConfig, confirmation: Optional[str]) -> None:
    if confirmation != EXECUTE_CONFIRMATION:
        raise RuntimeError(
            "Execute confirmation is missing. Re-run with "
            f"--confirm {EXECUTE_CONFIRMATION}")
    if not bool(cfg.live.execute_enabled):
        raise RuntimeError("Set live.execute_enabled=true only after the safety review")

    required_positive = (
        "max_translation_step_m",
        "max_rotation_step_rad",
        "max_force_n",
        "max_torque_nm",
    )
    for key in required_positive:
        value = cfg.safety.get(key)
        if value is None or float(value) <= 0:
            raise RuntimeError(f"safety.{key} must be explicitly set to a positive value")
    for key in ("workspace_min", "workspace_max"):
        value = cfg.safety.get(key)
        if value is None or len(value) != 3 or any(item is None for item in value):
            raise RuntimeError(f"safety.{key} must contain three explicit coordinates")

    if cfg.task.action_dim == 10:
        backend = str(cfg.live.get("backend", "direct"))
        if backend == "remote_desired":
            remote_mode = str(cfg.gripper.get(
                "remote_desired_mode", "hold_current"))
            if remote_mode == "hold_current":
                if not bool(cfg.gripper.get(
                        "allow_remote_desired_hold_current", False)):
                    raise RuntimeError(
                        "The RemoteDesired arm driver has no gripper endpoint. "
                        "Set gripper.allow_remote_desired_hold_current=true only "
                        "after gripping the valve peg manually, or configure a "
                        "reviewed ros_float64_width endpoint")
            elif remote_mode in ("ros_float64_width", "franka_action"):
                if remote_mode == "ros_float64_width" and not cfg.gripper.get("ros_command_topic"):
                    raise RuntimeError(
                        "ros_float64_width requires gripper.ros_command_topic")
                if not bool(cfg.gripper.mapping_confirmed):
                    raise RuntimeError(
                        "Valve execution requires gripper.mapping_confirmed=true")
                if remote_mode == "franka_action":
                    GripperActionSettings.from_payload(OmegaConf.to_container(cfg.gripper, resolve=True))
            else:
                raise RuntimeError(
                    "gripper.remote_desired_mode must be hold_current or "
                    "ros_float64_width or franka_action")
        else:
            if not bool(cfg.gripper.mapping_confirmed):
                raise RuntimeError(
                    "Valve execution requires gripper.mapping_confirmed=true after "
                    "checking the dataset command semantics")
            for key in (
                    "open_width_m", "closed_width_m", "velocity_mps",
                    "force_limit_n"):
                value = cfg.gripper.get(key)
                if value is None or float(value) < 0:
                    raise RuntimeError(f"gripper.{key} must be explicitly set")


def validate_ft_bias_configuration(cfg: DictConfig) -> FTBiasSettings:
    remote = cfg.live.get('remote_desired', {})
    values = remote.get('ft_bias', {})
    if OmegaConf.is_config(values):
        values = OmegaConf.to_container(values, resolve=True)
    settings = FTBiasSettings.from_payload(values)
    if settings.enabled and (cfg.live.get('backend') != 'remote_desired'
                             or not remote.get('require_enable', True)):
        raise ValueError('FT bias requires remote_desired with require_enable=true')
    return settings


def validate_live_configuration(cfg: DictConfig) -> None:
    """Fail fast on live-only dependencies before loading a large checkpoint."""
    validate_ft_bias_configuration(cfg)
    backend = str(cfg.live.get("backend", "direct"))
    if backend == "direct":
        if not cfg.live.get("external_camera_serial") or not cfg.live.get(
                "wrist_camera_serial"):
            raise RuntimeError(
                "Set live.external_camera_serial and live.wrist_camera_serial "
                "after confirming their physical roles")
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "Direct live mode requires pyrealsense2 in this environment"
            ) from error
        connected_serials = {
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
            if device.get_info(rs.camera_info.name).lower() != "platform camera"
        }
        configured_serials = {
            str(cfg.live.external_camera_serial),
            str(cfg.live.wrist_camera_serial),
        }
        if len(configured_serials) != 2:
            raise RuntimeError("External and wrist camera serials must differ")
        missing_serials = sorted(configured_serials - connected_serials)
        if missing_serials:
            raise RuntimeError(
                f"Configured RealSense cameras are not connected: {missing_serials}; "
                f"connected: {sorted(connected_serials)}")
        robot_url = (
            f"http://{cfg.live.robot_server_ip}:{int(cfg.live.robot_server_port)}")
        try:
            response = requests.get(
                f"{robot_url}/get_current_robot_states",
                timeout=float(cfg.live.http_timeout_seconds),
            )
            response.raise_for_status()
            state = response.json()
            pose7d_to_pose9d(state["leftRobotTCP"])
            if np.asarray(state["leftRobotTCPWrench"]).shape != (6,):
                raise ValueError("leftRobotTCPWrench is not 6D")
        except (requests.RequestException, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Franka HTTP state preflight failed at {robot_url}") from error
    elif backend == "rdp_ros":
        rdp_root = Path(cfg.live.rdp_root).expanduser().resolve()
        if not (rdp_root / "reactive_diffusion_policy").is_dir():
            raise RuntimeError(f"RDP source tree not found: {rdp_root}")
        if str(rdp_root) not in sys.path:
            sys.path.insert(0, str(rdp_root))
        try:
            import rclpy  # noqa: F401
            import message_filters  # noqa: F401
            import reactive_diffusion_policy  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "rdp_ros backend requires ROS 2 and RDP in this Python environment"
            ) from error
    elif backend == "remote_desired":
        bridge = cfg.live.get("remote_desired", {})
        rgb_keys = tuple(str(key) for key in cfg.task.get(
            "rgb_obs_keys", ("external_img", "left_wrist_img")))
        validate_chunk_window(bridge.get("chunk_start_index", 0), bridge["chunk_size"])
        if not 0 <= int(bridge["prefetch_steps"]) < int(bridge["chunk_size"]):
            raise ValueError("prefetch_steps must be smaller than the selected chunk_size")
        bridge_script = Path(str(bridge.get(
            "bridge_script",
            Path(__file__).resolve().parents[2]
            / "p2p_remote_desired_real_bridge.py",
        ))).expanduser().resolve()
        if not bridge_script.is_file():
            raise RuntimeError(
                f"RemoteDesired bridge script does not exist: {bridge_script}")
        bridge_python = Path(str(bridge.get(
            "bridge_python", "/usr/bin/python3"))).expanduser()
        if not bridge_python.is_file() or not os.access(bridge_python, os.X_OK):
            raise RuntimeError(
                f"RemoteDesired ROS Python is not executable: {bridge_python}")
        ros_setup_file = bridge.get("ros_setup_file")
        if ros_setup_file and not Path(str(ros_setup_file)).expanduser().is_file():
            raise RuntimeError(
                f"RemoteDesired ROS setup file does not exist: {ros_setup_file}")
        import_check = "from iris_teleop.msg import TeleopState; import rclpy"
        if rgb_keys:
            import_check += "; from sensor_msgs.msg import CompressedImage"
        if (int(cfg.task.action_dim) == 10
                and cfg.gripper.get("remote_desired_mode") == "franka_action"):
            GripperActionSettings.from_payload(OmegaConf.to_container(cfg.gripper, resolve=True))
            import_check += "; from franka_msgs.action import Move, Grasp; from rclpy.action import ActionClient"
        if ros_setup_file:
            check_command = [
                "/bin/bash", "-c",
                'set -e; source "$1"; exec "$2" -c "$3"',
                "dp-p2p-preflight",
                str(Path(str(ros_setup_file)).expanduser().resolve()),
                str(bridge_python),
                import_check,
            ]
        else:
            check_command = [str(bridge_python), "-c", import_check]
        result = subprocess.run(
            check_command, capture_output=True, text=True, check=False,
            timeout=10.0)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                "RemoteDesired ROS Python cannot import rclpy and IRIS "
                f"messages: {detail}")
        required_topics = (
            "remote_state", "remote_desired", "remote_desired_shadow",
            "enable", "status", "dry_run_action", *rgb_keys,
        )
        topics = bridge.get("topics", {})
        missing = [name for name in required_topics if not topics.get(name)]
        if missing:
            raise RuntimeError(
                f"live.remote_desired.topics is missing: {missing}")
        if str(bridge.get("worker_host", "127.0.0.1")) not in {
                "127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(
                "live.remote_desired.worker_host must be loopback because the "
                "local bridge protocol uses trusted pickle payloads")
    else:
        raise RuntimeError(
            f"Unknown live.backend={backend!r}; expected 'direct', 'rdp_ros', "
            "or 'remote_desired'")


def validate_action_step(
        target_pose: np.ndarray,
        previous_pose: np.ndarray,
        latest_wrench: np.ndarray,
        safety_cfg: DictConfig,
        ) -> Dict[str, float]:
    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(9)
    previous_pose = np.asarray(previous_pose, dtype=np.float64).reshape(9)
    latest_wrench = np.asarray(latest_wrench, dtype=np.float64).reshape(-1)
    if not np.isfinite(target_pose).all():
        raise RuntimeError("Target pose is not finite")

    workspace_min = np.asarray(safety_cfg.workspace_min, dtype=np.float64)
    workspace_max = np.asarray(safety_cfg.workspace_max, dtype=np.float64)
    if np.any(target_pose[:3] < workspace_min) or np.any(target_pose[:3] > workspace_max):
        raise RuntimeError(
            f"Target xyz {target_pose[:3]} is outside the configured workspace")

    translation_step = float(np.linalg.norm(target_pose[:3] - previous_pose[:3]))
    rotation_step = rotation_distance(previous_pose, target_pose)
    force = float(np.linalg.norm(latest_wrench[:3]))
    torque = float(np.linalg.norm(latest_wrench[3:6]))
    if translation_step > float(safety_cfg.max_translation_step_m):
        raise RuntimeError(
            f"Translation step {translation_step:.4f} m exceeds limit "
            f"{float(safety_cfg.max_translation_step_m):.4f} m")
    if rotation_step > float(safety_cfg.max_rotation_step_rad):
        raise RuntimeError(
            f"Rotation step {rotation_step:.4f} rad exceeds limit "
            f"{float(safety_cfg.max_rotation_step_rad):.4f} rad")
    if force > float(safety_cfg.max_force_n):
        raise RuntimeError(
            f"Measured force {force:.2f} N exceeds limit "
            f"{float(safety_cfg.max_force_n):.2f} N")
    if torque > float(safety_cfg.max_torque_nm):
        raise RuntimeError(
            f"Measured torque {torque:.2f} Nm exceeds limit "
            f"{float(safety_cfg.max_torque_nm):.2f} Nm")
    return {
        "translation_step_m": translation_step,
        "rotation_step_rad": rotation_step,
        "force_n": force,
        "torque_nm": torque,
    }


def gripper_command_to_width(command: float, gripper_cfg: DictConfig) -> float:
    if quantize_gripper_command(command, gripper_cfg.command_threshold) == 1:
        return float(gripper_cfg.closed_width_m)
    return float(gripper_cfg.open_width_m)


def run_offline_check(
        policy: torch.nn.Module,
        train_cfg: DictConfig,
        deploy_cfg: DictConfig,
        sample_index: int,
        output_dir: Path,
        ) -> Dict[str, Any]:
    dataset_cfg = copy.deepcopy(train_cfg.task.dataset)
    dataset_cfg.load_to_memory = False
    dataset = hydra.utils.instantiate(dataset_cfg)
    if len(dataset) == 0:
        raise RuntimeError("The configured training dataset has no samples")
    index = int(sample_index) % len(dataset)
    item = dataset[index]

    observation = {
        key: value[None]
        for key, value in item["obs"].items()
    }
    relative_actions = predict_relative_actions(
        policy, observation, deploy_cfg.get("inference_seed"))

    raw_sample = dataset.sampler.sample_sequence(index)
    absolute_pose_history = dataset._select_observations(
        raw_sample[dataset.tcp_pose_key]).astype(np.float32)
    base_pose = absolute_pose_history[-1]
    absolute_actions = relative_action_to_absolute(relative_actions, base_pose)
    start = int(policy.action_start_offset)
    ground_truth = item["action"][start:start + len(relative_actions)].numpy()
    action_mse = float(np.mean((relative_actions - ground_truth) ** 2))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "offline_prediction.npz",
        base_pose=base_pose,
        predicted_relative_action=relative_actions,
        predicted_absolute_action=absolute_actions,
        ground_truth_relative_action=ground_truth,
    )
    report = {
        "mode": "offline",
        "inference_seed": deploy_cfg.get("inference_seed"),
        "num_inference_steps": int(policy.num_inference_steps),
        "task": str(deploy_cfg.task.name),
        "checkpoint": str(Path(deploy_cfg.checkpoint).expanduser().resolve()),
        "weights": str(deploy_cfg.weights),
        "dataset_path": str(train_cfg.task.dataset.dataset_path),
        "dataset_index": index,
        "dataset_size": len(dataset),
        "action_shape": list(relative_actions.shape),
        "relative_action_mse_single_sample": action_mse,
        "base_pose": base_pose,
        "first_predicted_relative_action": relative_actions[0],
        "first_predicted_absolute_action": absolute_actions[0],
        "warning": str(deploy_cfg.checkpoint_warning),
    }
    write_json(output_dir / "offline_report.json", report)
    return report


class RDPFrankaInterface:
    """Observation/command adapter around the existing Franka ROS stack.

    Construction subscribes to sensors but deliberately sends no robot or
    gripper command.  Commands are only exposed through explicit methods used
    by execute mode.
    """

    def __init__(self, cfg: DictConfig):
        rdp_root = Path(cfg.live.rdp_root).expanduser().resolve()
        if not (rdp_root / "reactive_diffusion_policy").is_dir():
            raise FileNotFoundError(f"RDP source tree not found: {rdp_root}")
        if str(rdp_root) not in sys.path:
            sys.path.insert(0, str(rdp_root))
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from reactive_diffusion_policy.env.real_bimanual.real_env import (
                RealRobotEnvironment,
            )
            from reactive_diffusion_policy.real_world.real_world_transforms import (
                RealWorldTransforms,
            )
        except ImportError as error:
            raise RuntimeError(
                "Live mode needs ROS 2 and the RDP real-robot dependencies in "
                "the same Python environment. See docs/p2p_franka_deployment.md"
            ) from error

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)

        calibration_path = cfg.live.get("calibration_path")
        transform_option = None
        if calibration_path:
            transform_option = OmegaConf.create(
                {"calibration_path": str(Path(calibration_path).expanduser())})
        transforms = RealWorldTransforms(option=transform_option)
        processing = OmegaConf.create({
            "mode": "single_arm_two_realsense_no_tactile",
            "image_resize_shape": [320, 240],
            "use_6d_rotation": True,
            "marker_dimension": 2,
            "pca_param_dict": None,
            "debug": False,
        })
        self.env = RealRobotEnvironment(
            robot_server_ip=str(cfg.live.robot_server_ip),
            robot_server_port=int(cfg.live.robot_server_port),
            transforms=transforms,
            device_mapping_server_ip=str(cfg.live.device_mapping_server_ip),
            device_mapping_server_port=int(cfg.live.device_mapping_server_port),
            data_processing_params=processing,
            max_fps=int(cfg.live.sensor_fps),
            use_force_control_for_gripper=bool(cfg.gripper.close_with_force),
            max_gripper_width=float(cfg.gripper.get("open_width_m") or 0.078),
            min_gripper_width=float(cfg.gripper.get("closed_width_m") or 0.0),
            grasp_force=float(cfg.gripper.get("force_limit_n") or 5.0),
            gripper_control_width_precision=float(cfg.gripper.width_precision_m),
            enable_exp_recording=False,
            debug=False,
        )
        self.executor = MultiThreadedExecutor(num_threads=int(cfg.live.ros_threads))
        self.executor.add_node(self.env)
        self._spin_thread = threading.Thread(
            target=self.executor.spin, name="franka-ros-spin", daemon=True)
        self._spin_thread.start()

    def get_observation(
            self,
            obs_steps: int,
            temporal_downsample_ratio: int,
            timeout_seconds: float,
            ) -> Dict[str, np.ndarray]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            observation = self.env.get_obs(
                obs_steps=obs_steps,
                temporal_downsample_ratio=temporal_downsample_ratio,
            )
            if observation and all(
                    key in observation for key in (
                        "external_img",
                        "left_wrist_img",
                        "left_robot_tcp_pose",
                        "left_robot_tcp_wrench",
                    )):
                return observation
            time.sleep(0.05)
        raise TimeoutError(
            "Timed out waiting for synchronized Franka/camera observations")

    def send_tcp_pose(self, pose9d: np.ndarray) -> None:
        self.env.send_command(
            "/move_tcp/left",
            {"target_tcp": pose9d_to_pose7d(pose9d).tolist()},
        )

    def send_gripper_width(self, width: float, cfg: DictConfig) -> None:
        data = {
            "width": float(width),
            "velocity": float(cfg.velocity_mps),
            "force_limit": float(cfg.force_limit_n),
        }
        if bool(cfg.close_with_force) and math.isclose(
                width, float(cfg.closed_width_m), abs_tol=1e-6):
            self.env.send_command("/move_gripper_force/left", data)
        else:
            self.env.send_command("/move_gripper/left", data)

    def close(self) -> None:
        try:
            self.executor.remove_node(self.env)
            self.executor.shutdown(timeout_sec=2.0)
        finally:
            self.env.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()
            self._spin_thread.join(timeout=2.0)


class DirectFrankaInterface:
    """Read two RealSense cameras and Franka HTTP state without ROS 2.

    Camera frames stay BGR, matching the RDP recording path
    (RealSense ``bgr8`` -> OpenCV JPEG decode). Robot state is sampled in a
    background thread and nearest-neighbor aligned to the selected camera
    timestamps.
    """

    def __init__(self, cfg: DictConfig):
        external_serial = cfg.live.get("external_camera_serial")
        wrist_serial = cfg.live.get("wrist_camera_serial")
        if not external_serial or not wrist_serial:
            raise RuntimeError(
                "Direct live mode requires live.external_camera_serial and "
                "live.wrist_camera_serial. Camera roles must be explicit.")
        if str(external_serial) == str(wrist_serial):
            raise RuntimeError("External and wrist camera serials must differ")
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "Direct live mode requires pyrealsense2 in the DP environment"
            ) from error

        self.cfg = cfg
        self.session = requests.Session()
        self.robot_url = (
            f"http://{cfg.live.robot_server_ip}:{int(cfg.live.robot_server_port)}")
        self._state_lock = threading.Lock()
        self._states = deque(maxlen=max(64, int(cfg.live.state_poll_fps) * 10))
        self._stop_event = threading.Event()
        self._state_error: Optional[BaseException] = None
        self._camera_lock = threading.Lock()
        self._camera_errors: Dict[str, BaseException] = {}
        self._camera_buffers = {
            str(external_serial): deque(maxlen=max(64, int(cfg.live.sensor_fps) * 2)),
            str(wrist_serial): deque(maxlen=max(64, int(cfg.live.sensor_fps) * 2)),
        }
        self._camera_pipelines: Dict[str, Any] = {}
        self._camera_threads = []

        connected_serials = {
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
            if device.get_info(rs.camera_info.name).lower() != "platform camera"
        }
        requested_serials = set(self._camera_buffers)
        missing_serials = sorted(requested_serials - connected_serials)
        if missing_serials:
            raise RuntimeError(
                f"Configured RealSense cameras are not connected: {missing_serials}; "
                f"connected: {sorted(connected_serials)}")

        width, height = (int(item) for item in cfg.live.camera_resolution)
        try:
            for serial in self._camera_buffers:
                pipeline = rs.pipeline()
                camera_config = rs.config()
                camera_config.enable_device(serial)
                camera_config.enable_stream(
                    rs.stream.color,
                    width,
                    height,
                    rs.format.bgr8,
                    int(cfg.live.sensor_fps),
                )
                profile = pipeline.start(camera_config)
                color_sensor = profile.get_device().first_color_sensor()
                if color_sensor.supports(rs.option.global_time_enabled):
                    color_sensor.set_option(rs.option.global_time_enabled, 1.0)
                exposure = cfg.live.get("camera_exposure")
                if exposure is None:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
                else:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
                    color_sensor.set_option(rs.option.exposure, float(exposure))
                    if color_sensor.supports(rs.option.gain):
                        color_sensor.set_option(rs.option.gain, 0.0)
                white_balance = cfg.live.get("camera_white_balance")
                if white_balance is None:
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
                else:
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
                    color_sensor.set_option(rs.option.white_balance, float(white_balance))
                self._camera_pipelines[serial] = pipeline

            for serial, pipeline in self._camera_pipelines.items():
                thread = threading.Thread(
                    target=self._capture_camera,
                    args=(serial, pipeline),
                    name=f"realsense-{serial}",
                    daemon=True,
                )
                thread.start()
                self._camera_threads.append(thread)
        except BaseException:
            for pipeline in self._camera_pipelines.values():
                pipeline.stop()
            raise

        self._state_thread = threading.Thread(
            target=self._poll_robot_state,
            name="franka-http-state",
            daemon=True,
        )
        self._state_thread.start()

    def _capture_camera(self, serial: str, pipeline: Any) -> None:
        while not self._stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data()).copy()
                sample = {
                    "timestamp": np.float64(time.time()),
                    "color": frame,
                }
                with self._camera_lock:
                    self._camera_buffers[serial].append(sample)
                    self._camera_errors.pop(serial, None)
            except RuntimeError as error:
                if not self._stop_event.is_set():
                    with self._camera_lock:
                        self._camera_errors[serial] = error

    def _request_state(self) -> Dict[str, np.ndarray]:
        response = self.session.get(
            f"{self.robot_url}/get_current_robot_states",
            timeout=float(self.cfg.live.http_timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        try:
            pose = pose7d_to_pose9d(payload["leftRobotTCP"])
            wrench = np.asarray(payload["leftRobotTCPWrench"], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Franka state response must contain leftRobotTCP (7D wxyz) "
                "and leftRobotTCPWrench (6D)") from error
        if wrench.shape != (6,):
            raise RuntimeError(f"Expected 6D Franka wrench, got {wrench.shape}")
        return {
            "timestamp": np.float64(time.time()),
            "pose": pose,
            "wrench": wrench,
        }

    def _poll_robot_state(self) -> None:
        period = 1.0 / float(self.cfg.live.state_poll_fps)
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                state = self._request_state()
                with self._state_lock:
                    self._states.append(state)
                    self._state_error = None
            except BaseException as error:
                with self._state_lock:
                    self._state_error = error
            self._stop_event.wait(max(0.0, period - (time.monotonic() - started)))

    @staticmethod
    def _nearest_indices(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) == 0:
            raise RuntimeError("Camera returned no timestamps")
        return np.asarray(
            [int(np.argmin(np.abs(timestamps - target))) for target in targets],
            dtype=np.int64,
        )

    def get_observation(
            self,
            obs_steps: int,
            temporal_downsample_ratio: int,
            timeout_seconds: float,
            ) -> Dict[str, np.ndarray]:
        raw_span = (obs_steps - 1) * temporal_downsample_ratio
        deadline = time.monotonic() + timeout_seconds
        last_error: Optional[BaseException] = None

        while time.monotonic() < deadline:
            try:
                external_serial = str(self.cfg.live.external_camera_serial)
                wrist_serial = str(self.cfg.live.wrist_camera_serial)
                with self._camera_lock:
                    external_samples = list(self._camera_buffers[external_serial])
                    wrist_samples = list(self._camera_buffers[wrist_serial])
                    camera_errors = dict(self._camera_errors)
                minimum_samples = raw_span + 1
                if len(external_samples) < minimum_samples or len(
                        wrist_samples) < minimum_samples:
                    if camera_errors:
                        raise RuntimeError(
                            f"RealSense capture failed: {camera_errors}")
                    raise RuntimeError("Waiting for RealSense frame history")
                external_timestamps = np.asarray(
                    [sample["timestamp"] for sample in external_samples])
                wrist_timestamps = np.asarray(
                    [sample["timestamp"] for sample in wrist_samples])
                latest_time = min(
                    float(external_timestamps[-1]),
                    float(wrist_timestamps[-1]),
                )
                targets = latest_time - (
                    np.arange(obs_steps - 1, -1, -1, dtype=np.float64)
                    * temporal_downsample_ratio / float(self.cfg.live.sensor_fps))
                external_indices = self._nearest_indices(external_timestamps, targets)
                wrist_indices = self._nearest_indices(wrist_timestamps, targets)

                with self._state_lock:
                    states = list(self._states)
                    state_error = self._state_error
                if not states:
                    if state_error is not None:
                        raise RuntimeError("Franka state polling failed") from state_error
                    raise RuntimeError("No Franka state samples are available yet")
                state_timestamps = np.asarray(
                    [state["timestamp"] for state in states], dtype=np.float64)
                state_indices = self._nearest_indices(state_timestamps, targets)

                selected_external_times = external_timestamps[external_indices]
                selected_wrist_times = wrist_timestamps[wrist_indices]
                selected_state_times = state_timestamps[state_indices]
                max_skew = float(self.cfg.live.max_sync_skew_seconds)
                observed_skew = max(
                    float(np.max(np.abs(selected_external_times - targets))),
                    float(np.max(np.abs(selected_wrist_times - targets))),
                    float(np.max(np.abs(selected_state_times - targets))),
                )
                if observed_skew > max_skew:
                    raise RuntimeError(
                        f"Sensor synchronization skew {observed_skew:.4f}s "
                        f"exceeds {max_skew:.4f}s")

                return {
                    "timestamp": targets[:, None],
                    "external_img": np.stack(
                        [external_samples[index]["color"]
                         for index in external_indices]),
                    "left_wrist_img": np.stack(
                        [wrist_samples[index]["color"]
                         for index in wrist_indices]),
                    "left_robot_tcp_pose": np.stack(
                        [states[index]["pose"] for index in state_indices]),
                    "left_robot_tcp_wrench": np.stack(
                        [states[index]["wrench"] for index in state_indices]),
                }
            except BaseException as error:
                last_error = error
                time.sleep(0.05)
        raise TimeoutError(
            f"Timed out waiting for synchronized direct sensors: {last_error}")

    def _send_command(
            self,
            endpoint: str,
            data: Dict[str, Any],
            accept_read_timeout: bool = False,
            ) -> None:
        try:
            response = self.session.post(
                f"{self.robot_url}{endpoint}",
                json=data,
                timeout=float(self.cfg.live.http_timeout_seconds),
            )
        except requests.exceptions.ReadTimeout:
            # FrankaHand calls are blocking in the existing server. A short
            # client timeout does not mean the already accepted motion failed.
            if accept_read_timeout:
                return
            raise
        response.raise_for_status()

    def send_tcp_pose(self, pose9d: np.ndarray) -> None:
        self._send_command(
            "/move_tcp/left",
            {"target_tcp": pose9d_to_pose7d(pose9d).tolist()},
        )

    def send_gripper_width(self, width: float, cfg: DictConfig) -> None:
        data = {
            "width": float(width),
            "velocity": float(cfg.velocity_mps),
            "force_limit": float(cfg.force_limit_n),
        }
        endpoint = "/move_gripper/left"
        if bool(cfg.close_with_force) and math.isclose(
                width, float(cfg.closed_width_m), abs_tol=1e-6):
            endpoint = "/move_gripper_force/left"
        self._send_command(endpoint, data, accept_read_timeout=True)

    def close(self) -> None:
        self._stop_event.set()
        self._state_thread.join(timeout=2.0)
        for thread in self._camera_threads:
            thread.join(timeout=2.0)
        for pipeline in self._camera_pipelines.values():
            pipeline.stop()
        self.session.close()


def _write_pickle_frame(stream: Any, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!I", len(data)) + data)
    stream.flush()


def _read_pickle_frame(stream: Any) -> Any:
    header = stream.read(4)
    if len(header) != 4:
        raise EOFError("Incomplete bridge request header")
    (size,) = struct.unpack("!I", header)
    body = stream.read(size)
    if len(body) != size:
        raise EOFError("Incomplete bridge request body")
    return pickle.loads(body)


def _decode_compressed_bgr(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode a compressed camera image")
    return image


def build_live_env_observation(
        states: np.ndarray,
        images: Dict[str, list[bytes]],
        shape_meta: DictConfig,
        goal_pose_world_9d: Optional[np.ndarray] = None,
        ) -> Dict[str, np.ndarray]:
    """Map the bridge protocol into exactly the checkpoint's observation dict."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 13:
        raise ValueError(f"observation.state must be [T,13], got {states.shape}")
    if not np.isfinite(states).all():
        raise ValueError("observation.state contains NaN or infinite values")

    rgb_keys, lowdim_keys = observation_key_sets(shape_meta)
    missing_images = [key for key in rgb_keys if key not in images]
    if missing_images:
        raise ValueError(f"Bridge images are missing checkpoint keys: {missing_images}")
    for key in rgb_keys:
        if len(images[key]) != len(states):
            raise ValueError(
                f"State and {key} history lengths differ: "
                f"{len(states)} vs {len(images[key])}")

    pose_history = np.stack([
        pose7d_xyzw_to_pose9d(row[:7]) for row in states
    ]).astype(np.float32)
    observation: Dict[str, np.ndarray] = {
        key: np.stack([_decode_compressed_bgr(data) for data in images[key]])
        for key in rgb_keys
    }
    for key in lowdim_keys:
        if key == "left_robot_tcp_pose":
            observation[key] = pose_history
        elif key == "left_robot_tcp_wrench":
            observation[key] = states[:, 7:13].copy()
        elif key == "left_goal_pose_rel":
            if goal_pose_world_9d is None:
                raise ValueError(
                    "Checkpoint requires left_goal_pose_rel but no world goal was configured")
            goal = np.asarray(goal_pose_world_9d, dtype=np.float32).reshape(1, 9)
            relative_goal = absolute_pose9d_to_relative(
                goal, pose_history[-1])[0]
            # Training declares this key in single_step_obs_keys: the current
            # relative goal is repeated across the full observation window.
            observation[key] = np.repeat(
                relative_goal[None], len(states), axis=0)
        else:
            raise ValueError(
                f"No live observation mapping is defined for checkpoint key {key!r}")
    return observation


class _RemoteDesiredWorkerState:
    def __init__(
            self,
            policy: torch.nn.Module,
            train_cfg: DictConfig,
            deploy_cfg: DictConfig,
            output_dir: Path,
            instance_id: str,
            ) -> None:
        self.policy = policy
        self.train_cfg = train_cfg
        self.deploy_cfg = deploy_cfg
        self.output_dir = output_dir
        self.instance_id = instance_id
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.requests = 0
        self.actions_returned = 0
        self.first_observation_saved = False
        self.log_path = output_dir / "remote_desired_worker_events.jsonl"
        _, lowdim_keys = observation_key_sets(train_cfg.shape_meta)
        goal = deploy_cfg.task.get("goal_pose_world_9d")
        self.goal_pose_world_9d = (
            None if goal is None else np.asarray(goal, dtype=np.float32))
        if "left_goal_pose_rel" in lowdim_keys:
            if (self.goal_pose_world_9d is None
                    or self.goal_pose_world_9d.shape != (9,)
                    or not np.isfinite(self.goal_pose_world_9d).all()):
                raise ValueError(
                    "Goal-conditioned checkpoint requires a finite 9D world goal")

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if int(request.get("protocol_version", -1)) != 1:
            raise ValueError("RemoteDesired protocol version must be 1")
        states = np.asarray(request["observation.state"], dtype=np.float32)
        env_observation = build_live_env_observation(
            states=states,
            images=request.get("images", {}),
            shape_meta=self.train_cfg.shape_meta,
            goal_pose_world_9d=self.goal_pose_world_9d,
        )

        started = time.monotonic()
        with self.lock:
            observation, base_pose = prepare_policy_observation(
                env_observation, self.train_cfg.shape_meta)
            relative_actions = predict_relative_actions(
                self.policy, observation, self.deploy_cfg.get("inference_seed"))
        absolute_actions = relative_action_to_absolute(relative_actions, base_pose)
        remote = self.deploy_cfg.live.remote_desired
        chunk_start = remote.get("chunk_start_index", 0)
        selected_actions = select_action_chunk(
            absolute_actions, chunk_start, remote.chunk_size)
        actions = []
        remote_gripper_mode = str(self.deploy_cfg.gripper.get(
            "remote_desired_mode", "hold_current"))
        for action in selected_actions:
            command = pose9d_to_pose7d_xyzw(action[:9]).tolist()
            if len(action) == 10 and remote_gripper_mode in ("ros_float64_width", "franka_action"):
                command.append(gripper_command_to_width(
                    float(action[9]), self.deploy_cfg.gripper))
            actions.append(command)
        inference_time = time.monotonic() - started

        event = {
            "ft_bias": request.get("ft_bias"),
            "last_policy_wrench": states[-1, 7:13],
            "last_uncorrected_wrench": np.asarray(request.get("observation.state_uncorrected", states))[-1, 7:13],
            "inference_seed": self.deploy_cfg.get("inference_seed"),
            "num_inference_steps": int(self.policy.num_inference_steps),
            "time": time.time(),
            "sequence": int(request["sequence"]),
            "base_pose": base_pose,
            "first_relative_action": relative_actions[0],
            "first_absolute_action": absolute_actions[0],
            "chunk_start_index": chunk_start,
            "chunk_end_index_exclusive": chunk_start + len(selected_actions),
            "predicted_action_count": len(absolute_actions),
            "first_selected_relative_action": relative_actions[chunk_start],
            "first_selected_absolute_action": selected_actions[0],
            "action_count": len(actions),
            "inference_time_sec": inference_time,
        }
        with self.log_lock:
            if not self.first_observation_saved:
                np.savez_compressed(
                    self.output_dir / "remote_desired_first_observation.npz",
                    **env_observation,
                    uncorrected_wrench=np.asarray(request.get("observation.state_uncorrected", states))[:, 7:13],
                    predicted_relative_action=relative_actions,
                    predicted_absolute_action=absolute_actions,
                    selected_absolute_action=selected_actions,
                )
                self.first_observation_saved = True
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")
            self.requests += 1
            self.actions_returned += len(actions)
        return {
            "ok": True,
            "protocol_version": 1,
            "sequence": int(request["sequence"]),
            "action_representation": "absolute_pose",
            "chunk_start_index": chunk_start,
            "actions": actions,
            "inference_time_sec": inference_time,
        }


def _make_remote_desired_handler(state: _RemoteDesiredWorkerState) -> Any:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            try:
                request = _read_pickle_frame(self.rfile)
                if request.get("request_type") == "heartbeat":
                    if request.get("instance_id") != state.instance_id:
                        raise ValueError("Bridge instance ID mismatch")
                    response = {"ok": True, "protocol_version": 1}
                elif request.get("request_type") == "predict":
                    response = state.predict(request)
                else:
                    raise ValueError("Unknown bridge request type")
            except Exception as error:  # noqa: BLE001
                response = {"ok": False, "error": str(error)}
            _write_pickle_frame(self.wfile, response)

    return Handler


class _ThreadingWorkerServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _remote_desired_bridge_payload(
        policy: torch.nn.Module,
        deploy_cfg: DictConfig,
        mode: str,
        train_cfg: Optional[DictConfig] = None,
        ) -> Dict[str, Any]:
    remote = deploy_cfg.live.remote_desired
    validate_chunk_window(remote.get("chunk_start_index", 0), remote.chunk_size,
                          getattr(policy, "n_action_steps", None))
    ft_settings = validate_ft_bias_configuration(deploy_cfg)
    topics = OmegaConf.to_container(remote.topics, resolve=True)
    if train_cfg is None:
        rgb_keys = tuple(str(key) for key in deploy_cfg.task.get(
            "rgb_obs_keys", ("external_img", "left_wrist_img")))
        expected_keys = tuple(str(key) for key in deploy_cfg.task.get(
            "expected_observation_keys", ()))
    else:
        rgb_keys, _ = observation_key_sets(train_cfg.shape_meta)
        expected_keys = tuple(str(key) for key in train_cfg.shape_meta.obs.keys())
    camera_topics = {}
    for key in rgb_keys:
        if not topics.get(key):
            raise ValueError(f"No ROS camera topic is configured for RGB key {key!r}")
        camera_topics[key] = topics[key]

    wrench_filter_cfg = remote.get("wrench_filter", {})
    wrench_filter_enabled = bool(wrench_filter_cfg.get("enabled", False))
    uses_wrench = "left_robot_tcp_wrench" in expected_keys
    if wrench_filter_enabled and not uses_wrench:
        raise ValueError(
            "Wrench filtering was enabled for a checkpoint without a wrench input")
    wrench_filter_payload: Dict[str, Any] = {"enabled": False}
    if wrench_filter_enabled:
        from scipy.signal import butter, lfilter_zi

        cutoff_hz = float(wrench_filter_cfg["cutoff_hz"])
        sample_rate_hz = float(wrench_filter_cfg["sample_rate_hz"])
        order = int(wrench_filter_cfg.get("order", 2))
        if cutoff_hz <= 0 or cutoff_hz >= sample_rate_hz / 2:
            raise ValueError(
                "Wrench LPF cutoff must be positive and below Nyquist")
        if order <= 0:
            raise ValueError("Wrench LPF order must be positive")
        b, a = butter(
            order, cutoff_hz, btype="low", fs=sample_rate_hz)
        wrench_filter_payload = {
            "enabled": True,
            "cutoff_hz": cutoff_hz,
            "order": order,
            "sample_rate_hz": sample_rate_hz,
            "resample": bool(wrench_filter_cfg.get("resample", True)),
            "b": b.tolist(),
            "a": a.tolist(),
            "zi": lfilter_zi(b, a).tolist(),
        }
    safety = {
        "require_enable": bool(remote.require_enable),
        "allow_real_robot": bool(deploy_cfg.live.execute_enabled),
        "workspace_min": deploy_cfg.safety.workspace_min,
        "workspace_max": deploy_cfg.safety.workspace_max,
        "max_translation_step_m": deploy_cfg.safety.max_translation_step_m,
        "max_rotation_step_rad": deploy_cfg.safety.max_rotation_step_rad,
        "max_force_n": deploy_cfg.safety.max_force_n,
        "max_torque_nm": deploy_cfg.safety.max_torque_nm,
    }
    if mode == "shadow":
        # Shadow output cannot actuate the robot and remains useful before the
        # task-specific execution envelope has been approved.
        for key in (
                "workspace_min", "workspace_max", "max_translation_step_m",
                "max_rotation_step_rad", "max_force_n", "max_torque_nm"):
            safety[key] = None
    return {
        "observation": {
            "obs_steps": int(policy.n_obs_steps),
            "obs_downsample_ratio": int(deploy_cfg.live.obs_downsample_ratio),
            "camera_topics": camera_topics,
            "wrench_filter": wrench_filter_payload,
        },
        "topics": topics,
        "ft_bias": vars(ft_settings),
        "timing": {
            "action_rate_hz": float(deploy_cfg.live.control_fps),
            "command_publish_rate_hz": float(remote.command_publish_rate_hz),
            "state_timeout_sec": float(remote.state_timeout_seconds),
            "image_timeout_sec": float(remote.image_timeout_seconds),
            "command_target_timeout_sec": float(
                remote.command_target_timeout_seconds),
            "worker_timeout_sec": float(remote.worker_timeout_seconds),
            "worker_heartbeat_period_sec": float(
                remote.worker_heartbeat_period_seconds),
            "worker_disconnect_timeout_sec": float(
                remote.worker_disconnect_timeout_seconds),
        },
        "action": {
            "chunk_size": int(remote.chunk_size),
            "chunk_start_index": remote.get("chunk_start_index", 0),
            "prefetch_steps": int(remote.prefetch_steps),
        },
        "safety": safety,
        "gripper": {
            **OmegaConf.to_container(deploy_cfg.gripper, resolve=True),
            "mode": (
                str(deploy_cfg.gripper.get("remote_desired_mode", "hold_current"))
                if int(deploy_cfg.task.action_dim) == 10 else "disabled"
            ),
            "ros_command_topic": (
                deploy_cfg.gripper.get("ros_command_topic")
                if int(deploy_cfg.task.action_dim) == 10 else None
            ),
        },
    }


def run_remote_desired_live(
        policy: torch.nn.Module,
        train_cfg: DictConfig,
        deploy_cfg: DictConfig,
        mode: str,
        duration_seconds: float,
        output_dir: Path,
        ) -> Dict[str, Any]:
    """Run DP inference here and the ROS bridge in its own Python environment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    remote = deploy_cfg.live.remote_desired
    # Validate policy-relative slicing before starting any background resources.
    bridge_payload = _remote_desired_bridge_payload(
        policy, deploy_cfg, mode, train_cfg=train_cfg)
    bridge_payload['ft_bias_report_path'] = str((output_dir/'ft_bias_calibration.json').resolve())
    bridge_script = Path(str(remote.bridge_script)).expanduser().resolve()
    bridge_python = Path(str(remote.bridge_python)).expanduser().resolve()
    instance_id = uuid.uuid4().hex
    pid_file = Path("/tmp/dp_p2p_remote_desired") / f"bridge-{instance_id}.json"
    state = _RemoteDesiredWorkerState(
        policy, train_cfg, deploy_cfg, output_dir, instance_id)
    server = _ThreadingWorkerServer(
        (str(remote.worker_host), 0), _make_remote_desired_handler(state))
    worker_port = int(server.server_address[1])
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    bridge_command = [
        str(bridge_python), str(bridge_script),
        "--worker-host", str(remote.worker_host),
        "--worker-port", str(worker_port),
        "--instance-id", instance_id,
        "--pid-file", str(pid_file),
        "--real-robot",
        "--publish" if mode == "execute" else "--echo-only",
        "--duration-sec", str(duration_seconds),
    ]
    if remote.get("ros_setup_file"):
        command = [
            "/bin/bash", "-c",
            'set -e; source "$1"; shift; exec "$@"',
            "dp-p2p-bridge",
            str(Path(str(remote.ros_setup_file)).expanduser().resolve()),
            *bridge_command,
        ]
    else:
        command = bridge_command
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["DP_P2P_BRIDGE_CONFIG_JSON"] = json.dumps(
        _jsonable(bridge_payload),
        separators=(",", ":"),
    )
    repository_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = repository_root + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else "")

    started = time.monotonic()
    process: Optional[subprocess.Popen] = None
    return_code: Optional[int] = None
    try:
        process = subprocess.Popen(command, env=environment)
        try:
            return_code = process.wait(timeout=max(10.0, duration_seconds + 5.0))
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "RemoteDesired bridge did not stop after its requested duration"
            ) from error
        if return_code != 0:
            raise RuntimeError(
                f"RemoteDesired bridge exited with code {return_code}")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        server.shutdown()
        server.server_close()
        try:
            pid_file.unlink()
        except OSError:
            pass

    report = {
        "mode": mode,
        "task": str(deploy_cfg.task.name),
        "backend": "remote_desired",
        "chunk_start_index": remote.get("chunk_start_index", 0),
        "chunk_size": int(remote.chunk_size),
        "prefetch_steps": int(remote.prefetch_steps),
        "inference_seed": deploy_cfg.get("inference_seed"),
        "num_inference_steps": int(policy.num_inference_steps),
        "checkpoint": str(Path(deploy_cfg.checkpoint).expanduser().resolve()),
        "duration_seconds": time.monotonic() - started,
        "policy_requests": state.requests,
        "actions_returned": state.actions_returned,
        "command_topic": str(
            remote.topics.remote_desired
            if mode == "execute" else remote.topics.remote_desired_shadow),
        "worker_event_log": str(state.log_path),
        "bridge_return_code": return_code,
        "warning": str(deploy_cfg.checkpoint_warning),
    }
    write_json(output_dir / f"{mode}_report.json", report)
    return report


def run_live(
        policy: torch.nn.Module,
        train_cfg: DictConfig,
        deploy_cfg: DictConfig,
        mode: str,
        duration_seconds: float,
        output_dir: Path,
        confirmation: Optional[str] = None,
        ) -> Dict[str, Any]:
    if mode not in ("shadow", "execute"):
        raise ValueError(f"Unsupported live mode: {mode}")
    if mode == "execute":
        validate_execute_configuration(deploy_cfg, confirmation)

    action_dim = int(train_cfg.shape_meta.action.shape[0])
    if action_dim != int(deploy_cfg.task.action_dim):
        raise RuntimeError(
            f"Checkpoint action_dim={action_dim} does not match deployment "
            f"config action_dim={deploy_cfg.task.action_dim}")

    backend = str(deploy_cfg.live.get("backend", "direct"))
    if backend == "remote_desired":
        return run_remote_desired_live(
            policy=policy,
            train_cfg=train_cfg,
            deploy_cfg=deploy_cfg,
            mode=mode,
            duration_seconds=duration_seconds,
            output_dir=output_dir,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{mode}_events.jsonl"
    backend = str(deploy_cfg.live.get("backend", "direct"))
    if backend == "direct":
        interface = DirectFrankaInterface(deploy_cfg)
    elif backend == "rdp_ros":
        interface = RDPFrankaInterface(deploy_cfg)
    else:
        raise ValueError(
            f"Unknown live.backend={backend!r}; expected 'direct', 'rdp_ros', "
            "or 'remote_desired'")
    started = time.monotonic()
    iterations = 0
    commands_sent = 0
    last_gripper_width: Optional[float] = None
    first_observation_saved = False

    try:
        with log_path.open("a", encoding="utf-8") as log_stream:
            while time.monotonic() - started < duration_seconds:
                loop_started = time.monotonic()
                env_observation = interface.get_observation(
                    obs_steps=int(policy.n_obs_steps),
                    temporal_downsample_ratio=int(deploy_cfg.live.obs_downsample_ratio),
                    timeout_seconds=float(deploy_cfg.live.observation_timeout_seconds),
                )
                policy_observation, base_pose = prepare_policy_observation(
                    env_observation, train_cfg.shape_meta)
                relative_actions = predict_relative_actions(
                    policy, policy_observation, deploy_cfg.get("inference_seed"))
                absolute_actions = relative_action_to_absolute(relative_actions, base_pose)
                latest_wrench = np.asarray(
                    env_observation["left_robot_tcp_wrench"][-1], dtype=np.float32)

                if not first_observation_saved:
                    np.savez_compressed(
                        output_dir / f"{mode}_first_observation.npz",
                        **{key: np.asarray(value) for key, value in env_observation.items()
                           if key in train_cfg.shape_meta.obs},
                        predicted_relative_action=relative_actions,
                        predicted_absolute_action=absolute_actions,
                    )
                    first_observation_saved = True

                event: Dict[str, Any] = {
                    "time": time.time(),
                    "iteration": iterations,
                    "mode": mode,
                    "base_pose": base_pose,
                    "latest_wrench": latest_wrench,
                    "first_relative_action": relative_actions[0],
                    "first_absolute_action": absolute_actions[0],
                }

                if mode == "execute":
                    count = min(
                        int(deploy_cfg.live.actions_per_inference),
                        len(absolute_actions),
                    )
                    previous_pose = base_pose.copy()
                    step_metrics = []
                    for step_index in range(count):
                        action = absolute_actions[step_index]
                        metrics = validate_action_step(
                            action[:9], previous_pose, latest_wrench,
                            deploy_cfg.safety)
                        interface.send_tcp_pose(action[:9])
                        if action_dim == 10:
                            width = gripper_command_to_width(
                                action[9], deploy_cfg.gripper)
                            if last_gripper_width is None or abs(
                                    width - last_gripper_width) >= float(
                                        deploy_cfg.gripper.width_precision_m):
                                interface.send_gripper_width(width, deploy_cfg.gripper)
                                last_gripper_width = width
                        previous_pose = action[:9].copy()
                        commands_sent += 1
                        step_metrics.append(metrics)
                        if step_index + 1 < count:
                            time.sleep(1.0 / float(deploy_cfg.live.control_fps))
                    event["safety"] = step_metrics
                    event["commands_sent"] = count
                else:
                    event["commands_sent"] = 0

                log_stream.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")
                log_stream.flush()
                iterations += 1
                elapsed = time.monotonic() - loop_started
                time.sleep(max(0.0, 1.0 / float(deploy_cfg.live.inference_fps) - elapsed))
    finally:
        # The Franka server exposes no TCP-stop endpoint; ceasing waypoints is
        # the only software action available here. The physical E-stop must be
        # held by the operator throughout execution.
        interface.close()

    report = {
        "mode": mode,
        "task": str(deploy_cfg.task.name),
        "checkpoint": str(Path(deploy_cfg.checkpoint).expanduser().resolve()),
        "duration_seconds": time.monotonic() - started,
        "iterations": iterations,
        "inference_seed": deploy_cfg.get("inference_seed"),
        "num_inference_steps": int(policy.num_inference_steps),
        "commands_sent": commands_sent,
        "event_log": str(log_path),
        "warning": str(deploy_cfg.checkpoint_warning),
    }
    write_json(output_dir / f"{mode}_report.json", report)
    return report
