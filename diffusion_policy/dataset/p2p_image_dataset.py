from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def _array_to_stats(array: np.ndarray) -> Dict[str, np.ndarray]:
    array = np.asarray(array, dtype=np.float32)
    return {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
    }


def _concatenate_normalizers(
        normalizers: Sequence[SingleFieldLinearNormalizer]
        ) -> SingleFieldLinearNormalizer:
    scale = torch.cat(
        [normalizer.params_dict["scale"] for normalizer in normalizers], dim=-1)
    offset = torch.cat(
        [normalizer.params_dict["offset"] for normalizer in normalizers], dim=-1)
    stats = {}
    for key in ("min", "max", "mean", "std"):
        stats[key] = torch.cat([
            normalizer.params_dict["input_stats"][key]
            for normalizer in normalizers
        ], dim=-1)
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stats,
    )


def _pose9d_normalizer(poses: np.ndarray) -> SingleFieldLinearNormalizer:
    """Match RDP normalization: scale XYZ to [-1, 1], keep rotation-6D as-is."""
    poses = np.asarray(poses, dtype=np.float32).reshape(-1, 9)
    position = get_range_normalizer_from_stat(_array_to_stats(poses[:, :3]))
    rotation = get_identity_normalizer_from_stat(_array_to_stats(poses[:, 3:9]))
    return _concatenate_normalizers([position, rotation])


def _action_normalizer(
        actions: np.ndarray,
        normalize_gripper_action: bool,
        ) -> SingleFieldLinearNormalizer:
    """Normalize a 9D TCP action, optionally followed by a gripper command."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] not in (9, 10):
        raise ValueError(f"Expected 9D or 10D actions, got {actions.shape}")

    normalizers = [_pose9d_normalizer(actions[..., :9])]
    if actions.shape[-1] == 10:
        gripper_stats = _array_to_stats(actions[..., 9:10].reshape(-1, 1))
        if normalize_gripper_action:
            gripper = get_range_normalizer_from_stat(gripper_stats)
        else:
            gripper = get_identity_normalizer_from_stat(gripper_stats)
        normalizers.append(gripper)
    return _concatenate_normalizers(normalizers)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(vector, axis=1, keepdims=True)
    return vector / np.maximum(magnitude, 1e-8)


def _ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    """Convert RDP's two-column rotation-6D representation to rotation matrices."""
    x = _normalize_vector(ortho6d[:, 0:3])
    z = _normalize_vector(np.cross(x, ortho6d[:, 3:6]))
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=2)


def _pose9d_to_matrix(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1] != 9:
        raise ValueError(f"Expected (N, 9) pose array, got {poses.shape}")
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(poses), axis=0)
    matrices[:, :3, 3] = poses[:, :3]
    matrices[:, :3, :3] = _ortho6d_to_rotation_matrix(poses[:, 3:9])
    return matrices


def _matrix_to_pose9d(matrices: np.ndarray) -> np.ndarray:
    result = np.empty((len(matrices), 9), dtype=np.float64)
    result[:, :3] = matrices[:, :3, 3]
    result[:, 3:9] = matrices[:, :3, :2].swapaxes(1, 2).reshape(len(matrices), 6)
    return result


def absolute_pose9d_to_relative(
        poses: np.ndarray,
        base_pose: np.ndarray,
        ) -> np.ndarray:
    """Express an absolute pose trajectory in the coordinate frame of base_pose."""
    poses = np.asarray(poses, dtype=np.float64)
    base_pose = np.asarray(base_pose, dtype=np.float64).reshape(1, 9)
    base_inverse = np.linalg.inv(_pose9d_to_matrix(base_pose))[0]
    relative_matrices = base_inverse[None] @ _pose9d_to_matrix(poses)
    return _matrix_to_pose9d(relative_matrices).astype(np.float32)


def _resolve_replay_buffer_path(path: str | Path) -> Path:
    path = Path(os.path.expanduser(str(path)))
    if (path / "replay_buffer.zarr").is_dir():
        return path / "replay_buffer.zarr"
    if (path / ".zgroup").is_file():
        return path
    raise FileNotFoundError(
        f"Expected a Zarr group or replay_buffer.zarr under {path}")


def absolute_action_to_relative(
        actions: np.ndarray,
        base_pose: np.ndarray,
        ) -> np.ndarray:
    """Make the TCP portion relative while preserving an optional gripper."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] not in (9, 10):
        raise ValueError(f"Expected (N, 9) or (N, 10) actions, got {actions.shape}")
    result = actions.copy()
    result[:, :9] = absolute_pose9d_to_relative(actions[:, :9], base_pose)
    return result


def relative_action_to_absolute(
        actions: np.ndarray,
        base_pose: np.ndarray,
        ) -> np.ndarray:
    """Restore relative TCP actions to the robot-base frame.

    This is the inverse used at real-robot deployment time.  As in
    :func:`absolute_action_to_relative`, an optional tenth gripper command is
    copied without modification because it is not part of the SE(3) pose.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] not in (9, 10):
        raise ValueError(f"Expected (N, 9) or (N, 10) actions, got {actions.shape}")

    base_pose = np.asarray(base_pose, dtype=np.float64).reshape(1, 9)
    base_matrix = _pose9d_to_matrix(base_pose)[0]
    result = actions.copy()
    absolute_matrices = base_matrix[None] @ _pose9d_to_matrix(actions[:, :9])
    result[:, :9] = _matrix_to_pose9d(absolute_matrices).astype(np.float32)
    return result


class P2PImageDataset(BaseImageDataset):
    """DP adapter for single-arm p2p ReplayBuffers.

    The on-disk buffer remains untouched. This adapter reproduces the RDP data
    semantics used by ``real_square_peg_wrench_at_30fps``:

    * two RGB observations (external and wrist cameras),
    * 9D TCP pose and 6D wrench observations,
    * temporally downsampled observation history,
    * 9D TCP actions expressed relative to the latest observation,
    * an optional absolute 1D gripper command appended to each action,
    * episode-level train/validation splitting.

    RDP's ``extended_obs`` is deliberately omitted because standard Diffusion
    Policy has no corresponding input path.
    """

    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            horizon: int = 32,
            pad_before: int = 0,
            pad_after: int = 0,
            n_obs_steps: int = 4,
            obs_temporal_downsample_ratio: int = 2,
            n_latency_steps: int = 0,
            seed: int = 42,
            val_ratio: float = 0.1,
            max_train_episodes: Optional[int] = None,
            relative_action: bool = True,
            relative_tcp_obs_for_relative_action: bool = True,
            normalize_gripper_action: bool = True,
            load_to_memory: bool = False,
            tcp_pose_key: str = "left_robot_tcp_pose",
            single_step_obs_keys: Optional[Sequence[str]] = None,
            action_dataset_path: Optional[str] = None,
            ):
        super().__init__()

        if horizon < 1:
            raise ValueError("horizon must be positive")
        if n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive")
        if obs_temporal_downsample_ratio < 1:
            raise ValueError("obs_temporal_downsample_ratio must be positive")
        if n_obs_steps % obs_temporal_downsample_ratio != 0:
            raise ValueError(
                "n_obs_steps must be divisible by obs_temporal_downsample_ratio")

        zarr_path = _resolve_replay_buffer_path(dataset_path)

        obs_shape_meta = shape_meta["obs"]
        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attributes in obs_shape_meta.items():
            modality = attributes.get("type", "low_dim")
            if modality == "rgb":
                self.rgb_keys.append(key)
            elif modality == "low_dim":
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported observation type {modality!r} for {key}")

        self.single_step_obs_keys = frozenset(single_step_obs_keys or [])
        unknown_single_step_keys = self.single_step_obs_keys.difference(
            self.lowdim_keys)
        if unknown_single_step_keys:
            raise ValueError(
                "single_step_obs_keys must be low-dimensional observation "
                f"keys, got unknown keys: {sorted(unknown_single_step_keys)}")

        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.tcp_pose_key = tcp_pose_key
        if relative_action:
            if self.action_dim not in (9, 10):
                raise ValueError(
                    "Relative p2p actions require action shape [9] or [10]")
            if tcp_pose_key not in self.lowdim_keys:
                raise ValueError(
                    f"relative_action requires {tcp_pose_key!r} in low-dimensional observations")

        observation_keys = list(dict.fromkeys(self.rgb_keys + self.lowdim_keys))
        if action_dataset_path is None:
            load_keys = observation_keys + ["action"]
            if load_to_memory:
                replay_buffer = ReplayBuffer.copy_from_path(
                    str(zarr_path), keys=load_keys)
            else:
                replay_buffer = ReplayBuffer.create_from_path(
                    str(zarr_path), mode="r")
        else:
            action_zarr_path = _resolve_replay_buffer_path(action_dataset_path)
            source_buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="r")
            action_buffer = ReplayBuffer.create_from_path(
                str(action_zarr_path), mode="r")
            if "action" not in action_buffer:
                raise KeyError(
                    f"Action override {action_zarr_path} has no 'action' array")
            source_ends = np.asarray(source_buffer.episode_ends[:])
            action_ends = np.asarray(action_buffer.episode_ends[:])
            if not np.array_equal(source_ends, action_ends):
                raise ValueError(
                    "Observation and action-override episode boundaries differ")

            if load_to_memory:
                replay_buffer = ReplayBuffer.copy_from_path(
                    str(zarr_path), keys=observation_keys)
                replay_buffer.data["action"] = np.asarray(
                    action_buffer["action"][:])
            else:
                replay_buffer = ReplayBuffer(root={
                    "data": {
                        **{key: source_buffer[key] for key in observation_keys},
                        "action": action_buffer["action"],
                    },
                    "meta": {"episode_ends": source_buffer.episode_ends},
                })

        self._validate_shapes(replay_buffer, shape_meta)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.horizon = int(horizon)
        self.n_latency_steps = int(n_latency_steps)
        self.sequence_length = self.horizon + self.n_latency_steps
        self.n_raw_obs_steps = int(n_obs_steps)
        self.obs_downsample_ratio = int(obs_temporal_downsample_ratio)
        self.n_obs_steps = self.n_raw_obs_steps // self.obs_downsample_ratio
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.relative_action = bool(relative_action)
        self.relative_tcp_obs_for_relative_action = bool(
            relative_tcp_obs_for_relative_action)
        self.normalize_gripper_action = bool(normalize_gripper_action)
        self.action_dataset_path = action_dataset_path
        self.val_mask = val_mask
        self.train_mask = train_mask
        self._normalizer = None

        self.sampler = self._make_sampler(train_mask)

    def _validate_shapes(self, replay_buffer: ReplayBuffer, shape_meta: dict) -> None:
        available = set(replay_buffer.keys())
        required = set(self.rgb_keys + self.lowdim_keys + ["action"])
        missing = sorted(required - available)
        if missing:
            raise KeyError(f"ReplayBuffer is missing required fields: {missing}")

        for key in self.rgb_keys:
            channels, height, width = tuple(shape_meta["obs"][key]["shape"])
            actual = tuple(replay_buffer[key].shape[1:])
            expected = (height, width, channels)
            if actual != expected:
                raise ValueError(f"{key} has disk shape {actual}, expected {expected}")
        for key in self.lowdim_keys:
            expected = tuple(shape_meta["obs"][key]["shape"])
            actual = tuple(replay_buffer[key].shape[1:])
            if actual != expected:
                raise ValueError(f"{key} has shape {actual}, expected {expected}")
        action_shape = tuple(replay_buffer["action"].shape[1:])
        expected_action_shape = tuple(shape_meta["action"]["shape"])
        if action_shape != expected_action_shape:
            raise ValueError(
                f"action has shape {action_shape}, expected {expected_action_shape}")

    def _make_sampler(self, episode_mask: np.ndarray) -> SequenceSampler:
        key_first_k = {
            key: self.n_raw_obs_steps
            for key in self.rgb_keys + self.lowdim_keys
        }
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.sequence_length,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
            keys=self.rgb_keys + self.lowdim_keys + ["action"],
            key_first_k=key_first_k,
        )

    def get_validation_dataset(self):
        validation_set = copy.copy(self)
        validation_set.train_mask = self.val_mask.copy()
        validation_set.sampler = validation_set._make_sampler(self.val_mask)
        validation_set._normalizer = None
        return validation_set

    def _select_observations(self, array: np.ndarray) -> np.ndarray:
        # Right-aligned selection matches RDP's x[:K][::-ratio][::-1].
        selected = array[:self.n_raw_obs_steps]
        selected = selected[::-self.obs_downsample_ratio][::-1]
        if len(selected) != self.n_obs_steps:
            raise RuntimeError(
                f"Expected {self.n_obs_steps} observations, selected {len(selected)}")
        return np.ascontiguousarray(selected)

    def _convert_lowdim_sample(
            self,
            pose_sequence: np.ndarray,
            action_sequence: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:
        pose = self._select_observations(pose_sequence).astype(np.float32)
        action = action_sequence.astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        if self.relative_action:
            base_pose = pose[-1]
            action = absolute_action_to_relative(action, base_pose)
            if self.relative_tcp_obs_for_relative_action:
                pose = absolute_pose9d_to_relative(pose, base_pose)
        return pose, action

    def _sample_array_from_index(
            self,
            array,
            index: np.ndarray,
            ) -> np.ndarray:
        """Sample a small low-dimensional array using SequenceSampler's rules."""
        buffer_start, buffer_end, sample_start, sample_end = map(int, index)
        sample = np.asarray(array[buffer_start:buffer_end])
        if sample_start == 0 and sample_end == self.sequence_length:
            return sample

        result = np.empty(
            (self.sequence_length,) + tuple(array.shape[1:]), dtype=array.dtype)
        if sample_start > 0:
            result[:sample_start] = sample[0]
        result[sample_start:sample_end] = sample
        if sample_end < self.sequence_length:
            result[sample_end:] = sample[-1]
        return result

    def _relative_normalizer_data(self) -> tuple[np.ndarray, np.ndarray]:
        indices = self.sampler.indices
        pose_array = self.replay_buffer[self.tcp_pose_key]
        action_array = self.replay_buffer["action"]
        poses = np.empty(
            (len(indices), self.n_obs_steps, 9), dtype=np.float32)
        actions = np.empty(
            (len(indices), self.horizon, self.action_dim), dtype=np.float32)

        for i, index in enumerate(indices):
            pose_sequence = self._sample_array_from_index(pose_array, index)
            action_sequence = self._sample_array_from_index(action_array, index)
            poses[i], actions[i] = self._convert_lowdim_sample(
                pose_sequence=pose_sequence,
                action_sequence=action_sequence,
            )
        return poses, actions

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if self._normalizer is not None:
            return self._normalizer

        normalizer = LinearNormalizer()
        if self.relative_action:
            relative_pose, relative_action = self._relative_normalizer_data()
            normalizer[self.tcp_pose_key] = _pose9d_normalizer(relative_pose)
            normalizer["action"] = _action_normalizer(
                relative_action,
                normalize_gripper_action=self.normalize_gripper_action,
            )
        else:
            normalizer[self.tcp_pose_key] = _pose9d_normalizer(
                self.replay_buffer[self.tcp_pose_key][:])
            normalizer["action"] = _action_normalizer(
                self.replay_buffer["action"][:],
                normalize_gripper_action=self.normalize_gripper_action,
            )

        for key in self.lowdim_keys:
            if key == self.tcp_pose_key:
                continue
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key][:], **kwargs)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        self._normalizer = normalizer
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.replay_buffer["action"][:]))

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(index)

        obs = {}
        for key in self.rgb_keys:
            image = self._select_observations(sample[key])
            obs[key] = np.ascontiguousarray(
                np.moveaxis(image, -1, 1).astype(np.float32) / 255.0)
        for key in self.lowdim_keys:
            values = self._select_observations(sample[key]).astype(np.float32)
            if key in self.single_step_obs_keys:
                # Match RDP: use only the current value and broadcast it over
                # the observation window so every modality keeps the same T.
                values = np.repeat(values[-1:], self.n_obs_steps, axis=0)
            obs[key] = values

        pose, action = self._convert_lowdim_sample(
            pose_sequence=sample[self.tcp_pose_key],
            action_sequence=sample["action"],
        )
        obs[self.tcp_pose_key] = pose

        result = {
            "obs": obs,
            "action": action,
        }
        return dict_apply(result, torch.from_numpy)
