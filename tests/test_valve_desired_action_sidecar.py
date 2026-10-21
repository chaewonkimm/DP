import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from diffusion_policy.dataset.p2p_image_dataset import P2PImageDataset
from scripts.build_valve_insertion_desired_actions import (
    causal_match_indices,
    pose7d_to_pose9d,
)


def _write_buffer(path: Path, pose: np.ndarray, action: np.ndarray,
                  episode_ends: np.ndarray) -> None:
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.create_dataset("left_robot_tcp_pose", data=pose)
    data.create_dataset("action", data=action)
    meta.create_dataset("episode_ends", data=episode_ends, compressor=None)


class TestValveDesiredActionSidecar(unittest.TestCase):
    def test_causal_matching_never_selects_a_future_sample(self):
        source = np.asarray([0.0, 0.01, 0.02, 0.03])
        target = np.asarray([0.0, 0.009, 0.01, 0.025, 0.03])
        indices = causal_match_indices(source, target)
        np.testing.assert_array_equal(indices, [0, 0, 1, 2, 3])
        self.assertTrue(np.all(source[indices] <= target))

    def test_pose_conversion_uses_tcp_pose9d_convention(self):
        pose = pose7d_to_pose9d(
            np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(
            pose, [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_dataset_replaces_only_action_with_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = 6
            pose = np.zeros((count, 9), dtype=np.float32)
            source_action = np.zeros((count, 9), dtype=np.float32)
            desired_action = np.full((count, 9), 7.0, dtype=np.float32)
            episode_ends = np.asarray([count], dtype=np.int64)
            observation_path = root / "observations.zarr"
            action_path = root / "desired.zarr"
            _write_buffer(
                observation_path, pose, source_action, episode_ends)
            _write_buffer(action_path, pose, desired_action, episode_ends)

            dataset = P2PImageDataset(
                shape_meta={
                    "obs": {
                        "left_robot_tcp_pose": {
                            "shape": [9],
                            "type": "low_dim",
                        },
                    },
                    "action": {"shape": [9]},
                },
                dataset_path=str(observation_path),
                action_dataset_path=str(action_path),
                horizon=2,
                n_obs_steps=2,
                obs_temporal_downsample_ratio=1,
                val_ratio=0.0,
                relative_action=False,
                load_to_memory=False,
            )

            np.testing.assert_array_equal(
                dataset.replay_buffer["left_robot_tcp_pose"][:], pose)
            np.testing.assert_array_equal(
                dataset.replay_buffer["action"][:], desired_action)


if __name__ == "__main__":
    unittest.main()
