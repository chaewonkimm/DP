import unittest
from pathlib import Path
import sys

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diffusion_policy.workspace.train_diffusion_unet_image_workspace  # noqa: F401,E402
from scripts.build_next_state_action_sidecar import next_state_actions  # noqa: E402


CONFIG_DIR = str(
    Path(__file__).resolve().parents[1] / "diffusion_policy" / "config")


def compose_config(name: str):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    return cfg


class TestNextStateActionBuilder(unittest.TestCase):
    def test_shift_is_episode_local_and_terminal_pose_is_held(self):
        pose = np.zeros((5, 9), dtype=np.float32)
        pose[:, 0] = np.arange(5)
        actions = next_state_actions(pose, np.asarray([3, 5]))
        np.testing.assert_array_equal(actions[:, 0], [1, 2, 2, 4, 4])

    def test_invalid_episode_boundaries_are_rejected(self):
        pose = np.zeros((5, 9), dtype=np.float32)
        with self.assertRaises(ValueError):
            next_state_actions(pose, np.asarray([3, 4]))


class TestNextStateTrainingConfigs(unittest.TestCase):
    CONFIGS = {
        "train_diffusion_unet_peg_proprio_next_state_workspace": {
            "task_name": "peg_proprio_goal_next_state",
            "obs": {"left_robot_tcp_pose", "left_goal_pose_rel"},
            "uses_sidecar": True,
        },
        "train_diffusion_unet_peg_proprio_wrench_next_state_workspace": {
            "task_name": "peg_proprio_goal_wrench_next_state",
            "obs": {
                "left_robot_tcp_pose",
                "left_goal_pose_rel",
                "left_robot_tcp_wrench",
            },
            "uses_sidecar": True,
        },
        "train_diffusion_unet_valve_insertion_p2p_image_next_state_workspace": {
            "task_name": "valve_insertion_p2p_image_next_state",
            "obs": {
                "external_img",
                "left_wrist_img",
                "left_robot_tcp_pose",
                "left_robot_tcp_wrench",
            },
            "uses_sidecar": False,
        },
    }

    def test_all_variants_keep_the_common_schedule(self):
        for config_name, expected in self.CONFIGS.items():
            with self.subTest(config=config_name):
                cfg = compose_config(config_name)
                self.assertEqual(cfg.task.name, expected["task_name"])
                self.assertEqual(
                    set(cfg.task.shape_meta.obs.keys()), expected["obs"])
                self.assertEqual(list(cfg.task.shape_meta.action.shape), [9])
                self.assertEqual(cfg.horizon, 32)
                self.assertEqual(cfg.dataset_obs_steps, 4)
                self.assertEqual(cfg.n_obs_steps, 2)
                self.assertEqual(cfg.n_action_steps, 29)
                self.assertEqual(cfg.policy.action_start_offset, 3)
                self.assertEqual(cfg.training.num_epochs, 800)
                self.assertTrue(cfg.training.use_ema)
                self.assertTrue(cfg.training.validation_use_ema)
                self.assertEqual(
                    list(cfg.checkpoint.milestone_epochs),
                    [200, 400, 600, 800],
                )
                self.assertIn("next_state", list(cfg.logging.tags))
                if expected["uses_sidecar"]:
                    self.assertIn("action_dataset_path", cfg.task)
                    self.assertIn("action_dataset_path", cfg.task.dataset)
                else:
                    self.assertNotIn("action_dataset_path", cfg.task)
                    self.assertNotIn("action_dataset_path", cfg.task.dataset)

    def test_next_state_variants_do_not_change_observation_sources(self):
        desired_peg = compose_config(
            "train_diffusion_unet_peg_proprio_workspace")
        next_peg = compose_config(
            "train_diffusion_unet_peg_proprio_next_state_workspace")
        self.assertEqual(
            desired_peg.task.dataset_path, next_peg.task.dataset_path)
        self.assertEqual(
            desired_peg.task.shape_meta.obs, next_peg.task.shape_meta.obs)

        desired_valve = compose_config(
            "train_diffusion_unet_valve_insertion_p2p_image_workspace")
        next_valve = compose_config(
            "train_diffusion_unet_valve_insertion_p2p_image_next_state_workspace")
        self.assertEqual(
            desired_valve.task.dataset_path, next_valve.task.dataset_path)
        self.assertEqual(
            desired_valve.task.shape_meta.obs, next_valve.task.shape_meta.obs)
        self.assertIn("action_dataset_path", desired_valve.task)
        self.assertNotIn("action_dataset_path", next_valve.task)


if __name__ == "__main__":
    unittest.main()
