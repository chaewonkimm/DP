import unittest
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Registers the ${eval:...} resolver used by the training configs.
import diffusion_policy.workspace.train_diffusion_unet_image_workspace  # noqa: F401


CONFIG_DIR = str(
    Path(__file__).resolve().parents[1] / "diffusion_policy" / "config")


class TestValveInsertionCheckpointConfig(unittest.TestCase):
    def test_insertion_config_saves_train_and_validation_best(self):
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(
                config_name=(
                    "train_diffusion_unet_valve_insertion_p2p_image_workspace"))

        self.assertTrue(cfg.training.use_ema)
        self.assertTrue(cfg.training.validation_use_ema)
        self.assertEqual(cfg.training.num_epochs, 800)
        self.assertEqual(
            cfg.task.name, "valve_insertion_p2p_image_desired")
        self.assertTrue(
            cfg.task.action_dataset_path.endswith(
                "valve_insertion_after_grasp_desired_actions_30f"))
        self.assertEqual(
            set(cfg.task.shape_meta.obs.keys()),
            {
                "external_img",
                "left_wrist_img",
                "left_robot_tcp_pose",
                "left_robot_tcp_wrench",
            },
        )
        self.assertNotIn("goal", " ".join(cfg.task.shape_meta.obs.keys()).lower())
        self.assertEqual(list(cfg.task.shape_meta.action.shape), [9])
        self.assertEqual(cfg.checkpoint.topk.monitor_key, "train_loss")
        self.assertEqual(
            cfg.checkpoint.validation_topk.monitor_key, "val_loss")
        self.assertEqual(cfg.checkpoint.topk.k, 1)
        self.assertEqual(cfg.checkpoint.validation_topk.k, 1)
        self.assertEqual(
            list(cfg.checkpoint.milestone_epochs), [200, 400, 600, 800])
        self.assertEqual(
            cfg.checkpoint.milestone_format_str,
            "epoch={epoch:04d}-milestone.ckpt")
        self.assertTrue(cfg.checkpoint.save_last_ckpt)

    def test_base_config_keeps_original_checkpoint_behavior(self):
        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            cfg = compose(
                config_name="train_diffusion_unet_p2p_image_workspace")

        self.assertNotIn("validation_use_ema", cfg.training)
        self.assertNotIn("validation_topk", cfg.checkpoint)
        self.assertNotIn("milestone_epochs", cfg.checkpoint)
        self.assertEqual(cfg.checkpoint.topk.monitor_key, "train_loss")


if __name__ == "__main__":
    unittest.main()
