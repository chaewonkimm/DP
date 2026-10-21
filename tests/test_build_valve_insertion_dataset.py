import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_valve_insertion_dataset import find_episode_crops


class TestValveInsertionCrop(unittest.TestCase):
    def test_find_episode_crops_after_settle_margin(self):
        first = np.r_[-np.ones(5), np.ones(9)]
        second = np.r_[-np.ones(7), np.ones(10)]
        commands = np.r_[first, second].astype(np.float32)
        crops = find_episode_crops(
            gripper_commands=commands,
            episode_ends=np.array([len(first), len(first) + len(second)]),
            settle_frames=2,
            minimum_frames=3,
        )

        self.assertEqual([crop.close_offset for crop in crops], [5, 7])
        self.assertEqual([crop.crop_start_offset for crop in crops], [7, 9])
        self.assertEqual([crop.output_length for crop in crops], [7, 8])
        self.assertEqual(crops[1].source_crop_start, len(first) + 9)

    def test_find_episode_crops_rejects_reopen(self):
        commands = np.array(
            [-1, -1, 1, 1, -1, 1, 1, 1], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_episode_crops(
                gripper_commands=commands,
                episode_ends=np.array([len(commands)]),
                settle_frames=0,
                minimum_frames=1,
            )


if __name__ == "__main__":
    unittest.main()
