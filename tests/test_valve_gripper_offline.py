from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from omegaconf import OmegaConf

from scripts.check_valve_gripper_offline import (
    gripper_metrics, target_indices, evaluate_episode, save_results,
)
from scripts.reconstruct_valve_episode import previous_indices


def test_always_open_is_not_good_close_prediction():
    metrics = gripper_metrics([-1, -1, -1, 1], [-1, -1, -1, -1])
    assert metrics['accuracy'] == 0.75
    assert metrics['close_recall'] == 0
    assert metrics['balanced_accuracy'] == 0.5
    assert metrics['confusion']['true_close_pred_open'] == 1


def test_no_close_labels_is_not_perfect_close_recall():
    metrics = gripper_metrics([-1, -1], [-1, -1])
    assert metrics['close_recall'] is None
    assert metrics['balanced_accuracy'] is None
    assert gripper_metrics([1], [0])['close_recall'] == 1  # deployed >= threshold


def test_padding_and_latency_alignment():
    indices, valid = target_indices([10, 14, 2, 6], 0, 8)
    np.testing.assert_array_equal(indices, np.arange(8, 16))
    np.testing.assert_array_equal(valid, [False, False, True, True, True, True, False, False])
    indices, valid = target_indices([10, 14, 2, 6], 1, 4, latency=2)
    np.testing.assert_array_equal(indices, [11, 12, 13, 14])
    np.testing.assert_array_equal(valid, [True, True, True, False])


def test_previous_sample_never_uses_future():
    np.testing.assert_array_equal(previous_indices(np.array([10, 20, 30]), [10, 19, 25]), [0, 0, 1])
    with pytest.raises(ValueError):
        previous_indices(np.array([10, 20]), [9])


@pytest.mark.parametrize('latency', [0, 2])
@pytest.mark.parametrize('chunk_start', [0, 2])
def test_real_dataset_alignment_and_artifacts(tmp_path, monkeypatch, latency, chunk_start):
    from diffusion_policy.dataset.p2p_image_dataset import P2PImageDataset
    import diffusion_policy.real_world.p2p_franka_deployment as deployment
    count = 18
    pose = np.zeros((count, 9), dtype=np.float32)
    pose[:, 3], pose[:, 7] = 1, 1
    commands = np.where(np.arange(count) < 9, -1, 1).astype(np.float32)
    root = zarr.open_group(str(tmp_path/'recording'), mode='w')
    data, meta = root.create_group('data'), root.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array([count]))
    data.create_dataset('left_robot_tcp_pose', data=pose)
    data.create_dataset('action', data=np.c_[pose, commands])
    data.create_dataset('external_img', data=np.zeros((count, 2, 3, 3), dtype=np.uint8))
    shape = dict(obs=dict(left_robot_tcp_pose=dict(shape=[9]),
                          external_img=dict(shape=[3, 2, 3], type='rgb')),
                 action=dict(shape=[10]))
    dataset = P2PImageDataset(shape, str(tmp_path/'recording'), horizon=8,
                              pad_before=3+latency, pad_after=4, n_obs_steps=4,
                              obs_temporal_downsample_ratio=2, n_latency_steps=latency)
    cfg = OmegaConf.create(dict(live=dict(remote_desired=dict(chunk_start_index=chunk_start,
                                                             chunk_size=5-chunk_start)),
                                gripper=dict(command_threshold=0, open_width_m=.078, closed_width_m=0)))
    monkeypatch.setattr(deployment, 'predict_relative_actions',
                        lambda *args: np.tile(np.r_[np.zeros(9), -1.0], (5, 1)))
    report, arrays = evaluate_episode(SimpleNamespace(action_start_offset=3, n_action_steps=5),
                                      dataset, cfg, seed=42)
    valid = arrays['valid']
    np.testing.assert_array_equal(arrays['ground_truth'][valid], commands[arrays['target_frame'][valid]])
    np.testing.assert_array_equal(arrays['target_frame'][:, 0], arrays['observation_frame']+latency+chunk_start)
    assert report['all_selected_actions']['close_recall'] == 0
    save_results(tmp_path/'result', report, arrays, cfg.gripper)
    for name in ['report.json', 'predictions.csv', 'predictions.npz', 'gripper_comparison.png']:
        assert (tmp_path/'result'/name).stat().st_size > 0
    with pytest.raises(FileExistsError):
        save_results(tmp_path/'result', report, arrays, cfg.gripper)
