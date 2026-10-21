import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from diffusion_policy.real_world import p2p_franka_deployment as deployment
from diffusion_policy.real_world.p2p_remote_desired_bridge_core import (
    AbsoluteChunkPlayer, BridgeSettings, select_action_chunk, validate_chunk_window,
)


ROOT = Path(__file__).resolve().parents[1]


def config_and_settings():
    cfg = OmegaConf.load(ROOT / 'diffusion_policy/config/deploy/p2p_peg_franka.yaml')
    cfg.live.remote_desired.chunk_start_index = 4
    cfg.live.remote_desired.chunk_size = 4
    cfg.live.remote_desired.prefetch_steps = 3
    payload = deployment._remote_desired_bridge_payload(
        SimpleNamespace(n_obs_steps=2, n_action_steps=29), cfg, 'shadow')
    return cfg, BridgeSettings.from_payload(json.loads(json.dumps(deployment._jsonable(payload))))


def pose_actions():
    actions = np.zeros((29, 7))
    actions[:, 0] = np.arange(29)
    actions[:, 6] = 1
    return actions


@pytest.mark.parametrize('elapsed,expected,skipped', [
    (0, [4, 5, 6, 7], 0), (2, [4, 5, 6, 7], 0),
    (4, [4, 5, 6, 7], 0), (6, [6, 7], 2), (8, [], 4), (20, [], 4),
])
def test_slice_respects_original_timeline(elapsed, expected, skipped):
    _, settings = config_and_settings()
    player = AbsoluteChunkPlayer(settings)
    seq = player.start_request(100)
    player.on_chunk(seq, select_action_chunk(pose_actions(), 4, 4), 100 + elapsed)
    assert [a[0] for a in player.queue] == expected
    assert player.stale_entries_skipped == skipped


def test_prefetch_waiting_slice_and_boundary():
    _, settings = config_and_settings()
    player = AbsoluteChunkPlayer(settings)
    player.queue.extend(pose_actions()[:4].copy())
    assert not player.should_request()
    assert player.pop(100)[0] == 0
    assert player.should_request()
    seq = player.start_request(101)
    player.on_chunk(seq, select_action_chunk(pose_actions(), 4, 4), 103)
    assert not player.should_request()  # Already buffered next chunk.
    assert [player.pop(t)[0] for t in [103, 104, 105]] == [1, 2, 3]
    assert player.pop(106)[0] == 5  # Original index 4 expired while waiting.
    assert player.stale_entries_skipped == 1


@pytest.mark.parametrize('start,size,available', [(-1, 4, 29), (4, 0, 29),
    (26, 4, 29), (4.5, 4, 29), (True, 4, 29)])
def test_invalid_slice_rejected(start, size, available):
    with pytest.raises(ValueError):
        validate_chunk_window(start, size, available)


@pytest.mark.parametrize('start,size', [(0, 29), (4, 4)])
@pytest.mark.parametrize('action_dim', [9, 10])
def test_worker_selects_and_logs_exact_actions(tmp_path, monkeypatch, start, size, action_dim):
    cfg, _ = config_and_settings()
    cfg.live.remote_desired.chunk_start_index = start
    cfg.live.remote_desired.chunk_size = size
    cfg.task.action_dim = action_dim
    if action_dim == 10:
        cfg.gripper.remote_desired_mode = 'franka_action'
        cfg.gripper.open_width_m = 0.078
        cfg.gripper.closed_width_m = 0.0
    relative = np.zeros((29, action_dim), dtype=np.float32)
    relative[:, 0] = np.arange(29) * 0.001
    relative[:, 3] = relative[:, 7] = 1
    if action_dim == 10:
        relative[:, 9] = np.where(np.arange(29) % 2, 1, -1)
    base = np.array([0.5, 0, 0.3, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    monkeypatch.setattr(deployment, 'prepare_policy_observation', lambda *a: ({}, base))
    monkeypatch.setattr(deployment, 'predict_relative_actions', lambda *a: relative)
    shape_meta = OmegaConf.create({
        'obs': {
            'external_img': {'shape': [3, 8, 8], 'type': 'rgb'},
            'left_wrist_img': {'shape': [3, 8, 8], 'type': 'rgb'},
            'left_robot_tcp_pose': {'shape': [9], 'type': 'low_dim'},
            'left_robot_tcp_wrench': {'shape': [6], 'type': 'low_dim'},
        },
        'action': {'shape': [action_dim]},
    })
    worker = deployment._RemoteDesiredWorkerState(
        SimpleNamespace(num_inference_steps=8),
        SimpleNamespace(shape_meta=shape_meta), cfg, tmp_path, 'slice-test')
    states = np.zeros((2, 13), dtype=np.float32)
    states[:, 6] = 1
    ok, jpeg = cv2.imencode('.jpg', np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    request = {'protocol_version': 1, 'sequence': 1, 'observation.state': states,
               'images': {k: [jpeg.tobytes()] * 2 for k in ['external_img', 'left_wrist_img']}}
    reply = worker.predict(request)
    assert reply['chunk_start_index'] == start
    assert np.asarray(reply['actions']).shape == (size, 8 if action_dim == 10 else 7)
    if action_dim == 10:
        np.testing.assert_allclose(np.asarray(reply['actions'])[:, 7],
            np.where(np.arange(start, start + size) % 2, 0.0, 0.078))
    np.testing.assert_allclose(np.asarray(reply['actions'])[:, 0],
                               0.5 + np.arange(start, start + size) * 0.001)
    event = json.loads(worker.log_path.read_text())
    assert event['action_count'] == size
    assert event['predicted_action_count'] == 29
    assert event['chunk_end_index_exclusive'] == start + size
    with np.load(tmp_path / 'remote_desired_first_observation.npz') as saved:
        np.testing.assert_array_equal(saved['selected_absolute_action'],
                                      saved['predicted_absolute_action'][start:start + size])


def test_slice_out_of_policy_range_fails_before_launch():
    cfg, _ = config_and_settings()
    with pytest.raises(ValueError, match='exceeds'):
        deployment._remote_desired_bridge_payload(
            SimpleNamespace(n_obs_steps=2, n_action_steps=6), cfg, 'execute')
