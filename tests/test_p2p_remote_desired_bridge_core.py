import numpy as np
import pytest

from diffusion_policy.real_world.p2p_remote_desired_bridge_core import (
    AbsoluteChunkPlayer,
    BridgeSettings,
    ObservationBuffer,
    validate_target,
)


def make_settings(**overrides):
    values = {
        "obs_steps": 2,
        "obs_downsample_ratio": 2,
        "action_rate_hz": 30.0,
        "chunk_size": 4,
        "prefetch_steps": 1,
        "state_timeout_sec": 0.1,
        "image_timeout_sec": 0.2,
        "command_target_timeout_sec": 0.2,
        "require_enable": True,
        "workspace_min": (0.0, -1.0, 0.0),
        "workspace_max": (1.0, 1.0, 1.0),
        "max_translation_step_m": 0.02,
        "max_rotation_step_rad": 0.1,
        "max_force_n": 20.0,
        "max_torque_nm": 3.0,
    }
    values.update(overrides)
    return BridgeSettings(**values)


def test_observation_request_preserves_training_downsample_spacing():
    buffer = ObservationBuffer(make_settings(), ("external_img", "left_wrist_img"))
    for tick in range(3):
        state = np.zeros(13)
        state[0] = tick
        state[6] = 1.0
        buffer.push_state(state, tick / 30.0)
        buffer.push_image("external_img", f"e{tick}".encode(), tick / 30.0)
        buffer.push_image("left_wrist_img", f"w{tick}".encode(), tick / 30.0)
        buffer.sample_tick()

    request = buffer.request(7)
    np.testing.assert_allclose(request["observation.state"][:, 0], [0.0, 2.0])
    assert request["images"]["external_img"] == [b"e0", b"e2"]
    assert request["images"]["left_wrist_img"] == [b"w0", b"w2"]


def test_absolute_chunk_player_discards_targets_missed_during_inference():
    player = AbsoluteChunkPlayer(make_settings())
    sequence = player.start_request(tick=10)
    actions = np.zeros((4, 7))
    actions[:, 0] = [0.1, 0.2, 0.3, 0.4]
    actions[:, 6] = 1.0
    player.on_chunk(sequence, actions, now_tick=12)

    assert player.stale_entries_skipped == 2
    assert player.pop(12)[0] == pytest.approx(0.3)
    assert player.pop(13)[0] == pytest.approx(0.4)


def test_target_safety_uses_previous_target_and_measured_wrench():
    settings = make_settings()
    reference = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    target = reference.copy()
    target[0] += 0.01
    metrics = validate_target(target, reference, np.zeros(6), settings)
    assert metrics["translation_step_m"] == pytest.approx(0.01)

    target[0] += 0.02
    with pytest.raises(RuntimeError, match="translation_step_limit"):
        validate_target(target, reference, np.zeros(6), settings)

    with pytest.raises(RuntimeError, match="force_limit"):
        validate_target(reference, reference, [21, 0, 0, 0, 0, 0], settings)
