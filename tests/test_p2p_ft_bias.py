from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from diffusion_policy.real_world.p2p_ft_bias import FTBiasSettings, StartupFTBias, rotation_xyzw
from diffusion_policy.real_world.p2p_franka_deployment import _remote_desired_bridge_payload


def state(wrench=(2, -1, .5, .1, -.2, .05), quat=(0, 0, 0, 1)):
    return np.r_[.5, 0, .3, quat, wrench].astype(float)


def calibrator(**kwargs):
    return StartupFTBias(FTBiasSettings.from_payload(dict(
        enabled=True, free_space_confirmed=True, **kwargs)), 0)


def collect(bias, value=None):
    value = state() if value is None else value
    for now in np.linspace(0, .2, 7):
        bias.observe(value, np.zeros(7), now)


def test_point_two_seconds_mean_and_freeze():
    bias = calibrator()
    value = state()
    for t in [0, .04, .08, .12, .16]:
        bias.observe(value, np.zeros(7), t)
    assert not bias.ready
    bias.observe(value, np.zeros(7), .2)
    assert bias.ready
    np.testing.assert_allclose(bias.bias, value[7:])
    raw = np.stack([value, value])
    saved = raw.copy()
    corrected = bias.correct_states(raw)
    np.testing.assert_allclose(corrected[:, 7:], 0, atol=1e-12)
    np.testing.assert_array_equal(raw, saved)
    contact = value.copy()
    contact[7] += 4
    bias.observe(contact, np.ones(7), 100)  # Never re-zero after ready.
    np.testing.assert_allclose(bias.correct_states(contact[None])[0, 7:], [4, 0, 0, 0, 0, 0], atol=1e-12)


def test_rotated_tool_bias_and_contact_not_removed():
    bias = calibrator()
    collect(bias)
    q = [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]
    rot = rotation_xyzw(q)
    value = state(np.r_[rot@bias.bias[:3], rot@bias.bias[3:]], q)
    value[7] += 3  # A new world-frame external contact force must remain.
    np.testing.assert_allclose(bias.correct_states(value[None])[0, 7:], [3, 0, 0, 0, 0, 0], atol=1e-12)


def test_rotation_during_initial_window_is_rejected():
    bias = calibrator()
    bias.observe(state(), np.zeros(7), 0)
    q = [0, 0, np.sin(.1), np.cos(.1)]
    bias.observe(state(quat=q), np.zeros(7), .1)
    assert not bias.ready and bias.resets == 1


def test_movement_gap_noise_and_excessive_residual():
    bias = calibrator()
    bias.observe(state(), np.ones(7), 0)
    assert bias.reason == 'robot_moving' and not bias.samples
    bias.observe(state(), np.zeros(7), .01)
    bias.observe(state(), np.zeros(7), .15)
    assert len(bias.samples) == 1 and bias.resets == 2
    bias = calibrator()
    for i, t in enumerate(np.linspace(0, .2, 7)):
        bias.observe(state(wrench=[(-1)**i*2, 0, 0, 0, 0, 0]), np.zeros(7), t)
    assert not bias.ready and bias.reason == 'wrench_unstable'
    bias = calibrator()
    bias.observe(state(wrench=[11, 0, 0, 0, 0, 0]), np.zeros(7), 0)
    assert bias.error == 'residual_exceeds_calibration_limit'
    collect(bias)
    assert not bias.ready  # Requires restart, cannot acknowledge away contact.


def test_timeout_missing_data_nonfinite_and_minimum_samples():
    bias = calibrator()
    bad = state(); bad[7] = np.nan
    bias.observe(bad, np.zeros(7), 0)
    assert not bias.ready and bias.reason == 'invalid_sample'
    bias.check_timeout(11)
    assert bias.error == 'calibration_timeout'
    with pytest.raises(RuntimeError):
        bias.correct_states(state()[None])
    bias = calibrator(min_samples=20)
    collect(bias)
    assert not bias.ready


def test_disabled_is_exact_noop():
    bias = StartupFTBias(FTBiasSettings(), 0)
    raw = state()[None]
    np.testing.assert_array_equal(bias.correct_states(raw), raw)
    assert bias.ready and bias.status()['bias_tool_axes'] is None


@pytest.mark.parametrize('override', [dict(free_space_confirmed=False),
    dict(duration_seconds=0), dict(min_samples=1), dict(max_force_std_n=float('nan')),
    dict(timeout_seconds=.1), dict(enabled='true')])
def test_invalid_settings(override):
    values = dict(enabled=True, free_space_confirmed=True)
    values.update(override)
    with pytest.raises(ValueError):
        FTBiasSettings.from_payload(values)


def test_payload_gates_and_safety_unchanged():
    cfg = OmegaConf.load('diffusion_policy/config/deploy/p2p_valve_franka.yaml')
    before = _remote_desired_bridge_payload(SimpleNamespace(n_obs_steps=2), cfg, 'execute')
    cfg.live.remote_desired.ft_bias.enabled = True
    with pytest.raises(ValueError, match='confirm-ft'):
        _remote_desired_bridge_payload(SimpleNamespace(n_obs_steps=2), cfg, 'execute')
    cfg.live.remote_desired.ft_bias.free_space_confirmed = True
    after = _remote_desired_bridge_payload(SimpleNamespace(n_obs_steps=2), cfg, 'execute')
    assert after['ft_bias']['duration_seconds'] == .2
    assert before['safety'] == after['safety']
    assert before['gripper'] == after['gripper']
    cfg.live.remote_desired.require_enable = False
    with pytest.raises(ValueError, match='require_enable'):
        _remote_desired_bridge_payload(SimpleNamespace(n_obs_steps=2), cfg, 'execute')
