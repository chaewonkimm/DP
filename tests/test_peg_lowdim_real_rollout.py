from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.signal import butter, lfilter, lfilter_zi

from diffusion_policy.real_world.p2p_franka_deployment import (
    _remote_desired_bridge_payload,
    build_live_env_observation,
    validate_checkpoint_deployment_contract,
)
from diffusion_policy.real_world.p2p_remote_desired_bridge_core import (
    StreamingWrenchFilter,
    WrenchFilterSettings,
)


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_ROT6D = [1, 0, 0, 0, 1, 0]


def shape_meta(with_wrench=False):
    obs = {
        "left_robot_tcp_pose": {"shape": [9], "type": "low_dim"},
        "left_goal_pose_rel": {"shape": [9], "type": "low_dim"},
    }
    if with_wrench:
        obs["left_robot_tcp_wrench"] = {"shape": [6], "type": "low_dim"}
    return OmegaConf.create({"obs": obs, "action": {"shape": [9]}})


def states():
    result = np.zeros((2, 13), dtype=np.float32)
    result[:, :3] = [[0.4, -0.2, 0.3], [0.5, -0.2, 0.3]]
    result[:, 6] = 1.0  # identity xyzw quaternion
    result[:, 7:13] = [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]
    return result


@pytest.mark.parametrize("with_wrench", [False, True])
def test_live_lowdim_observation_repeats_current_relative_goal(with_wrench):
    goal = np.asarray([0.6, -0.2, 0.3, *IDENTITY_ROT6D], dtype=np.float32)
    observation = build_live_env_observation(
        states(), {}, shape_meta(with_wrench), goal)
    assert "external_img" not in observation
    np.testing.assert_allclose(
        observation["left_goal_pose_rel"][:, :3], [[0.1, 0, 0], [0.1, 0, 0]],
        atol=1e-6)
    np.testing.assert_allclose(
        observation["left_goal_pose_rel"][:, 3:], [IDENTITY_ROT6D] * 2,
        atol=1e-6)
    assert ("left_robot_tcp_wrench" in observation) is with_wrench


def test_streaming_iir_matches_scipy_lfilter_without_resampling():
    b, a = butter(2, 15.0, btype="low", fs=1001.74)
    settings = WrenchFilterSettings.from_payload({
        "enabled": True,
        "sample_rate_hz": 1001.74,
        "resample": False,
        "b": b,
        "a": a,
        "zi": lfilter_zi(b, a),
    })
    stream = StreamingWrenchFilter(settings, n_channels=1)
    values = np.asarray([2.0, 3.0, -1.0, 4.0, 0.5])
    outputs = [stream.reset(0.0, [values[0]])[0]]
    for index, value in enumerate(values[1:], start=1):
        outputs.append(stream.push(index / 1001.74, [value])[0])
    expected_tail, _ = lfilter(
        b, a, values[1:], zi=lfilter_zi(b, a) * values[0])
    np.testing.assert_allclose(outputs[0], values[0])
    np.testing.assert_allclose(outputs[1:], expected_tail, rtol=1e-12, atol=1e-12)


VARIANTS = {
    "p2p_peg_goal_prop_desired_franka.yaml": (
        "peg_proprio_goal_desired", False),
    "p2p_peg_goal_prop_next_state_franka.yaml": (
        "peg_proprio_goal_next_state", False),
    "p2p_peg_goal_prop_wrench_desired_franka.yaml": (
        "peg_proprio_goal_wrench_desired", True),
    "p2p_peg_goal_prop_wrench_next_state_franka.yaml": (
        "peg_proprio_goal_wrench_next_state", True),
}


@pytest.mark.parametrize("filename,expected", VARIANTS.items())
def test_four_configs_lock_checkpoint_contract_and_camera_free_payload(
        filename, expected):
    task_name, with_wrench = expected
    deploy = OmegaConf.load(ROOT / "diffusion_policy/config/deploy" / filename)
    train = OmegaConf.create({
        "task": {"name": task_name},
        "shape_meta": shape_meta(with_wrench),
    })
    validate_checkpoint_deployment_contract(train, deploy)
    payload = _remote_desired_bridge_payload(
        SimpleNamespace(n_obs_steps=2, n_action_steps=29),
        deploy, "shadow", train_cfg=train)
    assert payload["observation"]["camera_topics"] == {}
    assert payload["observation"]["wrench_filter"]["enabled"] is with_wrench
    assert deploy.weights == "ema_model"
    assert Path(str(deploy.checkpoint)).name.startswith("epoch=0780-train_loss=")


def test_wrong_checkpoint_variant_is_rejected():
    deploy = OmegaConf.load(
        ROOT / "diffusion_policy/config/deploy/"
        "p2p_peg_goal_prop_desired_franka.yaml")
    train = OmegaConf.create({
        "task": {"name": "peg_proprio_goal_next_state"},
        "shape_meta": shape_meta(False),
    })
    with pytest.raises(RuntimeError, match="does not match"):
        validate_checkpoint_deployment_contract(train, deploy)
