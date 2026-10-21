"""Regression for configured workspace bounds crossing the DP/ROS boundary."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from diffusion_policy.real_world.p2p_franka_deployment import (
    _jsonable, _remote_desired_bridge_payload,
)
from diffusion_policy.real_world.p2p_remote_desired_bridge_core import BridgeSettings


@pytest.mark.parametrize('task', ['peg', 'valve'])
def test_configured_workspace_survives_bridge_json(task):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / f'diffusion_policy/config/deploy/p2p_{task}_franka.yaml')
    cfg.safety.workspace_min = [0.2, -0.6, 0.1]
    cfg.safety.workspace_max = [0.7, 0.1, 0.8]
    payload = _remote_desired_bridge_payload(SimpleNamespace(n_obs_steps=2), cfg, 'execute')
    decoded = json.loads(json.dumps(_jsonable(payload)))
    settings = BridgeSettings.from_payload(decoded)
    assert settings.workspace_min == (0.2, -0.6, 0.1)
    assert settings.workspace_max == (0.7, 0.1, 0.8)
    assert decoded['gripper']['ros_command_topic'] is None


def test_valve_action_parameters_and_mapping_gate():
    from diffusion_policy.real_world.p2p_franka_deployment import (
        validate_execute_configuration, EXECUTE_CONFIRMATION,
    )
    from diffusion_policy.real_world.p2p_gripper_actions import GripperActionSettings
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / 'diffusion_policy/config/deploy/p2p_valve_franka.yaml')
    cfg.live.execute_enabled = True  # In-memory only; never run a bridge.
    cfg.gripper.remote_desired_mode = 'franka_action'  # Independent of user's arm-only default.
    cfg.gripper.mapping_confirmed = False
    with pytest.raises(RuntimeError, match='mapping_confirmed'):
        validate_execute_configuration(cfg, EXECUTE_CONFIRMATION)
    cfg.gripper.mapping_confirmed = True
    validate_execute_configuration(cfg, EXECUTE_CONFIRMATION)
    payload = _jsonable(_remote_desired_bridge_payload(SimpleNamespace(n_obs_steps=2), cfg, 'execute'))
    assert payload['gripper']['mode'] == 'franka_action'
    settings = GripperActionSettings.from_payload(payload['gripper'])
    assert settings.force_limit_n == cfg.gripper.force_limit_n
    assert settings.velocity_mps == cfg.gripper.velocity_mps
    assert settings.action_namespace == '/rightarm/remote_right_gripper'
