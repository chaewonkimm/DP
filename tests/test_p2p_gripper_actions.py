from concurrent.futures import Future
from types import SimpleNamespace
from pathlib import Path
import os
import subprocess

import pytest

from diffusion_policy.real_world.p2p_gripper_actions import (
    GripperActionController, GripperActionSettings, quantize_gripper_command,
)


def payload(**changes):
    data = dict(action_namespace='/test/gripper', open_width_m=0.078,
                closed_width_m=0.0, velocity_mps=0.1, force_limit_n=7.0,
                grasp_epsilon_inner_m=0.07, grasp_epsilon_outer_m=0.07,
                close_with_force=True, command_stability_seconds=0.1,
                action_timeout_seconds=5.0)
    data.update(changes)
    return data


class Goal:
    def __init__(self):
        self.epsilon = SimpleNamespace(inner=0, outer=0)


class FakeClient:
    def __init__(self, *args):
        self.calls = []
        self.futures = []
        self.available = True

    def server_is_ready(self):
        return self.available

    def send_goal_async(self, goal):
        future = Future()
        self.calls.append(goal)
        self.futures.append(future)
        return future


class Handle:
    accepted = True

    def __init__(self):
        self.result = Future()
        self.cancel_count = 0

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancel_count += 1
        future = Future()
        future.set_result(SimpleNamespace(goals_canceling=[1]))
        return future

    def finish(self, status=4, success=True):
        self.result.set_result(SimpleNamespace(status=status,
            result=SimpleNamespace(success=success, error='' if success else 'test failure')))


def controller(**changes):
    now = [0.0]
    actions = GripperActionController(None, GripperActionSettings.from_payload(payload(**changes)),
        client_factory=FakeClient, action_types=(SimpleNamespace(Goal=Goal), SimpleNamespace(Goal=Goal)),
        clock=lambda: now[0])
    return actions, now


def issue(actions, now, width):
    actions.tick(width)
    now[0] += 0.11
    actions.tick(width)


def accept(actions, kind):
    handle = Handle()
    actions.clients[kind].futures[-1].set_result(handle)
    return handle


@pytest.mark.parametrize('value,expected', [(-1.3, -1), (-0.01, -1),
    (0, 1), (0.01, 1), (0.9, 1), (1.2, 1)])
def test_float_gripper_quantization(value, expected):
    assert quantize_gripper_command(value) == expected


@pytest.mark.parametrize('value,threshold', [(float('nan'), 0), (float('inf'), 0),
    (-float('inf'), 0), (0, float('nan'))])
def test_invalid_gripper_quantization(value, threshold):
    with pytest.raises(ValueError, match='finite'):
        quantize_gripper_command(value, threshold)


def test_float_predictions_send_example_actions_on_sign_changes_only():
    from omegaconf import OmegaConf
    from diffusion_policy.real_world.p2p_franka_deployment import gripper_command_to_width
    from eval_real_p2p_common import apply_valve_gripper_action_profile
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] /
                         'diffusion_policy/config/deploy/p2p_valve_franka.yaml')
    original_safety = OmegaConf.to_container(cfg.safety)
    original_live = OmegaConf.to_container(cfg.live)
    original_mapping = cfg.gripper.mapping_confirmed
    apply_valve_gripper_action_profile(cfg, 'valve')
    assert OmegaConf.to_container(cfg.safety) == original_safety
    assert OmegaConf.to_container(cfg.live) == original_live
    assert cfg.gripper.mapping_confirmed == original_mapping
    actions, now = controller(**{k: v for k, v in OmegaConf.to_container(cfg.gripper).items()
                                 if k in payload()})
    for value in [-.9, -.2, -.001, .01, .7, 1.2, -.1, -.8]:
        previous = actions.goals_sent
        actions.tick(gripper_command_to_width(value, cfg.gripper))
        if actions.goals_sent > previous:
            accept(actions, 'grasp' if value >= 0 else 'move').finish()
    assert actions.goals_sent == 3  # initial open, close, open
    assert len(actions.clients['move'].calls) == 2
    assert len(actions.clients['grasp'].calls) == 1
    opened = actions.clients['move'].calls[0]
    closed = actions.clients['grasp'].calls[0]
    assert opened.width == .08 and opened.speed == .1
    assert closed.width == 0 and closed.speed == .1 and closed.force == 120
    assert closed.epsilon.inner == closed.epsilon.outer == .07
    assert actions.status()['desired_command'] == -1
    assert actions.status()['last_goal'] == dict(action='/rightarm/remote_right_gripper/move',
                                                command=-1, width=.08, speed=.1)
    assert actions.status()['last_result']['success'] is True


def test_action_profile_cannot_enable_peg_or_bypass_mapping():
    from omegaconf import OmegaConf
    from eval_real_p2p_common import apply_valve_gripper_action_profile
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root/'diffusion_policy/config/deploy/p2p_valve_franka.yaml')
    cfg.live.execute_enabled = False
    cfg.gripper.mapping_confirmed = False
    apply_valve_gripper_action_profile(cfg, 'valve')
    assert cfg.live.execute_enabled is False and cfg.gripper.mapping_confirmed is False
    with pytest.raises(ValueError, match='only'):
        apply_valve_gripper_action_profile(cfg, 'peg')


def test_move_grasp_parameters_dedup_and_no_automatic_release():
    actions, now = controller()
    issue(actions, now, 0.078)
    goal = actions.clients['move'].calls[0]
    assert goal.width == 0.078 and goal.speed == 0.1
    accept(actions, 'move').finish()
    for _ in range(100):
        actions.tick(0.078)
    assert actions.goals_sent == 1
    issue(actions, now, 0.0)
    goal = actions.clients['grasp'].calls[0]
    assert goal.width == 0 and goal.force == 7 and goal.speed == 0.1
    assert goal.epsilon.inner == goal.epsilon.outer == 0.07
    handle = accept(actions, 'grasp')
    handle.finish()
    actions.disable()
    assert handle.cancel_count == 0  # Completed grasp is not released on disable.
    assert actions.goals_sent == 2


def test_close_without_force_uses_move():
    actions, now = controller(close_with_force=False)
    issue(actions, now, 0.0)
    assert len(actions.clients['move'].calls) == 1
    assert not actions.clients['grasp'].calls


def test_disable_before_acceptance_cancels_only_owned_goal():
    actions, now = controller()
    issue(actions, now, 0.0)
    actions.disable()
    handle = accept(actions, 'grasp')
    actions.disable()
    assert handle.cancel_count == 1
    handle.finish(status=5, success=False)
    assert not actions.busy and actions.error is None


def test_inflight_commands_do_not_overlap_or_replay_history():
    actions, now = controller()
    issue(actions, now, 0.0)
    handle = accept(actions, 'grasp')
    issue(actions, now, 0.078)
    issue(actions, now, 0.0)
    assert actions.goals_sent == 1
    handle.finish()
    actions.tick(0.0)
    assert actions.goals_sent == 1  # Superseded open command was never queued.


def test_rejection_failure_timeout_and_reset():
    actions, now = controller()
    issue(actions, now, 0.0)
    actions.clients['grasp'].futures[-1].set_result(SimpleNamespace(accepted=False))
    assert actions.error == 'goal_rejected'
    issue(actions, now, 0.0)
    assert actions.goals_sent == 1
    actions.reset_error()
    issue(actions, now, 0.0)
    handle = accept(actions, 'grasp')
    handle.finish(status=6, success=False)
    assert actions.error.startswith('action_failed')
    actions.reset_error()
    issue(actions, now, 0.0)
    handle = accept(actions, 'grasp')
    now[0] += 6
    actions.tick(0.0)
    assert actions.error == 'action_timeout' and handle.cancel_count == 1


def test_unavailable_server_and_invalid_width_do_not_send():
    actions, now = controller()
    actions.clients['move'].available = False
    issue(actions, now, 0.0)
    assert actions.error == 'action_server_unavailable' and actions.goals_sent == 0
    actions.reset_error()
    actions.tick(float('nan'))
    assert actions.error == 'invalid_binary_width' and actions.goals_sent == 0


@pytest.mark.parametrize('key,value', [('force_limit_n', 0), ('velocity_mps', -1),
    ('open_width_m', float('nan')), ('closed_width_m', 0.09),
    ('grasp_epsilon_outer_m', -1), ('action_namespace', ''), ('close_with_force', 'true')])
def test_invalid_settings_rejected(key, value):
    with pytest.raises(ValueError):
        GripperActionSettings.from_payload(payload(**{key: value}))


@pytest.mark.skipif(os.environ.get('DP_TEST_ROS') != '1', reason='requires local ROS installation')
@pytest.mark.parametrize('example_profile', [False, True])
@pytest.mark.parametrize('ft_bias', [False, True])
def test_real_ros_action_serialization(tmp_path, example_profile, ft_bias):
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / 'diffusion_policy/config/deploy/p2p_valve_franka.yaml')
    remote = cfg.live.remote_desired
    env = dict(os.environ, ROS_DOMAIN_ID='187', ROS_LOCALHOST_ONLY='1',
               ROS_LOG_DIR=str(tmp_path / 'logs'), PYTHONPATH=str(root),
               DP_TEST_GRIPPER_EXAMPLE='1' if example_profile else '0')
    env['DP_TEST_FT_BIAS'] = '1' if ft_bias else '0'
    result = subprocess.run([
        '/bin/bash', '-c', 'set -e; source "$1"; shift; exec "$@"',
        'dp-gripper-test', str(remote.ros_setup_file), str(remote.bridge_python),
        str(root / 'tests/gripper_ros_probe.py')], env=env,
        capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: isolated' in result.stdout
