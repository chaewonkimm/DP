"""Invoked by an opt-in test in an isolated ROS domain, never against hardware."""
import time
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import tempfile
import threading

import rclpy
from rclpy.action import ActionServer
from franka_msgs.action import Move, Grasp
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from iris_teleop.msg import TeleopState
from rclpy.qos import qos_profile_sensor_data
from p2p_remote_desired_real_bridge import _read_frame, _write_frame

from diffusion_policy.real_world.p2p_gripper_actions import (
    GripperActionController, GripperActionSettings,
)


def check_bridge(node, settings, received):
    """Run the actual ROS bridge against fake sensors/worker/action servers."""
    prefix = '/dp_mock_bridge/'
    names = ('remote_state', 'remote_desired', 'remote_desired_shadow', 'enable',
             'dry_run_action', 'status', 'external_img', 'left_wrist_img')
    topics = {name: prefix + name for name in names}
    gripper = dict(vars(settings), mode='franka_action')
    use_ft_bias = os.environ.get('DP_TEST_FT_BIAS') == '1'
    payload = dict(
        observation=dict(obs_steps=2, obs_downsample_ratio=2,
                         camera_topics={k: topics[k] for k in ('external_img', 'left_wrist_img')}),
        topics=topics,
        timing=dict(action_rate_hz=30, command_publish_rate_hz=1000,
                    state_timeout_sec=0.1, image_timeout_sec=0.2,
                    command_target_timeout_sec=0.2, worker_timeout_sec=1,
                    worker_heartbeat_period_sec=0.1, worker_disconnect_timeout_sec=1),
        action=dict(chunk_size=4, chunk_start_index=4, prefetch_steps=3),
        safety=dict(require_enable=True, allow_real_robot=True,
                    workspace_min=None, workspace_max=None, max_translation_step_m=None,
                    max_rotation_step_rad=None, max_force_n=None, max_torque_nm=None),
        gripper=gripper)
    if use_ft_bias:
        payload['ft_bias'] = dict(enabled=True, free_space_confirmed=True, duration_seconds=.2)
    predictions_checked = []

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            request = _read_frame(self.request)
            reply = {'ok': True}
            if request.get('request_type') == 'predict':
                if use_ft_bias:
                    import numpy as np
                    corrected = request['observation.state']
                    uncorrected = request['observation.state_uncorrected']
                    np.testing.assert_allclose(corrected[:, 7:], 0, atol=1e-6)
                    np.testing.assert_allclose(uncorrected[:, 7], 2)
                    assert request['ft_bias']['ready']
                    predictions_checked.append(True)
                reply.update(action_representation='absolute_pose', chunk_start_index=4,
                             actions=[[0.5, 0, 0.3, 0, 0, 0, 1, 0.0]] * 4,
                             inference_time_sec=0.001)
            _write_frame(self.request, reply)

    pubs = {name: node.create_publisher(typ, topics[name], 10) for name, typ in
            [('remote_state', TeleopState), ('external_img', CompressedImage),
             ('left_wrist_img', CompressedImage), ('enable', Bool)]}
    states, commands = [], []
    subs = [node.create_subscription(String, topics['status'], lambda m: states.append(json.loads(m.data)), 10)]
    for name in ('remote_desired', 'remote_desired_shadow'):
        subs.append(node.create_subscription(TeleopState, topics[name], commands.append, qos_profile_sensor_data))
    state = TeleopState()
    state.pose.position.x, state.pose.position.z = 0.5, 0.3
    state.pose.orientation.w = 1.0
    state.dq = [0.] * 7
    if use_ft_bias:
        state.wrench.force.x = 2.0
    frame = CompressedImage()
    frame.format, frame.data = 'jpeg', b'mock-worker-does-not-decode'
    with tempfile.TemporaryDirectory(prefix='dp-gripper-probe-') as folder, \
            socketserver.ThreadingTCPServer(('127.0.0.1', 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for mode in ('execute', 'shadow'):
                states.clear()
                commands.clear()
                before = len(received)
                env = dict(os.environ, DP_P2P_BRIDGE_CONFIG_JSON=json.dumps(payload))
                if use_ft_bias:
                    payload['ft_bias_report_path'] = folder + '/' + mode + '_ft_bias.json'
                    env['DP_P2P_BRIDGE_CONFIG_JSON'] = json.dumps(payload)
                command = [sys.executable, '-c',
                    'import sys; import p2p_remote_desired_real_bridge as b; '
                    'b.LOCK_FILE=sys.argv.pop(1); raise SystemExit(b.main())',
                    folder + '/bridge.lock', '--worker-host', '127.0.0.1',
                    '--worker-port', str(server.server_address[1]), '--instance-id', mode,
                    '--pid-file', folder + '/bridge.json', '--duration-sec', '2',
                    '--real-robot', '--publish' if mode == 'execute' else '--echo-only']
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                enabled = False
                deadline = time.monotonic() + 7
                try:
                    while process.poll() is None and time.monotonic() < deadline:
                        for name in ('external_img', 'left_wrist_img'):
                            pubs[name].publish(frame)
                        pubs['remote_state'].publish(state)
                        if states and not enabled:
                            status = states[-1]
                            if (status['instance_id'] == mode and status['last_reason'] == 'disabled'
                                    and status['worker_healthy'] and status['ft_bias']['ready']):
                                if mode == 'shadow' or status['gripper_action_status']['ready']:
                                    pubs['enable'].publish(Bool(data=True))
                                    enabled = True
                        rclpy.spin_once(node, timeout_sec=0.01)
                        time.sleep(0.002)
                    assert process.poll() is not None, 'Bridge exit timeout'
                    output = process.communicate(timeout=1)[0]
                    assert process.returncode == 0, output
                    assert enabled and commands, (mode, states, output)
                    if use_ft_bias:
                        assert predictions_checked
                        report = json.loads(Path(payload['ft_bias_report_path']).read_text())
                        assert report['calibration']['ready']
                        assert report['calibration']['bias_tool_axes'][0] == 2
                        # Safety still measures the unzeroed 2 N, NOT the zero policy input.
                        assert any(s['last_safety_metrics'].get('force_n') == 2 for s in states)
                    assert all(not msg.clutch2.data for msg in commands)
                    if mode == 'execute':
                        assert len(received) == before + 1, (received, states)
                        assert received[-1][0] == 'grasp'
                        assert received[-1][1].force == settings.force_limit_n
                    else:
                        assert len(received) == before, 'Shadow sent a gripper goal'
                        assert all(s['gripper_action_status'] is None for s in states)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=2)
        finally:
            server.shutdown()
            thread.join(timeout=2)
    for sub in subs:
        node.destroy_subscription(sub)
    for pub in pubs.values():
        node.destroy_publisher(pub)
    print('PASS: real bridge isolated execute sends one grasp, shadow sends none, clutch2 stays false')


def main():
    rclpy.init()
    node = rclpy.create_node('dp_mock_gripper_test')
    received = []

    def callback(kind, action_type):
        def execute(handle):
            received.append((kind, handle.request))
            handle.succeed()
            result = action_type.Result()
            result.success = True
            result.error = ''
            return result
        return execute

    servers = [ActionServer(node, typ, '/dp_mock_gripper/' + kind,
                            callback(kind, typ))
               for kind, typ in [('move', Move), ('grasp', Grasp)]]
    settings = GripperActionSettings.from_payload(dict(
        action_namespace='/dp_mock_gripper',
        open_width_m=0.08 if os.environ.get('DP_TEST_GRIPPER_EXAMPLE') == '1' else 0.078,
        closed_width_m=0.0, velocity_mps=0.1,
        force_limit_n=120.0 if os.environ.get('DP_TEST_GRIPPER_EXAMPLE') == '1' else 7.0,
        grasp_epsilon_inner_m=0.07, grasp_epsilon_outer_m=0.07,
        close_with_force=True,
        command_stability_seconds=0.0 if os.environ.get('DP_TEST_GRIPPER_EXAMPLE') == '1' else 0.01))
    controller = GripperActionController(node, settings)
    try:
        deadline = time.monotonic() + 5
        while not controller.ready and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
        assert controller.ready
        for width in (settings.open_width_m, 0.0):
            deadline = time.monotonic() + 3
            while controller.last_completed != width and time.monotonic() < deadline:
                controller.tick(width)
                rclpy.spin_once(node, timeout_sec=0.01)
                assert controller.error is None, controller.error
            assert controller.last_completed == width
        for _ in range(10):
            controller.tick(0.0)
            rclpy.spin_once(node, timeout_sec=0.01)
        assert [kind for kind, _ in received] == ['move', 'grasp']
        move, grasp = received[0][1], received[1][1]
        assert move.width == settings.open_width_m and move.speed == 0.1
        assert grasp.width == 0 and grasp.speed == 0.1 and grasp.force == settings.force_limit_n
        assert grasp.epsilon.inner == grasp.epsilon.outer == 0.07
        controller.disable()
        assert not controller.busy
        print('PASS: isolated Move/Grasp goals, speed/force/epsilon, deduplication')
        check_bridge(node, settings, received)
    finally:
        for client in controller.clients.values():
            client.destroy()
        for server in servers:
            server.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
