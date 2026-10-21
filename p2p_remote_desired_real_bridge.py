#!/usr/bin/env python3
"""ROS 2 bridge between a DP policy worker and ``Teleop_RemoteDesired``.

Run this with the iris_teleoperation ROS Python, not the Diffusion Policy
Python.  ``eval_real_valve.py`` normally starts it automatically.  Its design
is based on training/runtime/learned_remote_desired_real_bridge.py in the FR3
teleoperation repository, with an absolute-pose protocol for this repository's
relative-to-observation DP checkpoints.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pickle
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_policy.real_world.p2p_remote_desired_bridge_core import (
    AbsoluteChunkPlayer,
    BridgeSettings,
    ObservationBuffer,
    StreamingWrenchFilter,
    WrenchFilterSettings,
    validate_target,
)
from diffusion_policy.real_world.p2p_gripper_actions import (
    GripperActionController, GripperActionSettings,
)
from diffusion_policy.real_world.p2p_ft_bias import FTBiasSettings, StartupFTBias


LOCK_FILE = "/tmp/dp_p2p_remote_desired/bridge.lock"


def _write_frame(sock: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)) + data)


def _read_frame(sock: socket.socket) -> Any:
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise EOFError("Policy worker closed the connection")
        header += chunk
    (size,) = struct.unpack("!I", header)
    body = b""
    while len(body) < size:
        chunk = sock.recv(min(1 << 20, size - len(body)))
        if not chunk:
            raise EOFError("Policy worker closed the connection")
        body += chunk
    return pickle.loads(body)


class WorkerClient:
    def __init__(self, host: str, port: int, timeout_sec: float) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        with socket.create_connection(
                (self.host, self.port), timeout=self.timeout_sec) as sock:
            _write_frame(sock, payload)
            return _read_frame(sock)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--echo-only", action="store_true")
    parser.add_argument("--real-robot", action="store_true")
    parser.add_argument("--duration-sec", type=float)
    args = parser.parse_args(argv)

    payload = json.loads(os.environ["DP_P2P_BRIDGE_CONFIG_JSON"])
    settings = BridgeSettings.from_payload(payload)
    wrench_filter_settings = WrenchFilterSettings.from_payload(
        payload.get("observation", {}).get("wrench_filter"))
    ft_settings = FTBiasSettings.from_payload(payload.get('ft_bias'))
    if ft_settings.enabled and not settings.require_enable:
        raise ValueError('FT calibration requires an explicit enable gate')
    topics = payload["topics"]
    safety = payload["safety"]
    publish = bool(args.publish) and not args.echo_only
    gripper_payload = payload.get("gripper", {})
    gripper_mode = gripper_payload.get("mode", "disabled")
    gripper_settings = (GripperActionSettings.from_payload(gripper_payload)
                        if gripper_mode == "franka_action" else None)
    if publish and not (
            args.real_robot and bool(safety.get("allow_real_robot", False))):
        print(
            "[dp_p2p_bridge] publish requires --real-robot and "
            "safety.allow_real_robot=true", file=sys.stderr)
        return 2

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import Bool, Float64, Float64MultiArray, String

    try:
        from iris_teleop.msg import TeleopState
    except ImportError as error:  # pragma: no cover - robot PC only
        print(f"[dp_p2p_bridge] cannot import iris_teleop messages: {error}",
              file=sys.stderr)
        return 2

    Path(LOCK_FILE).parent.mkdir(parents=True, exist_ok=True)
    lock_handle = open(LOCK_FILE, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[dp_p2p_bridge] another bridge holds the lock", file=sys.stderr)
        return 3

    pid_file = Path(args.pid_file)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(json.dumps({
        "pid": os.getpid(),
        "instance_id": args.instance_id,
        "bridge_script": str(Path(__file__).resolve()),
        "worker_port": args.worker_port,
    }), encoding="utf-8")

    camera_topics = dict(payload["observation"]["camera_topics"])
    camera_names = tuple(camera_topics)
    client = WorkerClient(
        args.worker_host,
        args.worker_port,
        float(payload["timing"]["worker_timeout_sec"]),
    )
    stop_requested = threading.Event()

    class BridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("dp_p2p_remote_desired_bridge")
            self.lock = threading.Lock()
            self.observations = ObservationBuffer(settings, camera_names)
            self.wrench_filter = StreamingWrenchFilter(wrench_filter_settings)
            self.player = AbsoluteChunkPlayer(settings)
            self.enabled = not settings.require_enable
            self.fault_latched = False
            self.worker_healthy = True
            self.last_heartbeat_ok = time.monotonic()
            self.last_state_msg: Any = None
            self.latest_raw_wrench: np.ndarray | None = None
            self.current_target: np.ndarray | None = None
            self.current_target_time: float | None = None
            self.current_gripper: float | None = None
            self.last_reason = "startup"
            self.last_metrics: dict[str, float] = {}
            self.last_inference_sec: float | None = None
            self.tick_index = 0
            self.started = time.monotonic()
            self.ft_bias = StartupFTBias(ft_settings, self.started)
            self.ft_report_saved = False
            self.publisher_safe = True

            self.create_subscription(
                TeleopState, topics["remote_state"], self.on_state,
                qos_profile_sensor_data)
            for name, topic in camera_topics.items():
                self.create_subscription(
                    CompressedImage, topic,
                    lambda msg, key=name: self.on_image(key, msg),
                    qos_profile_sensor_data)
            self.create_subscription(Bool, topics["enable"], self.on_enable, 10)

            self.command_topic = (
                topics["remote_desired"] if publish
                else topics["remote_desired_shadow"])
            self.command_pub = self.create_publisher(
                TeleopState, self.command_topic, qos_profile_sensor_data)
            self.action_pub = self.create_publisher(
                Float64MultiArray, topics["dry_run_action"], 10)
            self.status_pub = self.create_publisher(String, topics["status"], 10)

            gripper_topic = payload.get("gripper", {}).get("ros_command_topic")
            self.gripper_pub = None
            self.gripper_actions = None
            if gripper_mode == "ros_float64_width" and gripper_topic and publish:
                self.gripper_pub = self.create_publisher(
                    Float64, str(gripper_topic), 10)
            if gripper_settings is not None and publish:
                self.gripper_actions = GripperActionController(self, gripper_settings)
                # Action network I/O never runs in the 1 kHz pose publisher.
                self.create_timer(0.05, self.on_gripper)

            self.create_timer(settings.dt_sec, self.on_tick)
            self.create_timer(
                1.0 / float(payload["timing"]["command_publish_rate_hz"]),
                self.on_command)
            self.create_timer(0.5, self.on_status)
            self.create_timer(
                float(payload["timing"]["worker_heartbeat_period_sec"]),
                self.on_heartbeat)
            mode = f"PUBLISH {self.command_topic}" if publish else (
                f"echo-only {self.command_topic}")
            self.get_logger().info(f"DP p2p bridge {args.instance_id}: {mode}")

        def clear_target(self, reason: str) -> None:
            self.player.queue.clear()
            self.player.next_chunk = None
            self.player.cancel_request()
            self.current_target = None
            self.current_target_time = None
            self.current_gripper = None
            self.last_reason = reason

        def latch_fault(self, reason: str) -> None:
            self.fault_latched = True
            self.enabled = False
            self.clear_target(reason)

        def on_state(self, msg: Any) -> None:
            p = msg.pose.position
            q = msg.pose.orientation
            force = msg.wrench.force
            torque = msg.wrench.torque
            raw_state = np.asarray([
                p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                force.x, force.y, force.z,
                torque.x, torque.y, torque.z,
            ], dtype=np.float64)
            with self.lock:
                self.last_state_msg = msg
                try:
                    now = time.monotonic()
                    self.latest_raw_wrench = raw_state[7:13].copy()
                    state = raw_state.copy()
                    state[7:13] = self.wrench_filter.push(
                        now, self.latest_raw_wrench)
                    self.observations.push_state(state, now)
                    if not self.enabled:
                        self.ft_bias.observe(self.observations.latest_state, msg.dq, time.monotonic())
                except ValueError as error:
                    if not self.ft_bias.ready:
                        self.ft_bias.reset_window('invalid_state')
                    self.latch_fault(f"invalid_state:{error}")

        def on_image(self, name: str, msg: Any) -> None:
            with self.lock:
                self.observations.push_image(
                    name, bytes(msg.data), time.monotonic())

        def on_enable(self, msg: Any) -> None:
            with self.lock:
                if not bool(msg.data):
                    # A deliberate false -> true cycle is required to clear a
                    # safety/worker fault; a repeated true publisher cannot.
                    self.fault_latched = False
                    self.enabled = False
                    self.clear_target("disabled")
                    if self.gripper_actions is not None:
                        self.gripper_actions.disable()
                        self.gripper_actions.reset_error()
                else:
                    if not self.ft_bias.ready or self.ft_bias.error:
                        self.enabled = False
                        self.clear_target('ft_bias_not_ready')
                        self.get_logger().warning('Enable rejected: startup FT bias not ready; check ft_bias status')
                        return
                    if self.gripper_actions is not None and (
                            not self.gripper_actions.ready or self.gripper_actions.error
                            or self.gripper_actions.busy):
                        self.latch_fault("gripper_not_ready")
                        return
                    self.enabled = self.worker_healthy and not self.fault_latched

        def on_heartbeat(self) -> None:
            try:
                reply = client.call({
                    "request_type": "heartbeat",
                    "protocol_version": 1,
                    "instance_id": args.instance_id,
                })
                healthy = bool(reply.get("ok"))
            except Exception as error:  # noqa: BLE001
                self.get_logger().warning(f"worker heartbeat failed: {error}")
                healthy = False
            with self.lock:
                self.worker_healthy = healthy
                if not healthy:
                    self.latch_fault("worker_unhealthy")
            if healthy:
                self.last_heartbeat_ok = time.monotonic()
            elif time.monotonic() - self.last_heartbeat_ok > float(
                    payload["timing"]["worker_disconnect_timeout_sec"]):
                self.get_logger().error("worker unavailable; shutting down")
                stop_requested.set()

        def send_request(self, sequence: int, request: dict[str, Any]) -> None:
            def work() -> None:
                try:
                    reply = client.call(request)
                    if not reply.get("ok", False):
                        raise RuntimeError(str(reply.get("error", "worker error")))
                    if reply.get("action_representation") != "absolute_pose":
                        raise RuntimeError("worker did not return absolute poses")
                    if reply.get("chunk_start_index", 0) != settings.chunk_start_index:
                        raise RuntimeError("worker/bridge chunk_start_index mismatch")
                    actions = np.asarray(reply["actions"], dtype=np.float64)
                    with self.lock:
                        self.player.on_chunk(
                            sequence, actions, self.tick_index)
                        self.last_inference_sec = float(
                            reply.get("inference_time_sec", 0.0))
                except Exception as error:  # noqa: BLE001
                    with self.lock:
                        self.player.cancel_request()
                        self.clear_target(f"worker_error:{error}")
                    self.get_logger().warning(f"policy request failed: {error}")

            threading.Thread(target=work, daemon=True).start()

        def on_tick(self) -> None:
            request = None
            sequence = None
            with self.lock:
                self.tick_index += 1
                now = time.monotonic()
                self.ft_bias.check_timeout(now)
                if self.ft_bias.error:
                    self.latch_fault('ft_bias:' + self.ft_bias.error)
                    return
                if publish:
                    self.publisher_safe = (
                        self.count_publishers(topics["remote_desired"]) <= 1)
                    if not self.publisher_safe:
                        self.latch_fault("competing_desired_publisher")

                stale = self.observations.freshness(now)
                if stale is not None:
                    if self.enabled:
                        self.latch_fault(stale)
                    else:
                        self.clear_target(stale)
                    return
                self.observations.sample_tick()
                if settings.require_enable and not self.enabled:
                    self.last_reason = "disabled"
                    return
                if not self.worker_healthy:
                    self.clear_target("worker_unhealthy")
                    return
                if self.observations.ready and self.player.should_request():
                    sequence = self.player.start_request(self.tick_index)
                    request = self.observations.request(sequence)
                    if ft_settings.enabled:
                        request['observation.state_uncorrected'] = request['observation.state'].copy()
                        request['observation.state'] = self.ft_bias.correct_states(request['observation.state'])
                        request['ft_bias'] = self.ft_bias.status()

                action = self.player.pop(self.tick_index)
                if action is None:
                    if (self.current_target_time is None or
                            now - self.current_target_time >
                            settings.command_target_timeout_sec):
                        self.current_target = None
                        self.current_gripper = None
                        self.last_reason = "underrun"
                else:
                    measured = self.observations.latest_state
                    safety_wrench = (
                        self.latest_raw_wrench
                        if self.latest_raw_wrench is not None else measured[7:13])
                    reference = (
                        self.current_target
                        if self.current_target is not None else measured[:7])
                    try:
                        self.last_metrics = validate_target(
                            action[:7], reference, safety_wrench, settings)
                    except RuntimeError as error:
                        self.latch_fault(f"safety:{error}")
                        self.get_logger().error(
                            f"command rejected and gate disabled: {error}")
                        return
                    self.current_target = action[:7].copy()
                    self.current_target_time = now
                    self.current_gripper = (
                        float(action[7]) if len(action) == 8 else None)
                    self.last_reason = "ok"
                    self.action_pub.publish(Float64MultiArray(
                        data=[float(value) for value in action]))
            if request is not None:
                self.send_request(int(sequence), request)

        def on_command(self) -> None:
            with self.lock:
                target = None if self.current_target is None else (
                    self.current_target.copy())
                target_time = self.current_target_time
                gripper = self.current_gripper
                state_msg = self.last_state_msg
                enabled = self.enabled or not settings.require_enable
            if (target is None or target_time is None or state_msg is None
                    or not enabled
                    or time.monotonic() - target_time >
                    settings.command_target_timeout_sec):
                return

            msg = TeleopState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = state_msg.header.frame_id
            msg.state = TeleopState.STATE_RUNNING
            msg.clutch1.data = True
            # --load-gripper's legacy bridge toggles on a rising clutch2 edge.
            # Never trigger it, including in hold_current and peg modes.
            msg.clutch2.data = False
            msg.clutch3.data = True
            msg.q = list(state_msg.q)
            msg.dq = list(state_msg.dq)
            msg.tau = list(state_msg.tau)
            msg.pose.position.x = float(target[0])
            msg.pose.position.y = float(target[1])
            msg.pose.position.z = float(target[2])
            msg.pose.orientation.x = float(target[3])
            msg.pose.orientation.y = float(target[4])
            msg.pose.orientation.z = float(target[5])
            msg.pose.orientation.w = float(target[6])
            msg.wrench = state_msg.wrench
            self.command_pub.publish(msg)
            if self.gripper_pub is not None and gripper is not None:
                self.gripper_pub.publish(Float64(data=float(gripper)))

        def on_gripper(self) -> None:
            with self.lock:
                active = (self.enabled and not self.fault_latched and self.worker_healthy
                          and self.publisher_safe and self.current_target_time is not None
                          and time.monotonic() - self.current_target_time <= settings.command_target_timeout_sec
                          and self.observations.freshness(time.monotonic()) is None)
                width = self.current_gripper if active else None
            previous_goals = self.gripper_actions.goals_sent
            self.gripper_actions.tick(width)
            if self.gripper_actions.goals_sent != previous_goals:
                self.get_logger().info(
                    'gripper action sent: ' + json.dumps(self.gripper_actions.last_goal))
            if self.gripper_actions.error:
                with self.lock:
                    first_fault = not self.fault_latched
                    self.latch_fault(f"gripper:{self.gripper_actions.error}")
                if first_fault:
                    self.get_logger().error(f"gripper fault: {self.gripper_actions.error}")

        def stop_gripper_actions(self) -> None:
            if self.gripper_actions is None:
                return
            with self.lock:
                self.enabled = False
                self.clear_target("shutdown")
            self.gripper_actions.disable()
            deadline = time.monotonic() + 1.0
            while rclpy.ok() and self.gripper_actions.busy and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            if self.gripper_actions.busy:
                self.get_logger().error("Gripper cancellation unconfirmed; shutdown is NOT an emergency stop")

        def on_status(self) -> None:
            with self.lock:
                status = {
                    "instance_id": args.instance_id,
                    "mode": "publish" if publish else "echo_only",
                    "command_topic": self.command_topic,
                    "enabled": self.enabled,
                    "fault_latched": self.fault_latched,
                    "worker_healthy": self.worker_healthy,
                    "publisher_safe": self.publisher_safe,
                    "last_reason": self.last_reason,
                    "tick": self.tick_index,
                    "queue_depth": self.player.depth,
                    "chunk_start_index": settings.chunk_start_index,
                    "chunk_size": settings.chunk_size,
                    "request_in_flight": self.player.pending_sequence is not None,
                    "stale_entries_skipped": self.player.stale_entries_skipped,
                    "underrun_ticks": self.player.underrun_ticks,
                    "last_inference_sec": self.last_inference_sec,
                    "last_safety_metrics": self.last_metrics,
                    "gripper_mode": gripper_mode,
                    "ft_bias": self.ft_bias.status(),
                    "wrench_filter": self.wrench_filter.status(),
                    "gripper_command_width_m": self.current_gripper,
                    "gripper_action_status": (self.gripper_actions.status()
                                              if self.gripper_actions is not None else None),
                    "uptime_sec": time.monotonic() - self.started,
                }
            self.status_pub.publish(String(data=json.dumps(status)))
            if ft_settings.enabled and (self.ft_bias.ready or self.ft_bias.error) and not self.ft_report_saved:
                self.get_logger().info('Startup FT calibration: ' + json.dumps(status['ft_bias']))
                report_path = payload.get('ft_bias_report_path')
                if report_path:
                    try:
                        with Path(report_path).open('x', encoding='utf-8') as stream:
                            json.dump(dict(instance_id=args.instance_id, settings=vars(ft_settings),
                                           calibration=status['ft_bias']), stream, indent=2)
                    except OSError as error:
                        with self.lock:
                            self.ft_bias.error = self.ft_bias.reason = 'calibration_report_write_failed'
                            self.latch_fault('ft_bias_report_write_failed')
                        self.get_logger().error(f'FT calibration report not saved: {error}')
                self.ft_report_saved = True
            if (args.duration_sec is not None
                    and time.monotonic() - self.started >= args.duration_sec):
                self.get_logger().info("duration reached; shutting down")
                stop_requested.set()

    node = None
    try:
        rclpy.init()
        node = BridgeNode()
        # Shutdown waits for executor callbacks to finish. Calling it inside a
        # timer callback makes that callback wait for itself on this ROS build.
        # Let the callback return, then perform teardown from the main loop.
        while rclpy.ok() and not stop_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.stop_gripper_actions()
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            try:
                pid_file.unlink()
            except OSError:
                pass
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
