"""Opt-in ROS subprocess regression; isolated domain/topics, no hardware.

Run with DP_TEST_ROS=1 in the DP environment on the ROS workstation.
"""
import fcntl
import json
import os
from pathlib import Path
import socketserver
import subprocess
import threading
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from diffusion_policy.real_world.p2p_franka_deployment import (
    _jsonable, _remote_desired_bridge_payload,
)
from p2p_remote_desired_real_bridge import _read_frame, _write_frame


@pytest.mark.skipif(os.environ.get("DP_TEST_ROS") != "1",
                    reason="requires workstation ROS installation and local sockets")
@pytest.mark.parametrize("healthy", [True, False])
def test_bridge_exits_and_releases_lock(tmp_path, healthy):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "diffusion_policy/config/deploy/p2p_peg_franka.yaml")
    payload = _jsonable(_remote_desired_bridge_payload(
        SimpleNamespace(n_obs_steps=2), cfg, "shadow"))
    for key in payload["topics"]:
        payload["topics"][key] = f"/dp_lifecycle_test/{key}"
    payload["observation"]["camera_topics"] = {
        key: payload["topics"][key] for key in ("external_img", "left_wrist_img")}
    payload["timing"]["worker_heartbeat_period_sec"] = 0.1
    payload["timing"]["worker_disconnect_timeout_sec"] = 0.2

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            _read_frame(self.request)
            _write_frame(self.request, {"ok": healthy})

    lock_path = tmp_path / "bridge.lock"
    pid_path = tmp_path / "bridge.json"
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = dict(os.environ)
        env.update(ROS_DOMAIN_ID="187", ROS_LOCALHOST_ONLY="1",
                   ROS_LOG_DIR=str(tmp_path / "ros_logs"),
                   PYTHONPATH=str(root),
                   DP_P2P_BRIDGE_CONFIG_JSON=json.dumps(payload))
        command = [
            "/bin/bash", "-c", 'set -e; source "$1"; shift; exec "$@"',
            "dp-lifecycle-test", str(cfg.live.remote_desired.ros_setup_file),
            str(cfg.live.remote_desired.bridge_python), "-c",
            "import sys; import p2p_remote_desired_real_bridge as b; "
            "b.LOCK_FILE=sys.argv.pop(1); raise SystemExit(b.main())",
            str(lock_path), "--worker-host", "127.0.0.1", "--worker-port",
            str(server.server_address[1]), "--instance-id", "lifecycle-test",
            "--pid-file", str(pid_path), "--echo-only", "--duration-sec",
            "0.5" if healthy else "10",
        ]
        try:
            result = subprocess.run(command, env=env, cwd=root, timeout=8,
                                    capture_output=True, text=True)
        finally:
            server.shutdown()
            thread.join(timeout=2)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = "duration reached" if healthy else "worker unavailable"
    assert expected in result.stdout + result.stderr
    assert not pid_path.exists()
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)
