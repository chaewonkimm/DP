#!/usr/bin/env python3
"""Check imports, ROS message ABI, and optionally real checkpoints without hardware.

Synthetic frames validate inference plumbing only, not policy performance.
No ROS node, camera stream or robot publisher is created.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', action='store_true')
    parser.add_argument('--task', choices=['peg', 'valve', 'both'], default='both')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy.real_world.p2p_franka_deployment import (
        _RemoteDesiredWorkerState, _jsonable, _remote_desired_bridge_payload,
        load_policy_from_checkpoint, validate_live_configuration, write_json,
    )
    from diffusion_policy.real_world.p2p_remote_desired_bridge_core import BridgeSettings

    packages = ['torch', 'torchvision', 'numpy', 'diffusers', 'hydra-core',
                'omegaconf', 'zarr', 'numcodecs', 'imagecodecs']
    report = {
        'python': sys.version,
        'executable': sys.executable,
        'packages': {name: importlib.metadata.version(name) for name in packages},
        'cuda_available': torch.cuda.is_available(),
        'cuda_runtime': torch.version.cuda,
        'checks': [],
        'hardware_commands_sent': 0,
        'observation_source': 'synthetic; not a policy accuracy evaluation',
    }
    if torch.cuda.is_available():
        report['gpu'] = torch.cuda.get_device_name(0)
        # Force actual kernel execution, not only driver discovery.
        matrix = torch.ones((64, 64), device='cuda')
        assert float((matrix @ matrix)[0, 0]) == 64.0
        torch.cuda.synchronize()
    elif args.checkpoints and args.device.startswith('cuda'):
        raise RuntimeError('CUDA is unavailable; use --device cpu only for CPU checks')

    tasks = ['peg', 'valve'] if args.task == 'both' else [args.task]
    for task in tasks:
        cfg = OmegaConf.load(ROOT / f'diffusion_policy/config/deploy/p2p_{task}_franka.yaml')
        cfg.checkpoint = str(ROOT / cfg.checkpoint)
        cfg.live.remote_desired.bridge_script = str(ROOT / cfg.live.remote_desired.bridge_script)
        validate_live_configuration(cfg)
        check = {'task': task, 'ros_imports': 'passed', 'checkpoint_exists': Path(cfg.checkpoint).is_file()}
        if args.checkpoints:
            started = time.monotonic()
            print(f'Loading {task} checkpoint on {args.device}', flush=True)
            policy, train_cfg = load_policy_from_checkpoint(
                cfg.checkpoint, args.device, cfg.weights, int(cfg.num_inference_steps))
            check['load_seconds'] = time.monotonic() - started
            check['dataset_path'] = str(train_cfg.task.dataset.dataset_path)
            check['dataset_exists'] = Path(check['dataset_path']).is_dir()
            count = int(policy.n_obs_steps)
            state = np.zeros((count, 13), dtype=np.float32)
            state[:, :3] = [0.5, 0.0, 0.3]
            state[:, 6] = 1.0  # ROS xyzw identity
            success, encoded = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
            assert success
            request = {'request_type': 'predict', 'protocol_version': 1, 'sequence': 1,
                       'observation.state': state,
                       'images': {key: [encoded.tobytes()] * count
                                  for key in ['external_img', 'left_wrist_img']}}
            with tempfile.TemporaryDirectory(prefix=f'dp-{task}-env-') as directory:
                worker = _RemoteDesiredWorkerState(policy, train_cfg, cfg, Path(directory), 'env-check')
                response = worker.predict(request)
                actions = np.asarray(response['actions'])
                width_channel = (int(cfg.task.action_dim) == 10 and cfg.gripper.remote_desired_mode
                                 in ('ros_float64_width', 'franka_action'))
                assert actions.shape == (int(cfg.live.remote_desired.chunk_size), 8 if width_channel else 7)
                assert np.isfinite(actions).all()
                check['absolute_action_shape'] = list(actions.shape)
                check['inference_seconds'] = response['inference_time_sec']
            # Exercise the actual serialization used to spawn the ROS bridge.
            # Use in-memory bounds to test ListConfig serialization; never save
            # or enable this configuration for hardware execution.
            payload_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            payload_cfg.safety.workspace_min = [0.0, -1.0, 0.0]
            payload_cfg.safety.workspace_max = [1.0, 1.0, 1.0]
            for mode in ['shadow', 'execute']:
                payload = json.loads(json.dumps(_jsonable(_remote_desired_bridge_payload(policy, payload_cfg, mode))))
                BridgeSettings.from_payload(payload)
            check['bridge_payload'] = 'passed'
            del worker, policy, train_cfg
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report['checks'].append(check)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
