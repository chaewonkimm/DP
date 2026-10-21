"""Shared CLI implementation for the task-specific Franka evaluation scripts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from diffusion_policy.real_world.p2p_franka_deployment import (
    EXECUTE_CONFIRMATION,
    load_policy_from_checkpoint,
    run_live,
    run_offline_check,
    validate_checkpoint_deployment_contract,
    validate_execute_configuration,
    validate_live_configuration,
    validate_inference_seed,
    validate_ft_bias_configuration,
)


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "peg": ROOT / "diffusion_policy/config/deploy/p2p_peg_franka.yaml",
    "peg_goal_prop_desired": ROOT / "diffusion_policy/config/deploy/p2p_peg_goal_prop_desired_franka.yaml",
    "peg_goal_prop_next_state": ROOT / "diffusion_policy/config/deploy/p2p_peg_goal_prop_next_state_franka.yaml",
    "peg_goal_prop_wrench_desired": ROOT / "diffusion_policy/config/deploy/p2p_peg_goal_prop_wrench_desired_franka.yaml",
    "peg_goal_prop_wrench_next_state": ROOT / "diffusion_policy/config/deploy/p2p_peg_goal_prop_wrench_next_state_franka.yaml",
    "valve": ROOT / "diffusion_policy/config/deploy/p2p_valve_franka.yaml",
}


def apply_valve_gripper_action_profile(cfg, task):
    """Opt-in, in-memory profile matching the user's ros2 action examples.

    Keep all execution/enable/safety/mapping gates intact. The normal valve YAML
    still owns the lower-force default when this profile is not requested.
    """
    if task != 'valve' or int(cfg.task.action_dim) != 10:
        raise ValueError('The gripper action profile is only for the 10D Valve policy')
    if str(cfg.live.backend) != 'remote_desired':
        raise ValueError('The gripper action profile requires remote_desired backend')
    cfg.gripper.update(dict(
        remote_desired_mode='franka_action', action_namespace='/rightarm/remote_right_gripper',
        command_threshold=0.0, open_width_m=0.08, closed_width_m=0.0,
        velocity_mps=0.1, force_limit_n=120.0, close_with_force=True,
        grasp_epsilon_inner_m=0.07, grasp_epsilon_outer_m=0.07,
        command_stability_seconds=0.0, ros_command_topic=None,
    ))


def parse_args(task: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Evaluate the {task} Diffusion Policy on a Franka")
    parser.add_argument(
        "--mode", choices=("offline", "shadow", "execute"), default="offline")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None, help="For example cuda:0 or cpu")
    parser.add_argument(
        "--ros-root",
        type=Path,
        default=None,
        help=(
            "Path to the iris_teleoperation workspace on the deployment PC. "
            "Overrides remote_desired.bridge_python and ros_setup_file from YAML."
        ),
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--inference-seed", type=int, default=None,
        help="Restart diffusion RNG with this seed for every chunk; overrides YAML inference_seed")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument('--ft-bias', action='store_true', help='Zero residual FT in policy inputs at startup (default window 0.2 s)')
    parser.add_argument('--confirm-ft-bias-free-space', action='store_true',
                        help='Confirm stationary, non-contact tool with fixed gripper and matching pose/wrench axes')
    parser.add_argument(
        '--gripper-action-profile', action='store_true',
        help='Valve only: opt into Move width=0.08 and Grasp force=120 N; float >=0 closes, <0 opens')
    parser.add_argument(
        "--confirm",
        default=None,
        help=f"Execute mode additionally requires: {EXECUTE_CONFIRMATION}",
    )
    return parser.parse_args()


def main(task: str) -> None:
    if task not in CONFIGS:
        raise ValueError(f"Unknown evaluation task: {task}")
    args = parse_args(task)
    config_path = (args.config or CONFIGS[task]).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    if args.ft_bias or args.confirm_ft_bias_free_space:
        if cfg.live.get('backend') != 'remote_desired':
            raise ValueError('Startup FT bias currently supports remote_desired only')
        if 'ft_bias' not in cfg.live.remote_desired:
            cfg.live.remote_desired.ft_bias = {}
        if args.ft_bias:
            cfg.live.remote_desired.ft_bias.enabled = True
        cfg.live.remote_desired.ft_bias.free_space_confirmed = args.confirm_ft_bias_free_space
    ft_settings = validate_ft_bias_configuration(cfg)
    if ft_settings.enabled and args.mode == 'offline':
        raise ValueError('Startup FT calibration requires shadow or execute; offline mode has no startup stream')
    if args.gripper_action_profile:
        apply_valve_gripper_action_profile(cfg, task)
        print('Valve gripper action profile: prediction <0 -> -1 Move(width=0.08, speed=0.1); '
              'prediction >=0 -> +1 Grasp(width=0, speed=0.1, force=120 N, epsilon=0.07/0.07). '
              'First command and subsequent state changes only; no automatic enable.', flush=True)
    if args.inference_seed is not None:
        cfg.inference_seed = args.inference_seed
    validate_inference_seed(cfg.get("inference_seed"))
    configured_checkpoint = Path(str(cfg.checkpoint)).expanduser()
    if not configured_checkpoint.is_absolute():
        configured_checkpoint = ROOT / configured_checkpoint
    cfg.checkpoint = str(configured_checkpoint.resolve())
    remote_desired = cfg.live.get("remote_desired")
    if args.ros_root is not None:
        if remote_desired is None:
            raise ValueError("--ros-root requires the remote_desired live backend")
        ros_root = args.ros_root.expanduser().resolve()
        remote_desired.bridge_python = str(
            ros_root / ".pixi/envs/default/bin/python")
        remote_desired.ros_setup_file = str(ros_root / "install/setup.bash")
    if remote_desired is not None and remote_desired.get("bridge_script"):
        bridge_script = Path(str(remote_desired.bridge_script)).expanduser()
        if not bridge_script.is_absolute():
            bridge_script = ROOT / bridge_script
        remote_desired.bridge_script = str(bridge_script.resolve())
    if args.checkpoint is not None:
        cfg.checkpoint = str(args.checkpoint.expanduser().resolve())
    if args.device is not None:
        cfg.device = args.device
    if args.num_inference_steps is not None:
        cfg.num_inference_steps = args.num_inference_steps
    duration_seconds = (
        float(args.duration_seconds)
        if args.duration_seconds is not None
        else float(cfg.live.duration_seconds)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        ROOT / "data/eval" / str(cfg.task.name) / f"{timestamp}_{args.mode}")

    if args.mode == "execute":
        # Check the cheap gates before deserializing a multi-GB checkpoint.
        validate_execute_configuration(cfg, args.confirm)
    if args.mode != "offline":
        validate_live_configuration(cfg)

    print(f"Loading {cfg.weights} from {cfg.checkpoint}")
    print(f"Inference seed: {cfg.get('inference_seed')} (null/None: random; integer: reset each chunk)")
    policy, train_cfg = load_policy_from_checkpoint(
        checkpoint_path=str(cfg.checkpoint),
        device=str(cfg.device),
        weights=str(cfg.weights),
        num_inference_steps=int(cfg.num_inference_steps),
    )
    validate_checkpoint_deployment_contract(train_cfg, cfg)
    if args.mode == "offline":
        report = run_offline_check(
            policy=policy,
            train_cfg=train_cfg,
            deploy_cfg=cfg,
            sample_index=args.sample_index,
            output_dir=output_dir,
        )
    else:
        report = run_live(
            policy=policy,
            train_cfg=train_cfg,
            deploy_cfg=cfg,
            mode=args.mode,
            duration_seconds=duration_seconds,
            output_dir=output_dir,
            confirmation=args.confirm,
        )
    print(json.dumps(report, indent=2, default=str))
