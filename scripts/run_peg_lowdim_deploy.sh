#!/usr/bin/env bash
# Deployment-PC launcher for the four low-dimensional square-peg DP policies.
#
# This starts the DP policy in the repository rollout environment. The policy
# process then starts p2p_remote_desired_real_bridge.py with the ROS Python from
# DP_ROS_ROOT. Robot endpoint startup and /rightarm/policy_enable are deliberately
# kept as separate, explicit operations.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_peg_lowdim_deploy.sh MODEL MODE [evaluation options]

MODEL:
  goal_prop_desired
  goal_prop_next_state
  goal_prop_wrench_desired
  goal_prop_wrench_next_state

MODE:
  shadow    Publish only to /rightarm/policy_debug/remote_desired_shadow
  execute   Publish to /rightarm/Teleop_RemoteDesired (all safety gates apply)

Environment:
  DP_ROS_ROOT          iris_teleoperation workspace
  CUDA_VISIBLE_DEVICES GPU visible to DP (default: 0)

Examples:
  bash scripts/run_peg_lowdim_deploy.sh goal_prop_desired shadow \
    --duration-seconds 30 --inference-seed 42

  bash scripts/run_peg_lowdim_deploy.sh goal_prop_wrench_desired execute \
    --duration-seconds 30 --inference-seed 42 \
    --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

MODEL="$1"
MODE="$2"
shift 2

case "${MODEL}" in
  goal_prop_desired)
    ENTRYPOINT="eval_real_peg_goal_prop_desired.py"
    ;;
  goal_prop_next_state)
    ENTRYPOINT="eval_real_peg_goal_prop_next_state.py"
    ;;
  goal_prop_wrench_desired)
    ENTRYPOINT="eval_real_peg_goal_prop_wrench_desired.py"
    ;;
  goal_prop_wrench_next_state)
    ENTRYPOINT="eval_real_peg_goal_prop_wrench_next_state.py"
    ;;
  *)
    echo "Unknown MODEL: ${MODEL}" >&2
    usage >&2
    exit 2
    ;;
esac

case "${MODE}" in
  shadow|execute)
    ;;
  *)
    echo "MODE must be shadow or execute, got: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

DP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_ROOT="${DP_ROS_ROOT:-/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation}"

if [[ ! -f "${ROS_ROOT}/install/setup.bash" ]]; then
  echo "ROS setup not found: ${ROS_ROOT}/install/setup.bash" >&2
  echo "Set DP_ROS_ROOT to the iris_teleoperation workspace." >&2
  exit 1
fi
if [[ ! -x "${ROS_ROOT}/.pixi/envs/default/bin/python" ]]; then
  echo "ROS Python not found: ${ROS_ROOT}/.pixi/envs/default/bin/python" >&2
  exit 1
fi
if [[ ! -f "${DP_ROOT}/.venv/bin/activate" ]]; then
  echo "DP rollout environment is missing." >&2
  echo "Create it once with: bash ${DP_ROOT}/scripts/setup_rollout_env.sh" >&2
  exit 1
fi

source "${DP_ROOT}/scripts/activate_rollout_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${DP_ROOT}"
exec python "${ENTRYPOINT}" \
  --mode "${MODE}" \
  --ros-root "${ROS_ROOT}" \
  "$@"
