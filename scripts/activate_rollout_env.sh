# Source this file in Bash: source scripts/activate_rollout_env.sh
DP_ROLLOUT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${DP_ROLLOUT_ROOT}/.venv/bin/activate" ]]; then
  echo 'Run bash scripts/setup_rollout_env.sh first.' >&2
  return 1
fi
source "${DP_ROLLOUT_ROOT}/.venv/bin/activate"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_launch_logs}"
# Avoid excessive CPU thread creation during checkpoint construction/inference.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="${DP_ROLLOUT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
