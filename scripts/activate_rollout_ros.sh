# Source in a separate terminal for ros2 CLI / remote camera tools.
# Do not source this into the DP policy terminal.
DP_ROS_ROOT="${DP_ROS_ROOT:-/home/iristele/work/seongsu/iris_learning_from_teleop_fr3/iris_teleoperation}"
if [[ ! -f "${DP_ROS_ROOT}/.pixi/envs/default/setup.bash" || ! -f "${DP_ROS_ROOT}/install/setup.bash" ]]; then
  echo "ROS installation not found under ${DP_ROS_ROOT}" >&2
  return 1
fi
source "${DP_ROS_ROOT}/.pixi/envs/default/setup.bash"
source "${DP_ROS_ROOT}/install/setup.bash"
export PATH="${DP_ROS_ROOT}/.pixi/envs/default/bin:${PATH}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_launch_logs}"
