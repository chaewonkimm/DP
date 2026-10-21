#!/usr/bin/env bash
# Valve launcher: follow YAML; hold_current currently disables DP gripper output.
# Do not inject --gripper-action-profile: that legacy profile overrides these
# settings with width=0 and force=120 N. No subprocess per action or auto-enable.
# Usage (moves the arm after enable; gripper output depends on YAML/CLI):
# bash scripts/run_valve_gripper_actions.sh --mode execute --duration-seconds 10 \
#   --inference-seed 42 --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA
# For a non-actuating bridge use --mode shadow instead (no confirmation needed).
set -e
DP_VALVE_LAUNCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${DP_VALVE_LAUNCH_ROOT}/scripts/activate_rollout_env.sh"
exec python "${DP_VALVE_LAUNCH_ROOT}/eval_real_valve.py" "$@"
