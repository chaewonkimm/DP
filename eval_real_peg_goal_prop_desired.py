#!/usr/bin/env python3
"""FR3 rollout for goal + proprioception DP trained on desired commands."""

# Shadow:
# CUDA_VISIBLE_DEVICES=0 python eval_real_peg_goal_prop_desired.py --mode shadow
#
# 실제 로봇 실행 (YAML safety/execute_enabled 설정 및 shadow 검증 후):
# CUDA_VISIBLE_DEVICES=0 python eval_real_peg_goal_prop_desired.py \
#   --mode execute --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

from eval_real_p2p_common import main


if __name__ == "__main__":
    main("peg_goal_prop_desired")
