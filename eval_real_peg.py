#!/usr/bin/env python3
"""Evaluate the arm-only, 9D square-peg policy via Franka RemoteDesired."""

# 기본 live backend는 split RemoteDesired bridge입니다. 먼저 --mode shadow로
# /rightarm/policy_debug/remote_desired_shadow와 status를 확인하세요.
# Peg는 gripper가 없으며 TCP pose만 전송합니다.
# 실제 로봇 실행 (별도 터미널에서 /rightarm/policy_enable=true 필요)
# CUDA_VISIBLE_DEVICES=0 python eval_real_peg.py \
#   --mode execute \
#   --duration-seconds 3 \
#   --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

from eval_real_p2p_common import main


if __name__ == "__main__":
    main("peg")
