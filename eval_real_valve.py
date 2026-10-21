#!/usr/bin/env python3
"""Evaluate the valve peg-in-hole Diffusion Policy on a Franka FR3."""

# 기본 live backend는 split RemoteDesired bridge입니다. 먼저 --mode shadow로
# /rightarm/policy_debug/remote_desired_shadow와 status를 확인하세요.
# 실제 로봇 실행 (별도 터미널에서 /rightarm/policy_enable=true 필요)
# CUDA_VISIBLE_DEVICES=0 python eval_real_valve.py \
#   --mode execute \
#   --duration-seconds 3 \
#   --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

from eval_real_p2p_common import main


if __name__ == "__main__":
    main("valve")
