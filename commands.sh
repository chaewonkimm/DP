#!/usr/bin/env bash
# Command notebook. Copy only the command block needed for one terminal.
# Running this file directly intentionally performs no deployment action.
: <<'DP_COMMAND_NOTES'

# peg in hole (vision + wrench + prop, next state) - w/o inference proprocessing

source scripts/activate_rollout_env.sh
python eval_real_peg.py --mode execute \
  --duration-seconds 120 \
  --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA


# =============================================================================
# Low-dimensional peg DP: deployment desktop commands
# =============================================================================
: <<'PEG_LOWDIM_DEPLOY_COMMANDS'

# [배포 데스크탑] validation-best + train-best 가중치 다운로드
cd /home/iristele/work/chaewon/DP

# 먼저 실제 전송 없이 대상 8개 확인
bash scripts/download_peg_lowdim_checkpoints.sh --dry-run

# 확인 후 실제 다운로드. 이미 동일한 파일은 다시 받지 않는다.
bash scripts/download_peg_lowdim_checkpoints.sh


# DP rollout 가상환경 생성
cd /home/iristele/work/chaewon/DP
bash scripts/setup_rollout_env.sh


# [Terminal 1] FR3 remote endpoint 실행
cd /home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation/src
./remote_runfile/start_fr3_remote.sh \
  --endpoint-only \
  --start-from-current

# [Terminal 2] Shadow rollout
cd /home/iristele/work/chaewon/DP
export DP_ROS_ROOT=/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation

# 1) goal + prop, desired-command target
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_desired shadow \
  --duration-seconds 120 \
  --inference-seed 42

# 2) goal + prop, next-state target
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_next_state shadow \
  --duration-seconds 120 \
  --inference-seed 42

# 3) goal + prop + wrench, desired-command target
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_wrench_desired shadow \
  --duration-seconds 120 \
  --inference-seed 42

# 4) goal + prop + wrench, next-state target
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_wrench_next_state shadow \
  --duration-seconds 120 \
  --inference-seed 42


# [Terminal 3] ROS status와 enable gate
cd /home/iristele/work/chaewon/DP
export DP_ROS_ROOT=/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation
source scripts/activate_rollout_ros.sh

# 정책 시작 전 또는 중지할 때 gate 내리기
ros2 topic pub --once \
  /rightarm/policy_enable std_msgs/msg/Bool \
  "{data: false}"

# 상태 확인
ros2 topic echo \
  /rightarm/policy_debug/status std_msgs/msg/String \
  --field data --full-length

# Shadow/execute 준비 상태를 확인한 뒤 gate 올리기
ros2 topic pub --once \
  /rightarm/policy_enable std_msgs/msg/Bool \
  "{data: true}"


# [선택 사항] Execute용 로컬 config 복사
# 기본 config는 기존 vision+wrench+prop next-state peg 기준과 동일하게
# execute_enabled=true 및 safety 값이 설정되어 있다. 데스크탑별 override를
# 별도 보관하고 싶을 때만 아래 복사본을 사용한다.
cd /home/iristele/work/chaewon/DP

cp diffusion_policy/config/deploy/p2p_peg_goal_prop_desired_franka.yaml \
  diffusion_policy/config/deploy/local_peg_goal_prop_desired_franka.yaml

cp diffusion_policy/config/deploy/p2p_peg_goal_prop_next_state_franka.yaml \
  diffusion_policy/config/deploy/local_peg_goal_prop_next_state_franka.yaml

cp diffusion_policy/config/deploy/p2p_peg_goal_prop_wrench_desired_franka.yaml \
  diffusion_policy/config/deploy/local_peg_goal_prop_wrench_desired_franka.yaml

cp diffusion_policy/config/deploy/p2p_peg_goal_prop_wrench_next_state_franka.yaml \
  diffusion_policy/config/deploy/local_peg_goal_prop_wrench_next_state_franka.yaml


# [Terminal 2] 실제 로봇 execute
# 아래 명령 전 Terminal 3에서 policy_enable=false로 두고, 정책과 status를 확인한
# 다음에만 policy_enable=true를 보낸다. 한 번에 모델 하나만 실행한다.
cd /home/iristele/work/chaewon/DP
export DP_ROS_ROOT=/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation

# 1) goal + prop, desired-command target
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_desired execute \
  --duration-seconds 120 \
  --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

# 2) goal + prop, next-state target
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_next_state execute \
  --duration-seconds 120 \
  --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

# 3) goal + prop + wrench, desired-command target
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_wrench_desired execute \
  --duration-seconds 120 \
  --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

# 4) goal + prop + wrench, next-state target
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_wrench_next_state execute \
  --duration-seconds 120 \
  --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA


# [중지] Terminal 3에서 먼저 gate를 내리고 Terminal 2에서 Ctrl-C
ros2 topic pub --once \
  /rightarm/policy_enable std_msgs/msg/Bool \
  "{data: false}"

PEG_LOWDIM_DEPLOY_COMMANDS

DP_COMMAND_NOTES
