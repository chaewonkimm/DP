rsync -a \
  --info=progress2 \
  --itemize-changes \
  --partial \
  --exclude '/data/' \
  iris@irislab.asuscomm.com:/home/iris/Workspace/junsu/DP/ \
  /home/iristele/work/chaewon/DP/

bash scripts/download_peg_lowdim_checkpoints.sh --dry-run

rsync -aR \
  --info=progress2 \
  --partial \
  iris@irislab.asuscomm.com:/home/iris/Workspace/junsu/DP/./data/outputs/2026.10.11/03.23.44_train_diffusion_unet_peg_proprio_goal_desired_peg_proprio_goal_desired/checkpoints/epoch=0019-val_loss=0.00528.ckpt \
  iris@irislab.asuscomm.com:/home/iris/Workspace/junsu/DP/./data/outputs/2026.10.13/16.47.06_train_diffusion_unet_peg_proprio_goal_next_state_peg_proprio_goal_next_state/checkpoints/epoch=0019-val_loss=0.00437.ckpt \
  iris@irislab.asuscomm.com:/home/iris/Workspace/junsu/DP/./data/outputs/2026.10.11/03.24.37_train_diffusion_unet_peg_proprio_goal_wrench_desired_peg_proprio_goal_wrench_desired/checkpoints/epoch=0019-val_loss=0.00509.ckpt \
  iris@irislab.asuscomm.com:/home/iris/Workspace/junsu/DP/./data/outputs/2026.10.13/16.47.32_train_diffusion_unet_peg_proprio_goal_wrench_next_state_peg_proprio_goal_wrench_next_state/checkpoints/epoch=0019-val_loss=0.00434.ckpt \
  /home/iristele/work/chaewon/DP/
  
source scripts/activate_rollout_env.sh
python eval_real_peg.py --mode execute \
  --duration-seconds 120 \
  --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

source scripts/activate_rollout_ros.sh
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool "{data: true}"

source scripts/activate_rollout_env.sh
bash scripts/run_valve_gripper_actions.sh \
  --mode execute \
  --duration-seconds 120 \
  --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

export DP_ROS_ROOT=/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation
source scripts/activate_rollout_ros.sh

ros2 topic echo /rightarm/policy_debug/status std_msgs/msg/String \
  --field data --full-length --once

ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool "{data: true}"

bash scripts/run_valve_gripper_actions.sh \
  --mode execute --duration-seconds 120 \
  --inference-seed 42 \
  --ft-bias --confirm-ft-bias-free-space \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA

./remote_runfile/start_fr3_remote.sh --endpoint-only --load-gripper

cd /home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation/src

./remote_runfile/start_fr3_remote.sh \
  --endpoint-only \
  --load-gripper \
  --start-from-current

cd /home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation/src

./remote_runfile/start_fr3_remote.sh \
  --endpoint-only \
  --load-gripper \
  --start-from-current