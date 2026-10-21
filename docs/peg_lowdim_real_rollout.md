# Peg-in-hole low-dimensional DP real rollout

이 문서는 FR3에서 다음 네 Diffusion Policy를 실행하는 절차를 정리한다.

| 실행 파일 | Observation | 학습 target | 기본 checkpoint |
|---|---|---|---|
| `eval_real_peg_goal_prop_desired.py` | relative TCP + relative goal | desired command | train-loss-best EMA (epoch 780) |
| `eval_real_peg_goal_prop_next_state.py` | relative TCP + relative goal | next state | train-loss-best EMA (epoch 780) |
| `eval_real_peg_goal_prop_wrench_desired.py` | relative TCP + relative goal + wrench | desired command | train-loss-best EMA (epoch 780) |
| `eval_real_peg_goal_prop_wrench_next_state.py` | relative TCP + relative goal + wrench | next state | train-loss-best EMA (epoch 780) |

모든 모델은 vision과 gripper action을 사용하지 않는다. ROS bridge는
`/rightarm/Teleop_Remote`에서 TCP pose와 wrench를 받고, shadow에서는
`/rightarm/policy_debug/remote_desired_shadow`, execute에서는
`/rightarm/Teleop_RemoteDesired`에 absolute TCP pose를 보낸다.

## Goal과 action 변환

학습 데이터의 고정 world-frame goal은 다음 9D pose이다.

```text
[0.61761892, -0.37369144, 0.08422311,
 0.99915916,  0.03782048, 0.01582829,
 0.03767658, -0.99924678, 0.00929329]
```

매 inference마다 최신 TCP pose를 기준으로 변환한다.

```text
T_goal_relative = inverse(T_tcp_current) @ T_goal_world
```

학습의 `single_step_obs_keys=[left_goal_pose_rel]`와 동일하게 이 최신
relative goal을 두 observation step에 반복한다. 정책의 9D relative action은
inference 기준 TCP pose로 다시 absolute pose로 복원한 뒤 ROS에 전달한다.
desired-command 모델과 next-state 모델 모두 이 출력 복원 방식은 동일하며,
checkpoint task name 검사를 통과해야 실행된다.

고정 goal은 robot base와 hole fixture가 데이터 수집 때와 같은 좌표 관계일
때만 유효하다. 둘 중 하나라도 움직였거나 재영점했다면 먼저 goal을 다시
측정하고 네 deploy YAML의 `task.goal_pose_world_9d`를 갱신해야 한다.

## Wrench preprocessing

두 wrench 모델은 학습 데이터와 동일하게 raw `Teleop_Remote` wrench를
1001.74 Hz uniform grid로 causal resampling하고, 15 Hz 2차 Butterworth LPF를
적용한다. 필터는 첫 샘플의 steady state로 초기화하며 미래 샘플을 사용하는
`filtfilt`가 아니다. Prop-only 모델은 wrench를 정책에 전달하지 않지만,
raw wrench는 execute safety gate에서 계속 사용한다.

## 최초 1회 설정

각 deploy YAML에서 로봇 PC에 맞게 다음 두 경로를 확인한다.

```yaml
live:
  remote_desired:
    bridge_python: /path/to/ROS/python
    ros_setup_file: /path/to/iris_teleoperation/install/setup.bash
```

실제 구동 전에는 해당 YAML에 fixture별 안전 범위를 직접 입력한다.

```yaml
live:
  execute_enabled: true
safety:
  workspace_min: [x_min, y_min, z_min]
  workspace_max: [x_max, y_max, z_max]
  max_translation_step_m: ...
  max_rotation_step_rad: ...
  max_force_n: ...
  max_torque_nm: ...
```

제공된 YAML은 위 값이 `null`이고 `execute_enabled=false`이므로 그대로는
실제 로봇을 움직이지 않는다. 기록 데이터의 전체 범위를 안전 한계로 그대로
복사하지 말고, 느린 hand-guided 점검과 controller 한계를 기준으로 정한다.

## 실행 순서

DP 환경을 활성화한다.

```bash
cd /home/iris/Workspace/junsu/DP
conda activate /home/iris/Workspace/junsu/.conda/miniforge3/envs/robodiff
```

먼저 dataset sample을 사용하는 offline inference로 checkpoint 로딩과 shape를
확인한다. 아래에서 실행 파일만 바꾸면 네 모델 모두 같은 절차이다.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_real_peg_goal_prop_desired.py --mode offline
CUDA_VISIBLE_DEVICES=0 python eval_real_peg_goal_prop_next_state.py --mode offline
CUDA_VISIBLE_DEVICES=0 python eval_real_peg_goal_prop_wrench_desired.py --mode offline
CUDA_VISIBLE_DEVICES=0 python eval_real_peg_goal_prop_wrench_next_state.py --mode offline
```

ROS graph와 실제 state stream이 준비된 뒤 shadow를 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_real_peg_goal_prop_desired.py \
  --mode shadow --duration-seconds 30
```

Shadow도 observation 수집 후 inference를 시작하려면 enable gate가 필요하지만,
출력은 shadow topic으로만 나가므로 로봇 명령 topic에는 publish하지 않는다.

```bash
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool '{data: true}'
```

별도 ROS terminal에서 다음을 확인한다.

```bash
ros2 topic echo /rightarm/policy_debug/status
ros2 topic echo /rightarm/policy_debug/remote_desired_shadow
```

점검을 마치면 echo를 중단하고 gate를 다시 false로 둔다.

```bash
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool '{data: false}'
```

다음을 모두 확인하기 전에는 execute로 전환하지 않는다.

- shadow pose가 workspace 내부이며 quaternion이 정상인지
- 첫 target이 현재 TCP에서 허용 step 이내인지
- fixed goal이 현재 fixture 좌표와 일치하는지
- wrench 모델의 status에서 `wrench_filter.enabled=true`, `ready=true`인지
- `/rightarm/Teleop_RemoteDesired`에 경쟁 publisher가 없는지
- 로봇 근처에 사람이 없고 hardware emergency stop을 즉시 사용할 수 있는지

Execute는 먼저 enable을 false로 둔 상태에서 시작한다.

```bash
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool '{data: false}'
```

```bash
CUDA_VISIBLE_DEVICES=0 python eval_real_peg_goal_prop_desired.py \
  --mode execute \
  --duration-seconds 30 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA
```

상태와 shadow 검증이 끝났을 때만 별도 terminal에서 enable한다.

```bash
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool '{data: true}'
```

중지하려면 먼저 gate를 내린다.

```bash
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool '{data: false}'
```

그 다음 inference process를 `Ctrl-C`로 종료한다. 네 모델 사이를 바꿀 때는
항상 이전 process가 완전히 종료되고 gate가 false인지 확인한 후 새 process를
시작한다.

## 네 모델별 실제 실행 파일

```text
eval_real_peg_goal_prop_desired.py
eval_real_peg_goal_prop_next_state.py
eval_real_peg_goal_prop_wrench_desired.py
eval_real_peg_goal_prop_wrench_next_state.py
```

각 파일 상단에도 해당 모델의 shadow/실제 로봇 명령이 주석으로 들어 있다.
checkpoint를 바꾸려면 `--checkpoint /absolute/path/to/file.ckpt`를 사용할 수
있지만, 저장된 checkpoint task/observation 계약이 해당 실행 파일과 다르면
ROS bridge를 시작하기 전에 실패하도록 되어 있다.
