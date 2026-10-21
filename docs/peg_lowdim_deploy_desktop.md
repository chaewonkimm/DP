# DP square-peg deployment desktop commands

이 문서는 `download.sh`에 기록된 배포 데스크탑 경로를 기준으로 네 개의
low-dimensional DP 정책을 FR3에 연결하는 명령을 정리한다.

`scripts/run_peg_lowdim_deploy.sh`는 DP 정책을 rollout 가상환경에서 실행하고,
ROS workspace의 Python으로 `p2p_remote_desired_real_bridge.py`를 자동 실행한다.
따라서 ROS bridge를 별도로 직접 실행하지 않는다.

## 최초 1회

DP 코드와 사용할 checkpoint를 배포 데스크탑으로 복사한 뒤 rollout 환경을
만든다.

현재 `download.sh`의 첫 `rsync`는 `data/` 전체를 제외하므로, 코드 복사 후
네 모델의 validation-best와 train-best checkpoint 8개는 별도로 복사해야 한다.
배포 데스크탑에서 먼저 dry-run을 실행한다.

```bash
cd /home/iristele/work/chaewon/DP
bash scripts/download_peg_lowdim_checkpoints.sh --dry-run
```

대상 목록을 확인한 후 실제 다운로드한다. 동일한 파일이 이미 있으면 rsync가
다시 전송하지 않는다.

```bash
bash scripts/download_peg_lowdim_checkpoints.sh
```

```bash
cd /home/iristele/work/chaewon/DP
bash scripts/setup_rollout_env.sh
```

네 deploy YAML의 checkpoint 상대경로 아래에 해당 checkpoint가 실제로 있어야
한다. 네 low-dimensional YAML은 기존 vision+wrench+prop next-state peg 설정과
동일하게 다음 execute 및 safety 값을 사용한다.

```yaml
live:
  execute_enabled: true
safety:
  workspace_min: [-1, -1, 0]
  workspace_max: [1, 1, 1]
  max_translation_step_m: 0.5
  max_rotation_step_rad: 0.5
  max_force_n: 150
  max_torque_nm: 5
```

데스크탑별로 더 보수적인 안전값을 사용하려면 기본 YAML을 바로 수정하는 대신
복사본을 만들고 실행 시 `--config`로 지정할 수 있다. 위 translation step
`0.5 m`는 넓은 비교 기준이므로, 실기 환경에서는 더 작은 값으로 낮추는 것을
권장한다.

```bash
cp diffusion_policy/config/deploy/p2p_peg_goal_prop_desired_franka.yaml \
  diffusion_policy/config/deploy/local_peg_goal_prop_desired_franka.yaml
```

## Terminal 1: FR3 remote endpoint

Peg task는 gripper action을 사용하지 않으므로 `--load-gripper`가 필요 없다.

```bash
cd /home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation/src
./remote_runfile/start_fr3_remote.sh \
  --endpoint-only \
  --start-from-current
```

## Terminal 2: DP policy와 ROS bridge

먼저 shadow로 확인한다.

```bash
cd /home/iristele/work/chaewon/DP
export DP_ROS_ROOT=/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_desired shadow \
  --duration-seconds 30 \
  --inference-seed 42
```

첫 번째 인자만 바꾸면 네 모델을 선택할 수 있다.

```text
goal_prop_desired
goal_prop_next_state
goal_prop_wrench_desired
goal_prop_wrench_next_state
```

Shadow 검증과 안전 설정을 마친 후 실제 로봇 모드로 실행한다. 아래 명령만으로
로봇이 즉시 움직이지는 않으며, Terminal 3에서 enable을 별도로 올려야 한다.

```bash
cd /home/iristele/work/chaewon/DP
export DP_ROS_ROOT=/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation
CUDA_VISIBLE_DEVICES=0 bash scripts/run_peg_lowdim_deploy.sh \
  goal_prop_desired execute \
  --duration-seconds 120 \
  --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA
```

## Terminal 3: 상태 확인 및 enable gate

```bash
cd /home/iristele/work/chaewon/DP
export DP_ROS_ROOT=/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation
source scripts/activate_rollout_ros.sh
```

먼저 gate를 내리고 status를 확인한다.

```bash
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool "{data: false}"
ros2 topic echo /rightarm/policy_debug/status std_msgs/msg/String \
  --field data --full-length
```

Shadow 또는 execute 정책이 준비된 뒤에만 gate를 올린다.

```bash
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool "{data: true}"
```

중지할 때는 먼저 gate를 내린 다음 Terminal 2에서 `Ctrl-C`로 정책을 종료한다.

```bash
ros2 topic pub --once /rightarm/policy_enable std_msgs/msg/Bool "{data: false}"
```

모델을 바꿀 때도 반드시 이전 정책 프로세스가 완전히 종료되고 enable이
false인지 확인한다. Wrench 모델의 LPF를 포함한 observation 전처리는 기존
inference 코드와 동일한 경로를 사용하며, 안전 판정에는 raw wrench가 사용된다.
