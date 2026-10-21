# DP next-state target 3종 실험

## 목적

기존 세 desired-command DP와 observation, episode split, 모델 구조 및 학습
조건을 유지하면서 target만 next measured TCP state로 변경한다.

```text
desired 모델:    action[t] = 기록된 Teleop_RemoteDesired pose[t]
next-state 모델: action[t] = measured TCP pose[t+1]
```

각 episode의 마지막 action에는 마지막 TCP pose를 반복한다. 저장된 absolute
9D next state는 기존 `P2PImageDataset`에서 최신 TCP observation 기준 relative
9D action으로 변환된다.

## 세 모델

| 모델 | Observation | Goal | Target |
|---|---|---|---|
| Peg prop | relative TCP pose | relative goal pose | next TCP state |
| Peg prop+wrench | relative TCP pose, wrench | relative goal pose | next TCP state |
| Valve insertion | external/wrist RGB, relative TCP pose, wrench | 없음 | next TCP state |

세 모델 모두 gripper action을 출력하지 않는다.

## Peg 데이터 구성

Peg observation은 desired 모델과 동일한 원본 Zarr를 직접 읽는다.

```text
/home/iris/Workspace/junsu/Data/replay_buffer/
square_peg_in_hole_21mm_20260916_p2p_goal_desired_zarr
```

Target만 다음 compact sidecar에서 교체한다. RGB나 observation은 복사하지
않는다.

```text
/home/iris/Workspace/junsu/DP/data/datasets/
square_peg_goal_desired_observations_next_state_actions_30f
```

Sidecar 재생성 명령은 다음과 같다. exporter는 기존 출력 덮어쓰기를
거부한다.

```bash
cd /home/iris/Workspace/junsu/DP
python scripts/build_next_state_action_sidecar.py \
  --input /home/iris/Workspace/junsu/Data/replay_buffer/square_peg_in_hole_21mm_20260916_p2p_goal_desired_zarr \
  --output /home/iris/Workspace/junsu/DP/data/datasets/square_peg_goal_desired_observations_next_state_actions_30f
```

생성 결과는 98 episodes, 43,312 frames, action `(43312, 9)`이다. 이는
desired-command Peg 실험과 동일한 observation 및 episode boundary다.

`Data/replay_buffer/...p2p_goal_zarr`에도 next-state action이 있지만 43,685
frames로 desired Zarr와 구성이 다르므로 target-only 비교에는 사용하지
않는다.

## Valve 데이터 구성

Valve는 기존 post-grasp cropped Zarr의 action이 이미 next state이므로 별도
action sidecar를 붙이지 않는다.

```text
/home/iris/Workspace/junsu/DP/data/datasets/
valve_insertion_after_grasp_30f
```

따라서 desired variant는 `action_dataset_path`로 desired sidecar를 읽고,
next-state variant는 cropped Zarr의 내장 action을 직접 읽는다. 두 variant의
observation과 105 episode, 51,809 frames는 동일하다. Valve에는 goal pose를
추가하지 않는다.

## 학습 명령어

```bash
cd /home/iris/Workspace/junsu/DP
conda activate /home/iris/Workspace/junsu/.conda/miniforge3/envs/robodiff

# Peg: goal + prop
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=train_diffusion_unet_peg_proprio_next_state_workspace

# Peg: goal + prop + wrench
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config-name=train_diffusion_unet_peg_proprio_wrench_next_state_workspace

# Valve insertion-only: RGB + prop + wrench, no goal
CUDA_VISIBLE_DEVICES=2 python train.py \
  --config-name=train_diffusion_unet_valve_insertion_p2p_image_next_state_workspace
```

GPU 번호는 실행 당시 빈 GPU에 맞게 변경한다.

## 공통 학습 조건

- horizon 32
- raw observation 4 frames에서 stride 2로 2 frames 사용
- `action_start_offset=3`, 예측 action 29 steps
- 800 epochs
- EMA 학습 및 EMA validation
- episode-level validation ratio 0.1, seed 42
- train-loss best, validation-loss best, latest checkpoint 저장
- 완료 epoch 200, 400, 600, 800 milestone 저장

Config/task/run 이름과 WandB tag에 `next_state`를 사용하므로 기존 desired
checkpoint에서 resume하지 않는다. Desired와 next-state는 target 분포가
다르므로 validation loss 절댓값보다는 동일한 실물 성공률, 삽입 시간,
최대 wrench 및 경로 길이로 비교한다.
