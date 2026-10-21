# Peg-in-hole / Valve 힘 크기 분석

분석일: 2026-10-19. 대상: `/home/iris/Workspace/junsu/Data/` 아래 모든 replay buffer 3개 및 raw bag 데이터셋 3개.

학습용 replay buffer의 힘 크기 범위는 **Peg 0.00588–93.04552 N**, **Valve 약 0–80.11312 N**이다. 원본 bag의 실물 Peg는 **0.05035–306.96868 N**, Valve는 **0.02958–102.82148 N**으로 순간 최대값이 더 높다.

계산: `|F| = sqrt(Fx² + Fy² + Fz²)`. Replay의 `data/left_robot_tcp_wrench[:, :3]`, 원본의 `/lfd/rightarm/Teleop_Remote`에서 `wrench.force`를 사용했다. 토크 3성분은 크기 계산에 포함하지 않았다. 단위 N은 로컬 시각화 코드의 `force (N)` 표기와 F/T 처리 코드 기준이다.

모든 저장 샘플을 동일 가중치로 집계했다. 학습/검증 split, 접촉/비접촉, 작업 단계별 구분은 하지 않았다. 추가 필터링·영점 보정·정규화·이상치 제거도 적용하지 않았다. 모든 대상 힘 샘플에 NaN/Inf는 없었다.

## 전체 분포

단위: N. P95/P99는 전체 샘플의 각각 95%/99%가 그 값 이하라는 뜻이다.

| 데이터 | 에피소드 | 샘플 | 최소 | 최대 | 평균 | 중앙값 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Peg desired replay | 98 | 43,312 | 0.0058757 | 93.04552 | 9.07880 | 3.19693 | 39.22436 | 57.29664 |
| Peg goal replay | 98 | 43,685 | 0.0058757 | 93.04552 | 9.02687 | 3.19538 | 39.11899 | 57.21242 |
| Valve replay (no_offset_gripper) | 105 | 104,980 | 1.3841e-18 | 80.11312 | 7.42981 | 2.14142 | 30.70449 | 42.07531 |
| Peg p2p 원본 | 98 | 1,456,567 | 0.050352 | 306.96868 | 9.09375 | 3.19560 | 39.21972 | 57.85771 |
| Peg fd_sim 원본 | 66 | 595,319 | 0 | 5.99294 | 0.50084 | 0.12879 | 2.22573 | 3.65073 |
| Valve 원본 | 105 | 3,506,432 | 0.029581 | 102.82148 | 13.92771 | 12.05627 | 36.46706 | 47.12958 |

## 학습용 데이터의 축별 범위

| 데이터 | Fx 범위 (N) | Fy 범위 (N) | Fz 범위 (N) |
|---|---:|---:|---:|
| Peg desired replay | -27.781 ~ 36.757 | -28.458 ~ 22.397 | -16.063 ~ 91.475 |
| Peg goal replay | -27.781 ~ 36.757 | -28.458 ~ 22.397 | -16.063 ~ 91.475 |
| Valve replay (no_offset_gripper) | -39.059 ~ 30.265 | -29.779 ~ 34.408 | -32.299 ~ 75.353 |

각 축의 최소·최대는 서로 다른 시점일 수 있으므로 축별 최댓값을 조합해 힘의 최대 크기를 계산하면 안 된다.

## 해석 및 비교 범위

- 기본 Peg replay는 샘플의 95%가 39.119 N 이하, 99%가 57.212 N 이하이다. Valve replay는 각각 30.704 N, 42.075 N 이하이다.
- 50 N 초과 비율은 기본 Peg replay 2.312%, Valve replay 0.236%이다. 최대값만으로 일반적인 분포를 대표하기 어렵다.
- Peg desired의 43,312개 샘플 모두 같은 replay episode index의 기본 Peg에 timestamp와 6축 wrench가 정확히 일치하는 샘플이 존재했다. 기본 Peg에는 373개 프레임이 더 있다. 두 버전을 독립된 추가 수집 데이터로 합산하지 않았다. `desired` 이름이 측정 힘을 목표 힘으로 바꾼다는 뜻은 아니다.
- Replay의 에피소드 내부 timestamp 중앙 간격은 약 1/30초이다. 원본의 에피소드별 유효 수신 빈도 중앙값은 실물 Peg/Valve 약 1,000 Hz, fd_sim 약 859 Hz이다.
- 로컬 Peg 동기화 코드는 causal 15 Hz Butterworth LPF 후 카메라 프레임에 맞춘다. Valve replay 폴더명은 `no_offset`이며, 로컬에는 센서 좌표계 첫 샘플 offset을 제거하는 변환 코드가 있다. 다만 이 Zarr들에는 생성 설정 속성이 없어 정확한 생성 이력을 전부 검증하지는 못했다. 원본과 replay의 차이는 필터링·샘플 선택·영점 보정 등의 영향을 구분해서 해석해야 하며, 원본 수치를 그대로 모델 입력 범위로 사용하면 안 된다.
- Peg 원본 최대 306.969 N은 `episode_45`에서 확인했다. 전후 수 ms에도 220.721, 273.627, 248.949 N 등이 기록되어 있다. 전체 CDR 파서로 다시 읽어 수치를 확인했다. 실제 충격인지 센서 이상인지는 이 통계만으로 판정하지 않았다.
- fd_sim은 별도 데이터셋이며 실물 Peg와 합산하지 않았다. 힘이 정확히 0인 샘플은 224,362개(약 37.69%)이다.
- `Data/` 외부의 `DP/data/datasets/valve_insertion_after_grasp_30f` 등 파생 학습 세트는 이번 범위에 포함하지 않았다. 실제 학습 설정이 그 경로를 사용한다면 단계별 분포를 별도로 봐야 한다.

## 최대값 위치

| 데이터 | 에피소드 식별자 | 에피소드 내부 샘플 index (0-based) | 힘 크기 (N) |
|---|---|---:|---:|
| Peg desired replay | replay_index_5 | 528 | 93.04552 |
| Peg goal replay | replay_index_5 | 540 | 93.04552 |
| Valve replay (no_offset_gripper) | replay_index_85 | 1050 | 80.11312 |
| Peg p2p 원본 | episode_45 | 16846 | 306.96868 |
| Peg fd_sim 원본 | episode_47 | 9088 | 5.99294 |
| Valve 원본 | episode_34 | 28897 | 102.82148 |

Replay index는 0부터 시작하는 저장 순서이다. 원본 episode 번호와의 대응 메타데이터가 없어 두 식별자를 동일시하지 않았다.

## 데이터와 산출물

- Peg desired replay: `/home/iris/Workspace/junsu/Data/replay_buffer/square_peg_in_hole_21mm_20260916_p2p_goal_desired_zarr/replay_buffer.zarr`
- Peg goal replay: `/home/iris/Workspace/junsu/Data/replay_buffer/square_peg_in_hole_21mm_20260916_p2p_goal_zarr/replay_buffer.zarr`
- Valve replay (no_offset_gripper): `/home/iris/Workspace/junsu/Data/replay_buffer/valve_peg_in_hole_20260930_no_offset_gripper_zarr/replay_buffer.zarr`
- Peg p2p 원본: `/home/iris/Workspace/junsu/Data/raw_bag/square_peg_in_hole_21mm_20260916_p2p_goal`
- Peg fd_sim 원본: `/home/iris/Workspace/junsu/Data/raw_bag/square_peg_in_hole_21mm_20260917_fd_sim`
- Valve 원본: `/home/iris/Workspace/junsu/Data/raw_bag/valve_peg_in_hole_20260930`

- [전체 통계 JSON](statistics.json)
- [분포 그림 PNG](force_distribution.png)
- [분포 그림 PDF](force_distribution.pdf)
- [최대값 원본 재검증](peak_validation.json)
- 에피소드별 CSV: 이 폴더의 `*_episodes.csv`
- 원본 DB 크기/수정시각 목록: `*_source_manifest.json`. 각 DB를 읽기 전후 크기와 수정시각이 동일함을 확인했다.

## 방법 및 재실행

입력은 Zarr `mode=r`, SQLite `mode=ro`로 읽었다. 단일 프로세스, BLAS/Blosc 1 thread, nice 10 및 idle I/O 우선순위를 사용했다. 시스템 설정, 공용 환경 패키지, 원본 데이터는 수정하지 않았다.

workspace 루트에서 기존 환경으로 실행:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ionice -c 3 nice -n 10 .conda/miniforge3/envs/robodiff/bin/python chaewon_force_analysis_20261019/analyze.py
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .conda/miniforge3/envs/robodiff/bin/python chaewon_force_analysis_20261019/make_report.py
```

코드 근거: `iris_teleop_fr3_1007/visualize/raw_bag/cdr_reader.py`, `iris_teleop_fr3_1007/visualize/raw_bag/visualize_raw_bag.py`, `WorldDP/worlddp/Lpf2/sync_dataset_p2p.py`, `WorldDP/worlddp/Lpf2/remove_ft_offset.py`, `DP/docs/next_state_target_variants_ko.md`.
