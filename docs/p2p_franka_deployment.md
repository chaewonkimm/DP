# DP p2p policies on the Franka

This guide deploys the two **Diffusion Policy (DP)** checkpoints in this repo:

- square peg-in-hole: relative 9D TCP action;
- valve peg-in-hole: relative 9D TCP action plus one binary gripper command.

It does not deploy an RDP model or run an RDP second stage. The default live
path is the IRIS RemoteDesired stack:

```text
DP checkpoint
  -> eval_real_{valve,peg}.py policy worker
  -> p2p_remote_desired_real_bridge.py (ROS Python)
  -> /rightarm/Teleop_RemoteDesired
  -> remote_panda_driver.cpp Cartesian impedance
  -> Franka remote robot
```

The split worker/bridge layout follows
`training/runtime/learned_remote_desired_node.py` and
`learned_remote_desired_real_bridge.py` in
`/home/iristele/work/seongsu/iris_learning_from_teleop_fr3`. It is split because
the DP Python has Torch/Hydra but no `rclpy`, while the teleoperation Python has
ROS messages but no Torch. Unlike that project's protocol-v3 delta policy, this
repository's bridge transfers **absolute targets reconstructed at the policy
observation pose**. Applying these actions as deltas from a later measured pose
would silently change their meaning.

## Safety model

`eval_real_peg.py` and `eval_real_valve.py` default to `offline`. `shadow`
subscribes to live sensors and publishes only a non-actuating debug topic. `execute`
is blocked unless all of the following are true:

1. `live.execute_enabled` is explicitly changed to `true`;
2. workspace, per-command motion, force, and torque limits are all filled in;
3. valve gripper behavior is explicitly configured;
4. the exact CLI confirmation token is supplied.

These software checks do not replace the Franka Desk limits, collision behavior,
physical E-stop, a clear workcell, or an operator holding the E-stop. A bridge
timeout stops publishing new targets, but `remote_panda_driver` can continue
holding its last target. The physical E-stop is still required for an immediate
stop.

## What must match the demonstrations

- Same Franka/FrankaHand and the same base/flange convention.
- External and wrist cameras must be assigned to `external_img` and
  `left_wrist_img`, with the same mounting, viewpoint, color order, and scene.
- The live pipeline must provide `left_robot_tcp_pose` as xyz plus the same
  ortho-6D rotation used in the Zarr, and a 6D TCP wrench.
- Observations are taken at 30 Hz, with two model frames separated by two raw
  frames (about 66.7 ms). Images are 240x320.
- The latest absolute TCP pose is the base frame for both relative pose
  observations and all predicted relative actions.
- The fixture, initial robot pose, object pose, lighting, and gripper state need
  to stay inside the demonstration distribution. DP has no explicit recovery or
  force-control stage.

The peg setup has no gripper: its 9D policy sends only TCP poses and creates no
gripper client. Valve predicts `-1/+1`; its `franka_action` mode converts the
prediction into open/closed width and sends separate Franka Hand Move/Grasp
actions. The arm driver itself does not control the gripper. All DP arm messages
now set `clutch2=false` so the separate legacy gripper toggle bridge started by
`--load-gripper` cannot interpret them as toggle edges. This also applies to
`hold_current` and peg, which must not accidentally toggle a loaded gripper.

## Checkpoints currently configured

The task configs point to the available EMA states inside:

- peg: `epoch=0590-train_loss=0.000.ckpt`;
- valve: `epoch=0510-train_loss=0.000.ckpt`.

Both files were selected by training loss, not validation loss. The valve run
also stopped before 600 epochs. They are useful for checking the deployment
pipeline and shadow predictions, but they should not be treated as reliable
final robot policies. For a baseline result, train/save EMA checkpoints selected
by validation loss and compare multiple seeds.

## 1. Offline inference first

The local rollout environment is `.venv` (Python 3.11). Recreate it with
`bash scripts/setup_rollout_env.sh` if needed. It installs the pinned rollout
dependencies from `requirements-rollout.txt` and Torch 2.7.1 / torchvision
0.22.1 with CUDA 12.8 for the RTX 5080. The original Conda recipes describe the
older full training/simulation environment and are not used for this rollout.

Activate the policy environment in a fresh terminal:

```bash
cd /home/iristele/work/chaewon/DP
source scripts/activate_rollout_env.sh
python eval_real_peg.py --help
python scripts/check_rollout_env.py
```

The activation sets `ROS_DOMAIN_ID=10` and `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`
unless already configured. For ROS CLI and camera tools, use a **separate**
terminal and `source scripts/activate_rollout_ros.sh`. The policy launcher
automatically sources the configured ROS installation in its bridge subprocess.

Check both actual checkpoints on the GPU without ROS nodes or camera access:

```bash
python scripts/check_rollout_env.py --checkpoints \
  --output data/eval/environment_check.json
```

This uses synthetic images/pose/wrench solely to verify checkpoint loading,
inference and bridge serialization. It is not a recorded-data accuracy test.
The `--mode offline` commands below additionally require the original Zarr
datasets at the paths stored in each checkpoint; installing libraries does not
provide those datasets.

Run one recorded sample from each dataset. No ROS or robot connection is used:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_real_peg.py --mode offline --sample-index 0
CUDA_VISIBLE_DEVICES=0 python eval_real_valve.py --mode offline --sample-index 0
```

Each run writes `offline_report.json` and `offline_prediction.npz` under
`data/eval/<task>/<timestamp>_offline/`. Inspect that all values are finite,
the action shape is `(29,9)` or `(29,10)`, and the reconstructed first absolute
pose is close to the demonstrated motion. The single-sample MSE is a diagnostic,
not a robot-success metric.

Eight DDIM steps are used for deployment latency. Training validation used 100
steps, so compare offline results at 8 and 100 before choosing the runtime value:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_real_peg.py --mode offline \
  --num-inference-steps 100
```

## 2. Bring up the RemoteDesired stack

### Repeatable inference noise

Both task launchers accept `--inference-seed`, or a top-level `inference_seed`
in the deployment YAML. The CLI value takes precedence. For example:

```bash
python eval_real_peg.py --mode shadow --duration-seconds 60 --inference-seed 42
```

An integer recreates a device-local random generator with that seed **for every
inference chunk**, so the starting noise is identical for the same shape, dtype,
device and software environment. Any scheduler noise also uses that generator.
This is different from seeding once at rollout startup, which produces a
repeatable sequence of different noises. `inference_seed: null` (the default)
preserves the original random sampling. Seed zero is valid. The seed and number
of diffusion steps are recorded in reports and RemoteDesired worker events.

Fixed noise does not guarantee identical physical rollouts: camera/state inputs,
control timing, checkpoint, inference steps, and numerical nondeterminism still
matter. Bitwise reproducibility across devices/library versions is not promised.
Restart the policy process after changing the seed; no ROS rebuild is required.

### Selecting part of a predicted chunk

The RemoteDesired worker can return a slice of the policy's predicted actions:

```yaml
live:
  remote_desired:
    chunk_start_index: 4
    chunk_size: 4
    prefetch_steps: 3
```

This selects `policy.predict_action()["action"][4:8]` (zero-based): the last four
of the first eight predicted actions. This is opt-in; the peg deployment currently
uses the full chunk (`chunk_start_index: 0`, `chunk_size: 29`, `prefetch_steps: 8`).
It is additional to the checkpoint's `action_start_offset`; it is NOT an index
into the raw diffusion horizon or a change to the number of denoising steps.
The model still predicts its entire trajectory, so this does not reduce inference
compute. Both shadow and execute use this selection; offline inspection retains
the full trajectory. Valve retains the original full-chunk setting.

`prefetch_steps` must be less than the selected `chunk_size`: here, the next
inference is requested with three actions remaining. At 30 Hz, four actions are
about 133 ms of playback; short slices leave less latency margin.

Latency skipping retains the ORIGINAL predicted-action indices. For slice `[4:8]`,
six elapsed ticks skip selected indices 4 and 5, leaving 6 and 7. Eight elapsed
ticks discard the entire slice, including on a slow cold-start inference. If the
reply arrives earlier, the chosen slice is used when the old queue empties; the
player does not wait for index 4's original time. Selecting later targets can
increase the first target jump. Existing target checks remain in place. Validate
the changed behavior in shadow before any hardware trial.

Worker events and live reports include the selected start/count. The original
`first_*_action` fields still describe prediction index 0; use
`first_selected_*_action` for the selected first target. The first-observation NPZ
adds `selected_absolute_action` while retaining the full prediction arrays.
To restore the old behavior, use `chunk_start_index: 0`, `chunk_size: 29`, and
`prefetch_steps: 8`. A policy-process restart is sufficient; no ROS rebuild.

### Start the sensor and robot endpoints

Build/source the `iris_teleoperation` workspace, then launch the remote endpoint
with `mapping_type: Cartesian`. It must publish measured state on
`/rightarm/Teleop_Remote`, and the two camera publishers must publish compressed
JPEG on the configured topics. The checked-in defaults are:

```text
/rightarm/Teleop_Remote
/rightarm/cameras/iso/image_raw/compressed       -> external_img
/rightarm/cameras/wrist/image_raw/compressed     -> left_wrist_img
```

Verify the graph before starting the policy:

```bash
ros2 topic info /rightarm/Teleop_Remote
ros2 topic hz /rightarm/cameras/iso/image_raw/compressed
ros2 topic hz /rightarm/cameras/wrist/image_raw/compressed
ros2 topic info /rightarm/Teleop_RemoteDesired --verbose
```

Stop `TeleopControllerRemote` or any other publisher of
`/rightarm/Teleop_RemoteDesired`. The bridge refuses real publication if it
detects another publisher. Confirm the ROS interpreter and paths under
`live.remote_desired` in the task YAML. `eval_real_valve.py` starts that bridge
process automatically; do not start it separately.

The incoming `Teleop_Remote.pose` and outgoing desired pose must use the same
pelvis/world convention that was recorded into the Zarr. Check the numeric
initial pose against a recorded sample before enabling output. Quaternion order
at ROS boundaries is xyzw; checkpoint observations remain rotation-6D.

## 3. Shadow mode

With the robot stationary and the sensor stack running:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_real_peg.py --mode shadow \
  --duration-seconds 30
CUDA_VISIBLE_DEVICES=0 python eval_real_valve.py --mode shadow \
  --duration-seconds 30
```

Shadow mode publishes only
`/rightarm/policy_debug/remote_desired_shadow`, never
`/rightarm/Teleop_RemoteDesired`. It saves the first observation/prediction NPZ
and an event JSONL. Enable the policy gate to exercise chunk playback:

```bash
ros2 topic pub /rightarm/policy_enable std_msgs/msg/Bool "{data: true}" -r 1
ros2 topic echo /rightarm/policy_debug/status
ros2 topic echo /rightarm/policy_debug/remote_desired_shadow
```

Wait until the status reports fresh inputs before setting `policy_enable=true`.
A sensor, worker, competing-publisher, or safety fault is latched; clear it with
an explicit `false` and then re-enable only after fixing the cause.

Check:

- images are not swapped, mirrored, or BGR/RGB-changed;
- current absolute pose and wrench are plausible;
- reconstructed targets remain near the fixture and are temporally smooth;
- inference latency stays below the 29-step action horizon, with stale entries
  skipped as reported by bridge status;
- moving the robot by hand (only when safely permitted) changes the relative
  reconstruction as expected.

## 4. Configure guarded execution

Edit only the matching deployment YAML:

- `p2p_peg_franka.yaml`, or
- `p2p_valve_franka.yaml`.

Measure and fill in the same pelvis/world command frame carried by
`Teleop_Remote.pose`:

```yaml
live:
  execute_enabled: true
safety:
  workspace_min: [x_min, y_min, z_min]
  workspace_max: [x_max, y_max, z_max]
  max_translation_step_m: <approved per-command limit>
  max_rotation_step_rad: <approved per-command limit>
  max_force_n: <approved measured-wrench limit>
  max_torque_nm: <approved measured-wrench limit>
```

Do not copy dataset-wide extrema blindly: an execution box should be a tighter,
task-specific safe volume. The bridge enforces these limits before placing each
target on the 1 kHz hold publisher and disables its enable gate on a violation.

For valve with an active gripper, use the **junsu** workspace that provides
`--load-gripper` and `franka_msgs`. The valve configuration selects that ROS
Python/setup; the DP policy environment stays the same. From a fresh ROS terminal:

```bash
export DP_ROS_ROOT=/home/iristele/work/junsu/iris_learning_from_teleop_fr3/iris_teleoperation
source /home/iristele/work/chaewon/DP/scripts/activate_rollout_ros.sh
```

The hardware wrapper is `remote_runfile/start_fr3_remote.sh --endpoint-only
--load-gripper` inside that workspace. It starts/moves hardware: do not restart
an existing driver just to apply a DP change. `--endpoint-only` excludes the
competing TeleopControllerRemote. Do not run teleop and policy command sources
simultaneously. The legacy gripper bridge may remain running but must receive no
rising clutch2 edges; do not command its actions from another client during DP.

Read-only endpoint checks (do not send test goals to real hardware):

```bash
ros2 action list -t
ros2 action info /rightarm/remote_right_gripper/move
ros2 action info /rightarm/remote_right_gripper/grasp
```

The checked-in valve mode is `gripper.remote_desired_mode: franka_action`:

- `action_namespace`: `/rightarm/remote_right_gripper`.
- Negative predictions select `open_width_m`; nonnegative predictions select
  `closed_width_m` (threshold `command_threshold`, default zero).
- Opening uses Move; closing uses Grasp when `close_with_force=true`, otherwise
  Move. `velocity_mps` applies to both. Grasp receives `force_limit_n` (currently
  7 N), `grasp_epsilon_inner_m` and `grasp_epsilon_outer_m` (currently 0.07 m,
  matching the old teleop bridge; review for object thickness).
- The old teleop bridge's 120 N source default is NOT used by DP action clients.
- Commands must remain stable for `command_stability_seconds` (0.1 s); identical
  commands are deduplicated. Only one owned goal can be outstanding. While busy,
  only the latest desired width is kept; historical toggles are never queued.
- Rejection, unsuccessful results, and `action_timeout_seconds` (5 s) latch a
  fault and disable policy output; there is no automatic retry. Explicit false
  clears the error only after any in-flight goal finishes/cancels.
- Disable/stale inputs/fault/shutdown request cancellation of this client's
  unfinished goal, never cancellation of every client's goals. Shutdown waits
  up to one second and reports unconfirmed cancellation. A completed grasp is
  not automatically opened or stopped. Cancellation is not an emergency stop.
- Shadow returns the width in debug actions and status but creates NO gripper
  action client and sends NO gripper goal. Before enabling execute, require
  `gripper_action_status.ready=true`, `busy=false`, and `error=null`.

Set `mapping_confirmed` true only after verifying
that the checkpoint's last action channel came from the recorded `gripper_command`
stream (-1=open, +1=close) and reviewing the widths/force/tolerances. Gripper
imports and configuration are checked before checkpoint loading. The legacy
`/rightarm/gripper_command` publisher reflects its own toggle state, NOT DP's
actions; do not treat it as DP command telemetry. Use DP debug action/status and
the actual measured gripper joint states instead.

### Separate Valve action-example launcher

`scripts/run_valve_gripper_actions.sh` runs the same policy and single ROS bridge,
with an explicit in-memory `--gripper-action-profile` override. It does **not**
start a second gripper subscriber or execute a `ros2` shell subprocess per frame.
Its typed ROS ActionClient goals are equivalent to the supplied Move/Grasp CLI
examples. It leaves the normal valve YAML unchanged:

- Float prediction `< 0` becomes `-1`: Move(width=0.08, speed=0.1).
- Float prediction `>= 0` becomes `+1`: Grasp(width=0.0, speed=0.1,
  force=120.0, epsilon.inner=0.07, epsilon.outer=0.07).
- **120 N is an explicit increase from the normal YAML's 7 N**, not a safety
  recommendation. Review it for the object/setup before using this launcher.
- The first enabled valid command is sent once. Subsequent commands are sent
  only when the binary target differs from the last completed target; changes
  within one sign do not retrigger an action. This profile sets the additional
  stability dwell to zero; all normal enable, freshness, fault and pose guards
  remain. Nonfinite predictions are rejected.
- Only one goal can be in flight. While busy, the latest target wins; obsolete
  intermediate transitions are not replayed. Failures latch a fault and are
  not automatically retried. A completed grasp is not released on disable.

```bash
# First check plumbing without gripper action goals:
bash scripts/run_valve_gripper_actions.sh --mode shadow \
  --duration-seconds 30 --inference-seed 42

# Separate, reviewed live trial (may move arm AND gripper after enable):
bash scripts/run_valve_gripper_actions.sh --mode execute \
  --duration-seconds 10 --inference-seed 42 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA
```

Do not run either launcher alongside an existing DP bridge or manual gripper
action sender. The hardware endpoint still needs `--endpoint-only --load-gripper`.
The script does not launch hardware or publish policy enable; use the separate
ROS terminal and status/enable procedure above. Inspect `gripper_action_status`
fields `desired_command`, `last_goal`, `goals_sent`, and `last_result` for the
actual goal parameters and outcome. `last_goal` records the most recent attempted
goal, not proof of acceptance; `goals_sent` counts send requests, not successes.

### Optional startup residual FT zeroing (0.2 seconds)

The FR3 driver already applies tool gravity and fixed FT offsets before
publishing `Teleop_Remote.wrench`. DP must not apply a second gravity correction.
The optional startup calibration subtracts only the remaining bias from **policy
observations**. Raw here means the driver-published, already gravity-compensated
wrench, not the original sensor counts. Existing wrench safety checks and the
outgoing Teleop message's wrench remain unchanged.

The option is OFF by default. With the robot stationary and tool free of any
external contact, keep the gripper/tool configuration fixed and explicitly opt in:

```bash
source scripts/activate_rollout_env.sh
python eval_real_valve.py --mode shadow --duration-seconds 30 \
  --inference-seed 42 --ft-bias --confirm-ft-bias-free-space
```

The same flags work with `scripts/run_valve_gripper_actions.sh`; the launcher
currently leaves automatic gripper control disabled (`hold_current`). FT zeroing
does not re-enable it. After a separate review, the execute form is:

```bash
bash scripts/run_valve_gripper_actions.sh --mode execute \
  --duration-seconds 30 --inference-seed 42 \
  --ft-bias --confirm-ft-bias-free-space \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA
```

No automatic policy enable is sent. Before enabling, inspect the usual status
topic: `ft_bias.enabled=true`, `ft_bias.ready=true`, `ft_bias.error=null`, as well
as the existing worker, publisher and fault checks. Early enable is rejected,
not queued; send enable only after calibration completes. Keep the setup still
and non-contact until then. Startup calibration time counts toward run duration.

The calibration uses the first stable **0.2-second window** (at least 5 distinct
received samples). Motion, sample gaps or excessive FT variance restart the
window; a 10-second timeout or residual above the configured bounds blocks
execution. Defaults are in `live.remote_desired.ft_bias` in the Valve YAML.
These checks do **not** detect steady contact: operator non-contact confirmation
is necessary. A calibration failure requires fixing the cause and restarting,
not an enable toggle. Changing tool, load, grasp state or sensor setup also
requires a new reviewed calibration, never automatic re-zeroing during contact.

The mean force and torque are stored in tool-rotating axes, at the original
wrench origin, then rotated to the observation frame for each history sample.
There is no torque-origin translation. This assumes pose and wrench reference
axes coincide, as in the inspected FR3 configuration (`WorldToBase` rotation
zero, world/pelvis aligned). The confirmation also acknowledges this frame
contract; do not use unchanged with a different mount/TF configuration. A
single-pose zero cannot identify incorrect load mass/CoM or repair gravity
compensation over all orientations. The bias is frozen once calibrated and
persists through disable/enable; fresh external contact forces are retained.

Each enabled run saves `ft_bias_calibration.json` (including thresholds,
reference pose, sample count, mean and standard deviation) in its evaluation
directory. Worker events contain `ft_bias`, `last_uncorrected_wrench`, and
`last_policy_wrench`; the first-observation NPZ includes `uncorrected_wrench`.
Use these to compare with recorded training inputs: zeroing only deployment
inputs may introduce a train/deploy mismatch if training retained bias.
No FT sensor tare service, gravity-estimator motion, driver parameters, model
normalizer, or training data is changed. ROS-free and isolated mock ROS tests
verify the correction path; physical calibration quality is not established by
those tests. The generic offline checker does not invent a free-space window;
use its unmodified result as the recorded-data baseline.

### Offline gripper learning check (no robot commands)

`scripts/check_valve_gripper_offline.py` compares one recorded episode's gripper
labels with the checkpoint's predictions using the training dataset adapter,
checkpoint normalizer, observation downsampling, action offset, and deployment
chunk slice. It never creates a ROS node or enables a policy. For an existing
training-format dataset:

```bash
source scripts/activate_rollout_env.sh
python scripts/check_valve_gripper_offline.py \
  --dataset /path/to/valve_dataset \
  --episode 0 --inference-seed 42
```

Episode indices are zero-based. Without `--episode`, selection prefers an
episode containing both open and close labels, then a validation episode.
Use `--stride 4` or `--max-samples 100` for a smaller diagnostic; these can miss
brief transitions. The default evaluates every observation window in one
episode. `--num-inference-steps` overrides deployment sampling steps only for
this offline check. Existing output directories are never overwritten.

Results are saved under `data/eval/valve_gripper_offline/<timestamp>/`:

- `report.json`: close recall, open recall, confusion counts, constant-command
  baselines, raw prediction range, and metrics per future action offset.
- `predictions.csv` / `predictions.npz`: aligned ground truth and predictions;
  artificial end padding is excluded from metrics.
- `gripper_comparison.png`: first selected action over time and an example chunk.

If only a raw recording is available, reconstruct **one** episode in a new DP
output directory (original bags remain unchanged):

```bash
python scripts/reconstruct_valve_episode.py \
  --episode /path/to/recording_group/episode_01 \
  --output data/eval/valve_episode01_reconstructed
python scripts/check_valve_gripper_offline.py \
  --dataset data/eval/valve_episode01_reconstructed \
  --episode 0 --inference-seed 42
```

The reconstruction script uses the recording workspace's existing CDR parser
(override `--cdr-reader` if needed), read-only SQLite, bag receive timestamps,
latest-at-or-before samples on a 30 Hz grid, front/wrist RGB, and no pose offset.
It saves a camera preview and reconstruction provenance. This is a diagnostic
adapter, **not a replacement for the original training exporter**: exact
cropping, synchronization, filtering, and dataset split equivalence are not
verified. Its split is reported as unknown. High recorded-data accuracy is not
proof of closed-loop grasp success; low accuracy on reconstructed input alone
does not establish that training failed. Gripper action-server behavior and
the live stability/queue filters are not simulated.

If the intended trial is arm-only, prepare the grasp manually and explicitly use:

```yaml
gripper:
  remote_desired_mode: hold_current
  allow_remote_desired_hold_current: true
```

## 5. Execute a short trial

Start from the demonstrated initial pose, keep the E-stop in hand, and use a very
short duration:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_real_valve.py --mode execute \
  --duration-seconds 3 \
  --confirm I_UNDERSTAND_THIS_WILL_MOVE_THE_FRANKA
```

For the peg task, substitute `eval_real_peg.py`. Increase duration only after
reviewing the event log. Keep publishing `policy_enable=true` only while
the operator is ready; publish `false` to stop new bridge output. `Ctrl-C` and a
false gate are not emergency stops. Reset/reposition manually between trials; the runner
intentionally has no automatic go-home command.

## Output and troubleshooting

All files go below `data/eval/`; neither source Zarr is modified.

- `missing_state` or camera timeout: check the three ROS input topics and their
  namespaces/rates.
- `competing_desired_publisher`: stop `TeleopControllerRemote` or the stale
  policy bridge; never run two command publishers.
- Worker heartbeat failure: verify that the bridge uses the configured ROS
  Python and that localhost connections are allowed.
- Missing live imports under `rdp_ros`: use a single Python environment
  compatible with ROS 2 and DP; do not patch around a Python ABI mismatch.
- CUDA OOM while loading: the checkpoint is about 5.3 GB and deserializes model,
  EMA, and optimizer before retaining EMA. Stop training jobs or use a machine
  with more host/GPU memory.
- Targets outside workspace or step limits: treat this as a failed policy/sensor
  check. Do not widen limits until the pose frame, camera assignment, and relative
  transform have been verified.
- Immediate wrench rejection: inspect bias/sign conventions and zero the sensor
  using the same procedure as data collection; never disable the limit merely to
  make rollout start.
