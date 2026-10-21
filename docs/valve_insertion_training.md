# Valve insertion-only Diffusion Policy

This pipeline keeps the existing full valve and square-peg experiments intact.
It trains a separate policy for the motion after a rule-based valve grasp.

## Policy boundary

The runtime state machine should perform:

1. move to a verified pre-grasp pose with the gripper open;
2. move to the grasp pose and close with a force limit;
3. verify the grasp using measured gripper width/state;
4. hold briefly and refill the policy observation history;
5. hand control to the insertion-only policy while keeping the gripper closed.

The insertion policy observes both RGB cameras, relative TCP pose, and 6D TCP
wrench. It has no goal-pose input. Its target is the recorded 9D
`Teleop_RemoteDesired` TCP command, expressed relative to the latest observed
TCP pose by the dataset adapter. It never controls the gripper.

## Build the cropped dataset

The source is read-only. The command below creates a new Zarr under the DP
repository, drops 30 frames (1 second at 30 Hz) after each close transition,
and removes the tenth gripper action dimension.

First validate the crop plan without writing:

```bash
cd /home/iris/Workspace/junsu/DP
python scripts/build_valve_insertion_dataset.py \
  --input /home/iris/Workspace/junsu/Data/replay_buffer/valve_peg_in_hole_20260930_no_offset_gripper_zarr \
  --output /home/iris/Workspace/junsu/DP/data/datasets/valve_insertion_after_grasp_30f \
  --settle-frames 30 \
  --dry-run
```

Create the dataset:

```bash
python scripts/build_valve_insertion_dataset.py \
  --input /home/iris/Workspace/junsu/Data/replay_buffer/valve_peg_in_hole_20260930_no_offset_gripper_zarr \
  --output /home/iris/Workspace/junsu/DP/data/datasets/valve_insertion_after_grasp_30f \
  --settle-frames 30
```

The exporter refuses to overwrite an existing output. Its
`build_summary.json` records every source episode, close frame, crop start, and
retained length.

## Build the desired-command action sidecar

The cropped observation Zarr retains the old next-measured-state action for
compatibility. Build a compact sidecar that contains the desired command from
the original ROS bags; RGB data is not copied again.

```bash
python scripts/build_valve_insertion_desired_actions.py \
  --raw-root /home/iris/Workspace/junsu/Data/raw_bag/valve_peg_in_hole_20260930 \
  --insertion-dataset /home/iris/Workspace/junsu/DP/data/datasets/valve_insertion_after_grasp_30f \
  --output /home/iris/Workspace/junsu/DP/data/datasets/valve_insertion_after_grasp_desired_actions_30f
```

The command is aligned causally: each 30 Hz observation uses the most recent
desired command that had already arrived. The generated sidecar has 105
episodes and 51,809 actions, with exactly the same episode boundaries as the
observation dataset. The config combines observations from the cropped Zarr
with actions from this sidecar at load time.

## Train

```bash
cd /home/iris/Workspace/junsu/DP
conda activate /home/iris/Workspace/junsu/.conda/miniforge3/envs/robodiff
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=train_diffusion_unet_valve_insertion_p2p_image_workspace
```

This config now trains the desired-command target. Checkpoints produced by the
older next-state run are not interchangeable with the new policy; start a new
run and use a checkpoint whose saved config has task name
`valve_insertion_p2p_image_desired`.

The dedicated config uses a 0.1 episode-level validation split and saves three
checkpoint roles:

- `latest.ckpt`: periodic resume checkpoint;
- `epoch=...-train_loss=....ckpt`: lowest training loss;
- `epoch=...-val_loss=....ckpt`: lowest EMA validation loss.

Validation-best is checked every validation epoch, even though periodic/latest
checkpoints follow `checkpoint_every`. For rollout, load `ema_model` from the
validation-loss checkpoint.

Training runs for 800 epochs. Four non-replacing milestone checkpoints are also
saved after exactly 200, 400, 600, and 800 completed epochs:

```text
epoch=0200-milestone.ckpt
epoch=0400-milestone.ckpt
epoch=0600-milestone.ckpt
epoch=0800-milestone.ckpt
```

## Handoff requirements

- The rule-based grasp must end close to the post-grasp TCP pose distribution in
  the demonstrations.
- A close command alone is not proof of a successful grasp. Reject or retry when
  the measured width indicates that the fingers closed without the valve.
- Keep the gripper closed for the entire learned phase.
- Clear old observations at handoff. Hold the robot long enough to collect four
  fresh 30 Hz raw frames; the policy selects two observations at stride two.
- Tune the 30-frame crop margin only after reviewing images and robot state near
  the close transition. The source Zarr stores gripper command but not measured
  gripper width, so the exporter cannot prove physical grasp stability.
