#!/usr/bin/env bash
# Download validation-best and train-best checkpoints for the four
# low-dimensional square-peg DP variants. Identical files are skipped by rsync.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_peg_lowdim_checkpoints.sh [--dry-run] [--destination PATH]

Defaults:
  destination: /home/iristele/work/chaewon/DP

Environment overrides:
  DP_CHECKPOINT_REMOTE       default: iris@irislab.asuscomm.com
  DP_CHECKPOINT_REMOTE_ROOT  default: /home/iris/Workspace/junsu/DP
EOF
}

DESTINATION="/home/iristele/work/chaewon/DP"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --destination)
      if [[ $# -lt 2 ]]; then
        echo "--destination requires a path" >&2
        exit 2
      fi
      DESTINATION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REMOTE="${DP_CHECKPOINT_REMOTE:-iris@irislab.asuscomm.com}"
REMOTE_ROOT="${DP_CHECKPOINT_REMOTE_ROOT:-/home/iris/Workspace/junsu/DP}"

CHECKPOINTS=(
  "data/outputs/2026.10.11/03.23.44_train_diffusion_unet_peg_proprio_goal_desired_peg_proprio_goal_desired/checkpoints/epoch=0019-val_loss=0.00528.ckpt"
  "data/outputs/2026.10.11/03.23.44_train_diffusion_unet_peg_proprio_goal_desired_peg_proprio_goal_desired/checkpoints/epoch=0780-train_loss=0.00014.ckpt"
  "data/outputs/2026.10.11/03.24.37_train_diffusion_unet_peg_proprio_goal_wrench_desired_peg_proprio_goal_wrench_desired/checkpoints/epoch=0019-val_loss=0.00509.ckpt"
  "data/outputs/2026.10.11/03.24.37_train_diffusion_unet_peg_proprio_goal_wrench_desired_peg_proprio_goal_wrench_desired/checkpoints/epoch=0780-train_loss=0.00013.ckpt"
  "data/outputs/2026.10.13/16.47.06_train_diffusion_unet_peg_proprio_goal_next_state_peg_proprio_goal_next_state/checkpoints/epoch=0019-val_loss=0.00437.ckpt"
  "data/outputs/2026.10.13/16.47.06_train_diffusion_unet_peg_proprio_goal_next_state_peg_proprio_goal_next_state/checkpoints/epoch=0780-train_loss=0.00013.ckpt"
  "data/outputs/2026.10.13/16.47.32_train_diffusion_unet_peg_proprio_goal_wrench_next_state_peg_proprio_goal_wrench_next_state/checkpoints/epoch=0019-val_loss=0.00434.ckpt"
  "data/outputs/2026.10.13/16.47.32_train_diffusion_unet_peg_proprio_goal_wrench_next_state_peg_proprio_goal_wrench_next_state/checkpoints/epoch=0780-train_loss=0.00012.ckpt"
)

SOURCES=()
for checkpoint in "${CHECKPOINTS[@]}"; do
  SOURCES+=("${REMOTE}:${REMOTE_ROOT}/./${checkpoint}")
done

RSYNC_OPTIONS=(
  -aR
  --info=progress2
  --itemize-changes
  --partial
)
if [[ "${DRY_RUN}" == true ]]; then
  RSYNC_OPTIONS+=(--dry-run)
fi

mkdir -p "${DESTINATION}"
rsync "${RSYNC_OPTIONS[@]}" "${SOURCES[@]}" "${DESTINATION}/"
