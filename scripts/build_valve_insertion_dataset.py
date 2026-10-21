#!/usr/bin/env python3
"""Build an insertion-only valve dataset without modifying the source Zarr.

Each source episode contains one open-to-close gripper transition.  This script
starts a new episode a configurable number of frames after that transition,
copies the image/pose/wrench observations, and drops the gripper dimension from
the action.  The resulting action is a 9D absolute TCP pose; P2PImageDataset
converts it to a relative action during training exactly as it does for the
existing peg task.

The output is written through a temporary sibling directory and is only renamed
to the requested path after validation.  Existing output is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import uuid
from typing import Iterable, Optional

import numpy as np
import zarr


REQUIRED_COPY_KEYS = (
    "external_img",
    "left_wrist_img",
    "left_robot_tcp_pose",
    "left_robot_tcp_wrench",
)
OPTIONAL_AUDIT_KEYS = (
    "left_robot_gripper_command",
    "timestamp",
)


@dataclass(frozen=True)
class EpisodeCrop:
    episode_index: int
    source_episode_start: int
    source_episode_end: int
    close_offset: int
    crop_start_offset: int

    @property
    def source_crop_start(self) -> int:
        return self.source_episode_start + self.crop_start_offset

    @property
    def output_length(self) -> int:
        return self.source_episode_end - self.source_crop_start


def _as_1d(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(f"Expected a 1D command array, got {result.shape}")
    return result


def find_episode_crops(
        gripper_commands: np.ndarray,
        episode_ends: np.ndarray,
        settle_frames: int,
        minimum_frames: int,
        max_episodes: Optional[int] = None,
        ) -> list[EpisodeCrop]:
    """Return validated post-grasp source ranges for every selected episode."""
    commands = _as_1d(gripper_commands).astype(np.float32, copy=False)
    ends = np.asarray(episode_ends, dtype=np.int64)
    if settle_frames < 0:
        raise ValueError("settle_frames must be non-negative")
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be positive")
    if ends.ndim != 1 or len(ends) == 0:
        raise ValueError("episode_ends must be a non-empty 1D array")
    if np.any(np.diff(ends) <= 0) or int(ends[-1]) != len(commands):
        raise ValueError("episode_ends are inconsistent with gripper commands")

    n_episodes = len(ends)
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        n_episodes = min(n_episodes, int(max_episodes))

    crops: list[EpisodeCrop] = []
    starts = np.r_[0, ends[:-1]]
    for episode_index in range(n_episodes):
        start = int(starts[episode_index])
        end = int(ends[episode_index])
        episode = commands[start:end]
        if not np.all(np.isclose(np.abs(episode), 1.0, atol=1e-6)):
            values = np.unique(episode)
            raise ValueError(
                f"Episode {episode_index} has non-binary gripper values: {values}")
        if not np.isclose(episode[0], -1.0):
            raise ValueError(f"Episode {episode_index} does not start open (-1)")

        close_transitions = np.flatnonzero(
            np.isclose(episode[:-1], -1.0)
            & np.isclose(episode[1:], 1.0)
        ) + 1
        any_transitions = np.flatnonzero(~np.isclose(
            episode[:-1], episode[1:], atol=1e-6)) + 1
        if len(close_transitions) != 1 or len(any_transitions) != 1:
            raise ValueError(
                f"Episode {episode_index} must contain exactly one -1 to +1 "
                f"transition; got close={close_transitions.tolist()}, "
                f"all={any_transitions.tolist()}")

        close_offset = int(close_transitions[0])
        crop_start_offset = close_offset + int(settle_frames)
        remaining = len(episode) - crop_start_offset
        if remaining < minimum_frames:
            raise ValueError(
                f"Episode {episode_index} has only {remaining} frames after "
                f"close + settle, fewer than minimum_frames={minimum_frames}")
        if not np.all(np.isclose(episode[crop_start_offset:], 1.0)):
            raise ValueError(
                f"Episode {episode_index} reopens after the post-grasp crop")

        crops.append(EpisodeCrop(
            episode_index=episode_index,
            source_episode_start=start,
            source_episode_end=end,
            close_offset=close_offset,
            crop_start_offset=crop_start_offset,
        ))
    return crops


def _resolve_zarr(path: Path, *, must_exist: bool) -> Path:
    path = path.expanduser().resolve()
    if must_exist and (path / "replay_buffer.zarr").is_dir():
        path = path / "replay_buffer.zarr"
    if must_exist and not (path / ".zgroup").is_file():
        raise FileNotFoundError(
            f"Expected a Zarr group or replay_buffer.zarr under {path}")
    if not must_exist and path.name != "replay_buffer.zarr":
        path = path / "replay_buffer.zarr"
    return path


def _adjust_chunks(source: zarr.Array, output_shape: tuple[int, ...]) -> tuple[int, ...]:
    chunks = list(source.chunks)
    chunks[0] = min(chunks[0], output_shape[0])
    for axis in range(1, len(chunks)):
        chunks[axis] = min(chunks[axis], output_shape[axis])
    return tuple(chunks)


def _copy_segments(
        source: zarr.Array,
        destination: zarr.Array,
        crops: Iterable[EpisodeCrop],
        block_frames: int,
        column_slice=None,
        ) -> None:
    destination_offset = 0
    for crop in crops:
        source_offset = crop.source_crop_start
        frames_left = crop.output_length
        while frames_left:
            count = min(block_frames, frames_left)
            source_slice = slice(source_offset, source_offset + count)
            values = source[source_slice]
            if column_slice is not None:
                values = values[..., column_slice]
            destination[destination_offset:destination_offset + count] = values
            source_offset += count
            destination_offset += count
            frames_left -= count


def build_dataset(
        input_path: Path,
        output_path: Path,
        settle_frames: int,
        minimum_frames: int,
        block_frames: int,
        max_episodes: Optional[int],
        dry_run: bool,
        ) -> dict:
    input_zarr = _resolve_zarr(input_path, must_exist=True)
    output_zarr = _resolve_zarr(output_path, must_exist=False)
    source = zarr.open_group(str(input_zarr), mode="r")
    if "data" not in source or "meta" not in source:
        raise ValueError("Source Zarr must contain data and meta groups")
    source_data = source["data"]
    missing = sorted(
        set(REQUIRED_COPY_KEYS + ("action",)) - set(source_data.array_keys()))
    if missing:
        raise KeyError(f"Source dataset is missing required arrays: {missing}")

    action = source_data["action"]
    if action.ndim != 2 or action.shape[1] != 10:
        raise ValueError(f"Expected source action shape (N, 10), got {action.shape}")

    if "left_robot_gripper_command" in source_data:
        gripper = _as_1d(source_data["left_robot_gripper_command"][:])
        action_gripper = _as_1d(action[:, 9])
        if not np.array_equal(gripper, action_gripper):
            raise ValueError(
                "action[:, 9] differs from left_robot_gripper_command")
    else:
        gripper = _as_1d(action[:, 9])

    crops = find_episode_crops(
        gripper_commands=gripper,
        episode_ends=source["meta"]["episode_ends"][:],
        settle_frames=settle_frames,
        minimum_frames=minimum_frames,
        max_episodes=max_episodes,
    )
    lengths = np.asarray([crop.output_length for crop in crops], dtype=np.int64)
    output_episode_ends = np.cumsum(lengths)
    total_frames = int(output_episode_ends[-1])
    summary = {
        "source_zarr": str(input_zarr),
        "output_zarr": str(output_zarr),
        "settle_frames": int(settle_frames),
        "minimum_frames": int(minimum_frames),
        "episodes": len(crops),
        "frames": total_frames,
        "episode_length_min": int(lengths.min()),
        "episode_length_median": float(np.median(lengths)),
        "episode_length_max": int(lengths.max()),
        "action_shape": [total_frames, 9],
        "observation_keys": list(REQUIRED_COPY_KEYS),
        "uses_wrench": "left_robot_tcp_wrench" in REQUIRED_COPY_KEYS,
        "crops": [asdict(crop) for crop in crops],
    }
    if dry_run:
        return summary
    if output_zarr.parent.exists():
        # The caller names the dataset directory, while output_zarr points at
        # its replay_buffer.zarr child. Never replace an existing directory.
        raise FileExistsError(f"Output already exists: {output_zarr.parent}")

    output_parent = output_zarr.parent
    output_parent.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = output_parent.with_name(
        f".{output_parent.name}.tmp-{uuid.uuid4().hex}")
    temporary_zarr = temporary_parent / "replay_buffer.zarr"
    temporary_parent.mkdir(parents=False, exist_ok=False)

    try:
        target = zarr.open_group(str(temporary_zarr), mode="w")
        target_data = target.create_group("data")
        target_meta = target.create_group("meta")
        target.attrs.update({
            "dataset_type": "valve_insertion_after_rule_based_grasp",
            "source_zarr": str(input_zarr),
            "settle_frames": int(settle_frames),
            "gripper_semantics": "-1=open,+1=close",
            "action_semantics": "absolute 9D TCP pose; converted to relative by P2PImageDataset",
        })

        target_meta.create_dataset("episode_ends", data=output_episode_ends)
        target_meta.create_dataset(
            "source_episode_indices",
            data=np.asarray([crop.episode_index for crop in crops], dtype=np.int64))
        target_meta.create_dataset(
            "source_episode_starts",
            data=np.asarray([crop.source_episode_start for crop in crops], dtype=np.int64))
        target_meta.create_dataset(
            "source_close_offsets",
            data=np.asarray([crop.close_offset for crop in crops], dtype=np.int64))
        target_meta.create_dataset(
            "source_crop_start_offsets",
            data=np.asarray([crop.crop_start_offset for crop in crops], dtype=np.int64))

        copy_keys = list(REQUIRED_COPY_KEYS)
        copy_keys.extend(key for key in OPTIONAL_AUDIT_KEYS if key in source_data)
        for key in copy_keys + ["action"]:
            source_array = source_data[key]
            item_shape = source_array.shape[1:]
            column_slice = None
            if key == "action":
                item_shape = (9,)
                column_slice = slice(0, 9)
            output_shape = (total_frames,) + tuple(item_shape)
            destination = target_data.create_dataset(
                key,
                shape=output_shape,
                chunks=_adjust_chunks(source_array, output_shape),
                dtype=source_array.dtype,
                compressor=source_array.compressor,
            )
            print(f"Copying {key}: {source_array.shape} -> {output_shape}", flush=True)
            _copy_segments(
                source=source_array,
                destination=destination,
                crops=crops,
                block_frames=block_frames,
                column_slice=column_slice,
            )

        source_frame_index = target_data.create_dataset(
            "source_frame_index",
            shape=(total_frames,),
            chunks=(min(block_frames, total_frames),),
            dtype=np.int64,
            compressor=None,
        )
        destination_offset = 0
        for crop in crops:
            source_frame_index[
                destination_offset:destination_offset + crop.output_length
            ] = np.arange(
                crop.source_crop_start,
                crop.source_episode_end,
                dtype=np.int64,
            )
            destination_offset += crop.output_length

        # Validate the completed temporary dataset before making it visible.
        checked = zarr.open_group(str(temporary_zarr), mode="r")
        if tuple(checked["data"]["action"].shape) != (total_frames, 9):
            raise RuntimeError("Output action validation failed")
        if tuple(checked["data"]["left_robot_tcp_wrench"].shape) != (
                total_frames, 6):
            raise RuntimeError("Output wrench validation failed")
        if not np.array_equal(
                checked["meta"]["episode_ends"][:], output_episode_ends):
            raise RuntimeError("Output episode boundary validation failed")
        if "left_robot_gripper_command" in checked["data"]:
            output_command = checked["data"]["left_robot_gripper_command"][:]
            if not np.all(np.isclose(output_command, 1.0)):
                raise RuntimeError("Post-grasp output contains a non-close command")

        with (temporary_parent / "build_summary.json").open("w") as stream:
            json.dump(summary, stream, indent=2)
        os.replace(temporary_parent, output_parent)
    except Exception:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source dataset directory or replay_buffer.zarr",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New dataset directory; must not already contain files",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=30,
        help="Frames discarded after the -1 to +1 close transition (default: 30)",
    )
    parser.add_argument(
        "--minimum-frames",
        type=int,
        default=32,
        help="Minimum retained frames per episode (default: 32)",
    )
    parser.add_argument(
        "--block-frames",
        type=int,
        default=128,
        help="Maximum frames held in memory per copy block (default: 128)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional prefix of episodes for a small conversion test",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate transitions and print the crop plan without writing",
    )
    args = parser.parse_args()
    if args.block_frames < 1:
        parser.error("--block-frames must be positive")
    return args


def main() -> None:
    args = _parse_args()
    summary = build_dataset(
        input_path=args.input,
        output_path=args.output,
        settle_frames=args.settle_frames,
        minimum_frames=args.minimum_frames,
        block_frames=args.block_frames,
        max_episodes=args.max_episodes,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
