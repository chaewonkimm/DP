#!/usr/bin/env python3
"""Build a compact next-state TCP-action sidecar from an existing Zarr.

The observation ReplayBuffer remains untouched.  For every episode this tool
sets ``action[t]`` to ``left_robot_tcp_pose[t + 1]`` and repeats the terminal
TCP pose at the final frame.  Only the 9D action and matching episode metadata
are written, so image observations are not duplicated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr


def resolve_zarr(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "replay_buffer.zarr").is_dir():
        path = path / "replay_buffer.zarr"
    if not (path / ".zgroup").is_file():
        raise FileNotFoundError(
            f"Expected a Zarr group or replay_buffer.zarr under {path}")
    return path


def next_state_actions(
        tcp_pose: np.ndarray,
        episode_ends: np.ndarray,
        ) -> np.ndarray:
    """Return per-episode next TCP states with terminal hold padding."""
    pose = np.asarray(tcp_pose, dtype=np.float32)
    ends = np.asarray(episode_ends, dtype=np.int64)
    if pose.ndim != 2 or pose.shape[1] != 9:
        raise ValueError(f"Expected TCP pose shape (N, 9), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("TCP pose contains non-finite values")
    if ends.ndim != 1 or len(ends) == 0:
        raise ValueError("episode_ends must be a non-empty 1D array")
    if np.any(np.diff(ends) <= 0) or int(ends[-1]) != len(pose):
        raise ValueError("episode_ends are inconsistent with TCP poses")

    actions = pose.copy()
    starts = np.r_[0, ends[:-1]]
    for start, end in zip(starts, ends):
        start = int(start)
        end = int(end)
        if end - start > 1:
            actions[start:end - 1] = pose[start + 1:end]
        actions[end - 1] = pose[end - 1]
    return actions


def build_sidecar(
        input_path: Path,
        output_path: Path,
        pose_key: str = "left_robot_tcp_pose",
        dry_run: bool = False,
        ) -> dict:
    input_zarr = resolve_zarr(input_path)
    source = zarr.open_group(str(input_zarr), mode="r")
    if "data" not in source or "meta" not in source:
        raise ValueError("Source Zarr must contain data and meta groups")
    if pose_key not in source["data"]:
        raise KeyError(f"Source dataset has no {pose_key!r} array")
    if "episode_ends" not in source["meta"]:
        raise KeyError("Source dataset has no episode_ends metadata")

    poses = np.asarray(source["data"][pose_key][:], dtype=np.float32)
    episode_ends = np.asarray(
        source["meta"]["episode_ends"][:], dtype=np.int64)
    actions = next_state_actions(poses, episode_ends)
    starts = np.r_[0, episode_ends[:-1]]

    nonterminal = np.ones(len(actions), dtype=bool)
    nonterminal[episode_ends - 1] = False
    successor_indices = np.arange(len(actions)) + 1
    if not np.array_equal(
            actions[nonterminal], poses[successor_indices[nonterminal]]):
        raise RuntimeError("Non-terminal actions do not match the next TCP state")
    if not np.array_equal(actions[episode_ends - 1], poses[episode_ends - 1]):
        raise RuntimeError("Terminal actions do not hold the final TCP state")

    lengths = episode_ends - starts
    summary = {
        "dataset_type": "next_state_action_sidecar",
        "source_zarr": str(input_zarr),
        "pose_key": pose_key,
        "action_semantics": (
            "action[t] = TCP pose[t+1] within each episode; terminal holds"),
        "episodes": int(len(episode_ends)),
        "frames": int(len(actions)),
        "episode_length_min": int(lengths.min()),
        "episode_length_median": float(np.median(lengths)),
        "episode_length_max": int(lengths.max()),
        "action_shape": list(actions.shape),
    }
    if dry_run:
        return summary

    output_parent = output_path.expanduser().resolve()
    output_zarr = (output_parent if output_parent.name == "replay_buffer.zarr"
                   else output_parent / "replay_buffer.zarr")
    output_parent = output_zarr.parent
    if output_parent.exists():
        raise FileExistsError(f"Output already exists: {output_parent}")
    output_parent.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = output_parent.with_name(
        f".{output_parent.name}.tmp-{uuid.uuid4().hex}")
    temporary_zarr = temporary_parent / "replay_buffer.zarr"
    temporary_parent.mkdir(parents=False, exist_ok=False)

    try:
        target = zarr.open_group(str(temporary_zarr), mode="w")
        target.attrs.update(summary)
        target_data = target.create_group("data")
        target_meta = target.create_group("meta")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        target_data.create_dataset(
            "action",
            data=actions,
            chunks=(min(10000, len(actions)), 9),
            compressor=compressor,
        )
        target_meta.create_dataset(
            "episode_ends", data=episode_ends, compressor=None)

        checked = zarr.open_group(str(temporary_zarr), mode="r")
        if not np.array_equal(checked["meta"]["episode_ends"][:], episode_ends):
            raise RuntimeError("Sidecar episode boundaries failed validation")
        if not np.array_equal(checked["data"]["action"][:], actions):
            raise RuntimeError("Sidecar actions failed round-trip validation")
        with (temporary_parent / "build_summary.json").open("w") as stream:
            json.dump(summary, stream, indent=2)
        os.replace(temporary_parent, output_parent)
    except Exception:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pose-key", default="left_robot_tcp_pose",
        help="9D TCP pose array used to construct next-state actions",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_sidecar(
        input_path=args.input,
        output_path=args.output,
        pose_key=args.pose_key,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
