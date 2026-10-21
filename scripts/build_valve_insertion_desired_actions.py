#!/usr/bin/env python3
"""Build a compact desired-action sidecar for the valve insertion dataset.

The existing insertion dataset contains the correct cropped observations but
its ``action`` is the next measured TCP state.  This exporter reads
``/lfd/rightarm/Teleop_RemoteDesired`` directly from each original ROS 2 bag,
causally aligns the last command available at each retained camera timestamp,
and writes only the replacement 9D TCP action plus matching episode metadata.

No source file is modified and RGB arrays are not duplicated.  Pass the output
to ``P2PImageDataset.action_dataset_path``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import uuid
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
import zarr


TOPIC_REMOTE = "/lfd/rightarm/Teleop_Remote"
TOPIC_DESIRED = "/lfd/rightarm/Teleop_RemoteDesired"
TOPIC_EXTERNAL = "/rightarm/cameras/front/image_raw/compressed"
TOPIC_WRIST = "/rightarm/cameras/wrist/image_raw/compressed"
TELEOP_TYPE = "iris_teleop/msg/TeleopStateLite"


class _CDRReader:
    def __init__(self, payload: bytes):
        if payload[:2] != b"\x00\x01":
            raise ValueError("Only little-endian CDR payloads are supported")
        self.payload = payload
        self.position = 4

    def _align(self, alignment: int) -> None:
        relative = self.position - 4
        remainder = relative % alignment
        if remainder:
            self.position += alignment - remainder

    def u8(self) -> int:
        value = self.payload[self.position]
        self.position += 1
        return value

    def i32(self) -> int:
        self._align(4)
        value = struct.unpack_from("<i", self.payload, self.position)[0]
        self.position += 4
        return value

    def u32(self) -> int:
        self._align(4)
        value = struct.unpack_from("<I", self.payload, self.position)[0]
        self.position += 4
        return value

    def f64(self, count: int) -> tuple[float, ...]:
        self._align(8)
        values = struct.unpack_from(
            f"<{count}d", self.payload, self.position)
        self.position += 8 * count
        return values

    def string(self) -> str:
        length = self.u32()
        if length == 0:
            return ""
        value = self.payload[
            self.position:self.position + length - 1
        ].decode("utf-8", "replace")
        self.position += length
        return value


def _read_header(reader: _CDRReader) -> int:
    seconds = reader.i32()
    nanoseconds = reader.u32()
    reader.string()
    return seconds * 1_000_000_000 + nanoseconds


def parse_header_stamp_ns(payload: bytes) -> int:
    return _read_header(_CDRReader(payload))


def parse_teleop_pose(payload: bytes) -> tuple[int, np.ndarray]:
    reader = _CDRReader(payload)
    stamp_ns = _read_header(reader)
    reader.u8()  # state
    reader.u8()  # clutch1
    reader.u8()  # clutch2
    reader.u8()  # clutch3
    reader.f64(7)  # q
    reader.f64(7)  # dq
    reader.f64(7)  # tau
    pose = np.asarray(reader.f64(7), dtype=np.float64)
    return stamp_ns, pose


def pose7d_to_pose9d(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"Expected a finite 7D pose, got {pose}")
    quaternion = pose[3:7]
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError("Desired pose contains a zero quaternion")
    rotation = Rotation.from_quat(quaternion / norm).as_matrix()
    rotation_6d = rotation[:, :2].T.reshape(6)
    return np.concatenate((pose[:3], rotation_6d)).astype(np.float32)


def causal_match_indices(source_time: np.ndarray,
                         target_time: np.ndarray) -> np.ndarray:
    source_time = np.asarray(source_time)
    target_time = np.asarray(target_time)
    if source_time.ndim != 1 or len(source_time) == 0:
        raise ValueError("source_time must be a non-empty 1D array")
    if np.any(np.diff(source_time) < 0):
        raise ValueError("source_time must be sorted")
    indices = np.searchsorted(source_time, target_time, side="right") - 1
    return np.clip(indices, 0, len(source_time) - 1)


def _resolve_zarr(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "replay_buffer.zarr").is_dir():
        path = path / "replay_buffer.zarr"
    if not (path / ".zgroup").is_file():
        raise FileNotFoundError(
            f"Expected a Zarr group or replay_buffer.zarr under {path}")
    return path


def _bag_files(episode_dir: Path) -> list[Path]:
    files = sorted((episode_dir / "bag").glob("*.db3"))
    if not files:
        raise FileNotFoundError(f"No bag/*.db3 under {episode_dir}")
    return files


def _topic_spec(connection: sqlite3.Connection,
                topic_name: str) -> tuple[int, str]:
    row = connection.execute(
        "SELECT id,type FROM topics WHERE name=?", (topic_name,)).fetchone()
    if row is None:
        raise KeyError(f"Missing topic {topic_name!r}")
    return int(row[0]), str(row[1])


def _first_effective_time(bag_files: list[Path], topic_name: str) -> int:
    candidates = []
    for bag_file in bag_files:
        uri = bag_file.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            topic_id, _ = _topic_spec(connection, topic_name)
            row = connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp,id LIMIT 1", (topic_id,)).fetchone()
            if row is not None:
                bag_time = int(row[0])
                header_time = parse_header_stamp_ns(bytes(row[1]))
                candidates.append(header_time if header_time > 0 else bag_time)
    if not candidates:
        raise ValueError(f"Topic {topic_name!r} has no messages")
    return min(candidates)


def read_desired_series(
        episode_dir: Path,
        ) -> tuple[np.ndarray, np.ndarray, int]:
    bag_files = _bag_files(episode_dir)
    t0_ns = min(
        _first_effective_time(bag_files, TOPIC_REMOTE),
        _first_effective_time(bag_files, TOPIC_EXTERNAL),
        _first_effective_time(bag_files, TOPIC_WRIST),
    )

    rows: list[tuple[int, np.ndarray]] = []
    for bag_file in bag_files:
        uri = bag_file.as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            topic_id, topic_type = _topic_spec(connection, TOPIC_DESIRED)
            if topic_type != TELEOP_TYPE:
                raise TypeError(
                    f"{TOPIC_DESIRED} has type {topic_type!r}, expected "
                    f"{TELEOP_TYPE!r}")
            cursor = connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp,id", (topic_id,))
            for bag_time, blob in cursor:
                header_time, pose = parse_teleop_pose(bytes(blob))
                effective_time = header_time if header_time > 0 else int(bag_time)
                rows.append((effective_time, pose))
    if not rows:
        raise ValueError(f"No desired commands in {episode_dir}")
    rows.sort(key=lambda row: row[0])
    time = (np.asarray([row[0] for row in rows], dtype=np.int64) - t0_ns) / 1e9
    pose9d = np.stack([pose7d_to_pose9d(row[1]) for row in rows])
    return time, pose9d, t0_ns


def _episode_directory(raw_root: Path, zero_based_index: int) -> Path:
    episode_number = zero_based_index + 1
    direct = raw_root / f"episode_{episode_number:02d}"
    if direct.is_dir():
        return direct
    matches = [
        path for path in raw_root.glob("episode_*")
        if int(path.name.rsplit("_", 1)[-1]) == episode_number
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not resolve source episode {episode_number} in {raw_root}")
    return matches[0]


def build_desired_action_sidecar(
        raw_root: Path,
        insertion_dataset: Path,
        output_path: Path,
        max_desired_age_seconds: float = 0.03,
        max_episodes: Optional[int] = None,
        dry_run: bool = False,
        ) -> dict:
    raw_root = raw_root.expanduser().resolve()
    insertion_zarr = _resolve_zarr(insertion_dataset)
    source = zarr.open_group(str(insertion_zarr), mode="r")
    data = source["data"]
    meta = source["meta"]
    required = {"action", "timestamp"}
    missing = sorted(required.difference(data.array_keys()))
    if missing:
        raise KeyError(f"Insertion dataset is missing arrays: {missing}")
    if "source_episode_indices" not in meta:
        raise KeyError("Insertion dataset has no source_episode_indices metadata")

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    source_episode_indices = np.asarray(
        meta["source_episode_indices"][:], dtype=np.int64)
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        count = min(int(max_episodes), len(episode_ends))
        episode_ends = episode_ends[:count]
        source_episode_indices = source_episode_indices[:count]
    total_frames = int(episode_ends[-1])
    timestamps = np.asarray(data["timestamp"][:total_frames]).reshape(-1)
    old_action = np.asarray(data["action"][:total_frames, :9])
    desired_action = np.empty((total_frames, 9), dtype=np.float32)

    starts = np.r_[0, episode_ends[:-1]]
    episode_reports = []
    all_ages = []
    for output_index, (start, end, source_index) in enumerate(zip(
            starts, episode_ends, source_episode_indices)):
        episode_dir = _episode_directory(raw_root, int(source_index))
        desired_time, desired_pose, t0_ns = read_desired_series(episode_dir)
        target_time = timestamps[int(start):int(end)]
        indices = causal_match_indices(desired_time, target_time)
        ages = target_time - desired_time[indices]
        if np.any(ages < -1e-9):
            raise ValueError(
                f"Episode {source_index + 1}: desired matching used a future command")
        maximum_age = float(np.max(ages))
        if maximum_age > max_desired_age_seconds:
            raise ValueError(
                f"Episode {source_index + 1}: desired command age {maximum_age:.6f}s "
                f"exceeds {max_desired_age_seconds:.6f}s")
        matched = desired_pose[indices]
        desired_action[int(start):int(end)] = matched
        all_ages.append(ages)
        difference = np.abs(matched - old_action[int(start):int(end)])
        episode_reports.append({
            "output_episode_index": output_index,
            "source_episode_index": int(source_index),
            "raw_episode": str(episode_dir),
            "frames": int(end - start),
            "desired_messages": int(len(desired_time)),
            "sync_t0_ns": int(t0_ns),
            "desired_age_median_seconds": float(np.median(ages)),
            "desired_age_max_seconds": maximum_age,
            "desired_vs_next_state_max_abs": float(np.max(difference)),
        })

    all_ages_array = np.concatenate(all_ages)
    action_difference = np.abs(desired_action - old_action)
    summary = {
        "dataset_type": "valve_insertion_desired_action_sidecar",
        "raw_root": str(raw_root),
        "observation_zarr": str(insertion_zarr),
        "output_frames": total_frames,
        "episodes": int(len(episode_ends)),
        "action_shape": list(desired_action.shape),
        "action_semantics": "absolute Teleop_RemoteDesired TCP pose, causal previous match",
        "contains_goal_pose": False,
        "desired_age_median_seconds": float(np.median(all_ages_array)),
        "desired_age_max_seconds": float(np.max(all_ages_array)),
        "desired_vs_next_state_max_abs": float(np.max(action_difference)),
        "desired_vs_next_state_mean_abs": float(np.mean(action_difference)),
        "episode_reports": episode_reports,
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
        target.attrs.update({key: value for key, value in summary.items()
                             if key != "episode_reports"})
        target_data = target.create_group("data")
        target_meta = target.create_group("meta")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        target_data.create_dataset(
            "action", data=desired_action,
            chunks=(min(10000, total_frames), 9), compressor=compressor)
        target_data.create_dataset(
            "timestamp", data=timestamps,
            chunks=(min(10000, total_frames),), compressor=compressor)
        if "source_frame_index" in data:
            source_frame_index = np.asarray(
                data["source_frame_index"][:total_frames], dtype=np.int64)
            target_data.create_dataset(
                "source_frame_index", data=source_frame_index,
                chunks=(min(10000, total_frames),), compressor=compressor)
        target_meta.create_dataset(
            "episode_ends", data=episode_ends, compressor=None)
        target_meta.create_dataset(
            "source_episode_indices", data=source_episode_indices,
            compressor=None)

        checked = zarr.open_group(str(temporary_zarr), mode="r")
        if tuple(checked["data"]["action"].shape) != (total_frames, 9):
            raise RuntimeError("Desired action sidecar shape validation failed")
        if not np.isfinite(checked["data"]["action"][:]).all():
            raise RuntimeError("Desired action sidecar contains non-finite values")
        if not np.array_equal(
                checked["meta"]["episode_ends"][:], episode_ends):
            raise RuntimeError("Desired action episode boundary validation failed")
        with (temporary_parent / "build_summary.json").open("w") as stream:
            json.dump(summary, stream, indent=2)
        os.replace(temporary_parent, output_parent)
    except Exception:
        shutil.rmtree(temporary_parent, ignore_errors=True)
        raise
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--insertion-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-desired-age-seconds", type=float, default=0.03)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_desired_age_seconds <= 0:
        parser.error("--max-desired-age-seconds must be positive")
    return args


def main() -> None:
    args = _parse_args()
    summary = build_desired_action_sidecar(
        raw_root=args.raw_root,
        insertion_dataset=args.insertion_dataset,
        output_path=args.output,
        max_desired_age_seconds=args.max_desired_age_seconds,
        max_episodes=args.max_episodes,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
