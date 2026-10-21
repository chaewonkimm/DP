"""Summarize saved predictions, NOT approved limits or actual command history.

Reads existing peg shadow artifacts only; never connects to ROS or a robot.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Statistics require nonempty, finite data")
    return dict(count=len(values), min=float(values.min()),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(values.max()))


def bounds(poses):
    xyz = np.asarray(poses, dtype=float)[:, :3]
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Position bounds require nonempty, finite data")
    return dict(count=len(xyz), min_xyz_m=xyz.min(0).tolist(),
                max_xyz_m=xyz.max(0).tolist())


def rotations(poses):
    """Use the deployment's Gram-Schmidt two-column rotation-6D convention."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] < 9 or not np.isfinite(poses).all():
        raise ValueError("Expected finite [N, >=9] pose data")
    x = poses[:, 3:6].copy()
    xnorm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(xnorm < 1e-8):
        raise ValueError("Degenerate rotation-6D first column")
    x /= xnorm
    z = np.cross(x, poses[:, 6:9])
    znorm = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(znorm < 1e-8):
        raise ValueError("Degenerate rotation-6D second column")
    z /= znorm
    return Rotation.from_matrix(np.stack((x, np.cross(z, x), z), axis=2))


def pose_distances(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (np.linalg.norm(a[:, :3] - b[:, :3], axis=1),
            (rotations(a).inv() * rotations(b)).magnitude())


def chunk_metrics(actions, base):
    steps, angles = pose_distances(actions[:-1], actions[1:])
    offsets, rotations_from_base = pose_distances(np.asarray(base)[None, :], actions)
    return dict(translation_step_m=steps, rotation_step_rad=angles,
                translation_from_base_m=offsets, rotation_from_base_rad=rotations_from_base)


def summarize(root):
    runs, csv_rows, warnings = [], [], []
    metric_parts = {}
    targets, first_targets, bases, wrenches, inference_times = [], [], [], [], []
    for directory in sorted(root.glob("*_shadow")):
        event_path = directory / "remote_desired_worker_events.jsonl"
        snapshot = directory / "remote_desired_first_observation.npz"
        if not event_path.exists():
            continue
        # Fail on malformed input rather than silently omitting observations.
        events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
        if not events:
            continue
        report_path = directory / "shadow_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        if report and report.get("mode") != "shadow":
            raise ValueError(f"Non-shadow report: {report_path}")
        if not report:
            warnings.append(f"{directory.name}: final report absent; completion unconfirmed")
        with np.load(snapshot, allow_pickle=False) as data:
            actions = data["predicted_absolute_action"].astype(float)
            observations = data["left_robot_tcp_pose"].astype(float)
            wrench = data["left_robot_tcp_wrench"].astype(float)
        if wrench.ndim != 2 or wrench.shape[1] != 6 or not np.isfinite(wrench).all():
            raise ValueError(f"Invalid wrench data: {snapshot}")
        if events[0]["sequence"] != 1 or not np.allclose(
                actions[0], events[0]["first_absolute_action"], atol=1e-7, rtol=0):
            raise ValueError(f"First snapshot/event mismatch: {directory}")
        metrics = chunk_metrics(actions, observations[-1])
        event_targets = np.asarray([e["first_absolute_action"] for e in events], dtype=float)
        event_bases = np.asarray([e["base_pose"] for e in events], dtype=float)
        first_offsets, first_angles = pose_distances(event_bases, event_targets)
        metrics["event_first_target_from_base_m"] = first_offsets
        metrics["event_first_target_from_base_rad"] = first_angles
        for key, values in metrics.items():
            metric_parts.setdefault(key, []).append(values)
        targets.append(actions)
        first_targets.append(event_targets)
        bases.append(event_bases)
        wrenches.append(wrench)
        times = [e["inference_time_sec"] for e in events]
        inference_times.extend(times)
        run = dict(run=directory.name, event_count=len(events),
                   completion="unconfirmed" if report is None else report["bridge_return_code"],
                   source_files=[str(event_path), str(snapshot)],
                   first_chunk_target_bounds=bounds(actions),
                   metrics={key: stats(values) for key, values in metrics.items()},
                   force_norm_n=stats(np.linalg.norm(wrench[:, :3], axis=1)),
                   torque_norm_nm=stats(np.linalg.norm(wrench[:, 3:], axis=1)))
        runs.append(run)
        # Keep sequences separate; never compute artificial steps across runs/chunks.
        for index, action in enumerate(actions):
            csv_rows.append(dict(run=directory.name, source="first_chunk_snapshot",
                                 sequence=1, action_index=index,
                                 x_m=action[0], y_m=action[1], z_m=action[2]))
        # Sequence 1 was already included above; avoid double counting that pose.
        for event in events[1:]:
            action = event["first_absolute_action"]
            csv_rows.append(dict(run=directory.name, source="event_first_target_only",
                                 sequence=event["sequence"], action_index=0,
                                 x_m=action[0], y_m=action[1], z_m=action[2]))
    if not runs:
        raise ValueError(f"No saved shadow predictions in {root}")
    wrench = np.concatenate(wrenches)
    union = np.array([[row[k] for k in ("x_m", "y_m", "z_m")] for row in csv_rows])
    output = dict(
        purpose="Observed data for operator review; NOT safe limits or execution approval",
        units="position/translation: m; rotation: rad; force: N; torque: N*m",
        coordinate_frame="Same absolute pose convention as the recorded Teleop_Remote; frame calibration not verified",
        limitations=[
            "Full prediction arrays exist only for the first inference chunk of each run.",
            "Other events store only the first target; it may be skipped by playback latency handling.",
            "Within-chunk adjacent predictions are NOT the actual published command sequence.",
            "Skipped targets and transitions between chunks can change actual command step sizes.",
            "Wrench samples cover only the initial two frames per run, not contact/insertion or the full run.",
            "Observations cannot establish safe workspace, collision clearance, contact limits or closed-loop success.",
        ],
        warnings=warnings,
        coverage=dict(runs=len(runs), inference_events=sum(r["event_count"] for r in runs),
                      full_chunks=len(targets), full_chunk_targets=sum(map(len, targets)),
                      within_chunk_steps=sum(len(x)-1 for x in targets),
                      wrench_frames=len(wrench), unique_saved_targets=len(csv_rows)),
        aggregate=dict(saved_target_bounds=bounds(union),
                       observation_base_bounds=bounds(np.concatenate(bases)),
                       first_chunk_target_bounds=bounds(np.concatenate(targets)),
                       event_first_target_bounds=bounds(np.concatenate(first_targets)),
                       metrics={key: stats(np.concatenate(parts)) for key, parts in metric_parts.items()},
                       force_norm_n=stats(np.linalg.norm(wrench[:, :3], axis=1)),
                       torque_norm_nm=stats(np.linalg.norm(wrench[:, 3:], axis=1)),
                       inference_seconds=stats(inference_times)),
        runs=runs,
        recommended_safety_limits=None,
        deployment_configuration_modified=False,
    )
    return output, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, rows = summarize(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "observed_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output_dir / "saved_target_positions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: report[key] for key in ("coverage", "warnings", "aggregate")}, indent=2))


if __name__ == "__main__":
    main()
