#!/usr/bin/env python3
"""Offline diagnostic reconstruction of ONE SQLite ROS bag; not a training exporter.

Uses existing cdr_reader.py from the recording workspace, with read-only SQLite.
No ROS node or publisher is created. Original training conversion is not assumed
identical: use the original replay_buffer.zarr for exact training-set evaluation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import zarr
from numcodecs import Blosc


def previous_indices(times, queries):
    indices = np.searchsorted(times, queries, side='right') - 1
    if np.any(indices < 0):
        raise ValueError('Requested a sample before a stream starts')
    return indices


def read_stream(bags, topic, parser_module):
    rows = []
    for bag in bags:
        with sqlite3.connect(bag.as_uri() + '?mode=ro', uri=True) as db:
            spec = db.execute('SELECT id,type FROM topics WHERE name=?', (topic,)).fetchone()
            if spec is None:
                continue
            parser = parser_module.PARSERS[spec[1]]
            for timestamp, blob in db.execute(
                    'SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp', (spec[0],)):
                if bytes(blob[:2]) != b'\x00\x01':
                    raise ValueError('Existing CDR helper supports little-endian CDR only')
                msg = parser(bytes(blob))
                rows.append((int(timestamp), msg))
    if not rows:
        raise ValueError(f'Missing/empty required topic: {topic}')
    rows.sort(key=lambda row: row[0])
    return np.array([r[0] for r in rows], dtype=np.int64), [r[1] for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory, never overwrite')
    parser.add_argument('--cdr-reader', type=Path, default=Path(
        '/home/iristele/work/junsu/iris_learning_from_teleop_fr3/visualize/raw_bag/cdr_reader.py'))
    parser.add_argument('--fps', type=float, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps must be finite and positive')
    episode = args.episode.expanduser().resolve()
    bags = sorted((episode/'bag').glob('*.db3'))
    if not bags:
        parser.error('Expected episode/bag/*.db3')
    spec = importlib.util.spec_from_file_location('recording_cdr_reader', args.cdr_reader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    topics = dict(remote='/lfd/rightarm/Teleop_Remote',
                  desired='/lfd/rightarm/Teleop_RemoteDesired',
                  external_img='/rightarm/cameras/front/image_raw/compressed',
                  left_wrist_img='/rightarm/cameras/wrist/image_raw/compressed',
                  gripper='/rightarm/gripper_command', width='/rightarm/gripper_width_m')
    streams = {key: read_stream(bags, topic, module) for key, topic in topics.items()}
    t0 = max(times[0] for times, _ in streams.values())
    t1 = min(times[-1] for times, _ in streams.values())
    if t1 <= t0:
        raise ValueError('Streams do not overlap')
    timeline = t0 + np.round(np.arange(int((t1-t0)/1e9*args.fps)+1)/args.fps*1e9).astype(np.int64)
    indices = {key: previous_indices(times, timeline) for key, (times, _) in streams.items()}
    selected = {key: [msgs[i] for i in indices[key]] for key, (_, msgs) in streams.items()}
    def pose(msgs):
        xyz = np.asarray([m.pose_pos for m in msgs])
        rot6 = Rotation.from_quat([m.pose_quat for m in msgs]).as_matrix()[:, :, :2].swapaxes(1, 2).reshape(-1, 6)
        return np.concatenate([xyz, rot6], axis=1).astype(np.float32)
    remote_pose = pose(selected['remote'])
    desired_pose = pose(selected['desired'])
    commands = np.asarray(selected['gripper'], dtype=np.float32)
    if not np.isclose(np.abs(commands), 1, atol=1e-4).all():
        raise ValueError('Expected gripper -1/+1 commands')
    args.output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(args.output/'replay_buffer.zarr'), mode='w-')
    data, meta = group.create_group('data'), group.create_group('meta')
    meta.create_dataset('episode_ends', data=np.array([len(timeline)], dtype=np.int64))
    arrays = dict(left_robot_tcp_pose=remote_pose,
                  left_robot_tcp_wrench=np.asarray([m.wrench_force+m.wrench_torque for m in selected['remote']], dtype=np.float32),
                  action=np.c_[desired_pose, commands],
                  recorded_gripper_width_m=np.asarray(selected['width'], dtype=np.float32),
                  timestamp_ns=timeline)
    for key, values in arrays.items():
        data.create_dataset(key, data=values)
    previews = []
    for key in ('external_img', 'left_wrist_img'):
        dataset = data.create_dataset(key, shape=(len(timeline), 240, 320, 3), dtype='u1',
                                      chunks=(1, 240, 320, 3), compressor=Blosc(cname='lz4', clevel=3))
        for i, msg in enumerate(selected[key]):
            frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f'Cannot decode {key} frame {i}')
            rgb = cv2.cvtColor(cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            dataset[i] = rgb
        preview_indices = [0, int(np.flatnonzero(commands > 0)[0]) if (commands > 0).any() else len(commands)//2, len(commands)-1]
        previews.append(np.concatenate([dataset[i] for i in preview_indices], axis=1))
    cv2.imwrite(str(args.output/'camera_preview.jpg'), cv2.cvtColor(np.concatenate(previews), cv2.COLOR_RGB2BGR))
    report = dict(source_episode=str(episode), source_bags=[str(b) for b in bags],
                  hardware_commands_sent=0, fps=args.fps, frames=len(timeline), topics=topics,
                  method='ROS bag receive timestamps; latest sample at/before each 30Hz query; '
                         'overlapping stream interval; no pose offset; RGB resize 320x240 INTER_AREA; '
                         'remote pose/wrench observations; RemoteDesired pose plus gripper_command actions',
                  original_training_conversion_equivalence='NOT VERIFIED',
                  split='unknown: single raw episode, not original training episode ordering',
                  command_values=np.unique(commands).tolist(),
                  close_frames=int((commands > 0).sum()), open_frames=int((commands < 0).sum()),
                  transition_frames=(np.flatnonzero(np.diff(commands)!=0)+1).tolist(),
                  alignment_age_seconds={key: dict(
                      max=float(np.max(timeline-times[indices[key]])/1e9),
                      median=float(np.median(timeline-times[indices[key]])/1e9))
                      for key, (times, _) in streams.items()})
    with (args.output/'reconstruction.json').open('w') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
