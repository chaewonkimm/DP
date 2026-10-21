"""Read-only force analysis. Run from the workspace with the existing robodiff Python.

Only this script's directory receives outputs. No image arrays are loaded.
ROS bags use SQLite mode=ro, query_only and a small per-connection cache.
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/chaewon-force-matplotlib')
import numpy as np
import zarr
from numcodecs import blosc
blosc.set_nthreads(1)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
TOPIC = '/lfd/rightarm/Teleop_Remote'
spec = importlib.util.spec_from_file_location('force_cdr', ROOT / 'iris_teleop_fr3_1007/visualize/raw_bag/cdr_reader.py')
cdr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cdr
sys.dont_write_bytecode = True
spec.loader.exec_module(cdr)


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    good = values[np.isfinite(values)]
    if not len(good):
        raise ValueError('No finite values')
    q = np.percentile(good, [0, 1, 5, 50, 90, 95, 99, 99.9, 100])
    result = dict(zip(['min', 'p01', 'p05', 'median', 'p90', 'p95', 'p99', 'p999', 'max'], map(float, q)))
    result.update(samples=len(values), valid_samples=len(good), invalid_samples=int(len(values)-len(good)),
                  mean=float(good.mean()), std=float(good.std()), exact_zero_samples=int((good == 0).sum()))
    result['above_threshold_percent'] = {str(n): float((good > n).mean()*100) for n in [5, 10, 20, 30, 40, 50, 60, 80, 100]}
    return result


def axes_summary(forces):
    return {name: statistics(forces[:, i]) for i, name in enumerate(['Fx', 'Fy', 'Fz'])}


def episode_summary(name, f, t):
    mag = np.linalg.norm(f, axis=1)
    result = statistics(mag)
    imax = int(np.nanargmax(mag))
    result.update(episode=name, max_sample_index=imax, max_force_xyz=f[imax].tolist(),
                  max_timestamp=float(t[imax]))
    return result, mag


def write_csv(path, rows):
    simple = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(simple[0]))
        writer.writeheader()
        writer.writerows(simple)


report = {'formula': 'sqrt(Fx**2 + Fy**2 + Fz**2)', 'force_unit': 'N',
          'weighting': 'Each finite recorded sample has equal weight; episodes are not equally weighted.',
          'source_read_mode': 'Zarr r; SQLite mode=ro and query_only',
          'additional_processing': 'None: no offset removal, smoothing, normalization or contact segmentation.',
          'replay': [], 'raw': []}
plot_data = []
replay_data = {}
for path in sorted((ROOT / 'Data/replay_buffer').glob('*/replay_buffer.zarr')):
    name = path.parent.name
    group = zarr.open_group(str(path), mode='r')
    w = np.asarray(group['data/left_robot_tcp_wrench'][:], dtype=np.float64)
    t = group['data/timestamp'][:]
    ends = group['meta/episode_ends'][:].astype(np.int64)
    assert w.ndim == 2 and w.shape[1] == 6 and len(t) == len(w)
    assert ends[-1] == len(w) and np.all(np.diff(np.r_[0, ends]) > 0)
    rows, deltas = [], []
    start = 0
    for i, end in enumerate(ends):
        row, mag = episode_summary(f'replay_index_{i}', w[start:end, :3], t[start:end])
        row['global_max_sample_index'] = start + row['max_sample_index']
        row['max_seconds_since_first_saved_sample'] = row['max_timestamp'] - float(t[start])
        rows.append(row)
        deltas.extend(np.diff(t[start:end]).tolist())
        start = int(end)
    mag = np.linalg.norm(w[:, :3], axis=1)
    positive_dt = np.array(deltas)[np.array(deltas) > 0]
    entry = dict(name=name, path=str(path), field='data/left_robot_tcp_wrench[:, :3]', episodes=len(ends),
                 force_magnitude_N=statistics(mag), axes_N=axes_summary(w[:, :3]),
                 nonfinite_wrench_values=int((~np.isfinite(w)).sum()),
                 median_positive_dt_s=float(np.median(positive_dt)),
                 max_episode=max(rows, key=lambda r: r['max']),
                 episode_index_note='Replay indices are zero-based. Raw episode identity is not encoded in this Zarr metadata.')
    report['replay'].append(entry)
    write_csv(OUT / f'{name}_episodes.csv', rows)
    plot_data.append((name, mag))
    replay_data[name] = (w, t, ends)
    print('REPLAY', name, json.dumps(entry['force_magnitude_N']), flush=True)

# Compare both peg variants using the exact stored timestamp and six wrench values,
# separately within each corresponding replay episode index.
base = 'square_peg_in_hole_21mm_20260916_p2p_goal_zarr'
desired = 'square_peg_in_hole_21mm_20260916_p2p_goal_desired_zarr'
wb, tb, eb = replay_data[base]
wd, td, ed = replay_data[desired]
assert len(eb) == len(ed)
start_b = start_d = matched = 0
for end_b, end_d in zip(eb, ed):
    keys = {struct.pack('<7d', float(ts), *row) for ts, row in zip(tb[start_b:end_b], wb[start_b:end_b])}
    matched += sum(struct.pack('<7d', float(ts), *row) in keys for ts, row in zip(td[start_d:end_d], wd[start_d:end_d]))
    start_b, start_d = end_b, end_d
report['peg_variant_comparison'] = dict(desired_frames=len(wd), base_frames=len(wb),
    desired_frames_exactly_matching_base_timestamp_and_wrench_within_same_episode_index=int(matched))
del replay_data

for dataset in sorted((ROOT / 'Data/raw_bag').iterdir()):
    if not dataset.is_dir():
        continue
    episode_paths = sorted(dataset.glob('episode_*'), key=lambda p: int(p.name.split('_')[-1]))
    if not episode_paths:
        continue
    rows, magnitudes, force_arrays = [], [], []
    frames, source_manifest = set(), []
    for ep in episode_paths:
        fs, ts = [], []
        bags = sorted((ep / 'bag').glob('*.db3'))
        if not bags:
            raise ValueError(f'No bag for {ep}')
        for bag in bags:
            before = bag.stat()
            with sqlite3.connect(bag.resolve().as_uri()+'?mode=ro', uri=True) as db:
                db.execute('PRAGMA query_only=ON')
                db.execute('PRAGMA cache_size=-2048')
                topic = db.execute('SELECT id,type FROM topics WHERE name=?', (TOPIC,)).fetchone()
                if not topic or topic[1] != 'iris_teleop/msg/TeleopStateLite':
                    raise ValueError(f'Unexpected/missing topic {bag}: {topic}')
                # A sequential table scan avoids an additional index sort and does not deserialize images.
                for j, (stamp, blob) in enumerate(db.execute('SELECT timestamp,data FROM messages WHERE topic_id=?', (topic[0],))):
                    if blob[:2] != b'\x00\x01' or (len(blob)-4) % 8:
                        raise ValueError(f'Unsupported CDR {bag}')
                    wrench = struct.unpack_from('<6d', blob, len(blob)-48)
                    if j < 2:
                        parsed = cdr.parse_teleop_state_lite(blob)
                        assert np.array_equal(wrench, parsed.wrench_force+parsed.wrench_torque, equal_nan=True)
                        frames.add(parsed.frame_id)
                    fs.append(wrench[:3])
                    ts.append(stamp)
            after = bag.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'Source changed: {bag}'
            source_manifest.append(dict(path=str(bag), size=before.st_size, mtime_ns=before.st_mtime_ns))
        f = np.asarray(fs, dtype=np.float64)
        timestamps = np.asarray(ts, dtype=np.int64)
        order = np.argsort(timestamps, kind='stable')
        f, timestamps = f[order], timestamps[order]
        # Float seconds are only for human-readable summaries, not alignment.
        row, mag = episode_summary(ep.name, f, (timestamps-timestamps[0])/1e9)
        row['max_bag_timestamp_ns'] = int(timestamps[row['max_sample_index']])
        row['duration_s'] = float((timestamps[-1]-timestamps[0])/1e9)
        row['effective_receive_rate_Hz'] = (len(f)-1)/row['duration_s'] if row['duration_s'] else 0.0
        rows.append(row)
        magnitudes.append(mag)
        force_arrays.append(f)
        if len(rows) % 10 == 0:
            print('RAW_PROGRESS', dataset.name, len(rows), '/', len(episode_paths), flush=True)
    f = np.concatenate(force_arrays)
    mag = np.concatenate(magnitudes)
    entry = dict(name=dataset.name, path=str(dataset), topic=TOPIC, episodes=len(rows),
                 force_magnitude_N=statistics(mag), axes_N=axes_summary(f),
                 header_frames_sampled=sorted(frames), max_episode=max(rows, key=lambda r: r['max']),
                 median_episode_effective_receive_rate_Hz=float(np.median([r['effective_receive_rate_Hz'] for r in rows])))
    report['raw'].append(entry)
    write_csv(OUT / f'{dataset.name}_raw_episodes.csv', rows)
    (OUT / f'{dataset.name}_source_manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
    print('RAW_DONE', dataset.name, json.dumps(entry['force_magnitude_N']), flush=True)
    # Only the replay distributions are plotted; retain no raw arrays across datasets.
    del f, mag, force_arrays, magnitudes, fs, ts
    (OUT / 'statistics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

(OUT / 'statistics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
labels = ['Peg desired', 'Peg goal', 'Valve']
for (name, values), label in zip(plot_data, labels):
    finite = values[np.isfinite(values)]
    axes[0].hist(finite, bins=np.arange(0, 101, 1), density=True, histtype='step', linewidth=1.5, label=label)
    ordered = np.sort(finite)
    axes[1].plot(ordered, np.arange(1, len(ordered)+1)/len(ordered)*100, label=label)
axes[0].set(xlabel='Force magnitude (N)', ylabel='Probability density (1/N)', title='Replay buffer distribution', xlim=(0, 100))
axes[1].set(xlabel='Force magnitude (N)', ylabel='Cumulative samples (%)', title='Replay buffer empirical CDF', xlim=(0, 100), ylim=(0, 100))
for ax in axes:
    ax.grid(alpha=.25)
    ax.legend()
fig.savefig(OUT / 'force_distribution.png', dpi=180)
fig.savefig(OUT / 'force_distribution.pdf')
plt.close(fig)
print('Saved', OUT, flush=True)
