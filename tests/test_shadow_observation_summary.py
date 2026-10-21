import json

import numpy as np
import pytest

from scripts.summarize_shadow_observations import (
    chunk_metrics, pose_distances, stats, summarize,
)


def pose(x=0.0):
    return np.array([x, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)


def test_distances_use_translation_norm_and_rotation_angle():
    a = pose()
    b = np.array([0.003, 0.004, 0, 0, 1, 0, -1, 0, 0], dtype=float)
    distance, angle = pose_distances(a[None], b[None])
    assert distance[0] == pytest.approx(0.005)
    assert angle[0] == pytest.approx(np.pi / 2)
    metrics = chunk_metrics(np.stack([a, b]), a)
    assert len(metrics['translation_step_m']) == 1
    assert len(metrics['translation_from_base_m']) == 2


def test_summary_deduplicates_first_target_without_cross_run_steps(tmp_path):
    for index, origin in enumerate([0.0, 10.0]):
        directory = tmp_path / f'{index}_shadow'
        directory.mkdir()
        actions = np.stack([pose(origin), pose(origin + 0.001)])
        observations = np.stack([pose(origin), pose(origin)])
        np.savez(directory / 'remote_desired_first_observation.npz',
                 predicted_absolute_action=actions,
                 left_robot_tcp_pose=observations,
                 left_robot_tcp_wrench=np.array([[3, 4, 0, 0, 0, 2]] * 2))
        events = [dict(sequence=i+1, base_pose=pose(origin).tolist(),
                       first_absolute_action=pose(origin).tolist(),
                       inference_time_sec=0.1) for i in range(2)]
        (directory / 'remote_desired_worker_events.jsonl').write_text(
            '\n'.join(json.dumps(e) for e in events))
    report, rows = summarize(tmp_path)
    assert len(rows) == 6
    assert report['coverage']['within_chunk_steps'] == 2
    assert report['aggregate']['metrics']['translation_step_m']['max'] == pytest.approx(0.001)
    assert report['aggregate']['force_norm_n']['max'] == 5
    assert report['aggregate']['torque_norm_nm']['max'] == 2
    assert len(report['warnings']) == 2
    assert report['recommended_safety_limits'] is None


def test_nonfinite_values_are_rejected():
    with pytest.raises(ValueError, match='finite'):
        stats([np.nan])
