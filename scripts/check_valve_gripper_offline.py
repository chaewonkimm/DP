#!/usr/bin/env python3
"""Compare Valve gripper predictions with one recorded Zarr episode. No ROS I/O."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gripper_metrics(truth, prediction, threshold=0.0):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError('Need matching, nonempty gripper arrays')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite gripper values')
    actual, predicted = truth >= threshold, prediction >= threshold
    closed, opened = int(actual.sum()), int((~actual).sum())
    close_recall = float(predicted[actual].mean()) if closed else None
    open_recall = float((~predicted[~actual]).mean()) if opened else None
    return dict(
        count=int(truth.size), accuracy=float((actual == predicted).mean()),
        close_recall=close_recall, open_recall=open_recall,
        balanced_accuracy=((close_recall + open_recall) / 2 if closed and opened else None),
        ground_truth_close_count=closed, ground_truth_open_count=opened,
        predicted_close_count=int(predicted.sum()),
        confusion=dict(true_close_pred_close=int((actual & predicted).sum()),
                       true_close_pred_open=int((actual & ~predicted).sum()),
                       true_open_pred_close=int((~actual & predicted).sum()),
                       true_open_pred_open=int((~actual & ~predicted).sum())),
        always_open_accuracy=float((~actual).mean()),
        always_close_accuracy=float(actual.mean()),
        mae=float(np.abs(truth - prediction).mean()),
        prediction_min=float(prediction.min()), prediction_max=float(prediction.max()),
    )


def target_indices(sampler_index, action_start, count, latency=0):
    """Map dataset action offsets back to disk; mask artificial edge padding."""
    buffer_start, buffer_end, sample_start, _ = map(int, sampler_index)
    indices = buffer_start - sample_start + latency + action_start + np.arange(count)
    return indices, (indices >= buffer_start) & (indices < buffer_end)


def choose_episode(dataset, requested):
    ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    inventory = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        command = np.asarray(dataset.replay_buffer['action'][int(start):int(end), 9])
        if not np.isfinite(command).all() or not np.isclose(np.abs(command), 1, atol=1e-4).all():
            raise ValueError(f'Episode {i}: expected recorded -1/+1 gripper commands; '
                             'check dataset conversion/mapping before evaluating')
        closed = command >= 0
        inventory.append(dict(episode=i, frames=int(end-start),
                              open_frames=int((~closed).sum()), close_frames=int(closed.sum()),
                              transitions=int(np.count_nonzero(np.diff(closed))),
                              reconstructed_split=('validation' if dataset.val_mask[i] else
                                                   'train' if dataset.train_mask[i] else 'unused')))
    if not inventory:
        raise ValueError('Dataset has no episodes')
    if requested is not None:
        if not 0 <= requested < len(ends):
            raise ValueError(f'--episode must be between 0 and {len(ends)-1} (zero-based)')
        selected = requested
    else:
        # Prefer both labels and then validation, so an always-open policy is visible.
        selected = max(range(len(ends)), key=lambda i: (
            bool(inventory[i]['open_frames'] and inventory[i]['close_frames']),
            bool(dataset.val_mask[i]), inventory[i]['transitions']))
    return selected, inventory, int(starts[selected]), int(ends[selected])


def evaluate_episode(policy, dataset, deploy_cfg, *, episode=None, stride=1, seed=42,
                     max_samples=None):
    from diffusion_policy.real_world.p2p_franka_deployment import predict_relative_actions

    if stride < 1 or (max_samples is not None and max_samples < 1):
        raise ValueError('stride and max_samples must be positive')
    if dataset.action_dim != 10:
        raise ValueError('Valve evaluation requires a 10D action including gripper')
    selected, inventory, start, end = choose_episode(dataset, episode)
    mask = np.zeros(len(inventory), dtype=bool)
    mask[selected] = True
    dataset.sampler = dataset._make_sampler(mask)
    offset = int(policy.action_start_offset)
    raw_count = int(policy.n_action_steps)
    chunk_start = int(deploy_cfg.live.remote_desired.get('chunk_start_index', 0))
    chunk_size = int(deploy_cfg.live.remote_desired.chunk_size)
    if chunk_start < 0 or chunk_size <= 0 or chunk_start + chunk_size > raw_count:
        raise ValueError('Deployment chunk slice exceeds predicted action length')
    candidates = []
    for i, index in enumerate(dataset.sampler.indices):
        observation_frame = int(index[0] - index[2] + dataset.n_raw_obs_steps - 1)
        targets, valid = target_indices(index, offset + chunk_start, chunk_size,
                                        dataset.n_latency_steps)
        if start <= observation_frame < end and (observation_frame-start) % stride == 0 and valid.any():
            candidates.append((i, observation_frame, targets, valid))
    if max_samples is not None and len(candidates) > max_samples:
        candidates = [candidates[i] for i in np.unique(
            np.linspace(0, len(candidates)-1, max_samples).astype(int))]
    if not candidates:
        raise ValueError('No usable observation/action windows in selected episode')
    print(f'Episode {selected}: {len(candidates)} windows; {inventory[selected]}', flush=True)
    predictions, truths, frames, valid_masks, observation_frames = [], [], [], [], []
    for j, (i, observation_frame, targets, valid) in enumerate(candidates):
        item = dataset[i]
        observation = {key: value[None] for key, value in item['obs'].items()}
        predicted = predict_relative_actions(policy, observation, seed)
        if predicted.shape != (raw_count, 10):
            raise ValueError(f'Unexpected policy output {predicted.shape}')
        predicted = predicted[chunk_start:chunk_start+chunk_size, 9]
        truth = item['action'][offset+chunk_start:offset+chunk_start+chunk_size, 9].numpy()
        if truth.shape != predicted.shape:
            raise ValueError('Ground-truth alignment is shorter than predicted chunk')
        predictions.append(predicted)
        truths.append(truth)
        frames.append(targets-start)
        valid_masks.append(valid)
        observation_frames.append(observation_frame-start)
        if (j+1) % 50 == 0 or j == len(candidates)-1:
            print(f'Inferred {j+1}/{len(candidates)} windows (no hardware)', flush=True)
    arrays = dict(prediction=np.stack(predictions), ground_truth=np.stack(truths),
                  target_frame=np.stack(frames), valid=np.stack(valid_masks),
                  observation_frame=np.asarray(observation_frames))
    threshold = float(deploy_cfg.gripper.command_threshold)
    first_valid = arrays['valid'][:, 0]
    first = (gripper_metrics(arrays['ground_truth'][first_valid, 0],
                            arrays['prediction'][first_valid, 0], threshold)
             if first_valid.any() else None)
    report = dict(episode=selected, episode_info=inventory[selected],
                  dataset_inventory=inventory, inference_windows=len(candidates),
                  stride=stride, max_samples=max_samples, inference_seed=seed,
                  action_start_offset=offset, n_latency_steps=dataset.n_latency_steps,
                  chunk_start_index=chunk_start, chunk_size=chunk_size,
                  command_threshold=threshold,
                  first_selected_action=first,
                  all_selected_actions=gripper_metrics(
                      arrays['ground_truth'][arrays['valid']],
                      arrays['prediction'][arrays['valid']], threshold),
                  per_chunk_offset=[
                      gripper_metrics(arrays['ground_truth'][arrays['valid'][:, k], k],
                                      arrays['prediction'][arrays['valid'][:, k], k], threshold)
                      if arrays['valid'][:, k].any() else None for k in range(chunk_size)])
    return report, arrays


def save_results(output, report, arrays, gripper_cfg):
    from diffusion_policy.real_world.p2p_franka_deployment import write_json
    from PIL import Image, ImageDraw, ImageFont

    # Never overwrite an earlier trial.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'report.json', report)
    np.savez_compressed(output / 'predictions.npz', **arrays)
    threshold = report['command_threshold']
    with (output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['observation_frame', 'chunk_offset', 'target_frame', 'true_gripper',
                         'predicted_gripper', 'true_close', 'predicted_close', 'predicted_width_m'])
        for row, col in zip(*np.nonzero(arrays['valid'])):
            predicted = float(arrays['prediction'][row, col])
            truth = float(arrays['ground_truth'][row, col])
            close = predicted >= threshold
            writer.writerow([arrays['observation_frame'][row], col,
                             arrays['target_frame'][row, col], truth, predicted,
                             int(truth >= threshold), int(close),
                             gripper_cfg.closed_width_m if close else gripper_cfg.open_width_m])
    canvas = Image.new('RGB', (1200, 800), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()

    def panel(top, x, truth, prediction, title):
        left, right, bottom = 80, 1150, top+235
        draw.text((left, top-55), title, fill='black', font=font)
        draw.text((left, top-30), 'Blue: recorded command | Orange: prediction | Gray: close threshold',
                  fill='black', font=font)
        if not len(x):
            draw.text((left, top+40), 'No valid first-action targets', fill='black', font=font)
            return
        xmin, xmax = float(x.min()), max(float(x.max()), float(x.min())+1)
        ymin = min(-1.2, float(prediction.min())-.1, threshold-.1)
        ymax = max(1.2, float(prediction.max())+.1, threshold+.1)
        def point(xv, yv):
            return (left+(float(xv)-xmin)/(xmax-xmin)*(right-left),
                    bottom-(float(yv)-ymin)/(ymax-ymin)*(bottom-top))
        for y in [-1, threshold, 1]:
            _, height = point(xmin, y)
            draw.line((left, height, right, height), fill='#cccccc', width=1)
            draw.text((15, height-10), f'{y:g}', fill='black', font=font)
        for value in np.linspace(xmin, xmax, 6):
            xpos, _ = point(value, ymin)
            draw.text((xpos-15, bottom+8), f'{value:.0f}', fill='black', font=font)
        for values, color in [(truth, '#2266bb'), (prediction, '#dd7722')]:
            points = [point(xv, yv) for xv, yv in zip(x, values)]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            else:
                px, py = points[0]
                draw.ellipse((px-3, py-3, px+3, py+3), fill=color)
        draw.rectangle((left, top, right, bottom), outline='black')
        draw.text((left+320, bottom+33), 'Target frame within episode (-1 open, +1 close)', fill='black', font=font)

    valid = arrays['valid'][:, 0]
    panel(85, arrays['target_frame'][valid, 0], arrays['ground_truth'][valid, 0],
          arrays['prediction'][valid, 0], f"Episode {report['episode']}: first selected action, recorded observations")
    # Display a chunk containing a close label, or the first chunk if none exist.
    close_rows = np.flatnonzero(((arrays['ground_truth'] >= threshold) & arrays['valid']).any(axis=1))
    row = int(close_rows[len(close_rows)//2]) if len(close_rows) else 0
    valid = arrays['valid'][row]
    panel(465, arrays['target_frame'][row, valid], arrays['ground_truth'][row, valid],
          arrays['prediction'][row, valid], f"Example predicted chunk from observation frame {arrays['observation_frame'][row]}")
    canvas.save(output / 'gripper_comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Training-format Zarr group or directory containing replay_buffer.zarr')
    parser.add_argument('--config', type=Path, default=ROOT/'diffusion_policy/config/deploy/p2p_valve_franka.yaml')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--episode', type=int, help='Zero-based episode; default prefers both labels, then validation')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-samples', type=int, help='Optional uniformly spaced window cap')
    parser.add_argument('--inference-seed', type=int, default=42)
    parser.add_argument('--num-inference-steps', type=int)
    parser.add_argument('--device', help='Default: deployment configuration device')
    parser.add_argument('--output', type=Path, help='Must not exist; default timestamped data/eval directory')
    args = parser.parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    if not ((dataset_path/'.zgroup').is_file() or (dataset_path/'replay_buffer.zarr'/'.zgroup').is_file()):
        parser.error(f'No training-format Zarr found at {dataset_path}. Raw ROS bags must be converted first.')
    if args.stride < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error('--stride and --max-samples must be positive')
    if args.num_inference_steps is not None and args.num_inference_steps < 1:
        parser.error('--num-inference-steps must be positive')
    output = args.output or ROOT/'data/eval/valve_gripper_offline'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output.exists():
        parser.error(f'Output already exists: {output}')
    import hydra
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy.real_world.p2p_franka_deployment import load_policy_from_checkpoint, validate_inference_seed
    validate_inference_seed(args.inference_seed)
    cfg = OmegaConf.load(args.config)
    checkpoint = (args.checkpoint or Path(cfg.checkpoint)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT/checkpoint
    device = args.device or str(cfg.device)
    steps = args.num_inference_steps or int(cfg.num_inference_steps)
    print(f'Offline only. Loading {checkpoint} on {device}', flush=True)
    policy, train_cfg = load_policy_from_checkpoint(str(checkpoint), device, str(cfg.weights), steps)
    dataset_cfg = copy.deepcopy(train_cfg.task.dataset)
    dataset_cfg.dataset_path = str(dataset_path)
    dataset_cfg.load_to_memory = False
    dataset = hydra.utils.instantiate(dataset_cfg)
    with torch.inference_mode():
        report, arrays = evaluate_episode(policy, dataset, cfg, episode=args.episode,
                                         stride=args.stride, seed=args.inference_seed,
                                         max_samples=args.max_samples)
    report.update(checkpoint=str(checkpoint.resolve()), weights=str(cfg.weights),
                  dataset_path=str(dataset_path), original_training_dataset=str(train_cfg.task.dataset.dataset_path),
                  device=device, num_inference_steps=steps, hardware_commands_sent=0,
                  gripper_settings=OmegaConf.to_container(cfg.gripper, resolve=True),
                  checkpoint_warning=str(cfg.get('checkpoint_warning', '')),
                  limitations=[
                      'Recorded observations, not closed-loop robot success or hardware diagnosis.',
                      'Split is reconstructed from checkpoint config; valid only with original episode ordering.',
                      'One episode cannot establish generalization. Training-episode agreement is only a fit check.',
                      'Padding excluded. Overlapping future chunks are counted repeatedly in all_selected_actions.',
                      'No action-server behavior, stability filtering, queue timing or physical grasp is simulated.',
                  ])
    reconstruction_path = (dataset_path.parent if dataset_path.name == 'replay_buffer.zarr'
                           else dataset_path) / 'reconstruction.json'
    if reconstruction_path.is_file():
        report['raw_reconstruction'] = json.loads(reconstruction_path.read_text())
        for entry in report['dataset_inventory']:
            entry['reconstructed_split'] = 'unknown (raw episode reconstruction)'
        report['episode_info']['reconstructed_split'] = 'unknown (raw episode reconstruction)'
        report['limitations'].append('Input reconstructed from raw bag, NOT the original training Zarr; '
                                     'exact synchronization/cropping/selection equivalence is unverified.')
    save_results(output, report, arrays, cfg.gripper)
    print(json.dumps({'output': str(output.resolve()), 'episode': report['episode'],
                      'first_selected_action': report['first_selected_action'],
                      'all_selected_actions': report['all_selected_actions']}, indent=2))


if __name__ == '__main__':
    main()
