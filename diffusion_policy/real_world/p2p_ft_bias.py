"""Startup residual FT zeroing for policy observations, never safety/control FT.

The upstream driver already removes gravity and fixed offsets. Residual sensor
bias is stored in rotating tool axes at the ORIGINAL wrench origin; no moment
arm translation or second gravity subtraction is performed. Pose and wrench
must have the same reference-frame orientation (current FR3 world/pelvis setup).
"""
from dataclasses import dataclass
import math

import numpy as np

from diffusion_policy.real_world.p2p_remote_desired_bridge_core import quaternion_distance


def rotation_xyzw(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError('Invalid FT reference quaternion')
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass(frozen=True)
class FTBiasSettings:
    enabled: bool = False
    free_space_confirmed: bool = False
    duration_seconds: float = 0.2
    timeout_seconds: float = 10.0
    min_samples: int = 5
    max_sample_gap_seconds: float = 0.1
    max_translation_m: float = 0.002
    max_rotation_rad: float = 0.02
    max_joint_velocity_rad_s: float = 0.02
    max_force_std_n: float = 0.5
    max_torque_std_nm: float = 0.05
    max_bias_force_n: float = 10.0
    max_bias_torque_nm: float = 1.0

    @classmethod
    def from_payload(cls, values=None):
        settings = cls(**dict(values or {}))
        for key in ('enabled', 'free_space_confirmed'):
            if not isinstance(getattr(settings, key), bool):
                raise ValueError(f'ft_bias.{key} must be boolean')
        if (isinstance(settings.min_samples, bool) or not isinstance(settings.min_samples, int)
                or settings.min_samples < 2):
            raise ValueError('ft_bias.min_samples must be an integer >=2')
        for key in cls.__dataclass_fields__:
            if key in ('enabled', 'free_space_confirmed', 'min_samples'):
                continue
            value = getattr(settings, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'ft_bias.{key} must be finite and positive')
        if settings.timeout_seconds <= settings.duration_seconds:
            raise ValueError('FT bias timeout must exceed collection duration')
        if settings.enabled and not settings.free_space_confirmed:
            raise ValueError('FT zeroing requires --confirm-ft-bias-free-space: stationary, '
                             'no external contact, fixed tool/gripper state and matching pose/wrench axes')
        return settings


class StartupFTBias:
    def __init__(self, settings, now):
        self.settings = settings
        self.ready = not settings.enabled
        self.error = None
        self.started = now
        self.samples = []
        self.reference = None
        self.window_started = None
        self.last_sample = None
        self.bias = np.zeros(6)
        self.std = None
        self.reason = 'collecting' if settings.enabled else 'disabled'
        self.resets = 0
        self.window_duration = 0.0

    def check_timeout(self, now):
        if not self.ready and not self.error and now-self.started > self.settings.timeout_seconds:
            self.error = 'calibration_timeout'
            self.reason = self.error

    def reset_window(self, reason):
        self.samples.clear()
        self.reference = self.window_started = self.last_sample = None
        self.reason = reason
        self.resets += 1

    def observe(self, state, dq, now):
        self.check_timeout(now)
        if self.ready or self.error:
            return
        state, dq = np.asarray(state), np.asarray(dq)
        if state.shape != (13,) or dq.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(dq).all():
            self.reset_window('invalid_sample')
            return
        if self.last_sample is not None:
            if now <= self.last_sample:
                return
            if now-self.last_sample > self.settings.max_sample_gap_seconds:
                self.reset_window('sample_gap')
        if np.max(np.abs(dq)) > self.settings.max_joint_velocity_rad_s:
            self.reset_window('robot_moving')
            return
        if self.reference is not None and (
                np.linalg.norm(state[:3]-self.reference[:3]) > self.settings.max_translation_m
                or quaternion_distance(state[3:7], self.reference[3:7]) > self.settings.max_rotation_rad):
            self.reset_window('pose_changed')
        if self.reference is None:
            self.reference = state[:7].copy()
            self.window_started = now
        rot = rotation_xyzw(state[3:7])
        local = np.r_[rot.T @ state[7:10], rot.T @ state[10:13]]
        if (np.linalg.norm(local[:3]) > self.settings.max_bias_force_n
                or np.linalg.norm(local[3:]) > self.settings.max_bias_torque_nm):
            self.error = self.reason = 'residual_exceeds_calibration_limit'
            return
        self.samples.append(local)
        self.last_sample = now
        self.window_duration = now-self.window_started
        self.reason = 'collecting'
        if self.window_duration + 1e-9 < self.settings.duration_seconds or len(self.samples) < self.settings.min_samples:
            return
        values = np.asarray(self.samples)
        self.std = values.std(axis=0)
        if (np.max(self.std[:3]) > self.settings.max_force_std_n
                or np.max(self.std[3:]) > self.settings.max_torque_std_nm):
            self.reset_window('wrench_unstable')
            return
        self.bias = values.mean(axis=0)
        self.ready = True
        self.reason = 'ready'

    def correct_states(self, states):
        result = np.asarray(states).copy()
        if not self.settings.enabled:
            return result
        if not self.ready or self.error:
            raise RuntimeError('FT bias is not calibrated')
        for row in result:
            rot = rotation_xyzw(row[3:7])
            row[7:10] -= rot @ self.bias[:3]
            row[10:13] -= rot @ self.bias[3:]
        return result

    def status(self):
        return dict(enabled=self.settings.enabled, ready=self.ready, reason=self.reason,
                    error=self.error, samples=len(self.samples), resets=self.resets,
                    window_seconds=self.window_duration,
                    bias_tool_axes=self.bias.tolist() if self.ready and self.settings.enabled else None,
                    std_tool_axes=None if self.std is None else self.std.tolist(),
                    reference_pose=None if self.reference is None else self.reference.tolist(),
                    application='policy_input_only; existing wrench origin; safety FT unchanged')
