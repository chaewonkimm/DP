"""ROS-free control core for the DP -> Teleop_RemoteDesired bridge.

This follows the lifecycle used by ``learned_remote_desired_real_bridge.py``
in iris_learning_from_teleop_fr3, but the command contract is deliberately
different: this DP policy returns absolute poses reconstructed at the policy
observation time.  Re-applying its relative actions to a later measured pose
would change the policy semantics, so the bridge queues absolute xyz+xyzw
targets instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np


POSE_DIM = 7
STATE_DIM = 13


@dataclass(frozen=True)
class WrenchFilterSettings:
    """Serializable causal-IIR settings for the ROS-side wrench stream.

    Coefficients are designed in the DP process (where SciPy is already a
    dependency) and sent to the ROS bridge.  The bridge therefore needs only
    NumPy while still using the exact Butterworth coefficients used by
    ``scipy.signal.lfilter`` during dataset generation.
    """

    enabled: bool = False
    sample_rate_hz: float | None = None
    resample: bool = True
    b: tuple[float, ...] = ()
    a: tuple[float, ...] = ()
    zi: tuple[float, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "WrenchFilterSettings":
        payload = payload or {}
        settings = cls(
            enabled=bool(payload.get("enabled", False)),
            sample_rate_hz=(
                None if payload.get("sample_rate_hz") is None
                else float(payload["sample_rate_hz"])
            ),
            resample=bool(payload.get("resample", True)),
            b=tuple(float(value) for value in payload.get("b", ())),
            a=tuple(float(value) for value in payload.get("a", ())),
            zi=tuple(float(value) for value in payload.get("zi", ())),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.sample_rate_hz is None or not np.isfinite(
                self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("Enabled wrench filter requires a positive sample_rate_hz")
        if len(self.a) < 2 or len(self.a) != len(self.b):
            raise ValueError("Wrench filter a/b coefficients must have equal length >= 2")
        if len(self.zi) != len(self.a) - 1:
            raise ValueError("Wrench filter zi must have len(a)-1 values")
        if not np.isfinite((*self.a, *self.b, *self.zi)).all():
            raise ValueError("Wrench filter coefficients must be finite")
        if abs(self.a[0] - 1.0) > 1e-9:
            raise ValueError("Wrench filter coefficients must be normalized to a[0] == 1")


class StreamingWrenchFilter:
    """Causal multi-channel IIR with optional uniform-grid resampling.

    The direct-form-II-transposed recurrence matches SciPy's ``lfilter``.
    Its initialization and interpolation order mirror
    ``RDP/real_rollout/core/live_wrench_filter.py``.
    """

    def __init__(self, settings: WrenchFilterSettings, n_channels: int = 6):
        self.settings = settings
        self.n_channels = int(n_channels)
        self.b = np.asarray(settings.b, dtype=np.float64)
        self.a = np.asarray(settings.a, dtype=np.float64)
        self.zi_base = np.asarray(settings.zi, dtype=np.float64)
        self.dt = (
            None if settings.sample_rate_hz is None
            else 1.0 / settings.sample_rate_hz
        )
        self.state: np.ndarray | None = None
        self.previous_time: float | None = None
        self.previous_wrench: np.ndarray | None = None
        self.latest_time: float | None = None
        self.latest_wrench: np.ndarray | None = None
        self.next_grid_time: float | None = None
        self.filtered: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.filtered is not None

    def reset(self, now_sec: float, wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench, dtype=np.float64).reshape(self.n_channels)
        if not np.isfinite(wrench).all():
            raise ValueError("Wrench filter input must be finite")
        self.previous_time = self.latest_time = float(now_sec)
        self.previous_wrench = self.latest_wrench = wrench.copy()
        self.next_grid_time = float(now_sec)
        self.filtered = wrench.copy()
        if self.settings.enabled:
            self.state = wrench[:, None] * self.zi_base[None, :]
        else:
            self.state = None
        return self.filtered.copy()

    def _step(self, wrench: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Wrench filter has not been reset")
        wrench = np.asarray(wrench, dtype=np.float64).reshape(self.n_channels)
        order = len(self.a) - 1
        output = self.b[0] * wrench + self.state[:, 0]
        if order > 1:
            self.state[:, :-1] = (
                self.b[1:order][None, :] * wrench[:, None]
                + self.state[:, 1:]
                - self.a[1:order][None, :] * output[:, None]
            )
        self.state[:, -1] = self.b[order] * wrench - self.a[order] * output
        return output

    def push(self, now_sec: float, wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench, dtype=np.float64).reshape(self.n_channels)
        if not np.isfinite(wrench).all():
            raise ValueError("Wrench filter input must be finite")
        now_sec = float(now_sec)
        if not self.ready:
            return self.reset(now_sec, wrench)
        if not self.settings.enabled:
            self.previous_time, self.previous_wrench = (
                self.latest_time, self.latest_wrench)
            self.latest_time, self.latest_wrench = now_sec, wrench.copy()
            self.filtered = wrench.copy()
            return self.filtered.copy()
        if now_sec < float(self.latest_time):
            raise ValueError("Wrench timestamps must be monotonic")

        self.previous_time, self.previous_wrench = (
            self.latest_time, self.latest_wrench)
        self.latest_time, self.latest_wrench = now_sec, wrench.copy()
        if not self.settings.resample:
            self.filtered = self._step(wrench)
            return self.filtered.copy()

        span = max(now_sec - float(self.previous_time), 1e-9)
        while self.next_grid_time is not None and self.next_grid_time <= now_sec:
            fraction = (self.next_grid_time - float(self.previous_time)) / span
            interpolated = self.previous_wrench + (
                self.latest_wrench - self.previous_wrench) * fraction
            self.filtered = self._step(interpolated)
            self.next_grid_time += float(self.dt)
        return self.filtered.copy()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "ready": self.ready,
            "sample_rate_hz": self.settings.sample_rate_hz,
            "resample": self.settings.resample,
        }


def validate_chunk_window(chunk_start_index, chunk_size, available_actions=None):
    """Indices refer to policy.predict_action()['action'], not the raw horizon."""
    for name, value in (("chunk_start_index", chunk_start_index), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if chunk_start_index < 0 or chunk_size <= 0:
        raise ValueError("chunk_start_index must be nonnegative and chunk_size positive")
    end = chunk_start_index + chunk_size
    if available_actions is not None and end > available_actions:
        raise ValueError(f"Requested action slice [{chunk_start_index}:{end}] exceeds "
                         f"policy action count {available_actions}")
    return int(chunk_start_index), int(end)


def select_action_chunk(actions, chunk_start_index, chunk_size):
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError("Expected a 2D action array")
    start, end = validate_chunk_window(chunk_start_index, chunk_size, len(actions))
    return actions[start:end]


def _unit_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Quaternion is not finite and non-zero")
    return quaternion / norm


def quaternion_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest rotation angle between two xyzw quaternions."""
    qa = _unit_quaternion_xyzw(a)
    qb = _unit_quaternion_xyzw(b)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


@dataclass(frozen=True)
class BridgeSettings:
    obs_steps: int
    obs_downsample_ratio: int
    action_rate_hz: float
    chunk_size: int
    prefetch_steps: int
    state_timeout_sec: float
    image_timeout_sec: float
    command_target_timeout_sec: float
    require_enable: bool
    workspace_min: tuple[float, float, float] | None
    workspace_max: tuple[float, float, float] | None
    max_translation_step_m: float | None
    max_rotation_step_rad: float | None
    max_force_n: float | None
    max_torque_nm: float | None
    chunk_start_index: int = 0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BridgeSettings":
        observation = payload["observation"]
        timing = payload["timing"]
        action = payload["action"]
        safety = payload["safety"]

        def optional_float(name: str) -> float | None:
            value = safety.get(name)
            return None if value is None else float(value)

        def optional_vector(name: str) -> tuple[float, float, float] | None:
            value = safety.get(name)
            if value is None:
                return None
            if len(value) != 3 or any(item is None for item in value):
                raise ValueError(f"safety.{name} must have three finite values")
            vector = tuple(float(item) for item in value)
            if not np.isfinite(vector).all():
                raise ValueError(f"safety.{name} must be finite")
            return vector

        settings = cls(
            obs_steps=int(observation["obs_steps"]),
            obs_downsample_ratio=int(observation["obs_downsample_ratio"]),
            action_rate_hz=float(timing["action_rate_hz"]),
            chunk_size=int(action["chunk_size"]),
            chunk_start_index=action.get("chunk_start_index", 0),
            prefetch_steps=int(action["prefetch_steps"]),
            state_timeout_sec=float(timing["state_timeout_sec"]),
            image_timeout_sec=float(timing["image_timeout_sec"]),
            command_target_timeout_sec=float(
                timing["command_target_timeout_sec"]),
            require_enable=bool(safety["require_enable"]),
            workspace_min=optional_vector("workspace_min"),
            workspace_max=optional_vector("workspace_max"),
            max_translation_step_m=optional_float("max_translation_step_m"),
            max_rotation_step_rad=optional_float("max_rotation_step_rad"),
            max_force_n=optional_float("max_force_n"),
            max_torque_nm=optional_float("max_torque_nm"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        validate_chunk_window(self.chunk_start_index, self.chunk_size)
        if self.obs_steps <= 0 or self.obs_downsample_ratio <= 0:
            raise ValueError("Observation counts must be positive")
        if self.action_rate_hz <= 0 or self.chunk_size <= 0:
            raise ValueError("Action rate and chunk size must be positive")
        if not 0 <= self.prefetch_steps < self.chunk_size:
            raise ValueError("prefetch_steps must be in [0, chunk_size)")
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("workspace_min and workspace_max must both be set")
        if self.workspace_min is not None and any(
                lo >= hi for lo, hi in zip(
                    self.workspace_min, self.workspace_max or ())):
            raise ValueError("workspace_min must be below workspace_max")
        for name in (
                "max_translation_step_m", "max_rotation_step_rad",
                "max_force_n", "max_torque_nm"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive when configured")

    @property
    def history_length(self) -> int:
        return (self.obs_steps - 1) * self.obs_downsample_ratio + 1

    @property
    def dt_sec(self) -> float:
        return 1.0 / self.action_rate_hz


@dataclass
class ObservationBuffer:
    settings: BridgeSettings
    camera_names: tuple[str, ...]
    latest_state: np.ndarray | None = None
    latest_state_time: float | None = None
    latest_images: dict[str, bytes] = field(default_factory=dict)
    latest_image_times: dict[str, float] = field(default_factory=dict)
    state_history: deque = field(init=False)
    image_history: dict[str, deque] = field(init=False)

    def __post_init__(self) -> None:
        self.state_history = deque(maxlen=self.settings.history_length)
        self.image_history = {
            name: deque(maxlen=self.settings.history_length)
            for name in self.camera_names
        }

    def push_state(self, state: np.ndarray, now_sec: float) -> None:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError("Remote state must be 13 finite pose/wrench values")
        state[3:7] = _unit_quaternion_xyzw(state[3:7])
        self.latest_state = state
        self.latest_state_time = float(now_sec)

    def push_image(self, name: str, data: bytes, now_sec: float) -> None:
        if name in self.image_history:
            self.latest_images[name] = bytes(data)
            self.latest_image_times[name] = float(now_sec)

    def freshness(self, now_sec: float) -> str | None:
        if self.latest_state is None:
            return "missing_state"
        if now_sec - float(self.latest_state_time) > self.settings.state_timeout_sec:
            return "state_timeout"
        for name in self.camera_names:
            if name not in self.latest_images:
                return f"missing_{name}"
            if now_sec - self.latest_image_times[name] > self.settings.image_timeout_sec:
                return f"{name}_timeout"
        return None

    def sample_tick(self) -> None:
        if self.latest_state is None:
            return
        self.state_history.append(self.latest_state.copy())
        for name in self.camera_names:
            if name in self.latest_images:
                self.image_history[name].append(self.latest_images[name])

    @property
    def ready(self) -> bool:
        size = self.settings.history_length
        return (
            len(self.state_history) == size
            and all(len(history) == size for history in self.image_history.values())
        )

    def request(self, sequence: int) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Observation history is not ready")
        indices = [
            index * self.settings.obs_downsample_ratio
            for index in range(self.settings.obs_steps)
        ]
        states = list(self.state_history)
        return {
            "request_type": "predict",
            "protocol_version": 1,
            "sequence": int(sequence),
            "observation.state": np.stack(
                [states[index] for index in indices]).astype(np.float32),
            "images": {
                name: [list(self.image_history[name])[index] for index in indices]
                for name in self.camera_names
            },
        }


@dataclass
class AbsoluteChunkPlayer:
    """Latency-aware absolute target chunk playback."""

    settings: BridgeSettings
    queue: deque = field(default_factory=deque)
    pending_sequence: int | None = None
    pending_tick: int | None = None
    next_chunk: tuple[int, np.ndarray] | None = None
    sequence: int = 0
    stale_entries_skipped: int = 0
    underrun_ticks: int = 0

    @property
    def depth(self) -> int:
        return len(self.queue)

    def should_request(self) -> bool:
        return (
            self.pending_sequence is None
            and self.next_chunk is None
            and self.depth <= self.settings.prefetch_steps
        )

    def start_request(self, tick: int) -> int:
        self.sequence += 1
        self.pending_sequence = self.sequence
        self.pending_tick = int(tick)
        return self.sequence

    def cancel_request(self) -> None:
        self.pending_sequence = None
        self.pending_tick = None

    def on_chunk(self, sequence: int, actions: np.ndarray, now_tick: int) -> None:
        if sequence != self.pending_sequence:
            return
        observed_tick = int(self.pending_tick)
        self.cancel_request()
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] not in (7, 8):
            raise ValueError(f"Absolute action chunk must be [N,7] or [N,8], got {actions.shape}")
        if len(actions) != self.settings.chunk_size:
            raise ValueError("Worker action count does not match configured chunk_size")
        if not np.isfinite(actions).all():
            raise ValueError("Absolute action chunk contains NaN or Inf")
        for row in actions:
            row[3:7] = _unit_quaternion_xyzw(row[3:7])
        self.next_chunk = (observed_tick, actions)
        if not self.queue:
            self._load(now_tick)

    def _load(self, now_tick: int) -> None:
        if self.next_chunk is None:
            return
        observed_tick, actions = self.next_chunk
        self.next_chunk = None
        # The worker already sliced the prefix. Preserve the ORIGINAL action
        # timeline: elapsed=6, offset=4 means discard 2 selected actions, not 6.
        start = max(0, int(now_tick) - observed_tick - self.settings.chunk_start_index)
        self.stale_entries_skipped += min(start, len(actions))
        for index in range(start, len(actions)):
            self.queue.append(actions[index].copy())

    def pop(self, now_tick: int) -> np.ndarray | None:
        if not self.queue:
            self._load(now_tick)
        if not self.queue:
            self.underrun_ticks += 1
            return None
        return self.queue.popleft()


def validate_target(
        target: np.ndarray,
        reference: np.ndarray,
        measured_wrench: np.ndarray,
        settings: BridgeSettings,
        ) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(7)
    reference = np.asarray(reference, dtype=np.float64).reshape(7)
    wrench = np.asarray(measured_wrench, dtype=np.float64).reshape(6)
    if not np.isfinite(target).all() or not np.isfinite(wrench).all():
        raise RuntimeError("Target or measured wrench contains NaN/Inf")

    if settings.workspace_min is not None:
        lower = np.asarray(settings.workspace_min)
        upper = np.asarray(settings.workspace_max)
        if np.any(target[:3] < lower) or np.any(target[:3] > upper):
            raise RuntimeError("target_outside_workspace")

    translation = float(np.linalg.norm(target[:3] - reference[:3]))
    rotation = quaternion_distance(target[3:7], reference[3:7])
    force = float(np.linalg.norm(wrench[:3]))
    torque = float(np.linalg.norm(wrench[3:]))
    checks = (
        (settings.max_translation_step_m, translation, "translation_step"),
        (settings.max_rotation_step_rad, rotation, "rotation_step"),
        (settings.max_force_n, force, "force"),
        (settings.max_torque_nm, torque, "torque"),
    )
    for limit, value, name in checks:
        if limit is not None and value > limit:
            raise RuntimeError(f"{name}_limit:{value:.6g}>{limit:.6g}")
    return {
        "translation_step_m": translation,
        "rotation_step_rad": rotation,
        "force_n": force,
        "torque_nm": torque,
    }
