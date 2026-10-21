"""Nonblocking Franka Hand actions. No ROS imports or hardware I/O on import."""
from dataclasses import dataclass
import math
import time


def quantize_gripper_command(value: float, threshold: float = 0.0) -> int:
    """Convert a model float to -1=open / +1=close; zero belongs to close."""
    value, threshold = float(value), float(threshold)
    if not math.isfinite(value) or not math.isfinite(threshold):
        raise ValueError('Gripper prediction and threshold must be finite')
    return 1 if value >= threshold else -1


@dataclass(frozen=True)
class GripperActionSettings:
    action_namespace: str
    open_width_m: float
    closed_width_m: float
    velocity_mps: float
    force_limit_n: float
    grasp_epsilon_inner_m: float
    grasp_epsilon_outer_m: float
    close_with_force: bool
    command_stability_seconds: float = 0.1
    action_timeout_seconds: float = 5.0

    @classmethod
    def from_payload(cls, payload):
        keys = ('open_width_m', 'closed_width_m', 'velocity_mps', 'force_limit_n',
                'grasp_epsilon_inner_m', 'grasp_epsilon_outer_m')
        values = {key: float(payload[key]) for key in keys}
        values.update({key: float(payload.get(key, default)) for key, default in
                       [('command_stability_seconds', 0.1), ('action_timeout_seconds', 5.0)]})
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('Gripper settings must be finite')
        if not 0 <= values['closed_width_m'] < values['open_width_m']:
            raise ValueError('Require 0 <= closed_width_m < open_width_m')
        for key in ('velocity_mps', 'force_limit_n', 'action_timeout_seconds'):
            if values[key] <= 0:
                raise ValueError(f'gripper.{key} must be positive')
        for key in ('grasp_epsilon_inner_m', 'grasp_epsilon_outer_m', 'command_stability_seconds'):
            if values[key] < 0:
                raise ValueError(f'gripper.{key} must be nonnegative')
        namespace = str(payload.get('action_namespace', '')).rstrip('/')
        if not namespace.startswith('/') or not namespace or any(c.isspace() for c in namespace):
            raise ValueError('gripper.action_namespace must be an absolute ROS namespace')
        if not isinstance(payload['close_with_force'], bool):
            raise ValueError('gripper.close_with_force must be boolean')
        return cls(action_namespace=namespace, close_with_force=payload['close_with_force'], **values)


class GripperActionController:
    """Single-executor state machine; send only on stable command changes.

    Never queues historical commands, retries failed grasps, homes automatically,
    or cancels other clients' goals. Disable cancels only this client's unfinished
    goal. Completed grasps are left untouched (no automatic release).
    """
    def __init__(self, node, settings, *, client_factory=None, action_types=None,
                 clock=time.monotonic):
        if client_factory is None:
            from rclpy.action import ActionClient
            client_factory = ActionClient
        if action_types is None:
            from franka_msgs.action import Move, Grasp
            action_types = (Move, Grasp)
        self.settings = settings
        self.clock = clock
        self.types = dict(zip(('move', 'grasp'), action_types))
        self.clients = {name: client_factory(node, typ, settings.action_namespace + '/' + name)
                        for name, typ in self.types.items()}
        self.desired = None
        self.stable_since = clock()
        self.last_completed = None
        self.inflight_width = None
        self.pending = None
        self.goal = None
        self.started = None
        self.cancel_requested = False
        self.cancel_sent = False
        self.error = None
        self.goals_sent = 0
        self.last_result = None
        self.last_goal = None

    @property
    def busy(self):
        return self.pending is not None or self.goal is not None

    @property
    def ready(self):
        return all(client.server_is_ready() for client in self.clients.values())

    def disable(self):
        self.desired = None
        if self.busy:
            self.cancel_requested = True
            self._cancel()

    def reset_error(self):
        if not self.busy:
            self.error = None

    def _fail(self, reason):
        self.error = reason
        self.disable()

    def _cancel(self):
        if self.goal is not None and not self.cancel_sent:
            self.cancel_sent = True
            try:
                self.goal.cancel_goal_async().add_done_callback(self._cancel_response)
            except Exception as error:
                self.error = f'cancel_error:{error}'

    def _cancel_response(self, future):
        try:
            response = future.result()
            if self.busy and not response.goals_canceling:
                self.error = 'cancel_rejected'
        except Exception as error:
            self.error = f'cancel_error:{error}'

    def tick(self, width):
        now = self.clock()
        if width is None:
            self.disable()
            return
        if self.error:
            self.disable()
            return
        width = float(width)
        if not math.isfinite(width) or not any(
                abs(width - target) <= 1e-6 for target in
                (self.settings.open_width_m, self.settings.closed_width_m)):
            self._fail('invalid_binary_width')
            return
        # Canonicalize float32 values before deduplication.
        width = min((self.settings.open_width_m, self.settings.closed_width_m),
                    key=lambda target: abs(width - target))
        if width != self.desired:
            self.desired, self.stable_since = width, now
        if self.busy:
            if now - self.started > self.settings.action_timeout_seconds:
                self._fail('action_timeout')
            return
        if width == self.last_completed or now - self.stable_since < self.settings.command_stability_seconds:
            return
        if not self.ready:
            self._fail('action_server_unavailable')
            return
        kind = ('grasp' if width == self.settings.closed_width_m
                and self.settings.close_with_force else 'move')
        goal = self.types[kind].Goal()
        goal.width, goal.speed = width, self.settings.velocity_mps
        if kind == 'grasp':
            goal.force = self.settings.force_limit_n
            goal.epsilon.inner = self.settings.grasp_epsilon_inner_m
            goal.epsilon.outer = self.settings.grasp_epsilon_outer_m
        self.last_goal = dict(action=self.settings.action_namespace + '/' + kind,
                              command=1 if width == self.settings.closed_width_m else -1,
                              width=goal.width, speed=goal.speed)
        if kind == 'grasp':
            self.last_goal.update(force=goal.force,
                                  epsilon=dict(inner=goal.epsilon.inner, outer=goal.epsilon.outer))
        self.inflight_width, self.started = width, now
        self.cancel_requested = self.cancel_sent = False
        try:
            self.pending = self.clients[kind].send_goal_async(goal)
            self.goals_sent += 1
            self.pending.add_done_callback(self._accepted)
        except Exception as error:
            self.pending = None
            self._fail(f'send_error:{error}')

    def _accepted(self, future):
        self.pending = None
        try:
            handle = future.result()
            if not handle.accepted:
                self._fail('goal_rejected')
                return
            self.goal = handle
            handle.get_result_async().add_done_callback(self._result)
            if self.cancel_requested or self.desired is None or self.error:
                self._cancel()
        except Exception as error:
            self._fail(f'goal_response_error:{error}')

    def _result(self, future):
        try:
            wrapped = future.result()
            self.last_result = {'status': int(wrapped.status),
                                'success': bool(wrapped.result.success),
                                'error': str(wrapped.result.error)}
            if wrapped.status == 4 and wrapped.result.success:  # STATUS_SUCCEEDED
                self.last_completed = self.inflight_width
            elif self.cancel_requested and wrapped.status == 5:  # STATUS_CANCELED
                # A partially executed move/grasp leaves the physical width unknown.
                self.last_completed = None
            else:
                self.error = f'action_failed:{self.last_result}'
        except Exception as error:
            self.error = f'result_error:{error}'
        finally:
            self.goal = None
            self.inflight_width = None
            self.cancel_sent = False

    def status(self):
        return dict(ready=self.ready, busy=self.busy, desired_width_m=self.desired,
                    desired_command=(None if self.desired is None else
                                     1 if self.desired == self.settings.closed_width_m else -1),
                    last_completed_width_m=self.last_completed, error=self.error,
                    goals_sent=self.goals_sent, last_goal=self.last_goal, last_result=self.last_result)
