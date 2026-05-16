"""Per-episode DeepRacer reward-param metrics, wired automatically.

What gets logged (every episode, averaged per rollout in TB/MLflow):

- ``dr/ep_reward``           — total episode reward (the sum your reward fn returned).
- ``dr/ep_length``           — episode length in env steps.
- ``dr/ep_offtrack_count``   — number of steps where the car was off-track
  (any of ``is_offtrack`` true, or ``all_wheels_on_track`` false).
- ``dr/ep_crash_count``      — number of steps where ``is_crashed`` was true.
- ``dr/ep_max_progress``     — peak ``progress`` value reached this episode.
- ``dr/ep_mean_speed``       — average ``speed`` across the episode's steps.
- ``dr/ep_mean_steering_abs``— average ``|steering_angle|`` (proxy for jerky driving).
- ``dr/ep_offtrack_rate``    — ``offtrack_count / steps`` (per-step rate).

Wiring is automatic. The orchestrator (``gym_dr/trainer.py``) wraps your
reward callable with a recorder and the env with an episode finalizer
before calling the trainer. Inside ``Sb3Trainer.fit`` a callback drains the
finalized summaries from each episode's ``info["dr_episode"]`` and pushes
them to the SB3 logger — TensorBoard picks them up directly and the
existing MLflow mirror callback re-publishes them as MLflow metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Tuple

import gymnasium as gym

if TYPE_CHECKING:
    from gym_dr.config import ExperimentConfig


@dataclass
class _EpisodeMetrics:
    """Stateful per-step accumulator; finalized on episode boundary."""

    steps: int = 0
    reward_sum: float = 0.0
    eval_reward_sum: float = 0.0
    offtrack_count: int = 0
    crash_count: int = 0
    max_progress: float = 0.0
    speed_sum: float = 0.0
    steering_abs_sum: float = 0.0

    def reset(self) -> None:
        self.steps = 0
        self.reward_sum = 0.0
        self.eval_reward_sum = 0.0
        self.offtrack_count = 0
        self.crash_count = 0
        self.max_progress = 0.0
        self.speed_sum = 0.0
        self.steering_abs_sum = 0.0

    def record_step(self, params: dict, reward: float, eval_reward: float = 0.0) -> None:
        self.steps += 1
        self.reward_sum += float(reward)
        self.eval_reward_sum += float(eval_reward)
        if params.get("is_offtrack", False) or not params.get("all_wheels_on_track", True):
            self.offtrack_count += 1
        if params.get("is_crashed", False):
            self.crash_count += 1
        progress = float(params.get("progress", 0.0))
        if progress > self.max_progress:
            self.max_progress = progress
        self.speed_sum += float(params.get("speed", 0.0))
        self.steering_abs_sum += abs(float(params.get("steering_angle", 0.0)))

    def summary(self) -> dict[str, float]:
        n = max(self.steps, 1)
        return {
            "dr/ep_reward": self.reward_sum,
            "dr/ep_eval_reward": self.eval_reward_sum,
            "dr/ep_length": float(self.steps),
            "dr/ep_offtrack_count": float(self.offtrack_count),
            "dr/ep_crash_count": float(self.crash_count),
            "dr/ep_max_progress": self.max_progress,
            "dr/ep_mean_speed": self.speed_sum / n,
            "dr/ep_mean_steering_abs": self.steering_abs_sum / n,
            "dr/ep_offtrack_rate": self.offtrack_count / n,
        }


def _wrap_reward(
    reward_fn: Callable[[dict], float],
    state: _EpisodeMetrics,
    eval_reward_fn: Callable[[dict], float] | None = None,
) -> Callable[[dict], float]:
    """Wrap the training reward so it also records per-step params + (optionally)
    runs the eval reward in parallel.

    The eval reward is computed on the same params dict but its return value is
    NOT used by the env — only recorded into the episode summary as
    ``dr/ep_eval_reward``. This lets HPO trials with different training rewards
    be compared on a fixed evaluation metric.
    """
    def wrapped(params: dict) -> float:
        r = reward_fn(params)
        if eval_reward_fn is None:
            state.record_step(params, r)
        else:
            try:
                er = float(eval_reward_fn(params))
            except Exception:
                # Don't let a buggy eval reward kill training.
                er = 0.0
            state.record_step(params, r, er)
        return r

    # Preserve identity for introspection / archival (inspect.getsource still finds the original).
    wrapped.__wrapped__ = reward_fn  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(reward_fn, "__name__", "reward")
    wrapped.__qualname__ = getattr(reward_fn, "__qualname__", "reward")
    wrapped.__module__ = getattr(reward_fn, "__module__", "?")
    return wrapped


class _MetricsEnvWrapper(gym.Wrapper):
    """Resets the metrics state on ``reset`` and stashes the summary in
    ``info["dr_episode"]`` on the terminal step. Otherwise transparent."""

    def __init__(self, env, state: _EpisodeMetrics) -> None:
        super().__init__(env)
        self._state = state

    def reset(self, **kwargs: Any):
        self._state.reset()
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = {**(info or {}), "dr_episode": self._state.summary()}
        return obs, reward, terminated, truncated, info


def install_metrics(experiment: "ExperimentConfig") -> Tuple["ExperimentConfig", Callable[[Any], Any]]:
    """Wire metrics around an experiment's reward + env.

    Returns ``(experiment_with_wrapped_reward, env_wrapper)``. The caller
    should build the env via ``experiment.env_factory(experiment_with_...)``
    then wrap with ``env_wrapper(env)`` before handing to the trainer.

    The wrapped reward records every call's params into a private state
    object; the env wrapper finalizes that state on each terminal step and
    stashes the summary in ``info["dr_episode"]`` for the SB3 callback to
    pick up.
    """
    state = _EpisodeMetrics()
    # The eval reward (if set) is computed in parallel to the training reward
    # but never returned to the env — it lands only in the episode summary as
    # ``dr/ep_eval_reward``, giving HPO trials with different training rewards
    # a comparable yardstick.
    eval_reward_fn = getattr(experiment, "eval_reward", None)
    wrapped_reward = _wrap_reward(experiment.reward, state, eval_reward_fn=eval_reward_fn)
    wrapped_experiment = experiment.with_overrides(reward=wrapped_reward)

    def wrap(env: Any) -> Any:
        return _MetricsEnvWrapper(env, state)

    return wrapped_experiment, wrap
