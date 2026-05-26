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
    """Stateful per-step accumulator; finalized on episode boundary.

    ``use_eval_reward`` is a global mode flag (NOT reset per episode). When
    True, the wrapped reward returns ``eval_reward_fn(params)`` to the env
    instead of the training reward — used by ``CtxEvalCallback`` so SB3's
    EvalCallback measures the policy against ``eval_reward`` (giving Optuna
    a yardstick comparable across trials that trained on different rewards).
    """

    steps: int = 0
    reward_sum: float = 0.0
    eval_reward_sum: float = 0.0
    offtrack_count: int = 0
    crash_count: int = 0
    max_progress: float = 0.0
    speed_sum: float = 0.0
    steering_abs_sum: float = 0.0
    use_eval_reward: bool = False

    def reset(self) -> None:
        self.steps = 0
        self.reward_sum = 0.0
        self.eval_reward_sum = 0.0
        self.offtrack_count = 0
        self.crash_count = 0
        self.max_progress = 0.0
        self.speed_sum = 0.0
        self.steering_abs_sum = 0.0
        # use_eval_reward is intentionally NOT reset — it's a mode flag
        # toggled by the eval callback around evaluation episodes.

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

    Behaviour depends on ``state.use_eval_reward``:
      * False (default, training mode) — env sees the training reward;
        ``dr/ep_eval_reward`` is still recorded in parallel for monitoring.
      * True (set by ``CtxEvalCallback`` during SB3 evaluation episodes) —
        env sees the eval reward instead, so ``last_mean_reward`` reflects
        the eval-reward yardstick and Optuna prunes on a fair, cross-trial
        metric. Both sums are still recorded.
    """
    def wrapped(params: dict) -> float:
        r_train = float(reward_fn(params))
        if eval_reward_fn is None:
            state.record_step(params, r_train)
            return r_train
        try:
            r_eval = float(eval_reward_fn(params))
        except Exception:
            # Don't let a buggy eval reward kill training.
            r_eval = 0.0
        state.record_step(params, r_train, r_eval)
        return r_eval if state.use_eval_reward else r_train

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


def install_metrics(
    experiment: "ExperimentConfig",
) -> Tuple["ExperimentConfig", Callable[[Any], Any], _EpisodeMetrics]:
    """Wire metrics around an experiment's reward + env.

    Returns ``(experiment_with_wrapped_reward, env_wrapper, state)``. The
    caller should build the env via ``experiment.env_factory(experiment_with_...)``
    then wrap with ``env_wrapper(env)`` before handing to the trainer. The
    ``state`` handle lets the trainer toggle ``state.use_eval_reward``
    around evaluation episodes (see ``CtxEvalCallback``).

    The wrapped reward records every call's params into the state object;
    the env wrapper finalizes that state on each terminal step and stashes
    the summary in ``info["dr_episode"]`` for the SB3 callback to pick up.
    """
    state = _EpisodeMetrics()
    eval_reward_fn = getattr(experiment, "eval_reward", None)
    wrapped_reward = _wrap_reward(experiment.reward, state, eval_reward_fn=eval_reward_fn)
    wrapped_experiment = experiment.with_overrides(reward=wrapped_reward)

    def wrap(env: Any) -> Any:
        return _MetricsEnvWrapper(env, state)

    return wrapped_experiment, wrap, state
