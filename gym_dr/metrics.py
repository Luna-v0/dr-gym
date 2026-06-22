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
- ``dr/ep_ended_offtrack``   — ``1.0`` if the episode ended with the car
  *fully* off the track (``is_offtrack`` — a real track-out reset, not just a
  wheel over the line), else ``0.0``. During evaluation the eval callbacks sum
  this per track into ``eval/<world>_offtrack_resets`` (and a global
  ``eval/offtrack_resets``).

Wiring is automatic. The orchestrator (``gym_dr/trainer.py``) wraps your
reward callable with a recorder and the env with an episode finalizer
before calling the trainer. Inside ``Sb3Trainer.fit`` a callback drains the
finalized summaries from each episode's ``info["dr_episode"]`` and pushes
them to the SB3 logger — TensorBoard picks them up directly and the
existing MLflow mirror callback re-publishes them as MLflow metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import gymnasium as gym

if TYPE_CHECKING:
    from gym_dr.config import ExperimentConfig
    from gym_dr.trace import TraceSink


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
    # Full-track-out status (``is_offtrack``) of the MOST RECENT step — the car
    # entirely off the track, not just a wheel over the line. Read at the
    # episode boundary to tell whether the episode ended *because* the car left
    # the track (vs a lap completion or time truncation), surfaced as
    # ``dr/ep_ended_offtrack``.
    last_offtrack: bool = False
    use_eval_reward: bool = False

    # --- Eval trajectory capture (optional; TrainingConfig.eval_path_plots) ---
    # When ``capture_path`` is on, each step appends the car's (x, y) so the eval
    # callbacks can plot the driven trajectory over the track skeleton. The
    # skeleton (centerline ``waypoints`` + ``track_width``) is grabbed once per
    # episode. ``capture_path`` is a mode flag (NOT reset per episode); the
    # ``path_*`` / ``wp_*`` buffers ARE reset on each episode boundary.
    capture_path: bool = False
    path_x: list = field(default_factory=list)
    path_y: list = field(default_factory=list)
    path_speed: list = field(default_factory=list)
    wp_x: list = field(default_factory=list)
    wp_y: list = field(default_factory=list)
    track_width: float = 0.0

    # --- Tier-1 trace sink (optional; see gym_dr/trace.py) -----------------
    # These are mode-level, NOT reset per episode. The trainer updates
    # world_name/chunk_index around each runtime track swap so every row knows
    # which hot-swapped world it belongs to (docs/trace-contract.md §2).
    sink: Optional["TraceSink"] = None
    world_name: Optional[str] = None
    chunk_index: int = 0
    run_id: Optional[str] = None

    def reset(self) -> None:
        self.steps = 0
        self.reward_sum = 0.0
        self.eval_reward_sum = 0.0
        self.offtrack_count = 0
        self.crash_count = 0
        self.max_progress = 0.0
        self.speed_sum = 0.0
        self.steering_abs_sum = 0.0
        self.last_offtrack = False
        self.path_x.clear()
        self.path_y.clear()
        self.path_speed.clear()
        self.wp_x.clear()
        self.wp_y.clear()
        self.track_width = 0.0
        # capture_path / use_eval_reward are intentionally NOT reset — mode flags
        # toggled by the eval callback around evaluation episodes.

    def record_step(self, params: dict, reward: float, eval_reward: float = 0.0) -> None:
        self.steps += 1
        self.reward_sum += float(reward)
        self.eval_reward_sum += float(eval_reward)
        # Per-step off-track RATE counts any wheel over the line (a partial
        # off-track) — the soft, pre-existing signal.
        if params.get("is_offtrack", False) or not params.get("all_wheels_on_track", True):
            self.offtrack_count += 1
        # The episode-end reset discriminator is the STRONGER condition: a full
        # track-out (``is_offtrack``), i.e. the car entirely off the track —
        # which is what actually terminates the episode. A wheel merely brushing
        # the line does NOT count here.
        self.last_offtrack = bool(params.get("is_offtrack", False))
        if params.get("is_crashed", False):
            self.crash_count += 1
        progress = float(params.get("progress", 0.0))
        if progress > self.max_progress:
            self.max_progress = progress
        self.speed_sum += float(params.get("speed", 0.0))
        self.steering_abs_sum += abs(float(params.get("steering_angle", 0.0)))

        if self.capture_path:
            x = params.get("x")
            y = params.get("y")
            if x is not None and y is not None:
                self.path_x.append(float(x))
                self.path_y.append(float(y))
                self.path_speed.append(float(params.get("speed", 0.0)))
            # Grab the skeleton once per episode — waypoints/width are constant
            # per world, so the first step that carries them is enough.
            if not self.wp_x:
                wps = params.get("waypoints") or []
                if wps:
                    self.wp_x = [float(p[0]) for p in wps]
                    self.wp_y = [float(p[1]) for p in wps]
                    self.track_width = float(params.get("track_width", 0.0))

        if self.sink is not None and self.sink.enabled:
            from gym_dr.trace import build_step_row

            self.sink.add(
                build_step_row(
                    params,
                    step=self.steps,
                    reward=float(reward),
                    eval_reward=float(eval_reward),
                    phase="eval" if self.use_eval_reward else "train",
                )
            )

    def flush_episode(self) -> None:
        """Flush the buffered episode to a trace shard, if a sink is wired."""
        if self.sink is None or not self.sink.enabled:
            return
        if self.run_id is None:
            self.run_id = _active_run_id()
        self.sink.flush_episode(
            world_name=self.world_name,
            chunk_index=self.chunk_index,
            run_id=self.run_id,
        )

    def path_payload(self) -> dict:
        """Snapshot the just-finished episode's trajectory + skeleton for plots.

        Returned at the terminal step in ``info["dr_episode_path"]`` (only when
        ``capture_path`` is set) for the eval callbacks to render. ``status`` is
        derived from the terminal off-track flag and peak progress so the chart
        legend can label each episode (off-track / lap-complete / ended)."""
        if self.last_offtrack:
            status = "off-track"
        elif self.max_progress >= 99.999:
            status = "lap-complete"
        else:
            status = "ended"
        return {
            "x": list(self.path_x),
            "y": list(self.path_y),
            "speed": list(self.path_speed),
            "wp_x": list(self.wp_x),
            "wp_y": list(self.wp_y),
            "track_width": self.track_width,
            "status": status,
            "progress": self.max_progress,
        }

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
            # 1.0 if this episode ended with the car off-track (a track-out
            # reset), 0.0 otherwise (lap completion / time truncation). Averaged
            # per rollout in training; summed per eval into eval/*_offtrack_resets.
            "dr/ep_ended_offtrack": 1.0 if self.last_offtrack else 0.0,
            # Success-criterion metrics: did the car reach the lap end, and did
            # it do so without ever leaving the track? Averaged per rollout in
            # training; turned into eval/<world>_(clean_)completion_rate in eval.
            "dr/ep_completed": 1.0 if self.max_progress >= 99.999 else 0.0,
            "dr/ep_completed_clean": (
                1.0 if (self.max_progress >= 99.999 and self.offtrack_count == 0) else 0.0
            ),
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
        # Drop any unflushed partial episode (e.g. a manual mid-episode reset).
        # In normal training the terminal step has already flushed, so this is
        # a no-op on an empty buffer.
        if self._state.sink is not None:
            self._state.sink.abandon_episode()
        self._state.reset()
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = {**(info or {}), "dr_episode": self._state.summary()}
            if self._state.capture_path:
                info["dr_episode_path"] = self._state.path_payload()
            # Flush BEFORE the vec-env auto-resets, so the buffer is the
            # just-finished episode.
            self._state.flush_episode()
        return obs, reward, terminated, truncated, info


def _active_run_id() -> Optional[str]:
    """The active MLflow run id, or None if MLflow is absent/inactive.

    Resolved lazily at first flush — the run isn't open when the env (and this
    state) are built; ``start_run`` opens it later in the orchestrator.
    """
    try:
        import mlflow
    except ImportError:
        return None
    run = mlflow.active_run()
    return run.info.run_id if run is not None else None


def install_metrics(
    experiment: "ExperimentConfig",
    run_dir: "Optional[Path]" = None,
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

    When ``run_dir`` is given and ``experiment.trace.enabled`` is set, a
    :class:`gym_dr.trace.TraceSink` is attached so every step is also written to
    per-episode Parquet shards under ``run_dir/trace/steps/`` (the Tier-1
    simtrace-equivalent; see ``docs/trace-contract.md``).
    """
    state = _EpisodeMetrics()

    trace_cfg = getattr(experiment, "trace", None)
    if run_dir is not None and trace_cfg is not None and getattr(trace_cfg, "enabled", False):
        from gym_dr.trace import TraceSink

        state.sink = TraceSink(run_dir, compression=trace_cfg.compression)

    eval_reward_fn = getattr(experiment, "eval_reward", None)
    wrapped_reward = _wrap_reward(experiment.reward, state, eval_reward_fn=eval_reward_fn)
    wrapped_experiment = experiment.with_overrides(reward=wrapped_reward)

    def wrap(env: Any) -> Any:
        return _MetricsEnvWrapper(env, state)

    return wrapped_experiment, wrap, state
