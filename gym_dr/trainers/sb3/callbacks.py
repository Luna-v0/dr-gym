from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)


def _make_eval_collector(
    capture_paths: bool = False,
) -> tuple[Any, dict[str, int], list[dict]]:
    """Build an ``evaluate_policy`` step-callback that mines finished episodes.

    ``evaluate_policy`` calls the callback once per env step with ``locals()``
    exposing the current ``done`` flag and ``info`` dict. On each *done* step we
    read what the metrics wrapper stamped into ``info``:

    - ``info["dr_episode"]["dr/ep_ended_offtrack"]`` — bumps ``count["n"]`` so it
      ends up as the number of eval episodes that terminated off-track.
    - ``info["dr_episode_path"]`` (only present when ``capture_path`` is on) — the
      episode's trajectory + skeleton, appended to ``episodes`` for plotting.

    Returns ``(callback, count, episodes)``; the caller reads them after
    ``evaluate_policy`` returns.
    """
    count = {"n": 0, "completed": 0, "clean": 0}
    episodes: list[dict] = []

    def _cb(locals_: dict, _globals: dict) -> None:
        if not locals_.get("done"):
            return
        info = locals_.get("info") or {}
        if not isinstance(info, dict):
            return
        summary = info.get("dr_episode")
        if summary:
            if summary.get("dr/ep_ended_offtrack", 0.0):
                count["n"] += 1
            if summary.get("dr/ep_completed", 0.0):
                count["completed"] += 1
            if summary.get("dr/ep_completed_clean", 0.0):
                count["clean"] += 1
        if capture_paths:
            path = info.get("dr_episode_path")
            if path:
                episodes.append(path)

    return _cb, count, episodes


def _log_eval_paths(logger: Any, world: str, timestep: int, episodes: list[dict]) -> None:
    """Render the eval trajectories for *world* and log them as TB images.

    One overlay figure (all episodes, colour + legend per episode) plus one
    figure per individual episode. No-op when there are no captured episodes.
    Matplotlib/plot imports are deferred here so they're only paid when
    ``TrainingConfig.eval_path_plots`` is enabled.
    """
    if not episodes:
        return
    from stable_baselines3.common.logger import Figure

    from gym_dr.trainers.sb3.plots import render_episode, render_overlay

    exclude = ("stdout", "log", "json", "csv")
    overlay = render_overlay(world, timestep, episodes)
    logger.record(f"eval_paths/{world}", Figure(overlay, close=True), exclude=exclude)
    for i, ep in enumerate(episodes):
        fig = render_episode(world, timestep, i, ep)
        logger.record(f"eval_paths/{world}/ep{i}", Figure(fig, close=True), exclude=exclude)


def update_training_status(run_dir: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_status.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _mastery_met(training: Any, n_offtrack: int, n_total: int) -> bool:
    """Has the car *mastered* the track in this eval round?

    Mastery = the fraction of the round's eval episodes that ended with the car
    off the track is within ``training.early_stop_max_offtrack_rate`` (``0.0`` =
    never left the track). Returns ``False`` when early stop is disabled or no
    eval episodes ran.
    """
    if not getattr(training, "early_stop_enabled", False) or n_total <= 0:
        return False
    return (n_offtrack / n_total) <= float(training.early_stop_max_offtrack_rate)


class _EarlyStopMixin:
    """Track-mastery early stop, shared by the eval callbacks.

    Both eval callbacks already tally the eval episodes that ended off-track; this
    turns that tally into a stop decision. When the car holds mastery (see
    :func:`_mastery_met`) for ``early_stop_patience`` consecutive eval rounds, the
    host ``_on_step`` returns ``False`` — the same early-exit SB3 honours for
    ``WallClockLimitCallback`` — which ends the chunk's ``model.learn`` and hands
    control back to ``Sb3Trainer.fit`` (advancing the rotation, or ending a
    single-track run). The streak resets per chunk via ``_on_training_start`` so
    mastering one track never pre-credits the next.
    """

    def _es_init(self) -> None:
        self._es_streak = 0
        self.mastered = False
        self.early_stops = 0  # non-reset: how many chunks this instance stopped

    def _on_training_start(self) -> None:
        super()._on_training_start()  # type: ignore[misc]
        self._es_streak = 0
        self.mastered = False

    def _apply_early_stop(self, n_offtrack: int, n_total: int) -> bool:
        """Fold this eval round into the mastery streak; return ``True`` when the
        chunk should stop now."""
        training = self._ctx.training  # type: ignore[attr-defined]
        if not getattr(training, "early_stop_enabled", False):
            return False
        if _mastery_met(training, n_offtrack, n_total):
            self._es_streak += 1
        else:
            self._es_streak = 0
        if self._es_streak < max(1, int(getattr(training, "early_stop_patience", 1))):
            return False
        self.mastered = True
        self.early_stops += 1
        rate = (n_offtrack / n_total) if n_total else 0.0
        steps = int(self.num_timesteps)  # type: ignore[attr-defined]
        print(
            f"[early-stop] track mastered at {steps} steps "
            f"(eval off-track rate {rate:.2f} <= "
            f"{training.early_stop_max_offtrack_rate}); ending chunk",
            flush=True,
        )
        update_training_status(
            self._ctx.run_dir,  # type: ignore[attr-defined]
            "early_stopped",
            {"timesteps_completed": steps, "eval_offtrack_rate": rate},
        )
        return True


class CtxCheckpointCallback(CheckpointCallback):
    """SB3 CheckpointCallback that routes saves through TrainingContext.

    Ensures every periodic checkpoint zip gets its `model_metadata.json`
    sibling, matching what `ctx.save_checkpoint` writes. When ``keep_last`` is
    set, prunes older checkpoints after each save so a long run doesn't fill
    the disk (``best_model``/``final_model``/``latest_model`` live elsewhere and
    are never touched).
    """

    def __init__(self, *args: Any, ctx, keep_last: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = ctx
        self._keep_last = keep_last

    def _on_step(self) -> bool:
        if self.save_freq > 0 and (self.n_calls + 1) % self.save_freq == 0:
            step = int(self.num_timesteps) + 1  # match super's post-save num_timesteps
            self._ctx.save_checkpoint(
                lambda p: self.model.save(str(p.with_suffix(""))),
                step=step,
                name_prefix=self.name_prefix,
            )
            self._prune_old_checkpoints()
            self.n_calls += 1
            return True
        proceed = super()._on_step()
        self._prune_old_checkpoints()
        return proceed

    def _prune_old_checkpoints(self) -> None:
        """Keep only the ``keep_last`` most recent checkpoints (by step number).

        Deletes each pruned ``<prefix>_<step>_steps.zip`` together with its
        ``.model_metadata.json`` sidecar. No-op when ``keep_last`` is ``None``."""
        if not self._keep_last or self._keep_last <= 0:
            return
        ckpt_dir = Path(self.save_path)
        zips = list(ckpt_dir.glob(f"{self.name_prefix}_*_steps.zip"))
        if len(zips) <= self._keep_last:
            return

        def _step(path: Path) -> int:
            try:
                return int(path.stem.split("_")[-2])  # <prefix>_<step>_steps
            except (ValueError, IndexError):
                return -1

        for old in sorted(zips, key=_step)[: -self._keep_last]:
            old.unlink(missing_ok=True)
            old.with_suffix(".model_metadata.json").unlink(missing_ok=True)


class CtxEvalCallback(_EarlyStopMixin, EvalCallback):
    """SB3 EvalCallback that calls ctx.report_eval after each evaluation.

    `ctx.report_eval` handles MLflow logging and Optuna pruning. We also write
    the metadata sidecar for `best_model.zip` whenever it gets saved.

    During evaluation we flip ``ctx.metrics_state.use_eval_reward`` so the env
    returns ``ExperimentConfig.eval_reward(params)`` instead of the training
    reward. Without this, ``last_mean_reward`` (and therefore the Optuna
    pruning signal) would be in units of the *training* reward — making
    cross-trial comparison meaningless whenever the HPO search sweeps the
    reward function. The flag is restored to False after the eval block so
    subsequent training rollouts go back to the training reward.
    """

    def __init__(self, *args: Any, ctx, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = ctx
        self._offtrack_resets = 0
        self._completed = 0
        self._clean = 0
        self._eval_episodes: list[dict] = []
        self._es_init()

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        # Keep SB3's success-rate bookkeeping, then additionally tally eval
        # episodes that ended with the car off-track (a track-out reset) and, if
        # path plots are enabled, collect each episode's trajectory.
        super()._log_success_callback(locals_, globals_)
        if not locals_.get("done"):
            return
        info = locals_.get("info") or {}
        if not isinstance(info, dict):
            return
        summary = info.get("dr_episode")
        if summary:
            if summary.get("dr/ep_ended_offtrack", 0.0):
                self._offtrack_resets += 1
            if summary.get("dr/ep_completed", 0.0):
                self._completed += 1
            if summary.get("dr/ep_completed_clean", 0.0):
                self._clean += 1
        path = info.get("dr_episode_path")
        if path:
            self._eval_episodes.append(path)

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        state = getattr(self._ctx, "metrics_state", None)
        if is_eval_step and state is not None:
            state.use_eval_reward = True
        if is_eval_step:
            self._offtrack_resets = 0
            self._completed = 0
            self._clean = 0
            self._eval_episodes = []
        try:
            proceed = super()._on_step()
        finally:
            if state is not None:
                state.use_eval_reward = False
        if is_eval_step:
            # Track-out resets for this eval. Single-world path, so the per-track
            # series and the global series coincide; log both so the metric name
            # is consistent with the multi-world callback.
            self.logger.record("eval/offtrack_resets", float(self._offtrack_resets))
            n_eps = max(1, self.n_eval_episodes)
            self.logger.record("eval/completion_rate", self._completed / n_eps)
            self.logger.record("eval/clean_completion_rate", self._clean / n_eps)
            world = getattr(state, "world_name", None)
            if world:
                self.logger.record(
                    f"eval/{world}_offtrack_resets", float(self._offtrack_resets)
                )
                self.logger.record(
                    f"eval/{world}_completion_rate", self._completed / n_eps
                )
                self.logger.record(
                    f"eval/{world}_clean_completion_rate", self._clean / n_eps
                )
            if self._eval_episodes and world:
                _log_eval_paths(
                    self.logger, world, int(self.num_timesteps), self._eval_episodes
                )
            if self.best_model_save_path is not None:
                best_zip = Path(self.best_model_save_path) / "best_model.zip"
                if best_zip.exists():
                    from gym_dr.action_space import write_model_metadata

                    write_model_metadata(
                        best_zip.with_suffix(".model_metadata.json"),
                        self._ctx.action_space,
                    )
            self._ctx.report_eval(float(self.last_mean_reward), int(self.num_timesteps))
            # Track-mastery early stop: end the chunk once the car stays on the
            # track across this eval round (advances the rotation / ends the run).
            if self._apply_early_stop(self._offtrack_resets, self.n_eval_episodes):
                return False
        return proceed


class MultiWorldEvalCallback(_EarlyStopMixin, BaseCallback):
    """Evaluate across an ordered, held-out set of worlds (track generalisation).

    Used by ``Sb3Trainer`` when ``ctx.eval_worlds`` is set (e.g. the
    :class:`~gym_dr.worlds.OrderedSplit` strategy). At each eval trigger it:

    1. flips ``metrics_state.use_eval_reward`` so the eval reward is measured,
    2. swaps the (shared) Gazebo env to each world in ``eval_worlds`` via
       ``set_world``, running ``n_eval_episodes`` per world,
    3. restores the world training was on (read from ``metrics_state.world_name``,
       which ``Sb3Trainer.fit`` keeps current per chunk),
    4. reports the mean-across-worlds as the eval metric (per-world means are
       logged as ``eval/<world>_mean_reward``), and saves ``best_model`` on
       improvement.

    Exposes ``last_mean_reward`` so the trainer can read the final eval score
    the same way it does for ``CtxEvalCallback``.
    """

    def __init__(
        self,
        *,
        ctx,
        eval_worlds: list[str],
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self._ctx = ctx
        self.eval_worlds = list(eval_worlds)
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.last_mean_reward = float("nan")
        self.best_mean_reward = -float("inf")
        self._es_init()

    def _swap(self, vec, world: str) -> None:
        import numpy as np

        vec.env_method("set_world", world)
        self.model._last_obs = vec.reset()
        self.model._last_episode_starts = np.ones((vec.num_envs,), dtype=bool)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        import numpy as np
        from stable_baselines3.common.evaluation import evaluate_policy

        state = getattr(self._ctx, "metrics_state", None)
        train_world = getattr(state, "world_name", None)
        vec = self.model.get_env()

        capture_paths = bool(getattr(self._ctx.training, "eval_path_plots", False))
        per_world: dict[str, float] = {}
        counts_per_world: dict[str, dict] = {}
        if state is not None:
            state.use_eval_reward = True
        try:
            for world in self.eval_worlds:
                self._swap(vec, world)
                eval_cb, counts, episodes = _make_eval_collector(capture_paths)
                mean_reward, _ = evaluate_policy(
                    self.model,
                    vec,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    warn=False,
                    callback=eval_cb,
                )
                per_world[world] = float(mean_reward)
                counts_per_world[world] = counts
                if capture_paths:
                    _log_eval_paths(self.logger, world, int(self.num_timesteps), episodes)
            # Restore the world training was on so the next rollout continues
            # on the right track (the env is shared with training).
            if train_world is not None:
                self._swap(vec, train_world)
        finally:
            if state is not None:
                state.use_eval_reward = False

        agg = float(np.mean(list(per_world.values()))) if per_world else float("nan")
        self.last_mean_reward = agg
        for world, m in per_world.items():
            self.logger.record(f"eval/{world}_mean_reward", m)
        self.logger.record("eval/mean_reward", agg)
        self.logger.record("eval/n_worlds", float(len(per_world)))
        # Track-out resets + completion rates, per-track and aggregated. The
        # (clean-)completion rate is the success-criterion yardstick: the fraction
        # of held-out eval episodes that finished the lap (cleanly, with no
        # off-track step). See docs/eval-protocol.md.
        n_eps = max(1, self.n_eval_episodes)
        for world, c in counts_per_world.items():
            self.logger.record(f"eval/{world}_offtrack_resets", float(c["n"]))
            self.logger.record(f"eval/{world}_completion_rate", c["completed"] / n_eps)
            self.logger.record(f"eval/{world}_clean_completion_rate", c["clean"] / n_eps)
        self.logger.record(
            "eval/offtrack_resets", float(sum(c["n"] for c in counts_per_world.values()))
        )
        if counts_per_world:
            self.logger.record(
                "eval/completion_rate",
                float(np.mean([c["completed"] / n_eps for c in counts_per_world.values()])),
            )
            self.logger.record(
                "eval/clean_completion_rate",
                float(np.mean([c["clean"] / n_eps for c in counts_per_world.values()])),
            )

        if per_world and agg > self.best_mean_reward:
            self.best_mean_reward = agg
            self._save_best()

        self._ctx.report_eval(agg, int(self.num_timesteps))
        # Track-mastery early stop, measured across all eval worlds this round.
        total_eps = self.n_eval_episodes * max(1, len(self.eval_worlds))
        if self._apply_early_stop(
            sum(c["n"] for c in counts_per_world.values()), total_eps
        ):
            return False
        return True

    def _save_best(self) -> None:
        if not self.best_model_save_path:
            return
        from gym_dr.action_space import write_model_metadata

        best_dir = Path(self.best_model_save_path)
        best_dir.mkdir(parents=True, exist_ok=True)
        best_zip = best_dir / "best_model.zip"
        self.model.save(str(best_zip.with_suffix("")))
        write_model_metadata(
            best_zip.with_suffix(".model_metadata.json"), self._ctx.action_space
        )


class StatusJsonCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        started_at: float,
        update_interval_steps: int,
        update_interval_seconds: int,
        max_train_seconds: int | None,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.started_at = started_at
        self.update_interval_steps = max(1, update_interval_steps)
        self.update_interval_seconds = max(1, update_interval_seconds)
        self.max_train_seconds = max_train_seconds
        self._last_report_step = 0
        self._last_report_time = started_at

    def _on_step(self) -> bool:
        now = time.monotonic()
        if (
            self.num_timesteps - self._last_report_step < self.update_interval_steps
            and now - self._last_report_time < self.update_interval_seconds
        ):
            return True
        elapsed = int(now - self.started_at)
        extra: dict[str, Any] = {
            "timesteps_completed": self.num_timesteps,
            "elapsed_seconds": elapsed,
        }
        if self.max_train_seconds is not None:
            extra["time_limit_seconds"] = self.max_train_seconds
            extra["time_remaining_seconds"] = max(0, self.max_train_seconds - elapsed)
        update_training_status(self.run_dir, "running", extra)
        self._last_report_step = self.num_timesteps
        self._last_report_time = now
        return True


class WallClockLimitCallback(BaseCallback):
    def __init__(self, run_dir: Path, started_at: float, max_train_seconds: int) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.started_at = started_at
        self.max_train_seconds = max_train_seconds
        self.time_limit_reached = False

    def _on_step(self) -> bool:
        elapsed = int(time.monotonic() - self.started_at)
        if elapsed < self.max_train_seconds:
            return True
        self.time_limit_reached = True
        update_training_status(
            self.run_dir,
            "time_limit_reached",
            {
                "timesteps_completed": self.num_timesteps,
                "elapsed_seconds": elapsed,
                "time_limit_seconds": self.max_train_seconds,
            },
        )
        print(
            f"Wall-clock training limit reached after {elapsed}s at {self.num_timesteps} timesteps",
            flush=True,
        )
        return False


class MlflowMirrorCallback(BaseCallback):
    """Mirror SB3 logger scalars to the active MLflow run on each rollout end."""

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        try:
            import mlflow
        except ImportError:
            return
        if mlflow.active_run() is None:
            return
        step = int(self.num_timesteps)
        for key, value in self.logger.name_to_value.items():
            try:
                mlflow.log_metric(key, float(value), step=step)
            except (TypeError, ValueError):
                continue


class RewardMetricsCallback(BaseCallback):
    """Drain per-episode DeepRacer metrics into the SB3 logger.

    The orchestrator wraps the env so every terminal step's ``info`` dict
    carries a ``dr_episode`` summary (see ``gym_dr/metrics.py``). This
    callback inspects ``self.locals["infos"]`` on each step, picks up any
    finalized summaries, and pushes their keys/values into the logger via
    ``record_mean`` — they then surface in TensorBoard scalars and (via
    ``MlflowMirrorCallback``) MLflow metrics, averaged per rollout.
    """

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            summary = info.get("dr_episode") if isinstance(info, dict) else None
            if not summary:
                continue
            for key, value in summary.items():
                try:
                    self.logger.record_mean(key, float(value))
                except (TypeError, ValueError):
                    continue
        return True
