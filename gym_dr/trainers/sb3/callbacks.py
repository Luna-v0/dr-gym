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


class CtxCheckpointCallback(CheckpointCallback):
    """SB3 CheckpointCallback that routes saves through TrainingContext.

    Ensures every periodic checkpoint zip gets its `model_metadata.json`
    sibling, matching what `ctx.save_checkpoint` writes.
    """

    def __init__(self, *args: Any, ctx, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = ctx

    def _on_step(self) -> bool:
        if self.save_freq > 0 and (self.n_calls + 1) % self.save_freq == 0:
            step = int(self.num_timesteps) + 1  # match super's post-save num_timesteps
            self._ctx.save_checkpoint(
                lambda p: self.model.save(str(p.with_suffix(""))),
                step=step,
                name_prefix=self.name_prefix,
            )
            self.n_calls += 1
            return True
        return super()._on_step()


class CtxEvalCallback(EvalCallback):
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

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        state = getattr(self._ctx, "metrics_state", None)
        if is_eval_step and state is not None:
            state.use_eval_reward = True
        try:
            proceed = super()._on_step()
        finally:
            if state is not None:
                state.use_eval_reward = False
        if is_eval_step:
            if self.best_model_save_path is not None:
                best_zip = Path(self.best_model_save_path) / "best_model.zip"
                if best_zip.exists():
                    from gym_dr.action_space import write_model_metadata

                    write_model_metadata(
                        best_zip.with_suffix(".model_metadata.json"),
                        self._ctx.action_space,
                    )
            self._ctx.report_eval(float(self.last_mean_reward), int(self.num_timesteps))
        return proceed


class MultiWorldEvalCallback(BaseCallback):
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

        per_world: dict[str, float] = {}
        if state is not None:
            state.use_eval_reward = True
        try:
            for world in self.eval_worlds:
                self._swap(vec, world)
                mean_reward, _ = evaluate_policy(
                    self.model,
                    vec,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic,
                    warn=False,
                )
                per_world[world] = float(mean_reward)
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

        if per_world and agg > self.best_mean_reward:
            self.best_mean_reward = agg
            self._save_best()

        self._ctx.report_eval(agg, int(self.num_timesteps))
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
