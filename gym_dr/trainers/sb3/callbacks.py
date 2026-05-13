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
    """

    def __init__(self, *args: Any, ctx, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = ctx

    def _on_step(self) -> bool:
        proceed = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
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
                mlflow.log_metric(key.replace("/", "_"), float(value), step=step)
            except (TypeError, ValueError):
                continue
