"""Orchestrator that wraps any `Trainer` implementation.

`run_training(experiment)`:

1. Builds the per-run artifact dir.
2. Generates `model_metadata.json` (DeepRacer schema) from the action space.
3. Renders the chosen reward factory's source for archival.
4. Builds the env via `experiment.env_factory(experiment)`.
5. Opens an MLflow run (nested if `MLFLOW_PARENT_RUN_ID` env is set).
6. Calls `experiment.trainer.fit(env, ctx)` where `ctx` is a `TrainingContext`
   that exposes save / metric / Optuna hooks the trainer can call.
7. Logs the full run dir as MLflow artifacts.

The trainer can be SB3 (default `Sb3Trainer`) or any user-supplied object that
implements the `Trainer` protocol from `gym_dr.trainers.base`.
"""
from __future__ import annotations

import inspect as py_inspect
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gym_dr.action_space import write_model_metadata
from gym_dr.config import ExperimentConfig
from gym_dr.mlflow_utils import log_run_artifacts, start_run
from gym_dr.trainers.base import TrainingContext, TrainResult


def _render_reward_source(reward_fn: Callable[[dict], float]) -> str:
    """Best-effort dump of the reward function source for archival.

    Falls back to ``repr(fn)`` for closures/lambdas whose source can't be
    retrieved (e.g. if the function was defined in a Jupyter cell).
    """
    name = getattr(reward_fn, "__qualname__", repr(reward_fn))
    module = getattr(reward_fn, "__module__", "?")
    header = f"# Reward function: {module}.{name}\n\n"
    try:
        return header + py_inspect.getsource(reward_fn)
    except (OSError, TypeError):
        return header + f"# source unavailable; repr = {reward_fn!r}\n"


def _project_root() -> Path:
    return Path(os.getenv("PROJECT_ROOT", "/workspace"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_run_paths(experiment: ExperimentConfig) -> dict[str, Path]:
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", str(_project_root() / "artifacts")))
    run_name = experiment.name or f"deepracer_{_timestamp()}"
    run_dir = artifacts_dir / run_name
    return {
        "artifacts_dir": artifacts_dir,
        "run_dir": run_dir,
        "export_dir": run_dir / "export_bundle",
    }


def _install_signal_handlers() -> None:
    def _raise_interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _update_status(run_dir: Path, status: str, extra: dict[str, Any] | None = None) -> None:
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


def run_training(experiment: ExperimentConfig, trial: Any | None = None) -> float:
    _install_signal_handlers()

    paths = _build_run_paths(experiment)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    run_dir = paths["run_dir"]
    export_dir = paths["export_dir"]

    write_model_metadata(run_dir / "model_metadata.json", experiment.action_space)
    write_model_metadata(export_dir / "model_metadata.json", experiment.action_space)
    reward_src = _render_reward_source(experiment.reward)
    (run_dir / "reward_function.py").write_text(reward_src, encoding="utf-8")
    (export_dir / "reward_function.py").write_text(reward_src, encoding="utf-8")
    _write_json(run_dir / "run_config.json", experiment.to_dict())
    _update_status(run_dir, "initialized")

    env = experiment.env_factory(experiment)
    ctx = TrainingContext(
        run_dir=run_dir,
        action_space=experiment.action_space,
        training=experiment.training,
        trial=trial,
    )

    started_at = time.monotonic()
    final_eval_reward = float("nan")

    with start_run(experiment):
        try:
            result: TrainResult = experiment.trainer.fit(env, ctx)
            final_eval_reward = float(result.final_eval_reward)
            elapsed = result.extra.get("elapsed_seconds", int(time.monotonic() - started_at))
            timesteps = result.extra.get("timesteps_completed")
            time_limit_reached = result.extra.get("time_limit_reached", False)
            _update_status(
                run_dir,
                "time_limit_reached" if time_limit_reached else "completed",
                {
                    "timesteps_completed": timesteps,
                    "elapsed_seconds": elapsed,
                    "time_limit_seconds": experiment.training.max_train_seconds,
                    "final_eval_reward": final_eval_reward,
                },
            )
        except KeyboardInterrupt as exc:
            _update_status(run_dir, "interrupted", {"reason": str(exc)})
            raise
        except Exception as exc:
            _update_status(run_dir, "failed", {"reason": repr(exc)})
            raise
        finally:
            try:
                close = getattr(env, "close", None)
                if close is not None:
                    close()
            except Exception:
                pass
            log_run_artifacts(run_dir)

    return final_eval_reward
