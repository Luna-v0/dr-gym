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


def _apply_global_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch globally.

    SB3 also calls ``set_random_seed(seed)`` internally when you pass
    ``seed=`` to the algorithm constructor — that re-seeds the same three
    RNGs at PPO build time. We do it here too so any torch-using code that
    runs *before* PPO is constructed (env wrappers with learned components,
    user-defined feature extractors, etc.) sees a deterministic RNG.
    Optuna's sampler is seeded separately via ``make_study(..., seed=...)``.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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

    if experiment.seed is not None:
        _apply_global_seed(experiment.seed)

    paths = _build_run_paths(experiment)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    run_dir = paths["run_dir"]
    export_dir = paths["export_dir"]

    write_model_metadata(run_dir / "model_metadata.json", experiment.action_space)
    write_model_metadata(export_dir / "model_metadata.json", experiment.action_space)

    # Archive the user's reward source BEFORE wrapping (the wrapped fn's
    # getsource would return the metric-recorder boilerplate, not the user's
    # code).
    reward_src = _render_reward_source(experiment.reward)
    (run_dir / "reward_function.py").write_text(reward_src, encoding="utf-8")
    (export_dir / "reward_function.py").write_text(reward_src, encoding="utf-8")

    # Wrap the reward with a metrics recorder and produce an env wrapper that
    # finalizes per-episode metrics into info['dr_episode']. See gym_dr/metrics.py.
    from gym_dr.metrics import install_metrics

    experiment, env_wrapper, metrics_state = install_metrics(experiment)

    _write_json(run_dir / "run_config.json", experiment.to_dict())
    _update_status(run_dir, "initialized")

    env = env_wrapper(experiment.env_factory(experiment))

    # Runtime multi-world rotation. When the host orchestrator runs a single
    # container that rotates worlds in-process (GYM_DR_ROTATE=1), expand the
    # WorldsConfig into a per-chunk plan the trainer walks via the env's
    # set_world() track swap. Otherwise (HPO workers, direct single-world
    # runs, the test stub) world_plan stays None and the trainer takes the
    # legacy single-``model.learn`` path.
    world_plan: list[str] | None = None
    chunk_steps: int | None = None
    if os.getenv("GYM_DR_ROTATE"):
        worlds = experiment.worlds
        world_plan = [w for _ in range(worlds.rotations) for w in worlds.names]
        chunk_steps = worlds.chunk_steps

    ctx = TrainingContext(
        run_dir=run_dir,
        action_space=experiment.action_space,
        training=experiment.training,
        trial=trial,
        seed=experiment.seed,
        metrics_state=metrics_state,
        world_plan=world_plan,
        chunk_steps=chunk_steps,
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
            # Optuna's MedianPruner raises TrialPruned to abort
            # underperforming trials early — that's the pruner working as
            # designed, not a real failure. Tag it accordingly so it doesn't
            # blend in with crashes in the artifact dir.
            status = "failed"
            try:
                import optuna

                if isinstance(exc, optuna.TrialPruned):
                    status = "pruned"
            except ImportError:
                pass
            _update_status(run_dir, status, {"reason": repr(exc)})
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
