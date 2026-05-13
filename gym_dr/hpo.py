from __future__ import annotations

import os
from typing import Any, Callable

from gym_dr.config import ExperimentConfig
from gym_dr.trainer import run_training


def _optuna():
    import optuna

    return optuna


def make_study(study_name: str, storage: str, eval_freq: int, seed: int | None = None):
    """Open (or join) a SQLite-backed Optuna study.

    ``seed`` seeds the TPE sampler so two workers given the same seed
    explore in lockstep — useful for reproducibility but undesirable for
    parallel search. The orchestrator currently passes the same seed to all
    workers; if that bites, set ``seed=None`` per worker (or offset by
    ``WORKER_INDEX``).
    """
    optuna = _optuna()
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=max(1, eval_freq) * 3,
        ),
    )


def build_objective(
    base_cfg: ExperimentConfig,
    search_space: Callable[[Any], dict[str, Any]],
) -> Callable[[Any], float]:
    def objective(trial) -> float:
        overrides = search_space(trial)
        trial_cfg = base_cfg.with_overrides(
            name=f"{base_cfg.name}_trial_{trial.number}",
            **overrides,
        )
        return run_training(trial_cfg, trial=trial)

    return objective


def run_worker(
    base_cfg: ExperimentConfig,
    search_space: Callable[[Any], dict[str, Any]],
    study_name: str,
    storage: str,
    n_trials: int,
) -> None:
    # Per-worker seed offset so concurrent workers don't TPE-sample in
    # lockstep on the same base seed.
    base_seed = base_cfg.seed
    worker_idx = int(os.getenv("WORKER_INDEX", "0"))
    sampler_seed = None if base_seed is None else int(base_seed) + worker_idx
    study = make_study(study_name, storage, base_cfg.training.eval_freq, seed=sampler_seed)
    objective = build_objective(base_cfg, search_space)
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))


def study_storage_default() -> str:
    return os.getenv("STUDY_STORAGE", "sqlite:////workspace/optuna.db")
