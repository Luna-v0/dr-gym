from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from gym_dr.config import ExperimentConfig
from gym_dr.trainer import run_training


def _optuna():
    import optuna

    return optuna


def make_study(study_name: str, storage: str, eval_freq: int):
    optuna = _optuna()
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True),
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
    study = make_study(study_name, storage, base_cfg.training.eval_freq)
    objective = build_objective(base_cfg, search_space)
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))


def study_storage_default() -> str:
    return os.getenv("STUDY_STORAGE", "sqlite:////workspace/optuna.db")
