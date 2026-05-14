from __future__ import annotations

import os
from typing import Any, Callable

from gym_dr.config import ExperimentConfig
from gym_dr.trainer import run_training


def _optuna():
    import optuna

    return optuna


# Pruner leniency. Trials need a real chance to learn before they're
# eligible for pruning — DeepRacer policies climb slowly. n_warmup_steps is
# expressed as a *fraction of the per-trial timestep budget*: no trial is
# pruned until it's at least PRUNE_WARMUP_FRAC of the way through training.
PRUNE_WARMUP_FRAC = 0.5
PRUNE_STARTUP_TRIALS = 8  # no pruning at all until this many trials finish


def make_study(
    study_name: str,
    storage: str,
    total_timesteps: int,
    seed: int | None = None,
):
    """Open (or join) a SQLite-backed Optuna study.

    The ``MedianPruner`` is deliberately lenient: ``n_warmup_steps`` is
    ``PRUNE_WARMUP_FRAC`` of ``total_timesteps`` (the per-trial budget), so a
    trial trains at least halfway before it can be killed, and
    ``n_startup_trials`` requires several full trials before pruning starts
    at all. DeepRacer reward curves are slow to separate — pruning early
    throws away trials that would have caught up.

    ``seed`` seeds the TPE sampler so two workers given the same seed
    explore in lockstep — useful for reproducibility but undesirable for
    parallel search. The orchestrator offsets it by ``WORKER_INDEX``.
    """
    optuna = _optuna()
    warmup_steps = max(1, int(total_timesteps * PRUNE_WARMUP_FRAC))
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNE_STARTUP_TRIALS,
            n_warmup_steps=warmup_steps,
        ),
    )


def build_objective(
    base_cfg: ExperimentConfig,
    search_space: Callable[[Any], dict[str, Any]],
) -> Callable[[Any], float]:
    def objective(trial) -> float:
        overrides = search_space(trial)
        run_name = f"{base_cfg.name}_trial_{trial.number}"
        trial_cfg = base_cfg.with_overrides(name=run_name, **overrides)
        # Tag the Optuna trial with the shared run name so it appears in
        # optuna-dashboard alongside the same identifier used by TB
        # (artifact subdir) and MLflow (run_name).
        trial.set_user_attr("run_name", run_name)
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
    study = make_study(
        study_name, storage, base_cfg.training.total_timesteps, seed=sampler_seed
    )
    objective = build_objective(base_cfg, search_space)
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))


def study_storage_default() -> str:
    return os.getenv("STUDY_STORAGE", "sqlite:////workspace/optuna.db")
