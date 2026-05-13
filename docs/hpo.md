# Hyperparameter optimization (HPO)

`gym_dr/hpo.py` and `gym_dr/docker_runner.py` run an Optuna study across many short trainings in parallel Docker workers. All workers share one Optuna study (SQLite-backed) and one MLflow tree.

```bash
uv run python experiments/hpo_example.py
```

## Script layout

The experiment script declares a base config, a search-space sampling function, and a `study(...)` call:

```python
# experiments/hpo_example.py
from gym_dr import ExperimentConfig, study, ...

base = ExperimentConfig(...)        # everything not swept

def search_space(trial) -> dict:
    return {
        "algorithm.kwargs.learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "reward.params.reward_center":    trial.suggest_float("reward_center", 10.0, 200.0),
        # ... flat dotted keys consumed by cfg.with_overrides(...)
    }

if __name__ == "__main__":
    study(base, search_space, study_name="hpo", n_trials=40, n_parallel=4)
```

The returned dict is applied via `cfg.with_overrides(**overrides)`. Dotted keys walk the dataclass tree; dict-typed fields (`algorithm.kwargs`, `reward.params`) accept the leaf.

## Orchestration

`gym_dr.app.study(...)` is the host orchestrator and the worker entrypoint — the same call, mode-detected via the `GYM_DR_WORKER` env var.

Host invocation (no `GYM_DR_WORKER`):

1. Pre-generates `model_metadata.json` on the host from `base.action_space`.
2. Opens a parent MLflow run named `hpo:<study_name>`.
3. Spawns N=`n_parallel` worker containers via `gym_dr/docker_runner.py`. Each container shares bind mounts on `mlruns/` and `optuna.db` and gets `GYM_DR_WORKER=1`, `EXPERIMENT_PATH=/workspace/experiments/<name>.py`, `STUDY_STORAGE`, `MLFLOW_PARENT_RUN_ID`, `N_TRIALS_PER_WORKER`.
4. Waits on all workers; surfaces non-zero exit codes.
5. SIGINT/SIGTERM → `docker kill` the workers.

Worker invocation (`GYM_DR_WORKER=1`, set by the host):

1. The container's CMD runs `python "$EXPERIMENT_PATH"` — the same experiment script.
2. Its `__main__` hits `study(...)`, which sees `GYM_DR_WORKER=1` and switches to `run_worker(base, search_space, study_name, storage, n_trials_per_worker)`.
3. Worker calls `study.optimize(objective, n_trials=N)`. Objective = `cfg.with_overrides(**search_space(trial))` → `run_training(cfg, trial=trial)`.
4. Trial trainings call `trial.report(eval_mean_reward, step)` from the eval callback and raise `optuna.TrialPruned` when `trial.should_prune()` says so. Pruner is `MedianPruner(n_startup_trials=5, n_warmup_steps=eval_freq*3)`.

## Live monitoring

```bash
optuna-dashboard sqlite:///$PWD/optuna.db
```

## Caveats

- **Action-space and `world_name` are fixed per study.** The simapp loads them at container start. To vary either, launch separate studies.
- **Network arch / policy class are out of the v1 search space.** Algorithm choice is fixed per study (it's a field on the base config, not a sampled param).
- **Trial artifacts** land under `artifacts/<name>_trial_<n>/` with the full standard layout — every checkpoint still gets its metadata sibling.
- If two workers grab `n_trials=N` each and only ~M trials are still pending in the study, Optuna's coordination via SQLite handles the duplicate work cleanly (the second worker just sees the study is full and stops).
