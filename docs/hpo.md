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

def make_reward(weight_close=100.0):
    def reward(p):
        return weight_close if p["distance_from_center"] <= 0.1 * p["track_width"] else 1e-3
    return reward

def search_space(trial) -> dict:
    return {
        "trainer.kwargs.learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "reward": make_reward(weight_close=trial.suggest_float("weight_close", 10.0, 200.0)),
        # ... flat dotted keys consumed by cfg.with_overrides(...)
    }

if __name__ == "__main__":
    study(base, search_space, study_name="hpo", n_trials=40, n_parallel=4)
```

The returned dict is applied via `cfg.with_overrides(**overrides)`. Dotted keys walk the dataclass tree; dict-typed fields (`trainer.kwargs`) accept the leaf. Whole fields (like `reward`) can be replaced by passing a non-dotted key.

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

- **Action-space and world are fixed per study.** The simapp loads the world at container start; HPO uses `base.worlds.names[0]`. To sweep across worlds, launch separate studies.
- **Network arch / policy class are out of the v1 search space.** Algorithm choice is fixed per study (it's a field on the base config, not a sampled param).
- **Trial artifacts** land under `artifacts/<name>_trial_<n>/` with the full standard layout — every checkpoint still gets its metadata sibling.
- If two workers grab `n_trials=N` each and only ~M trials are still pending in the study, Optuna's coordination via SQLite handles the duplicate work cleanly (the second worker just sees the study is full and stops).

## Oracle HPO (`experiments/oracle_hpo.py`) — 2026-06-28

Tunes the asymmetric-critic feature oracle: `learning_rate, ent_coef, n_steps, batch_size, gamma, gae_lambda, clip_range, n_epochs, target_kl`, the **observation-memory depth** `frame_stack ∈ {1,2,4,8}`, network width `∈ {64,128,256}`, and the `feature_noise` DR ceiling. Every key is config-driven (`trainer.*`, `environment.domain_randomization.*`), so the winning trial transplants directly into the multi-car production run `experiments/oracle_asym_multicar.py`.

- **Single-car base on purpose.** The multi-car oracle can't `set_world`, so it has no in-loop held-out objective for Optuna. The single-car asym oracle rotates the held-out worlds each chunk via the ACL curriculum, so each trial returns a real **held-out clean-completion** score (the maximised objective). The searched knobs transfer to the multi-car run unchanged.
- **`frame_stack` is in the search** because it's the lever that lets the policy infer an unobservable per-episode actuator bias; a mild ±5° bias is kept in the base so the search can see that benefit (the production run uses ±15°).
- **Short trials** (`GYM_DR_HPO_CHUNK`×`GYM_DR_HPO_NCHUNKS`, default 240k steps) rank configs by early held-out learning; MedianPruner kills laggards after the 50%-budget warmup. Env knobs: `GYM_DR_HPO_TRIALS` (40), `GYM_DR_HPO_PARALLEL` (2). Single-car trials are multi-hour — run on a many-core box with `n_parallel>1`.
