# DeepRacer SB3 Training Environment

A Python-first, pluggable RL training pipeline for `seresheim/deepracer-env` inside Docker. The user owns a single `app.py` that wires together an env, a trainer, a reward, and an action space. Everything else (artifact layout, MLflow tracking, per-checkpoint DeepRacer metadata sidecars, Optuna HPO across parallel Docker workers, TensorBoard) is provided by `gym_dr` and works for any user-supplied trainer/env that implements the interfaces.

```text
app.py        ←  the user edits this
   │
   └──► gym_dr.train(experiment)
            │
            ├──► env_factory(experiment)        ←  pluggable env (deepracer_env_v1, _v2, …, or your own)
            ├──► trainer.fit(env, ctx)          ←  pluggable trainer (Sb3Trainer, or anything you write)
            └──► MLflow + artifact layout + DeepRacer metadata sidecars (provided)
```

## Plug-in points

| What | Interface | Default | Where |
|---|---|---|---|
| Env | `Callable[[ExperimentConfig], gym.Env]` | `deepracer_env_v1` | `gym_dr/envs/` |
| Trainer | `Trainer` protocol: `fit(env, ctx) -> TrainResult` | `Sb3Trainer` (SB3 PPO/SAC/TD3/A2C/DDPG) | `gym_dr/trainers/` |
| Reward | `RewardFactory: (params) -> reward_fn(params) -> float` | `center_line` | `gym_dr/reward.py` |
| Action space | `ContinuousActionSpaceConfig` \| `DiscreteActionSpaceConfig` | continuous | `gym_dr/action_space.py` |
| Tracker | hard-wired to MLflow (file store under `./mlruns/`) | — | `gym_dr/mlflow_utils.py` |

## Prerequisites

- Docker (daemon running, `buildx` available)
- `git`
- [`uv`](https://github.com/astral-sh/uv)
- ~50 GB free in Docker's storage location for the first build

## First-time setup

```bash
cd /mnt/hd/Repos/gym-dr
./bootstrap.sh        # builds upstream simulator image + project image
uv sync               # host-side Python deps (launcher / HPO orchestrator)
```

Re-run `./bootstrap.sh` only after editing `pyproject.toml`. `./bootstrap.sh -h` for `-a gpu` etc.

## Running a training: `app.py`

The user edits `app.py`. Full file (this is the default `app.py` shipped with the repo):

```python
from gym_dr import (
    ContinuousActionSpaceConfig, ExperimentConfig, RewardConfig,
    Sb3Trainer, TrackingConfig, TrainingConfig, deepracer_env_v1, train,
)

experiment = ExperimentConfig(
    name="quick_test",
    world_name="reinvent_base",
    env_factory=deepracer_env_v1,
    trainer=Sb3Trainer(
        name="ppo", policy="MultiInputPolicy",
        kwargs={"n_steps": 256, "batch_size": 64, "learning_rate": 3e-4, "ent_coef": 0.01},
    ),
    reward=RewardConfig(factory="center_line", params={}),
    action_space=ContinuousActionSpaceConfig(),
    training=TrainingConfig(total_timesteps=5_000, eval_freq=2_500),
    tracking=TrackingConfig(),
)

if __name__ == "__main__":
    train(experiment)
```

Launch (Docker is wrapped by the helper script):

```bash
./run_cpu_training.sh             # uses ./app.py
./run_cpu_training.sh other_app.py
```

## Swapping the env

Add a new factory and point `env_factory` at it:

```python
# gym_dr/envs/deepracer.py  (or your own module)
def deepracer_env_v2(experiment):
    from deepracer_env_v2 import DeepRacerEnv as V2
    from gym_dr.reward import make_reward
    return V2(reward_fn=make_reward(experiment.reward), world=experiment.world_name)
```

```python
# app.py
from gym_dr.envs import deepracer_env_v2
experiment = ExperimentConfig(env_factory=deepracer_env_v2, ...)
```

The factory is any callable `(ExperimentConfig) -> gym.Env`.

## Swapping the trainer

`Sb3Trainer` is the default. To use a different RL library or a hand-rolled loop, implement `Trainer`:

```python
from gym_dr.trainers.base import Trainer, TrainingContext, TrainResult

class MyTrainer:
    """Anything with this method shape satisfies gym_dr.Trainer."""
    def __init__(self, lr: float = 1e-3):
        self.lr = lr

    def fit(self, env, ctx: TrainingContext) -> TrainResult:
        # ... your training loop ...
        for step in range(ctx.training.total_timesteps):
            ...
            if step % ctx.training.eval_freq == 0:
                ctx.report_eval(mean_reward, step=step)            # MLflow log + Optuna prune
            if step % ctx.training.checkpoint_freq == 0:
                ctx.save_checkpoint(self._save, step=step)         # zip + metadata sidecar
        return TrainResult(final_eval_reward=mean_reward)
```

```python
# app.py
experiment = ExperimentConfig(trainer=MyTrainer(lr=5e-4), ...)
```

The `TrainingContext` (see `gym_dr/trainers/base.py`) gives the trainer the four hooks it needs:

- `ctx.save_model(save_fn, name=...)` — writes `<run_dir>/<name>.zip` + `<name>.model_metadata.json`
- `ctx.save_checkpoint(save_fn, step=...)` — writes `checkpoints/<prefix>_<step>_steps.zip` + metadata sidecar
- `ctx.log_metric(name, value, step)` — mirrors to the active MLflow run
- `ctx.report_eval(mean_reward, step)` — `log_metric("eval_mean_reward", ...)` + Optuna `trial.report` / pruning

The orchestrator (`gym_dr/trainer.py:run_training`) handles everything around `fit`: run-dir setup, `model_metadata.json` generation, MLflow `start_run`, artifact upload at end, and `training_status.json` lifecycle.

## HPO

Same script-per-experiment pattern. Defines a `base` config + a `search_space(trial)` function + `study(...)` at the bottom. The same script runs unchanged inside each worker container — `study()` detects worker mode via `GYM_DR_WORKER`.

```bash
uv run python experiments/hpo_example.py
```

Details: [docs/hpo.md](docs/hpo.md).

## Inspect without running

```bash
uv run python -m gym_dr.cli inspect app.py
```

## Iterating without rebuilding

Project source is bind-mounted at `/workspace` in the container. Edit and re-run — no `docker build`:

- Hyperparameters / training control / algorithm choice → edit `app.py`
- Reward shaping → add a factory in `gym_dr/reward.py` and reference it from `app.py`
- Action space (continuous bounds or discrete action list) → edit `action_space` in `app.py`
- Custom trainer → drop a file anywhere, import it in `app.py`
- Custom env → drop a factory in `gym_dr/envs/` or your own module

Rebuild only when `pyproject.toml` changes.

## Internal layout

```text
gym_dr/
├── app.py             # train(), study(), inspect() — the user-facing call surface
├── trainer.py         # orchestrator: run_training() — wraps any Trainer
├── config.py          # ExperimentConfig and its sub-configs
├── action_space.py    # ContinuousActionSpaceConfig, DiscreteActionSpaceConfig
├── reward.py          # @register("...") factory registry
├── hpo.py             # Optuna study + objective + worker loop
├── docker_runner.py   # host-side parallel container spawner
├── mlflow_utils.py    # MLflow run / nested run helpers
├── cli.py             # internal: prepare-metadata + inspect (used by run_cpu_training.sh)
├── envs/
│   ├── __init__.py
│   └── deepracer.py   # deepracer_env_v1 (add _v2, _v3, … here)
└── trainers/
    ├── base.py        # Trainer protocol, TrainingContext, TrainResult
    └── sb3/           # default Sb3Trainer
        ├── __init__.py
        ├── algorithms.py
        └── callbacks.py
```

## Documentation

| Topic | File |
|---|---|
| `ExperimentConfig` anatomy, reward factories | [docs/configuration.md](docs/configuration.md) |
| Where artifacts go; per-checkpoint metadata guarantees | [docs/artifact-layout.md](docs/artifact-layout.md) |
| MLflow params/metrics/artifacts; TensorBoard | [docs/tracking.md](docs/tracking.md) |
| Optuna HPO + parallel Docker workers | [docs/hpo.md](docs/hpo.md) |
| Algorithm registry + off-policy caveats | [docs/algorithms.md](docs/algorithms.md) |
| TensorBoard launcher details | [docs/tensorboard.md](docs/tensorboard.md) |
| Physical-car export caveats | [docs/physical-car-integration-notes.md](docs/physical-car-integration-notes.md) |

## Resume training

Set `training.resume_from` in `app.py` to the **container** path of a previous checkpoint:

```python
training=TrainingConfig(
    resume_from="/workspace/artifacts/default_4h/latest_model.zip",
    total_timesteps=1_000_000_000,
    max_train_seconds=14_400,
)
```

`latest_model.zip` is the safest resume target — it is saved in the `finally` block even when training stops early. The sidecar `latest_model.model_metadata.json` travels with it.

## Detached long run with tmux

```bash
tmux new-session -s deepracer_train
./run_cpu_training.sh
# Detach: Ctrl-b d
# Reattach: tmux attach -t deepracer_train
```

## Check training status

```bash
cat artifacts/<run_name>/training_status.json
find artifacts/<run_name>/checkpoints -maxdepth 1 -type f | sort
cat artifacts/<run_name>/run_config.json | jq .
```
