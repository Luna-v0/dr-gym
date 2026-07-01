# The Study / Pipeline interface

This page is the entry point for **running an experiment** in dr-gym after the
Task-1/8 refactor (see [ADR-0004](decisions/0004-orchestrator-refactor-interface.md)).
Everything you run goes through one object — the **`Study`** — and every algorithm
plugs in behind one small abstract class — the **`Trainer`**. Nothing is locked to
Stable-Baselines3.

## One interface: `Study`

A `Study` is *always* defined over a set of hyperparameters. A plain training run is
simply a study whose hyperparameters are all **fixed**; add a search distribution and
the same call becomes a hyperparameter search. There is no separate `train()` vs
`study()` API.

```python
from gym_dr import Study, ExperimentConfig, EnvironmentConfig, FeatureObs, Float, OrderedSplit
from gym_dr import Sb3Trainer, TrainingConfig
from gym_dr.early_stopping import CleanCompletion

experiment = ExperimentConfig(
    name="oracle_feature",
    environment=EnvironmentConfig(
        observation=FeatureObs(),                    # or CameraObs()
        curriculum=OrderedSplit(train_worlds=[...], eval_worlds=[...]),
        n_cars=6,
    ),
    trainer=Sb3Trainer(name="ppo", device="cpu"),
    training=TrainingConfig(
        total_timesteps=3_000_000,
        early_stop=CleanCompletion(min_rate=1.0, patience=2),
    ),
)

# a single training run — every hyperparameter fixed
Study(experiment, master_seed=42).run()

# a hyperparameter search — add a distribution and n_trials (SAME interface)
Study(
    experiment,
    params={"trainer.kwargs.learning_rate": Float(1e-5, 1e-3, log=True)},
    master_seed=42, n_trials=40, n_replicates=3, n_parallel=4,
).run()
```

`Study(...).run()` dispatches exactly like the old entrypoints — it launches the
Docker container(s), runs the runtime world-rotation, and (for HPO) the Optuna
workers — so you don't manage any of that yourself.

### Reproducibility from one number

Every stochastic stream derives from `master_seed` via
[`SeedManager`](trainer-contract.md): each training replicate `k` is seeded from
`replicate(k).agent` (with `replicate(k).domain` reserved for environment
randomization), and the Optuna sampler is seeded from the same root. `master_seed`
is the single source of truth — record it and the study is reproducible.

### Hyperparameters — `params`

`params` maps dotted `ExperimentConfig` keys to values or search distributions from
`gym_dr.search`:

| Kind | Example | Meaning |
|---|---|---|
| constant | `"trainer.kwargs.gamma": 0.99` | fixed (coerced to `Fixed`) |
| `Float` | `Float(1e-5, 1e-3, log=True)` | continuous search dim |
| `Int` | `Int(1, 4)` | integer search dim |
| `Categorical` | `Categorical(["ppo", "sac"])` | categorical search dim |

A space that is **all constants** ⇒ a single run; **any** distribution ⇒ HPO over
`n_trials`. A legacy imperative `search_space(trial) -> dict` callable is also
accepted for `params`, to ease migrating existing HPO experiments.

## Bring your own algorithm: the `Trainer` ABC

`Trainer` is the abstract class you extend to plug in *any* RL algorithm — SB3 is
just one adapter. Implement `fit(env, ctx)`; the `TrainingContext` (`ctx`) hands you
the whole ecosystem — TensorBoard **and** MLflow logging, checkpointing with the
DeepRacer metadata sidecar, the held-out evaluation protocol, and Optuna pruning —
so a custom loop reuses all of it.

```python
from gym_dr.trainers.base import Trainer, TrainingContext, TrainResult

class MyTrainer(Trainer):
    def fit(self, env, ctx: TrainingContext) -> TrainResult:
        pipeline = self.build_pipeline()          # a gym_dr.Stage: obs -> action
        obs, _ = env.reset(seed=ctx.seed)
        for step in range(ctx.training.total_timesteps):
            action = pipeline(obs)                # literal Stage datapath — no SB3
            obs, reward, term, trunc, info = env.step(action)
            ctx.record_episode(info, step)        # drains dr/ep_* to TB + MLflow
            if step % ctx.training.eval_freq == 0:
                ctx.report_eval(mean_reward, step)  # MLflow + Optuna pruning
            if term or trunc:
                obs, _ = env.reset()
        return TrainResult(final_eval_reward=mean_reward)
```

Shipped adapters: `Sb3Trainer` (PPO/SAC/TD3/A2C/DDPG) and `FsrlTrainer` (safe-RL).
See `experiments/custom_trainer_example.py`.

## The explicit MDP pipeline: `Stage`

`Stage[I, O]` is a named, composable function `I -> O`. Compose the observation →
encode → policy → action flow with `>>`; the result is inspectable
(`len(p)`, `list(p)`, `repr(p)`) so a training pipeline can be printed and audited.

```python
from gym_dr import Stage, stage

@stage
def adr_input(obs): ...      # domain randomization on the observation
@stage
def encode(obs): ...         # CNN / feature extractor (a torch.nn.Module)
@stage
def policy(feat): ...        # policy head
@stage
def adr_output(action): ...  # domain randomization on the action

pipeline = adr_input >> encode >> policy >> adr_output   # Obs -> Action
action = pipeline(obs)
```

Two roles (ADR-0004/0005): a fast, vectorised adapter reads the composition to
*assemble* the training stack (so per-step cost stays in batched tensor code), and
the very same composed `Stage` is *called* directly for the light paths — inference,
ONNX export, on-car deployment, and the decoupled obs-encoder → policy evaluation.

## Early stopping: interchangeable strategies

`TrainingConfig.early_stop` takes an `EarlyStopStrategy` (or `None`). Strategies are
frozen dataclasses (so HPO can sweep e.g. `training.early_stop.threshold`):

- `OfftrackRate(max_offtrack_rate=0.0, patience=1)` — track mastery (the historical default);
- `CleanCompletion(min_rate=1.0, patience=2)` — the headline success criterion;
- `RewardThreshold(min_reward=...)`, `MetricThreshold(metric=..., threshold=..., mode="max"|"min")`;
- `AllOf(...)` / `AnyOf(...)` — combine the above.

An `EarlyStopController` (owned by the trainer's eval callback) requires the
strategy's `patience` consecutive qualifying eval rounds and resets per chunk.

## Curriculum

`EnvironmentConfig.curriculum` takes a `WorldStrategy` — `FixedWorlds`,
`OrderedSplit` (train on one list, evaluate on a held-out list), or `ACL`
(spaced-repetition curriculum). See [Configuration](configuration.md).
