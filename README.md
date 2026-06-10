# DeepRacer SB3 Training Environment

A Python-first, pluggable RL training pipeline for [`Luna-v0/deepracer-env`](https://github.com/Luna-v0/deepracer-env) (a fork of [`seresheim/deepracer-env`](https://github.com/seresheim/deepracer-env)) inside Docker. The user owns a single `app.py` that wires together an env, a trainer, a reward function, an action space, and a list of worlds. Everything else (artifact layout, MLflow tracking, per-checkpoint DeepRacer metadata sidecars, Optuna HPO across parallel Docker workers, TensorBoard, multi-world sequential rotation) is provided by `gym_dr`.

```text
app.py                                ← user edits
  │
  └── train(experiment)               ← host: orchestrates Docker chunks
        │
        └── (per world × rotation) docker run → python app.py
              │
              └── train(experiment)   ← container: runs one chunk
                    │
                    ├── env_factory(experiment)    ← pluggable (time_trial, …)
                    ├── trainer.fit(env, ctx)      ← pluggable (Sb3Trainer, …)
                    └── MLflow + metadata sidecars (provided)
```

## Plug-in points

| What | Interface | Default | Where to add yours |
|---|---|---|---|
| Env | `(ExperimentConfig) -> gym.Env` | `gym_dr.envs.time_trial` | `gym_dr/envs/` (or any module) |
| Trainer | `Trainer` protocol: `fit(env, ctx) -> TrainResult` | `gym_dr.trainers.Sb3Trainer` | `gym_dr/trainers/` (or any module) |
| Reward | `(params: dict) -> float` | `gym_dr.rewards.center_line` | written directly in `app.py` |
| Action space | `ContinuousActionSpaceConfig` or `DiscreteActionSpaceConfig` | continuous | configured in `app.py` |
| Worlds | `WorldsConfig(names, chunk_steps, rotations)` | `["reinvent_base"]` × 1 chunk | configured in `app.py` |

The reward is a plain Python function (no registry). For HPO over reward weights, write a closure factory like `make_center_line(weight=...)` and have your search space return a freshly-built callable each trial — see `experiments/hpo_example.py`.

## Prerequisites

- Docker (daemon running, `buildx` available)
- `git`
- [`uv`](https://github.com/astral-sh/uv) for host-side Python
- ~50 GB free in Docker's storage location for the first build

## First-time setup

```bash
cd /path/to/dr-gym
./bootstrap.sh           # upstream simulator image + project image (use -a gpu for GPU)
uv sync                  # host-side Python deps
```

`bootstrap.sh` tracks the upstream `deepracer-env` branch: a fresh machine
clones its latest tip, and re-runs fetch the branch and **prompt** you to pull
+ rebuild the base image when it has advanced. Pass `-a gpu` to build the GPU
image, `-b <branch>` to track a different upstream branch, and `--help` for all
options. Re-run it when `pyproject.toml` changes or when you want to pick up
upstream simulator updates.

## Running a training: `app.py`

The user edits `app.py`. Minimal shape:

```python
from gym_dr import (
    ContinuousActionSpaceConfig, ExperimentConfig, Sb3Trainer,
    TrackingConfig, TrainingConfig, WorldsConfig,
    center_line, time_trial, train,
)

experiment = ExperimentConfig(
    name="quick_test",
    env_factory=time_trial,
    trainer=Sb3Trainer(
        name="ppo", policy="MultiInputPolicy",
        kwargs={"n_steps": 256, "batch_size": 64, "learning_rate": 3e-4, "ent_coef": 0.01},
    ),
    reward=center_line,
    action_space=ContinuousActionSpaceConfig(),
    worlds=WorldsConfig(names=["reinvent_base"], chunk_steps=5_000, rotations=1),
    training=TrainingConfig(total_timesteps=5_000, eval_freq=2_500, n_eval_episodes=2),
    tracking=TrackingConfig(),
)

if __name__ == "__main__":
    train(experiment)
```

Run it directly from the host — `train()` handles Docker, multi-world rotation, MLflow:

```bash
uv run python app.py
```

### Ready-to-run example: simple time-trial training

For the quickest start, `experiments/time_trial_train.py` is a complete,
no-HPO single training run: pure time-trial on one track (`reinvent_base`),
a fixed set of PPO hyperparameters, and a straight `train(experiment)` call.
Nothing to wire up — just run it:

```bash
uv run python experiments/time_trial_train.py
```

It has the **GUI on by default** — connect a VNC client to
`vnc://localhost:5900` to watch the car as it trains — and runs the simulator
at **2x real time** (`rtf_override=2`). Four knobs at the top of the file
(`NAME`, `WORLD`, `TOTAL_TIMESTEPS`, `FRAME_STACK`) cover the common edits. It
defaults to GPU (`device="cuda"` + `use_gpu=True`), matching the
`my-deepracer-project:gpu` image; switch both to CPU if you only built that arch.

When it finishes, watch the trained policy with the evaluator:

```bash
uv run python scripts/evaluate.py \
    --model artifacts/time_trial_train/best_model/best_model.zip
```

## Writing a custom reward

Just write a function and pass it. The dict argument is the upstream DeepRacer reward-params dict — `track_width`, `distance_from_center`, `progress`, `speed`, `all_wheels_on_track`, `is_offtrack`, `waypoints`, etc. (See `.deepracer-env-upstream/deepracer_env/agent_ctrl/constants.py:108` for the full list.)

```python
def stay_centered_and_fast(params: dict) -> float:
    if not params["all_wheels_on_track"]:
        return 1e-3
    centeredness = 1.0 - params["distance_from_center"] / params["track_width"]
    return float(centeredness * params["speed"])

experiment = ExperimentConfig(reward=stay_centered_and_fast, ...)
```

The reward function's source is auto-archived into `artifacts/<run_name>/reward_function.py` via `inspect.getsource`.

## Multi-world training

`WorldsConfig` controls how worlds are rotated. A **single** container trains the whole rotation in-process: Gazebo loads `names[0]` at startup, and the in-container trainer hot-swaps the track between each `(rotation, world)` chunk at runtime via `DeepRacerEnv.set_world` — no per-chunk container restart. Because it's one process, the policy, optimizer state, and TensorBoard session persist across switches naturally.

```python
worlds=WorldsConfig(
    names=["reinvent_base", "Bowtie_track", "AmericasGeneratedInclStart"],
    chunk_steps=20_000,
    rotations=3,
)
```

This runs **9 chunks** in order `reinvent_base → Bowtie_track → AmericasGeneratedInclStart → reinvent_base → ...`, each 20k timesteps, all in one container with the track hot-swapped between them. All chunks log under one MLflow parent run (named after `experiment.name`); the UI nests them as children for easy comparison.

Valid world names are in `.deepracer-env-upstream/tracks.txt`. `gym_dr.TRACKS` is a `dict[world_name -> display_name]` covering every known track (re:Invent 2018, A to Z Speedway, Forever Raceway, etc.). To rotate through **every** track in one run:

```python
from gym_dr import ALL_TRACKS, WorldsConfig, existing_tracks

worlds = WorldsConfig(
    names=existing_tracks(),   # filters ALL_TRACKS against the simapp image's tracks.txt
    chunk_steps=10_000,
    rotations=1,
)
```

Use `ALL_TRACKS` directly if you want the unfiltered list; `existing_tracks()` is the safer default — it skips world names whose route file isn't in the simapp image, so the orchestrator doesn't crash on an unknown world halfway through.

## Plugging in a custom trainer (non-SB3)

`Sb3Trainer` is the default. Any object with a `fit(env, ctx) -> TrainResult` method is a trainer — no inheritance required (it's a `runtime_checkable` Protocol).

```python
from gym_dr.trainers.base import TrainingContext, TrainResult

class MyTrainer:
    def __init__(self, lr: float = 1e-3):
        self.lr = lr

    def fit(self, env, ctx: TrainingContext) -> TrainResult:
        ctx.save_model(self._save, name="initial_model")
        try:
            for step in range(ctx.training.total_timesteps):
                ...  # your training step
                if step % ctx.training.eval_freq == 0:
                    mean = self._evaluate(env)
                    ctx.report_eval(mean, step=step)     # MLflow + Optuna prune
                if step % ctx.training.checkpoint_freq == 0:
                    ctx.save_checkpoint(self._save, step=step)
            ctx.save_model(self._save, name="final_model")
            return TrainResult(final_eval_reward=mean)
        finally:
            ctx.save_model(self._save, name="latest_model")

experiment = ExperimentConfig(trainer=MyTrainer(lr=5e-4), ...)
```

`TrainingContext` (`gym_dr/trainers/base.py`) gives the trainer four hooks; using them is what makes MLflow logging + per-checkpoint DeepRacer metadata + Optuna pruning Just Work for your trainer too.

## Plugging in a custom env

Write a callable `(experiment) -> gym.Env`:

```python
# gym_dr/envs/object_avoidance.py (or any module)
def object_avoidance(experiment):
    from deepracer_env.environments.deepracer_env import DeepRacerEnv
    return DeepRacerEnv(
        reward_fn=experiment.reward,
        sensors=list(experiment.action_space.sensor),
        config={"race_type": "OBJECT_AVOIDANCE"},
    )

experiment = ExperimentConfig(env_factory=object_avoidance, ...)
```

The upstream `RaceType` enum has `TIME_TRIAL` (current default), `OBJECT_AVOIDANCE`, `HEAD_TO_BOT`, `HEAD_TO_MODEL`, `F1` — `.deepracer-env-upstream/deepracer_env/reset/constants.py:21`.

## Customizing the network

The policy/value network is modelled on the **real AWS DeepRacer** training
stack (the RoboMaker `markov` bundle + Intel rl-coach), not the community sim:

- **Separate actor & critic towers.** AWS's clipped PPO sets
  `use_separate_networks_per_head=True` — the policy and value each get their
  *own* CNN + FC tower (same spec, independent weights). We reproduce this
  with `policy_kwargs["share_features_extractor"] = False`, so SB3 builds two
  `DeepRacerCNN` instances.
- **Raw 0–255 input.** AWS feeds the model un-normalized grayscale uint8 — no
  `/255`. `policy_kwargs["normalize_images"] = False` matches that, and
  `time_trial`'s grayscale wrapper makes the obs single-channel (so the ONNX
  export's input matches what the car feeds).
- **CNN tower** — `gym_dr.networks.DeepRacerCNN`, a config-driven conv stack.
  Use a named DeepRacer preset or a custom stack:
  ```python
  from gym_dr.networks import DEEPRACER_CONV_PRESETS, DeepRacerCNN
  trainer = Sb3Trainer(
      name="ppo", policy="MultiInputPolicy",
      kwargs={"policy_kwargs": {
          "share_features_extractor": False,
          "normalize_images": False,
          "features_extractor_class": DeepRacerCNN,
          "features_extractor_kwargs": {
              # a named arch: "shallow" / "standard" / "deep"
              "conv_layers": DEEPRACER_CONV_PRESETS["shallow"],
              # ...or a custom ((filters, kernel, stride), ...) stack
              "features_dim": 512,
          },
          "net_arch": dict(pi=[512], vf=[512]),  # per-head FC, sized independently
      }},
  )
  ```
- **FC middleware** — `policy_kwargs["net_arch"] = dict(pi=[...], vf=[...])`,
  the layers between each CNN tower and its head. Sized independently per head.

`app.py`'s `search_space` sweeps all of it: the CNN arch (a named preset *or*
a sampled custom conv stack), `features_dim`, and the pi/vf FC widths/depths
(up to 1024 wide). `features_extractor_class` is a class — not Optuna-sweepable
— so it's fixed; everything inside `features_extractor_kwargs` + `net_arch`
varies. See `gym_dr/networks.py` for the AWS grounding and the preset specs.

## Evaluate (view mode)

Watch a trained model drive — no training, just inference + a live Gazebo GUI:

```bash
uv run python scripts/evaluate.py --model artifacts/hpo_trial_15/final_model.zip
```

Then point a VNC client at `localhost:5900`. Per-step and per-episode detail
(`dr/ep_reward`, `dr/ep_max_progress`, off-track count, mean speed, …) streams
to your terminal.

No `--app` needed — the experiment (env factory, reward, action space,
frame-stack depth, GPU flag) is reconstructed from the model's
`run_config.json`, which every training run writes. It's auto-discovered next
to the model, and — if the model lives in a subdir like `best_model/` — in the
nearest parent directory, so a nested checkpoint still finds its run's config.
Pass `--run-config <path>` to point at a specific one, or `--app <path>` only
if the run used callables defined *inline* in the experiment script.

The image is selected to match how the model was trained (`use_gpu` from the
run_config → `my-deepracer-project:gpu` or `:cpu`); override with `IMAGE_TAG`.

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--episodes N` | `5` | episodes to run (ignored with `--loop`) |
| `--loop` | off | run forever until Ctrl-C — just watch |
| `--world W` | model's training world | evaluate on a different track |
| `--rtf R` | `1.0` | simulator real-time factor — `1.0` is human-watchable real time; the training config's `rtf_override` (often 10+ for fast HPO) is **not** inherited |
| `--run-config PATH` | auto-discover (model dir, then nearest parent) | explicit `run_config.json` to reconstruct the experiment from |
| `--app PATH` | reconstruct from `run_config.json` | experiment-script override for inline-callable runs |

## HPO

`experiments/hpo_example.py` is a runnable script. Defines a `base` config + a `search_space(trial)` function + `study(...)` at the bottom. Run it from the host:

```bash
uv run python experiments/hpo_example.py
```

The same script runs unchanged inside each worker container — `study()` detects worker mode via `GYM_DR_WORKER`. See `docs/hpo.md` for details.

## Inspect without running

```bash
uv run python -c "from app import experiment; from gym_dr import inspect; inspect(experiment)"
```

## Iterating without rebuilding

Project source is bind-mounted at `/workspace` in the container. Edit and re-run — no `docker build`:

- Hyperparameters / training control / algorithm choice → edit `app.py`
- Reward shaping → write a function in `app.py`
- Action space / worlds → edit `app.py`
- Custom trainer / env → drop a file anywhere, import it in `app.py`

Rebuild only when `pyproject.toml` changes.

## Runtime world switching

Earlier versions rotated worlds by **respawning the container**, paying Gazebo's startup cost (~5–10 s) per switch. The [`Luna-v0/deepracer-env`](https://github.com/Luna-v0/deepracer-env) fork this project builds on now switches worlds *inside a single running container*, so the policy, optimizer state, and TB session all persist across the rotation.

This required, in the fork:

1. **A `TrackData` setter.** It was a singleton that read `WORLD_NAME` from `rospy` at first construction and cached waypoints from `routes/<name>.npy` for the process lifetime. The fork adds a setter that drops the cached waypoints and re-reads the routes file, called between episodes (between `env.reset()` and the first `env.step()` of the new episode), never mid-episode.
2. **A Gazebo world swap.** `gazebo_msgs/DeleteModel` on the current track meshes, then `gazebo_msgs/SpawnModel` on the new track's SDF.
3. **`DeepRacerEnv.set_world(name)`.** The public env hook that performs the world-swap RPC, updates `TrackData`, and resets the start position.

`gym_dr`'s orchestrator drives this: `train()` launches one container, and the in-container trainer calls `DeepRacerEnv.set_world(...)` between `WorldsConfig.chunk_steps`-sized chunks to walk the rotation — see `gym_dr/app.py`. `WorldsConfig` semantics are unchanged; only the mechanism is (no container respawn).

## Internal layout

```text
gym_dr/
├── app.py             # train(), study(), inspect() — user-facing call surface
├── trainer.py         # orchestrator: run_training() — wraps any Trainer
├── config.py          # ExperimentConfig + sub-configs (frozen dataclasses)
├── action_space.py    # ContinuousActionSpaceConfig, DiscreteActionSpaceConfig
├── rewards.py         # example reward functions (plain callables, no registry)
├── hpo.py             # Optuna study + objective + worker loop
├── docker_runner.py   # host-side container spawners (training chunks + HPO workers)
├── mlflow_utils.py    # MLflow run + parent-run helpers
├── envs/
│   ├── __init__.py
│   └── time_trial.py  # default env factory; add object_avoidance.py etc. here
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
| `ExperimentConfig` anatomy | [docs/configuration.md](docs/configuration.md) |
| Per-checkpoint metadata guarantees | [docs/artifact-layout.md](docs/artifact-layout.md) |
| MLflow + TensorBoard | [docs/tracking.md](docs/tracking.md) |
| Optuna HPO across parallel containers | [docs/hpo.md](docs/hpo.md) |
| Algorithm registry + off-policy caveats | [docs/algorithms.md](docs/algorithms.md) |
| Physical-car export caveats | [docs/physical-car-integration-notes.md](docs/physical-car-integration-notes.md) |

## Physical-car export

Once you have a trained model (SB3 `.zip`), package it for the on-device loader:

```bash
# from an SB3 zip — metadata is auto-picked up from the sibling .model_metadata.json:
uv run python scripts/export_bundle.py \
    --model artifacts/<run>/final_model.zip \
    --output bundle.tar.gz

# or, with explicit metadata from your app.py:
uv run python scripts/export_bundle.py \
    --model artifacts/<run>/final_model.zip \
    --app app.py \
    --output bundle.tar.gz

# or, packaging a pre-existing TF frozen-graph .pb verbatim:
uv run python scripts/export_bundle.py \
    --model my_model.pb \
    --metadata my_metadata.json \
    --output bundle.tar.gz
```

Bundle layout (all paths produce the same contract):

```text
bundle.tar.gz
├── model_metadata.json
└── agent/
    └── agent.{pb,onnx}
```

SB3 zips are exported to ONNX (`agent.onnx`); pre-existing `.pb` / `.onnx` files are packaged verbatim. `--bundle-filename` overrides the in-tar filename if your target expects something different. See [docs/physical-car-integration-notes.md](docs/physical-car-integration-notes.md) for the on-device caveats.

## Tests

```bash
uv run pytest tests/
```

The smoke suite wires a stub env in place of the upstream sim and exercises the whole orchestrator → `Sb3Trainer` → `TrainingContext` flow, plus the export-bundle script. Stable suite is 18 tests + 1 conditional skip.

## Resume training

Set `training.resume_from` in `app.py` to the **container** path of a previous checkpoint:

```python
training=TrainingConfig(
    resume_from="/workspace/artifacts/quick_test_rot0_reinvent_base/latest_model.zip",
    ...,
)
```

For multi-world rotations this happens automatically between chunks — only set it explicitly to resume a brand-new run from a previous one.

## Check training status

```bash
cat artifacts/<chunk_name>/training_status.json
find artifacts/<chunk_name>/checkpoints -maxdepth 1 -type f | sort
cat artifacts/<chunk_name>/run_config.json | jq .
```
