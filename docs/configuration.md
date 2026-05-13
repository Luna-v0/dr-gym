# Experiment configuration

The user owns a single `app.py` at the repo root. It defines an `experiment: ExperimentConfig` instance and ends with `if __name__ == "__main__": train(experiment)`. Run it directly with `uv run python app.py` — no CLI, no shell wrapper.

For HPO, write a separate script under `experiments/` that defines `base: ExperimentConfig` + `search_space(trial)` and ends with `study(base, search_space, ...)`. Same pattern.

## Anatomy

`gym_dr/config.py`, `gym_dr/action_space.py`, and `gym_dr/trainers/sb3/__init__.py` define the dataclass tree:

```text
ExperimentConfig
├── name: str                                          # used for artifacts/<name>_rot<r>_<world>/ and MLflow run name
├── env_factory: Callable[[ExperimentConfig], gym.Env] # default: gym_dr.envs.time_trial
├── trainer: Trainer                                   # default: Sb3Trainer()
├── reward: Callable[[dict], float]                    # default: gym_dr.rewards.center_line — plain function
├── action_space: ContinuousActionSpaceConfig | DiscreteActionSpaceConfig
│     # Continuous: steering_low/high, speed_low/high: float
│     # Discrete:   actions: list[DiscreteAction(steering_angle, speed)]
│     # Shared:     sensor, neural_network, version, training_algorithm
├── worlds: WorldsConfig
│     ├── names: list[str]                             # rotation list; single-world = list of one
│     ├── chunk_steps: int                             # timesteps per (world, rotation) chunk
│     └── rotations: int                               # full passes through `names`
├── training: TrainingConfig
│     ├── total_timesteps, checkpoint_freq, eval_freq, n_eval_episodes
│     ├── max_train_seconds, status_update_steps, status_update_seconds
│     ├── resume_from, rtf_override
└── tracking: TrackingConfig
      ├── mlflow_tracking_uri: str = "file:///workspace/mlruns"
      ├── mlflow_experiment: str = "gym-dr"
      ├── tensorboard: bool = True
      └── tags: dict[str, str]
```

`Sb3Trainer` (the default trainer) carries `name`, `policy`, `kwargs`, `device`. HPO sweeps its kwargs via dotted overrides like `"trainer.kwargs.learning_rate"`. To plug in your own trainer, see `gym_dr/trainers/base.py` and the README.

## Reward as a plain function

The reward is `Callable[[dict], float]`. No registry, no config dataclass — just write a function in `app.py` and pass it:

```python
def my_reward(params: dict) -> float:
    if params["is_offtrack"]:
        return 1e-3
    return float(params["progress"] * params["speed"])

experiment = ExperimentConfig(reward=my_reward, ...)
```

For HPO, write a closure factory and have the search space return a freshly-built closure each trial:

```python
def make_reward(weight: float):
    def reward(params):
        return weight * params["progress"]
    return reward

def search_space(trial):
    return {"reward": make_reward(weight=trial.suggest_float("weight", 0.1, 10.0))}
```

The chosen reward's source is auto-archived via `inspect.getsource` into `artifacts/<run_name>/reward_function.py`.

## Continuous vs discrete action space

Continuous:
```python
from gym_dr import ContinuousActionSpaceConfig
action_space = ContinuousActionSpaceConfig(
    steering_low=-30.0, steering_high=30.0,
    speed_low=0.1,    speed_high=4.0,
)
```

Discrete (matches the DeepRacer console export):
```python
from gym_dr import DiscreteActionSpaceConfig, DiscreteAction
action_space = DiscreteActionSpaceConfig(
    actions=[
        DiscreteAction(steering_angle=-30, speed=0.5),
        DiscreteAction(steering_angle=  0, speed=1.0),
        DiscreteAction(steering_angle= 30, speed=0.5),
    ],
)
```

`index` is auto-assigned (0-based, list order) when the JSON is rendered.

## Inspecting an experiment

```bash
uv run python -c "from app import experiment; from gym_dr import inspect; inspect(experiment)"
```

Prints the resolved tree and the flat MLflow param keys. No Docker, no sim.
