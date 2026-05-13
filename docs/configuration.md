# Experiment configuration

Every experiment is a runnable Python script under `experiments/`. It defines an `experiment: ExperimentConfig` instance (or for HPO, a `base: ExperimentConfig` + a `search_space(trial)` function) and ends with `if __name__ == "__main__": train(experiment)` (or `study(...)` for HPO). Run it directly — no CLI.

## Anatomy

`gym_dr/config.py` and `gym_dr/action_space.py` define the dataclass tree:

```text
ExperimentConfig
├── name: str                                          # artifacts/<name>/, MLflow run name
├── world_name: str                                    # DeepRacer track
├── env_factory: Callable[[ExperimentConfig], gym.Env] # default: gym_dr.envs.deepracer_env_v1
├── trainer: Trainer                                   # default: Sb3Trainer()
├── reward: RewardConfig
│     ├── factory: str                                 # key into gym_dr/reward.py registry
│     └── params: dict[str, float]                     # per-factory weights
├── action_space: ContinuousActionSpaceConfig | DiscreteActionSpaceConfig
│     # Continuous: steering_low/high, speed_low/high: float
│     # Discrete:   actions: list[DiscreteAction(steering_angle, speed)]
│     # Shared:     sensor, neural_network, version, training_algorithm
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

The default `trainer=Sb3Trainer()` carries its own fields (`name`, `policy`, `kwargs`, `device`); HPO can sweep them via dotted overrides like `"trainer.kwargs.learning_rate"`. To plug in your own trainer, see `gym_dr/trainers/base.py` and the README.

All dataclasses are `frozen=True`. To mutate a config, use `cfg.with_overrides(**flat_dotted_kwargs)` — see `gym_dr/config.py`.

## Continuous vs discrete action space

The action-space config determines which DeepRacer schema gets written to `model_metadata.json`.

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

## Reward factories

`gym_dr/reward.py` keeps a registry of `factory_name -> (params: dict) -> reward_function(params: dict) -> float`. The default factory is `"center_line"` (mirrors the original `reward.py`). Add a new factory:

```python
# gym_dr/reward.py
@register("speed_seeker")
def speed_seeker(params):
    weight = float(params.get("weight", 1.0))
    def reward_function(p):
        return weight * p["speed"]
    return reward_function
```

Then reference it from a config:

```python
reward=RewardConfig(factory="speed_seeker", params={"weight": 2.0})
```

The trainer renders the chosen factory's source (via `inspect.getsource`) into `artifacts/<run_name>/reward_function.py` for reproducibility.

## Inspecting an experiment

```bash
uv run python -m gym_dr.cli inspect experiments/quick.py
```

Prints the resolved tree and the flat MLflow param keys. No Docker or simulator needed.
