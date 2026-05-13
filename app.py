"""Single-training entrypoint. Edit and run.

Host (wraps it in Docker):    ./run_cpu_training.sh app.py
Inside the container:         python app.py

Plug-in points
--------------
- env_factory:  swap to a new sim version (e.g. `deepracer_env_v2`) or your own
                callable taking the ExperimentConfig and returning a gym env.
- trainer:      swap `Sb3Trainer(...)` for any object implementing the
                `Trainer` protocol — see gym_dr/trainers/base.py.
- reward:       reference any factory registered in gym_dr/reward.py
                (add new ones with `@register("my_reward")`).
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    RewardConfig,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    deepracer_env_v1,
    train,
)

experiment = ExperimentConfig(
    name="quick_test",
    world_name="reinvent_base",
    env_factory=deepracer_env_v1,
    trainer=Sb3Trainer(
        name="ppo",
        policy="MultiInputPolicy",
        kwargs={
            "n_steps": 256,
            "batch_size": 64,
            "learning_rate": 3.0e-4,
            "ent_coef": 0.01,
        },
        device="cpu",
    ),
    reward=RewardConfig(factory="center_line", params={}),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0,
        steering_high=30.0,
        speed_low=0.1,
        speed_high=4.0,
    ),
    training=TrainingConfig(
        total_timesteps=5_000,
        checkpoint_freq=1_000,
        eval_freq=2_500,
        n_eval_episodes=2,
    ),
    tracking=TrackingConfig(mlflow_experiment="gym-dr"),
)


if __name__ == "__main__":
    train(experiment)
