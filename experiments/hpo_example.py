"""Optuna HPO over PPO hyperparams + reward weights.

Run from the host (spawns parallel Docker workers):

    uv run python experiments/hpo_example.py

The same file runs unchanged inside each worker container — `study()` detects
worker mode via the GYM_DR_WORKER env var and switches to a single-process
trial loop.
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    RewardConfig,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    deepracer_env_v1,
    study,
)

base = ExperimentConfig(
    name="hpo",
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
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "n_epochs": 10,
        },
        device="cpu",
    ),
    reward=RewardConfig(factory="center_line", params={}),
    action_space=ContinuousActionSpaceConfig(),
    training=TrainingConfig(
        total_timesteps=100_000,
        checkpoint_freq=10_000,
        max_train_seconds=900,
        eval_freq=5_000,
        n_eval_episodes=3,
    ),
    tracking=TrackingConfig(mlflow_experiment="gym-dr-hpo"),
)


def search_space(trial) -> dict:
    return {
        "trainer.kwargs.learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "trainer.kwargs.ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
        "trainer.kwargs.n_steps": trial.suggest_categorical("n_steps", [128, 256, 512, 1024]),
        "trainer.kwargs.batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "trainer.kwargs.gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "trainer.kwargs.gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "trainer.kwargs.clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "reward.params.reward_center": trial.suggest_float("reward_center", 10.0, 200.0),
        "reward.params.reward_mid": trial.suggest_float("reward_mid", 0.1, 5.0),
        "reward.params.marker_1_frac": trial.suggest_float("marker_1_frac", 0.05, 0.2),
        "reward.params.marker_2_frac": trial.suggest_float("marker_2_frac", 0.2, 0.4),
        "reward.params.marker_3_frac": trial.suggest_float("marker_3_frac", 0.4, 0.6),
    }


# alias so the host-side prepare-metadata step can read the action space off this file
experiment = base


if __name__ == "__main__":
    study(
        base,
        search_space,
        study_name="hpo_example",
        n_trials=40,
        n_parallel=4,
    )
