"""Optuna HPO over PPO hyperparams + reward weights.

Run from the host (spawns parallel Docker workers):

    uv run python experiments/hpo_example.py

The same file runs unchanged inside each worker container — ``study()``
detects worker mode via the ``GYM_DR_WORKER`` env var and switches to a
single-process trial loop.

The search space mutates ``trainer.kwargs`` via dotted overrides; for the
reward, it swaps in a freshly-built closure each trial (``make_reward(...)``)
since the reward is now a plain callable, not a config object.
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    Study,
)


def make_center_line(
    threshold_close: float = 0.1,
    threshold_mid: float = 0.25,
    threshold_far: float = 0.5,
    reward_close: float = 100.0,
    reward_mid: float = 0.5,
    reward_far: float = 0.1,
    reward_off: float = 1e-3,
):
    """Parameterized center-line reward; returns a fresh callable per call."""
    def reward(params: dict) -> float:
        tw = params["track_width"]
        d = params["distance_from_center"]
        if d <= threshold_close * tw:
            return reward_close
        if d <= threshold_mid * tw:
            return reward_mid
        if d <= threshold_far * tw:
            return reward_far
        return reward_off
    return reward


base = ExperimentConfig(
    name="hpo",
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
    reward=make_center_line(),
    action_space=ContinuousActionSpaceConfig(),
    worlds=WorldsConfig(names=["reinvent_base"], chunk_steps=100_000, rotations=1),
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
        "reward": make_center_line(
            threshold_close=trial.suggest_float("threshold_close", 0.05, 0.2),
            reward_close=trial.suggest_float("reward_close", 10.0, 200.0),
            reward_mid=trial.suggest_float("reward_mid", 0.1, 5.0),
        ),
    }


# alias so the host-side metadata pre-gen can read the action space off this file
experiment = base


if __name__ == "__main__":
    Study(
        base,
        search_space,
        study_name="hpo_example",
        n_trials=40,
        n_parallel=4,
    ).run()
