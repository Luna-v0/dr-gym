"""Single-training entrypoint. Edit this file and run it.

Usage:

    uv run python app.py            # host-side: orchestrates Docker chunks
    python app.py                   # inside the container: runs one chunk

The same file runs in both modes; ``gym_dr.train`` checks the
``GYM_DR_IN_CONTAINER`` env var to decide which side it's on.

Plug-in points
--------------
- ``env_factory``: swap to a different env version or race type. See
  ``gym_dr/envs/`` for the time-trial default and add siblings for
  object_avoidance / head_to_bot / etc.
- ``trainer``: swap ``Sb3Trainer(...)`` for any object with
  ``fit(env, ctx) -> TrainResult`` (see ``gym_dr/trainers/base.py``).
- ``reward``: pass any ``Callable[[dict], float]``. The dict is the upstream
  DeepRacer reward-params dict; see ``gym_dr/rewards.py`` for the key list
  and example functions you can use as-is or adapt.
- ``worlds``: list of world names to rotate through. Single-world =
  list of one. See ``WorldsConfig`` for chunk_steps / rotations semantics.
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    center_line,
    time_trial,
    train,
)

experiment = ExperimentConfig(
    name="quick_test",
    env_factory=time_trial,
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
    reward=center_line,
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0,
        steering_high=30.0,
        speed_low=0.1,
        speed_high=4.0,
    ),
    worlds=WorldsConfig(
        names=["reinvent_base"],
        chunk_steps=5_000,
        rotations=1,
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
