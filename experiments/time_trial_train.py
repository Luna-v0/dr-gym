"""Simple single-run time-trial training — fixed hyperparameters, no HPO.

The minimal "just train a car" experiment: one track, pure time-trial
driving, a fixed set of PPO hyperparameters, and a straight ``train()`` call.
No Optuna, no world rotation, no search space — edit the constants below if
you want, then let it run.

Run it:

    uv run python experiments/time_trial_train.py

On the host this reconstructs the experiment, pre-generates
``model_metadata.json`` and ``docker run``s the sim container; inside the
container it trains and writes checkpoints + ``best_model`` into a run dir
under ``artifacts/``.

The GUI is on by default (``enable_gui=True``) — connect a VNC client to
``vnc://localhost:5900`` to watch the car as it trains. The sim runs at 2x
real time (``rtf_override=2``). Evaluate the result afterwards with::

    uv run python scripts/evaluate.py \\
        --model artifacts/time_trial_train/best_model/best_model.zip

This uses GPU (``device="cuda"`` + ``use_gpu=True``) to match the built
``my-deepracer-project:gpu`` image. Switch both to CPU if you only have the
cpu image built.
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

# --------------------------------------------------------------------------- #
# Knobs — edit these and re-run. Everything else below is wiring.
# --------------------------------------------------------------------------- #
NAME = "time_trial_demo"
WORLD = "reinvent_base"  # single track to train on (must be in gym_dr.TRACKS)
TOTAL_TIMESTEPS = 1_000_000  # total environment steps to train for
FRAME_STACK = 4  # temporal context (DeepRacerEnv emits single frames)


experiment = ExperimentConfig(
    name=NAME,
    env_factory=time_trial,  # pure time-trial (no object_avoidance)
    reward=center_line,  # a stable time-trial reward
    # Fixed PPO hyperparameters — sensible, non-swept defaults.
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
        frame_stack=FRAME_STACK,
        # device="cuda",
    ),
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0,
        steering_high=30.0,
        # Floor speed above the crawl bound so "stay on track by barely
        # moving" isn't a viable policy — the car has to actually drive.
        speed_low=1.0,
        speed_high=4.0,
    ),
    # Single track, no rotation. (Use a world_strategy like OrderedSplit for
    # multi-world train/eval splits — see experiments/ordered_split_example.py.)
    worlds=WorldsConfig(names=[WORLD]),
    training=TrainingConfig(
        total_timesteps=TOTAL_TIMESTEPS,
        checkpoint_freq=50_000,
        eval_freq=25_000,
        n_eval_episodes=3,
        rtf_override=10,  # run the sim at 10x real time
    ),
    tracking=TrackingConfig(mlflow_experiment=NAME),
    # Watch the car train over VNC: connect a client to vnc://localhost:5900.
    enable_gui=True,
    seed=42,
    # use_gpu=True,
)


if __name__ == "__main__":
    train(experiment)
