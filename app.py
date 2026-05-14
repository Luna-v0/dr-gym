"""HPO entrypoint. Edit and run.

Usage:

    uv run python app.py            # host-side: spawns N parallel worker containers
    python app.py                   # inside a worker container (auto via GYM_DR_WORKER=1)

This file defines:
  - ``base``: the base ``ExperimentConfig`` (everything not swept by HPO).
  - ``search_space(trial)``: returns a dotted-key overrides dict applied per
    trial via ``ExperimentConfig.with_overrides(**overrides)``.

The search includes:
  - PPO hyperparameters (learning_rate, ent_coef, n_steps, batch_size,
    gamma, gae_lambda, clip_range, n_epochs).
  - **Neural network architecture** — `layer_width` × `num_layers`,
    materialized as ``policy_kwargs.net_arch = dict(pi=..., vf=...)``.
    TPE explores small-to-large; the range is wide enough that it can
    walk toward bigger nets as it learns which help.

To turn this into a single (non-HPO) training run, swap the bottom-of-file
``study(...)`` for ``train(experiment)`` and remove ``search_space``. See
``experiments/hpo_example.py`` for the canonical reference.
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    center_line,
    study,
    time_trial,
)


# --------------------------------------------------------------------------- #
# Edit these to control the study. They're consumed only by the `study(...)`
# call at the bottom of the file (host orchestrator); the in-container worker
# reads N_TRIALS_PER_WORKER from env vars set by the host.
# --------------------------------------------------------------------------- #
STUDY_NAME = "hpo_app"
N_TRIALS = 20
N_PARALLEL = 1   # number of concurrent Docker workers (each runs its own simapp)
SEED = 42        # int for reproducibility; None for nondeterministic


base = ExperimentConfig(
    name="hpo",
    env_factory=time_trial,
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
    reward=center_line,
    action_space=ContinuousActionSpaceConfig(
        steering_low=-30.0,
        steering_high=30.0,
        speed_low=0.1,
        speed_high=4.0,
    ),
    # HPO uses worlds.names[0] for every trial; chunk_steps/rotations are
    # only consulted by the non-HPO host orchestrator (one chunk per track).
    worlds=WorldsConfig(names=["reinvent_base"], chunk_steps=100_000),
    training=TrainingConfig(
        # Per-trial budget in HPO mode; per-track budget == chunk_steps when
        # this experiment is run as a multi-world training instead. Bumped
        # from 20k — single-layer-free bigger nets need more steps to show
        # their ceiling, and the MedianPruner still kills weak trials early
        # so the wall-clock cost is mostly paid by the good trials.
        total_timesteps=100_000,
        checkpoint_freq=25_000,
        eval_freq=10_000,
        n_eval_episodes=3,
        rtf_override=10,
    ),
    tracking=TrackingConfig(mlflow_experiment="gym-dr-hpo"),
    enable_gui=True,   # watch the car: VNC client -> localhost:5900
    seed=SEED,
)


def search_space(trial) -> dict:
    """Per-trial overrides applied through ``ExperimentConfig.with_overrides``.

    Dotted keys walk into dataclasses and dicts; ``trainer.kwargs.*`` lands
    in the SB3 algorithm's constructor, and ``trainer.kwargs.policy_kwargs``
    replaces SB3's policy kwargs wholesale (including ``net_arch``).
    """
    # --- PPO hyperparameters ------------------------------------------------
    overrides: dict = {
        "trainer.kwargs.learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "trainer.kwargs.ent_coef":      trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
        "trainer.kwargs.n_steps":       trial.suggest_categorical("n_steps", [128, 256, 512, 1024]),
        "trainer.kwargs.batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "trainer.kwargs.gamma":         trial.suggest_float("gamma", 0.95, 0.999),
        "trainer.kwargs.gae_lambda":    trial.suggest_float("gae_lambda", 0.9, 0.99),
        "trainer.kwargs.clip_range":    trial.suggest_float("clip_range", 0.1, 0.3),
        "trainer.kwargs.n_epochs":      trial.suggest_int("n_epochs", 4, 12),
        # Frame stacking — upstream DeepRacerEnv emits single frames; stacking
        # gives the policy implicit temporal context (velocity / acceleration
        # cues). 1 = no stacking, 4 is the Atari-DQN classic default.
        "trainer.frame_stack":          trial.suggest_int("frame_stack", 2, 4),
    }

    # --- Neural network architecture ----------------------------------------
    # Two parts of the net are swept here:
    #   * net_arch       — the MLP *head* after the CNN. `dict(pi=..., vf=...)`
    #                      is the SB3 PPO convention.
    #   * cnn_output_dim — the width of the CNN feature embedding that feeds
    #                      that head. This is the easy "bigger CNN" lever
    #                      (no custom extractor class needed); see
    #                      gym_dr/extractors.py for the deeper-CNN option.
    # num_layers starts at 2 — single-layer heads underfit the camera input.
    layer_width = trial.suggest_categorical("layer_width", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    cnn_output_dim = trial.suggest_categorical("cnn_output_dim", [256, 512, 1024])
    overrides["trainer.kwargs.policy_kwargs"] = {
        "net_arch": dict(pi=[layer_width] * num_layers, vf=[layer_width] * num_layers),
        "features_extractor_kwargs": {"cnn_output_dim": cnn_output_dim},
    }
    return overrides


# Alias so the host-side `prepare-metadata` step (and `inspect`) can read
# the action space and other shared fields off this file.
experiment = base


if __name__ == "__main__":
    study(
        base,
        search_space,
        study_name=STUDY_NAME,
        n_trials=N_TRIALS,
        n_parallel=N_PARALLEL,
    )
