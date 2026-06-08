"""Pipeline-validation study: a tiny single-worker Optuna run that exercises
the *whole* dr-gym stack end-to-end, including the new Tier-1 trace sink.

Goal is validation, not hyperparameter results. It confirms, in one ~hour run:
  - the project container launches and the Gazebo/ROS simapp comes up,
  - PPO actually steps the real DeepRacerEnv,
  - MLflow logs runs/params/metrics, TensorBoard writes scalars,
  - Optuna pulls trials from the shared SQLite study,
  - the trace sink writes per-episode Parquet shards under
    artifacts/<run>/trace/steps/ (the simtrace-equivalent — see
    docs/trace-contract.md).

Run from the host (spawns one worker container):

    GYM_DR_EXPERIMENT_FILE=$PWD/experiments/trace_smoke_study.py \
      .venv/bin/python experiments/trace_smoke_study.py

The same file runs unchanged inside the worker (study() detects GYM_DR_WORKER).
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    TraceConfig,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    center_line,
    study,
)


base = ExperimentConfig(
    name="trace_smoke",
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
    action_space=ContinuousActionSpaceConfig(),
    worlds=WorldsConfig(names=["reinvent_base"], chunk_steps=1_000_000, rotations=1),
    # The Tier-1 trace — the whole point of this validation run.
    trace=TraceConfig(enabled=True),
    training=TrainingConfig(
        # total_timesteps is intentionally huge so the wall-clock cap governs;
        # each trial trains for max_train_seconds then stops cleanly.
        total_timesteps=1_000_000,
        max_train_seconds=900,        # ~15 min/trial -> ~45-50 min for 3 trials
        checkpoint_freq=20_000,
        eval_freq=5_000,
        n_eval_episodes=2,
    ),
    tracking=TrackingConfig(mlflow_experiment="trace-smoke"),
    seed=42,
)


def search_space(trial) -> dict:
    # Minimal sweep — just enough to give Optuna two knobs to vary across trials.
    return {
        "trainer.kwargs.learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 5e-4, log=True
        ),
        "trainer.kwargs.ent_coef": trial.suggest_float("ent_coef", 1e-3, 5e-2, log=True),
    }


# Host-side metadata pre-gen reads the action space off this attribute.
experiment = base


if __name__ == "__main__":
    study(
        base,
        search_space,
        study_name="trace_smoke",
        n_trials=3,
        n_parallel=1,
    )
