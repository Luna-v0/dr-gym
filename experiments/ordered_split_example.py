"""Train/eval track split via the world-strategy pattern.

`OrderedSplit` trains the policy on one ordered list of worlds and evaluates it
on a *different*, held-out ordered list — so the eval reward measures
generalisation to tracks the policy never trained on. Training proceeds strictly
in `train_worlds` order, hot-swapping the Gazebo track between chunks; at each
evaluation the policy is measured on every world in `eval_worlds` (per-world
eval metrics are logged as `eval/<world>_mean_reward`, and their mean drives the
best-model + Optuna signal).

Run it like any other experiment:

    ./run_cpu_training.sh experiments/ordered_split_example.py

To plug in a different schedule later, implement `gym_dr.worlds.WorldStrategy`
and pass it as `world_strategy=` — nothing else changes.
"""
from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    OrderedSplit,
    Sb3Trainer,
    TraceConfig,
    TrackingConfig,
    TrainingConfig,
    center_line,
)

experiment = ExperimentConfig(
    name="ordered_split",
    trainer=Sb3Trainer(name="ppo", policy="MultiInputPolicy", device="cpu"),
    reward=center_line,
    action_space=ContinuousActionSpaceConfig(),
    # The whole world schedule lives here. Train on these three, in order;
    # evaluate generalisation on two held-out tracks the policy never sees.
    world_strategy=OrderedSplit(
        train_worlds=["reinvent_base", "Bowtie_track", "Oval_track"],
        eval_worlds=["reInvent2019_track", "Spain_track"],
        chunk_steps=50_000,
        rotations=2,            # train order repeats: base→Bowtie→Oval ×2
    ),
    trace=TraceConfig(enabled=True),
    training=TrainingConfig(total_timesteps=300_000, eval_freq=10_000, n_eval_episodes=3),
    tracking=TrackingConfig(mlflow_experiment="ordered-split"),
)


if __name__ == "__main__":
    from gym_dr import train

    train(experiment)
