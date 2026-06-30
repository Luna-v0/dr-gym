"""Tiny end-to-end TRAINING smoke for the ROS 2 Lyrical / Jetty integration.

Runs the real gym_dr training pipeline (SB3 PPO + trainer + callbacks) for a
single short PPO update against the live sim — validates the whole stack beyond
env reset/step. Feature obs (camera-off, no rendering). Throwaway.
"""
from __future__ import annotations

from gym_dr import (
    ContinuousActionSpaceConfig,
    EnvironmentConfig,
    ExperimentConfig,
    FeatureObs,
    Sb3Trainer,
    TrainingConfig,
    train,
)


def main() -> int:
    exp = ExperimentConfig(
        name="smoke-train-lyrical",
        environment=EnvironmentConfig(
            # asymmetric_critic=True -> Dict{actor,critic} obs, matching the real
            # feature experiments (oracle_hpo) so the trainer's MultiInputPolicy
            # fits (a plain Box would need MlpPolicy).
            observation=FeatureObs(asymmetric_critic=True),
            action_space=ContinuousActionSpaceConfig(normalize_actions=True),
            n_cars=1,
        ),
        # Tiny PPO: one short rollout + update. The sim runs ~real-time (no RTF
        # accel wired yet), so keep n_steps small or the rollout outlasts the
        # smoke timeout. eval/checkpoint effectively disabled.
        trainer=Sb3Trainer(name="ppo", policy="MultiInputPolicy",
                            kwargs={"n_steps": 128, "batch_size": 32, "n_epochs": 2}),
        training=TrainingConfig(
            total_timesteps=140,
            checkpoint_freq=10_000_000,
            eval_freq=10_000_000,
            n_eval_episodes=1,
        ),
    )
    train(exp)
    print("[smoke-train] TRAIN SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
