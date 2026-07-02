"""Tiny 2-car TRAINING smoke — validates the BATCHED multi-car reset (B7 §1.1).

Exercises ``MultiAgentDeepRacerEnv.reset()``'s new batched ``set_pose_vector``
teleport (both cars teleported in one gz tick instead of 2 sequential blocking
``set_model_state`` calls) end-to-end on the live ROS 2 Lyrical / Jetty sim.
Feature obs (camera-off, no render → no storm), tiny budget. Throwaway.
"""
from __future__ import annotations

from gym_dr import (
    ContinuousActionSpaceConfig,
    EnvironmentConfig,
    ExperimentConfig,
    FeatureObs,
    FixedWorlds,
    Sb3Trainer,
    Study,
    TrainingConfig,
)


def main() -> int:
    exp = ExperimentConfig.from_environment(
        EnvironmentConfig(
            observation=FeatureObs(asymmetric_critic=True),
            action_space=ContinuousActionSpaceConfig(normalize_actions=True),
            # Short chunk so the rotation runs ~300 steps, not the 50k default.
            curriculum=FixedWorlds(chunk_steps=300),
            n_cars=2,
        ),
        name="smoke-multicar-lyrical",
        trainer=Sb3Trainer(name="ppo", policy="MultiInputPolicy",
                            kwargs={"n_steps": 64, "batch_size": 32, "n_epochs": 2}),
        training=TrainingConfig(total_timesteps=160, checkpoint_freq=10 ** 9,
                                eval_freq=10 ** 9, n_eval_episodes=1),
    )
    result = Study(exp).run()
    # A real gate: train() returns the run's latest_model path on rc=0, None if the
    # container aborted (e.g. a reset crash). Don't print OK on an abort.
    if not result.run_paths or result.run_paths[0] is None:
        print("[smoke-multicar] MULTICAR TRAIN SMOKE FAILED (container aborted)", flush=True)
        return 1
    print("[smoke-multicar] MULTICAR TRAIN SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
