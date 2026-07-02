"""n=5 CAMERA reset-storm validation for the batched reset (B7 §1.1).

Per D10 (docs/reports/camera-multicar-reset-storm.md) n>=5 camera training
reset-storms: the N sequential blocking per-car teleports cluster in a window, the
single-thread OGRE render falls behind, its DoubleBuffer times out, and the watchdog
kills the container. With the batched ``set_pose_vector`` reset (one gz tick instead
of N blocking round-trips) the reset window collapses — does the storm go away?

Runs a random-policy n=5 CAMERA training (GPU render) long enough to cluster many
resets. If it trains to completion → the storm is gone. If the container is killed
(watchdog / rc!=0) → the storm persists (need §1.4 gzserver / render throttling).
Throwaway. Needs the GPU image + --gpus (use_gpu=True) + GYM_DR_ALLOW_CAMERA_NCARS=1.
"""
from __future__ import annotations

import os

# Clear the dr-gym camera n>2 guard (module level so the re-imported container clears
# it too); the Lyrical launch spawns racecar_0..N with cameras.
os.environ["GYM_DR_ALLOW_CAMERA_NCARS"] = "1"

from gym_dr import (  # noqa: E402
    CameraObs,
    ContinuousActionSpaceConfig,
    EnvironmentConfig,
    ExperimentConfig,
    FixedWorlds,
    Sb3Trainer,
    Study,
    TrainingConfig,
)


def main() -> int:
    exp = ExperimentConfig.from_environment(
        EnvironmentConfig(
            observation=CameraObs(),
            action_space=ContinuousActionSpaceConfig(normalize_actions=True),
            # Enough steps that many episodes end + reset under camera render.
            curriculum=FixedWorlds(chunk_steps=800),
            n_cars=int(os.getenv("GYM_DR_STORM_NCARS", "5")),  # 5 = D10 storm boundary; 8 = original bug
        ),
        name="smoke-camera-storm",
        trainer=Sb3Trainer(name="ppo", policy="MultiInputPolicy", device="cpu",
                            kwargs={"n_steps": 64, "batch_size": 32, "n_epochs": 1}),
        training=TrainingConfig(total_timesteps=800, checkpoint_freq=10 ** 9,
                                eval_freq=10 ** 9, n_eval_episodes=1),
        use_gpu=True,  # --gpus all for the gz camera render
    )
    n_cars = int(os.getenv("GYM_DR_STORM_NCARS", "5"))
    result = Study(exp).run()
    if not result.run_paths or result.run_paths[0] is None:
        print(f"[smoke-camera-storm] CAMERA STORM SMOKE FAILED at n={n_cars} (container aborted/killed)",
              flush=True)
        return 1
    print(f"[smoke-camera-storm] CAMERA STORM SURVIVED — n={n_cars} camera trained without a reset-storm",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
