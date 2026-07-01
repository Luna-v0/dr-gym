"""Benchmark: does GPU policy inference improve CAMERA training throughput?

Runs one short camera PPO rollout+update (the trial-18 DeepRacerCNN) with the
policy on GYM_DR_DEVICE (cpu or cuda) and reports wall time + SB3 fps. The sim +
GPU camera render are identical between runs, so the delta isolates the CNN
forward (rollout) + backward (update) device cost. Run with --gpus all and
GYM_DR_RENDER=1; switch GYM_DR_DEVICE between runs. Throwaway.
"""
from __future__ import annotations

import os
import time

from gym_dr import (
    ContinuousActionSpaceConfig,
    EnvironmentConfig,
    ExperimentConfig,
    CameraObs,
    Sb3Trainer,
    TrainingConfig,
    Study,
)
from gym_dr.networks import DeepRacerCNN

CONV_LAYERS = [[16, 8, 4], [32, 4, 2], [32, 5, 1], [32, 5, 1], [32, 5, 1]]
DEVICE = os.getenv("GYM_DR_DEVICE", "cpu")


def main() -> int:
    exp = ExperimentConfig(
        name="bench-cam-train",
        environment=EnvironmentConfig(
            observation=CameraObs(), n_cars=2,
            action_space=ContinuousActionSpaceConfig(normalize_actions=True)),
        trainer=Sb3Trainer(
            name="ppo", policy="MultiInputPolicy", device=DEVICE,
            kwargs={
                "n_steps": 256, "batch_size": 64, "n_epochs": 6,
                "policy_kwargs": {
                    "share_features_extractor": False,
                    "normalize_images": False,
                    "features_extractor_class": DeepRacerCNN,
                    "features_extractor_kwargs": {"conv_layers": CONV_LAYERS, "features_dim": 256},
                    "net_arch": {"pi": [1024, 1024, 1024], "vf": [1024, 1024, 1024]},
                },
            }),
        training=TrainingConfig(total_timesteps=512, checkpoint_freq=10**9, eval_freq=10**9),
    )
    t0 = time.monotonic()
    Study(exp).run()
    print(f"[camtrain] device={DEVICE} total_wall={time.monotonic()-t0:.1f}s "
          f"(512 steps, trial-18 DeepRacerCNN, n_cars=2)", flush=True)
    print("[camtrain] CAMTRAIN DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
