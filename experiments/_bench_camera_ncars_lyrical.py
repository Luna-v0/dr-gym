"""Benchmark camera-obs throughput vs n_cars (find the OGRE render ceiling).

The camera path renders N car cameras on one OGRE thread; this measures where
that stops scaling. Steps a CameraObs VecEnv for N cars and reports aggregate
agent-steps/s + camera keep-up (a DoubleBuffer timeout = the render fell behind).
Run with --gpus all, GYM_DR_RENDER=1, GYM_DR_ALLOW_CAMERA_NCARS=1. Throwaway.
"""
from __future__ import annotations

import os
import time

import numpy as np

from gym_dr.config import ExperimentConfig
from gym_dr.environment import CameraObs, EnvironmentConfig
from gym_dr.envs.dispatch import build_env

os.environ.setdefault("GYM_DR_ALLOW_CAMERA_NCARS", "1")


def main() -> int:
    n_cars = int(os.getenv("N_CARS", "2"))
    n_steps = int(os.getenv("BENCH_STEPS", "150"))
    exp = ExperimentConfig.from_environment(EnvironmentConfig(observation=CameraObs(), n_cars=n_cars),
        name="cam-ncars-bench",
    )
    env = build_env(exp)
    n = env.num_envs
    env.reset()
    act = np.stack([env.action_space.sample() for _ in range(n)])
    for _ in range(8):  # warmup
        env.step(act)

    t0 = time.monotonic()
    for _ in range(n_steps):
        env.step(act)
    wall = time.monotonic() - t0
    line = (f"[cambench] n_cars={n_cars} env_steps={n_steps} wall={wall:.2f}s "
            f"| env_steps/s={n_steps/wall:.1f} | AGENT_steps/s={n_steps*n/wall:.1f} "
            f"| ms/env_step={wall/n_steps*1000:.0f}")
    print(line, flush=True)
    out = os.getenv("BENCH_OUT")
    if out:
        with open(out, "a") as fh:
            fh.write(line + "\n")
    env.close()
    print("[cambench] CAMBENCH DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
