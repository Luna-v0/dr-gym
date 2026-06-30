"""Smoke test for the GPU CAMERA path on ROS 2 Lyrical / Gazebo Jetty.

Builds the single-car camera env (camera_obs=True), which renders the car's
front camera on the GPU (EGL, --headless-rendering) and feeds grayscale frames as
the observation. Exercises reset()+step() and a few SB3-free steps to confirm
real camera frames arrive (no DoubleBuffer timeout). Run with --gpus all and
GYM_DR_RENDER=1. Throwaway.
"""
from __future__ import annotations

import time

import numpy as np

from gym_dr.config import ExperimentConfig
from gym_dr.envs.dispatch import build_env


def _shapes(obs):
    if isinstance(obs, dict):
        return {k: np.asarray(v).shape for k, v in obs.items()}
    return np.asarray(obs).shape


def main() -> int:
    exp = ExperimentConfig(name="smoke-camera", camera_obs=True, n_cars=1)
    env = build_env(exp)
    print("[cam] obs_space:", env.observation_space, flush=True)

    t0 = time.monotonic()
    obs, info = env.reset()
    print(f"[cam] reset OK in {time.monotonic()-t0:.1f}s; obs shapes={_shapes(obs)}", flush=True)

    for i in range(15):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if i < 2 or i == 14:
            arr = obs if not isinstance(obs, dict) else next(iter(obs.values()))
            arr = np.asarray(arr)
            print(f"[cam] step {i:2d} reward={float(r):+.3f} term={term} "
                  f"obs[min/mean/max]={arr.min()}/{arr.mean():.1f}/{arr.max()}", flush=True)
        if term or trunc:
            obs, info = env.reset()
    env.close()
    print("[cam] CAMERA SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
