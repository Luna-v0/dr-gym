"""Smoke test for the ROS 2 Lyrical / Gazebo Jetty integration.

Builds the real dr-gym feature-vector single-car env (no rendering) via the
factory and exercises reset()+step() against the live ROS 2 sim launched by the
container CMD. Validates the whole boundary: container -> ros2 launch
deepracer_env.launch.py -> deepracer_env.DeepRacerEnv -> gym loop. Throwaway.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from gym_dr.config import ExperimentConfig
from gym_dr.envs.dispatch import build_env


def main() -> int:
    print("[smoke] building feature single-car env (camera_obs=False, n_cars=1)", flush=True)
    exp = ExperimentConfig(name="smoke-lyrical", camera_obs=False, n_cars=1)
    env = build_env(exp)
    print("[smoke] obs_space:", env.observation_space, "act_space:", env.action_space, flush=True)

    t0 = time.monotonic()
    obs, info = env.reset()
    print(f"[smoke] reset OK in {time.monotonic()-t0:.1f}s; obs shape="
          f"{np.asarray(obs if not isinstance(obs, dict) else obs.get('actor', obs)).shape}", flush=True)

    n = int(os.getenv("SMOKE_STEPS", "30"))
    rsum = 0.0
    for i in range(n):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        rsum += float(r)
        if i < 3 or i == n - 1:
            print(f"[smoke] step {i:3d} reward={float(r):+.3f} term={term} trunc={trunc}", flush=True)
        if term or trunc:
            obs, info = env.reset()
    print(f"[smoke] stepped {n} times, reward_sum={rsum:.2f}", flush=True)
    env.close()
    print("[smoke] SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
