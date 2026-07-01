"""Smoke test: the perception dataset recorder actually collects camera->feature
data during a multi-car CAMERA rollout (the camera_cnn_dataset path).

Builds the n_cars=2 camera VecEnv with GYM_DR_PERCEPTION_OUT set, steps long
enough for episodes to flush shards, then loads one shard and verifies it holds
frames + feature targets + actions + DR meta. Needs --gpus all + GYM_DR_RENDER=1.
Throwaway.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from gym_dr.config import ExperimentConfig
from gym_dr.environment import CameraObs, EnvironmentConfig
from gym_dr.envs.dispatch import build_env

OUT = os.environ.setdefault("GYM_DR_PERCEPTION_OUT", "/out")


def main() -> int:
    exp = ExperimentConfig.from_environment(EnvironmentConfig(observation=CameraObs(), n_cars=2),
        name="rec-smoke",
    )
    env = build_env(exp)
    n = env.num_envs
    print(f"[rec] camera VecEnv n_envs={n}, recorder out={OUT}", flush=True)
    env.reset()
    n_steps = int(os.getenv("REC_STEPS", "80"))  # raise to reproduce the long-run crash
    for i in range(n_steps):
        actions = np.stack([env.action_space.sample() for _ in range(n)])
        env.step(actions)
        if i % 200 == 0 and i:
            print(f"[rec] step {i}/{n_steps} still alive", flush=True)
    env.close()

    shards = sorted(glob.glob(os.path.join(OUT, "**", "*.npz"), recursive=True))
    print(f"[rec] shards written: {len(shards)}", flush=True)
    if not shards:
        print("[rec] RECORDER SMOKE: NO SHARDS (collection FAILED)", flush=True)
        return 1
    d = np.load(shards[0], allow_pickle=True)
    print(f"[rec] shard0 keys: {list(d.keys())}", flush=True)
    print(f"[rec]   frames  {d['frames'].shape} {d['frames'].dtype}", flush=True)
    print(f"[rec]   targets {d['targets'].shape} cols={list(d['features'])}", flush=True)
    print(f"[rec]   actions {d['actions'].shape} cols={list(d['action_cols'])}", flush=True)
    print(f"[rec]   diag    {d['diag'].shape} cols={list(d['diag_cols'])}", flush=True)
    print(f"[rec]   meta    {str(d['meta'])[:240]}", flush=True)
    print("[rec] RECORDER SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
