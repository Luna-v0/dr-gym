"""Throughput benchmark: steps/s for the feature (camera-off) env.

Drives a gentle near-straight action to keep episodes long (so we measure step
throughput, not reset overhead), runs BENCH_STEPS steps, and reports aggregate
steps/s. n_cars from N_CARS (1 = single-car env; >1 = multi-car VecEnv). RTF is
set by the launch (RTF_OVERRIDE). Throwaway benchmark harness.
"""
from __future__ import annotations

import os
import time

import numpy as np

from gym_dr.config import ExperimentConfig
from gym_dr.environment import EnvironmentConfig, FeatureObs
from gym_dr.envs.dispatch import build_env


def main() -> int:
    n_cars = int(os.getenv("N_CARS", "1"))
    n_steps = int(os.getenv("BENCH_STEPS", "300"))
    exp = ExperimentConfig(
        name="bench",
        environment=EnvironmentConfig(observation=FeatureObs(), n_cars=n_cars),
    )
    env = build_env(exp)
    env.reset()

    # near-straight, moderate speed -> stays on track longer (fewer resets).
    if n_cars > 1:
        act = np.tile(np.array([0.0, -0.3], dtype=np.float32), (n_cars, 1))
    else:
        act = np.array([0.0, -0.3], dtype=np.float32)

    # warm up a few steps (JIT/first-frame costs), then time.
    for _ in range(10):
        out = env.step(act)
        _done = out[2] if n_cars > 1 else (out[2] or out[3])
        if n_cars == 1 and (out[2] or out[3]):
            env.reset()

    # Default: reset-inclusive (valid throughput). NOTE: stepping past 'done'
    # without reset is a degenerate no-op in gymnasium (step short-circuits) — it
    # reports a bogus ~2000 steps/s, so don't use it to "isolate" step cost.
    no_reset = os.getenv("BENCH_NO_RESET", "0") == "1"
    t0 = time.monotonic()
    env_steps = resets = 0
    for _ in range(n_steps):
        out = env.step(act)
        env_steps += 1
        if n_cars == 1 and not no_reset and (out[2] or out[3]):
            env.reset(); resets += 1
    wall = time.monotonic() - t0

    agent_sps = env_steps * n_cars / wall  # aggregate (per-agent) steps/s
    line = (f"[bench] n_cars={n_cars} RTF={os.getenv('RTF_OVERRIDE','(launch default)')} "
            f"env_steps={env_steps} resets={resets} wall={wall:.2f}s "
            f"| env_steps/s={env_steps/wall:.1f} | AGENT_steps/s={agent_sps:.1f}")
    print(line, flush=True)
    # Persist to a mounted file too, so the host harness never races container log
    # removal. BENCH_OUT is an absolute path inside the container (a bind mount).
    out_path = os.getenv("BENCH_OUT")
    if out_path:
        with open(out_path, "a") as fh:
            fh.write(line + "\n")
    env.close()
    print("[bench] BENCH DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
