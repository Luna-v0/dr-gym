"""Profile DeepRacerEnv.step() to see if the per-step path also spawns gz-CLI
subprocesses (the training hot loop). Feature obs (camera-off). Throwaway."""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np

from gym_dr.config import ExperimentConfig
from gym_dr.envs.dispatch import build_env


def main() -> int:
    exp = ExperimentConfig(name="prof-step", camera_obs=False, n_cars=1)
    env = build_env(exp)
    env.reset()

    # gentle near-straight slow action to stay on-track longer (fewer resets),
    # so we measure mostly pure step cost.
    act = np.array([0.0, -1.0], dtype=np.float32)  # normalized: ~0 steer, low speed

    N = 60
    pr = cProfile.Profile()
    t0 = time.monotonic()
    pr.enable()
    steps = resets = 0
    for _ in range(N):
        _o, _r, term, trunc, _i = env.step(act)
        steps += 1
        if term or trunc:
            env.reset()
            resets += 1
    pr.disable()
    wall = time.monotonic() - t0
    print(f"[prof] {steps} steps ({resets} resets) in {wall:.2f}s -> "
          f"{steps/wall:.1f} steps/s, {wall/steps*1000:.0f} ms/step", flush=True)

    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    st.sort_stats("cumulative").print_stats(22)
    print(s.getvalue(), flush=True)
    env.close()
    print("[prof] STEP PROFILE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
