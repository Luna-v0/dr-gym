"""Profile DeepRacerEnv.reset() to locate the ~1.3s/reset cost (seam refactor target).

Runs N resets under cProfile (cumulative + internal time) and prints the hottest
frames, then a wall-clock mean. Feature obs (camera-off, no rendering). Throwaway.
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

from gym_dr.config import ExperimentConfig
from gym_dr.envs.dispatch import build_env


def main() -> int:
    exp = ExperimentConfig(name="prof-reset", camera_obs=False, n_cars=1)
    env = build_env(exp)
    env.reset()  # warmup (first reset pays one-time costs)

    N = 8
    # wall-clock per reset
    times = []
    for _ in range(N):
        t0 = time.monotonic()
        env.reset()
        times.append(time.monotonic() - t0)
    print(f"[prof] wall-clock reset: mean={sum(times)/len(times):.3f}s "
          f"min={min(times):.3f}s max={max(times):.3f}s over {N}", flush=True)

    # cProfile a second batch
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(N):
        env.reset()
    pr.disable()

    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    print("\n[prof] ===== top by CUMULATIVE time =====", flush=True)
    st.sort_stats("cumulative").print_stats(25)
    print(s.getvalue(), flush=True)

    s2 = io.StringIO()
    st2 = pstats.Stats(pr, stream=s2)
    print("[prof] ===== top by INTERNAL (tottime) =====", flush=True)
    st2.sort_stats("tottime").print_stats(20)
    print(s2.getvalue(), flush=True)

    env.close()
    print("[prof] PROFILE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
