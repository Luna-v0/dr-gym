"""Offline reward ranking — filter reward functions WITHOUT training (no sim).

The D3 baseline converged to "floor it, crash at ~28%". Before spending sim time
on a training-based reward search, we can cheaply check the *shape* of each
candidate: a good reward must give a **clean full lap** a higher episode return
than a **fast early crash** (and than a slow crawl, and a zig-zag). This builds
synthetic trajectories with known reward_params and sums each reward over them.

    uv run --no-sync python scripts/reward_ranking.py

Reading: a reward whose argmax trajectory is `clean_lap` (and where
`clean_lap > fast_crash`) has the right shape and is worth putting in the
training search; one where `fast_crash` wins reproduces the D3 failure and
should be dropped. This is a necessary, not sufficient, filter.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def _make_track(n: int = 60) -> List[tuple]:
    """A closed-ish centerline: a straight, a 90-degree bend, another straight."""
    pts = []
    # straight east
    for i in range(n // 3):
        pts.append((float(i), 0.0))
    cx, cy = float(n // 3), 1.0  # arc centre
    r = 1.0
    for k in range(n // 3):
        a = -math.pi / 2 + (math.pi / 2) * (k / (n // 3))
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    # straight north
    x_top = pts[-1][0]
    for j in range(n - len(pts)):
        pts.append((x_top, 1.0 + float(j)))
    return pts


_WPS = _make_track()
_TW = 1.0  # track width


def _params_at(i: int, *, progress: float, steps: int, offset: float, speed: float,
               steering: float, heading_err_deg: float, offtrack: bool) -> dict:
    n = len(_WPS)
    prev_i, next_i = i % n, (i + 1) % n
    x0, y0 = _WPS[prev_i]
    x1, y1 = _WPS[next_i]
    tangent = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return {
        "all_wheels_on_track": not offtrack,
        "is_offtrack": offtrack,
        "distance_from_center": abs(offset),
        "is_left_of_center": offset < 0,
        "track_width": _TW,
        "heading": tangent + heading_err_deg,
        "waypoints": _WPS,
        "closest_waypoints": [prev_i, next_i],
        "progress": progress,
        "steps": steps,
        "speed": speed,
        "steering_angle": steering,
    }


def _trajectory(kind: str) -> List[dict]:
    n = len(_WPS)
    traj: List[dict] = []
    if kind == "clean_lap":               # full lap, centered, moderate speed, smooth
        T = n
        for i in range(T):
            traj.append(_params_at(i, progress=100.0 * (i + 1) / T, steps=i + 1, offset=0.03,
                                   speed=2.2, steering=3.0, heading_err_deg=4.0, offtrack=False))
    elif kind == "fast_crash":            # floor it, drift off, crash at ~28%
        T = int(0.28 * n)
        for i in range(T):
            off = 0.05 + 0.45 * (i / T)    # drifts toward the edge
            last = i == T - 1
            traj.append(_params_at(i, progress=100.0 * i / n, steps=i + 1, offset=off,
                                   speed=3.9, steering=12.0, heading_err_deg=20.0, offtrack=last))
    elif kind == "crawl":                 # full lap but slow (many steps)
        T = n * 3
        for i in range(T):
            traj.append(_params_at(i % n, progress=100.0 * (i + 1) / T, steps=i + 1, offset=0.03,
                                   speed=1.0, steering=3.0, heading_err_deg=4.0, offtrack=False))
    elif kind == "zigzag":                # full lap, on track, swervy
        T = n
        for i in range(T):
            traj.append(_params_at(i, progress=100.0 * (i + 1) / T, steps=i + 1,
                                   offset=0.2 * math.sin(i), speed=2.0,
                                   steering=25.0 * (1 if i % 2 else -1), heading_err_deg=8.0,
                                   offtrack=False))
    else:
        raise ValueError(kind)
    return traj


def _episode_return(reward_fn, traj: List[dict]) -> float:
    return float(sum(reward_fn(p) for p in traj))


def main() -> int:
    from gym_dr.rewards import REWARD_VARIANTS

    kinds = ["clean_lap", "fast_crash", "crawl", "zigzag"]
    trajs = {k: _trajectory(k) for k in kinds}

    print(f"{'reward':24s} " + " ".join(f"{k:>11s}" for k in kinds) + "   verdict")
    print("-" * 90)
    good: List[str] = []
    for name, fn in REWARD_VARIANTS.items():
        rets: Dict[str, float] = {k: _episode_return(fn, trajs[k]) for k in kinds}
        best = max(rets, key=rets.get)
        prefers_clean = best == "clean_lap" and rets["clean_lap"] > rets["fast_crash"]
        if prefers_clean:
            good.append(name)
        verdict = "OK clean-first" if prefers_clean else (
            "BAD: " + ("fast_crash wins" if best == "fast_crash" else f"{best} wins"))
        print(f"{name:24s} " + " ".join(f"{rets[k]:11.1f}" for k in kinds) + f"   {verdict}")

    print("-" * 90)
    print(f"shape-OK rewards (clean_lap is argmax AND > fast_crash): {good or '(none!)'}")
    print("Note: a necessary filter on reward SHAPE, not a substitute for the training search.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
