"""Scripted pure-pursuit baseline — is the env+reward actually drivable? (W1)

Method (base-prompt Q1a): if a hand-written controller can drive a track *inside
the env*, the env + reward + termination are sound and any learning failure is an
RL problem; if even a privileged controller can't, the env is the bug.

This controller is **privileged**: it ignores the camera and steers from the
ground-truth reward params (pose + waypoints) that the env passes to the reward
callback. That is fine here — we're validating the *environment*, not perception.
The deployed policy never sees these (see docs/guardrails); the perception net
(W-perception) is what bridges to camera-only control.

Run INSIDE the sim container (needs deepracer_env + a live Gazebo):

    uv run python scripts/scripted_baseline.py --world Oval_track --episodes 3

If it laps the track cleanly, the env is sound. If the car veers the wrong way,
flip --steer-sign (the steering-angle sign convention is the one thing we can't
confirm offline).
"""
from __future__ import annotations

import argparse
import math


def _angle_wrap_deg(a: float) -> float:
    """Wrap degrees to [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def pure_pursuit_action(params: dict, *, lookahead: int, speed: float,
                        steer_sign: float, steer_gain: float,
                        steer_limit: float = 30.0) -> list:
    """Steer toward a waypoint `lookahead` ahead of the car; constant speed.

    Uses params: x, y, heading (deg), waypoints [(x,y)...], closest_waypoints
    [prev, next]. Returns [steering_deg, speed]."""
    wps = params.get("waypoints") or []
    closest = params.get("closest_waypoints") or [0, 0]
    if len(wps) < 2:
        return [0.0, speed]
    n = len(wps)
    target = wps[(int(closest[1]) + lookahead) % n]
    dx = float(target[0]) - float(params.get("x", 0.0))
    dy = float(target[1]) - float(params.get("y", 0.0))
    desired_deg = math.degrees(math.atan2(dy, dx))
    err = _angle_wrap_deg(desired_deg - float(params.get("heading", 0.0)))
    steer = max(-steer_limit, min(steer_limit, steer_sign * steer_gain * err))
    return [steer, speed]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="Oval_track")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--lookahead", type=int, default=5, help="waypoints ahead to aim at")
    ap.add_argument("--speed", type=float, default=1.8, help="m/s, constant")
    ap.add_argument("--steer-sign", type=float, default=1.0, choices=[1.0, -1.0])
    ap.add_argument("--steer-gain", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()

    try:
        import numpy as np
        from deepracer_env.environments.deepracer_env import DeepRacerEnv
    except Exception as exc:  # noqa: BLE001
        print(f"[scripted_baseline] needs the sim container (deepracer_env): {exc}")
        return 2

    from gym_dr.envs.wrappers import ActionBounds

    latest: dict = {}

    def _record(params: dict) -> float:
        latest.clear()
        latest.update(params)
        return 0.0  # reward irrelevant; we only want the params

    env = DeepRacerEnv(reward_fn=_record, sensors=["FRONT_FACING_CAMERA"])
    env = ActionBounds(env, steering_low=-30.0, steering_high=30.0,
                       speed_low=1.0, speed_high=4.0)
    if hasattr(env, "set_world"):
        env.set_world(args.world)

    results = []
    for ep in range(args.episodes):
        env.reset()
        latest.clear()
        max_progress, offtrack_steps, steps = 0.0, 0, 0
        action = [0.0, args.speed]  # straight until we have params
        for _ in range(args.max_steps):
            _obs, _r, terminated, truncated, _info = env.step(np.array(action, dtype=np.float32))
            steps += 1
            if latest:
                max_progress = max(max_progress, float(latest.get("progress", 0.0)))
                if latest.get("is_offtrack") or not latest.get("all_wheels_on_track", True):
                    offtrack_steps += 1
                action = pure_pursuit_action(
                    latest, lookahead=args.lookahead, speed=args.speed,
                    steer_sign=args.steer_sign, steer_gain=args.steer_gain,
                )
            if terminated or truncated:
                break
        completed = max_progress >= 99.999
        clean = completed and offtrack_steps == 0
        results.append((max_progress, offtrack_steps, steps, completed, clean))
        print(f"[ep {ep}] progress={max_progress:5.1f}%  offtrack_steps={offtrack_steps:4d}  "
              f"steps={steps:4d}  completed={completed}  clean={clean}")

    env.close()
    mean_prog = sum(r[0] for r in results) / max(1, len(results))
    n_clean = sum(1 for r in results if r[4])
    print(f"\n[verdict] mean progress {mean_prog:.1f}% · clean laps {n_clean}/{len(results)} on {args.world}")
    print("  env is DRIVABLE -> learning failure is an RL problem" if n_clean
          else "  scripted controller could NOT lap cleanly -> check steer-sign, then suspect the env/reward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
