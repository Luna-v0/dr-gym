"""Random-action baseline — the lower bound for the env (W1).

Samples uniform random actions for a few episodes and reports progress, so the
scripted pure-pursuit baseline (scripts/scripted_baseline.py) has something to
beat. A sane env: random ≪ scripted ≪ a trained policy.

Run INSIDE the sim container:

    uv run python scripts/random_baseline.py --world Oval_track --episodes 3
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="Oval_track")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()

    try:
        from deepracer_env.environments.deepracer_env import DeepRacerEnv
    except Exception as exc:  # noqa: BLE001
        print(f"[random_baseline] needs the sim container (deepracer_env): {exc}")
        return 2

    from gym_dr.envs.wrappers import ActionBounds

    latest: dict = {}

    def _record(params: dict) -> float:
        latest.clear()
        latest.update(params)
        return 0.0

    env = DeepRacerEnv(reward_fn=_record, sensors=["FRONT_FACING_CAMERA"])
    env = ActionBounds(env, steering_low=-30.0, steering_high=30.0,
                       speed_low=1.0, speed_high=4.0)
    if hasattr(env, "set_world"):
        env.set_world(args.world)

    progresses = []
    for ep in range(args.episodes):
        env.reset()
        latest.clear()
        max_progress, steps = 0.0, 0
        for _ in range(args.max_steps):
            _obs, _r, terminated, truncated, _info = env.step(env.action_space.sample())
            steps += 1
            if latest:
                max_progress = max(max_progress, float(latest.get("progress", 0.0)))
            if terminated or truncated:
                break
        progresses.append(max_progress)
        print(f"[ep {ep}] progress={max_progress:5.1f}%  steps={steps}")

    env.close()
    print(f"\n[verdict] random mean progress {sum(progresses)/max(1,len(progresses)):.1f}% on {args.world}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
