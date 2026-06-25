"""Collect a supervised perception dataset (W-perception) — camera frame stacks
paired with free sim ground-truth labels (`gym_dr.perception.perception_targets`).

Needs the sim container (a live Gazebo + deepracer_env). Drives the car with the
privileged pure-pursuit controller from `scripts/scripted_baseline.py` so the data
is mostly on-track, but injects **epsilon-random** actions so the set also covers
near-edge / off-heading states — exactly the situations the perception net must
get right (the costs in `gym_dr/costs.py` fire there). Save as one `.npz`:

    obs      uint8  (N, 4, 120, 160)   the grayscale frame stack the policy sees
    targets  f32    (N, 6)             frame-local labels (PERCEPTION_FEATURES)
    features str array                 the feature column names

Run (inside the sim container):
    uv run python scripts/collect_perception_data.py --world Oval_track \
        --episodes 20 --epsilon 0.25 --out artifacts/perception/oval.npz

Collect across several worlds (varied geometry) and concatenate — the net should
see many tracks, like the policy. Then train with experiments/train_perception.py.
"""
from __future__ import annotations

import argparse
import os
import random
from collections import deque


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="Oval_track")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--epsilon", type=float, default=0.25,
                    help="probability of a random action (edge/off-heading coverage)")
    ap.add_argument("--speed", type=float, default=1.8)
    ap.add_argument("--lookahead", type=int, default=5)
    ap.add_argument("--steer-sign", type=float, default=1.0, choices=[1.0, -1.0])
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", action="store_true", default=True,
                    help="randomize the camera obs during collection so perception "
                         "trains on the noisy frames it'll face (labels stay ground-truth)")
    ap.add_argument("--no-dr", dest="dr", action="store_false")
    ap.add_argument("--out", default="artifacts/perception/dataset.npz")
    args = ap.parse_args()

    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"[collect] numpy missing: {exc}")
        return 2

    from gym_dr.perception import ALL_FEATURES, all_targets
    from scripts.scripted_baseline import pure_pursuit_action

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # Capture the privileged params the env hands the reward callback.
    latest: dict = {}

    def _record(params: dict) -> float:
        latest.clear()
        latest.update(params)
        return 0.0

    try:
        from gym_dr.config import ExperimentConfig
        from gym_dr.envs.time_trial import time_trial
    except Exception as exc:  # noqa: BLE001
        print(f"[collect] needs the sim container (deepracer_env): {exc}")
        return 2

    dr = None
    if args.dr:
        from gym_dr.domain_randomization import DomainRandomizationConfig

        # Static (non-ADR) obs noise so every frame is perturbed during
        # collection — matches the deployed policy's camera conditions. Actuator
        # noise also widens the visited-state distribution (more edge coverage).
        dr = DomainRandomizationConfig(
            obs_gaussian_std=8.0, obs_brightness_jitter=0.15,
            actuator_steering_std=2.0, actuator_speed_std=0.1, seed=args.seed,
        )
    experiment = ExperimentConfig(name="perception_collect", reward=_record,
                                  domain_randomization=dr)
    env = time_trial(experiment)
    if hasattr(env, "set_world"):
        env.set_world(args.world)

    # Locate the (single) grayscale image key in the Dict obs.
    import gymnasium as gym

    image_key = None
    for key, space in env.observation_space.spaces.items():
        if isinstance(space, gym.spaces.Box) and len(space.shape) == 3:
            image_key = key
            break
    if image_key is None:
        print("[collect] no image key in observation space; aborting")
        return 2

    def _gray_frame(obs) -> "np.ndarray":
        # (H, W, 1) uint8 -> (H, W) uint8
        return np.asarray(obs[image_key], dtype=np.uint8).squeeze(-1)

    obs_buf: list = []
    tgt_buf: list = []
    lo = env.action_space.low
    hi = env.action_space.high

    for ep in range(args.episodes):
        obs, _info = env.reset(seed=args.seed + ep)
        latest.clear()
        frames = deque(
            [_gray_frame(obs)] * args.frame_stack, maxlen=args.frame_stack
        )
        prev_params: dict | None = None
        action = None
        for _ in range(args.max_steps):
            if action is None or rng.random() < args.epsilon:
                action = np_rng.uniform(lo, hi).astype(np.float32)
            obs, _r, terminated, truncated, _info = env.step(action)
            frames.append(_gray_frame(obs))
            if latest:
                stack = np.stack(frames, axis=0)  # (4, H, W) uint8
                tgt = all_targets(latest, prev_params)  # core ⊕ dynamic candidates
                obs_buf.append(stack.astype(np.uint8))
                tgt_buf.append(tgt)
                prev_params = dict(latest)
                # next deliberate action from the privileged controller
                if rng.random() >= args.epsilon:
                    action = np.asarray(
                        pure_pursuit_action(
                            latest, lookahead=args.lookahead, speed=args.speed,
                            steer_sign=args.steer_sign, steer_gain=1.0,
                        ),
                        dtype=np.float32,
                    )
                    # the policy may act in [-1,1] (NormalizeActions); the env
                    # clips, so an eng-unit command is still valid input here.
            if terminated or truncated:
                break
        print(f"[ep {ep}] collected so far: {len(obs_buf)} samples")

    env.close()
    if not obs_buf:
        print("[collect] no samples captured; aborting")
        return 1

    obs_arr = np.stack(obs_buf, axis=0)
    tgt_arr = np.stack(tgt_buf, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, obs=obs_arr, targets=tgt_arr,
        features=np.array(ALL_FEATURES),
    )
    print(f"[collect] wrote {obs_arr.shape[0]} samples -> {args.out}  "
          f"(obs {obs_arr.shape}, targets {tgt_arr.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
