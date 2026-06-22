"""Validate FSRL PPO-Lagrangian (PID) on Safety-Gymnasium — the algorithm-trust
step before DeepRacer (D9, docs/reports/safe-rl-backend.md). NO sim needed.

This is the cheap, turnkey check that PID-Lagrangian PPO reproduces known safe-RL
behaviour (reward up while episode COST converges under the limit) on a standard
CMDP task — before we pay for the DeepRacer integration (FsrlTrainer custom CNN).

Install (FSRL/Tianshou pins clash with the main SB3 env, so use a SEPARATE venv):
    python3.10 -m venv .venv-safe && . .venv-safe/bin/activate
    pip install fast-safe-rl safety-gymnasium

Run:
    python scripts/validate_fsrl_safetygym.py --task SafetyPointGoal1-v0 --epoch 100

Watch the FSRL TensorBoard logs (logs/fsrl_validate): episode cost should track
toward --cost-limit while reward improves.
"""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="SafetyPointGoal1-v0")
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--train-envs", type=int, default=10)
    args = ap.parse_args()

    import gymnasium as gym
    import safety_gymnasium  # noqa: F401  — registers the Safety* tasks
    from tianshou.env import DummyVectorEnv
    from fsrl.agent import PPOLagAgent
    from fsrl.utils import TensorboardLogger

    logger = TensorboardLogger("logs/fsrl_validate", log_txt=True, name=args.task)
    agent = PPOLagAgent(gym.make(args.task), logger)
    train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.train_envs)])
    test_envs = DummyVectorEnv([lambda: gym.make(args.task)])
    agent.learn(train_envs, test_envs, epoch=args.epoch)
    print("done — inspect logs/fsrl_validate: cost should converge under the limit while reward rises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
