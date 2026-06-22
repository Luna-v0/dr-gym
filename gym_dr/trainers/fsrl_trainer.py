"""FSRL PPO-Lagrangian (PID) trainer backend — D9 (adopt FSRL ``PPOLagAgent``).

FSRL's ``PPOLagAgent`` is PPO + PID-Lagrangian in one algorithm
(`docs/reports/safe-rl-backend.md`). This plugs it in as a `Trainer` backend so the
rest of the pipeline (Docker dispatch, MLflow, artifacts, curriculum, the held-out
eval) is unchanged. The constraint **cost** comes from our graded
`gym_dr/costs.py` surfaced as ``info["cost"]`` by ``CostInfoWrapper`` (FSRL/Tianshou
read it there) — no deepracer-env change needed.

STATUS: scaffold. The cost bridge + Trainer-contract wiring are in place. Two bits
must be finalized against the *installed* FSRL (lazy-imported, so this module
imports without it):
  1. **Custom CNN net** for the Dict camera obs — FSRL's default actor/critic are
     MLPs; pass a Tianshou CNN feature extractor (see FSRL ``examples/customized``).
     The Safety-Gymnasium validation (vector obs) needs no custom net — do that
     first (`scripts/validate_fsrl_safetygym.py`) to trust the algorithm.
  2. **PPOLagAgent kwargs** (``cost_limit``, ``hidden_sizes``, seed) — confirm names
     against the installed version.

Install (likely its OWN venv — Tianshou pins clash with SB3): see that script.
"""
from __future__ import annotations

from typing import Any, Sequence

from gym_dr.trainers.base import Trainer, TrainingContext, TrainResult


class FsrlTrainer(Trainer):
    """Adopt FSRL ``PPOLagAgent`` (PID-Lagrangian PPO) behind the Trainer contract."""

    def __init__(self, *, epoch: int = 100, cost_limit: float = 10.0,
                 n_train_envs: int = 1, hidden_sizes: Sequence[int] = (256, 256)) -> None:
        self.epoch = epoch
        self.cost_limit = cost_limit          # the CMDP budget d (set empirically — dr/ep_mean_cost)
        self.n_train_envs = n_train_envs       # 1 = one sim/container; raise only with N-cars/parallel sim
        self.hidden_sizes = tuple(hidden_sizes)

    def fit(self, env: Any, ctx: TrainingContext) -> TrainResult:
        from tianshou.env import DummyVectorEnv  # noqa: F401  (lazy: needs the `safe` venv)
        from fsrl.agent import PPOLagAgent
        from fsrl.utils import TensorboardLogger

        from gym_dr.envs.wrappers import CostInfoWrapper

        # Surface the graded cost as info["cost"]; reuse the shared metrics tap.
        def make_env():
            return CostInfoWrapper(env, ctx.metrics_state)

        train_envs = DummyVectorEnv([make_env for _ in range(max(1, self.n_train_envs))])
        test_envs = DummyVectorEnv([make_env])
        logger = TensorboardLogger(str(ctx.run_dir / "tensorboard"), name="fsrl_ppolag")

        ctx.save_model(lambda p: p.write_bytes(b"fsrl-init"), name="initial_model")
        # TODO(custom-CNN + verify kwargs): pass a Tianshou CNN for Dict camera obs;
        # confirm PPOLagAgent's cost_limit/hidden_sizes/seed arg names.
        agent = PPOLagAgent(make_env(), logger)  # ← add cost_limit=self.cost_limit, hidden_sizes=..., once verified
        try:
            agent.learn(train_envs, test_envs, epoch=self.epoch)
        finally:
            # TODO: persist agent.policy.state_dict() instead of the placeholder.
            ctx.save_model(lambda p: p.write_bytes(b"fsrl-latest"), name="latest_model")
        ctx.save_model(lambda p: p.write_bytes(b"fsrl-final"), name="final_model")
        return TrainResult(final_eval_reward=float("nan"),
                           extra={"backend": "fsrl-ppolag", "epoch": self.epoch,
                                  "cost_limit": self.cost_limit})
