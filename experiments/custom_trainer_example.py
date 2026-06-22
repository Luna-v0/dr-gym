"""Skeleton: author your own (non-SB3) trainer.

Copy this and drop your algorithm into the marked spots — the system is not tied
to Stable-Baselines3. See ``docs/trainer-contract.md``. Plug it in with
``ExperimentConfig(trainer=MyTrainer())`` and the orchestrator (Docker dispatch,
MLflow, status JSON, artifact archival, Optuna, curriculum, held-out eval) keeps
working via the ``TrainingContext`` services.

This skeleton runs a raw-env episode loop with a random policy and exercises the
full lifecycle (save / log / record_episode / evaluate / checkpoint) so the wiring
is visible end-to-end. Replace ``act()`` and the ``# >>> your update`` block.
"""
from __future__ import annotations

import time

from gym_dr.trainers.base import Trainer, TrainingContext, TrainResult


class MyTrainer(Trainer):
    def __init__(self, lr: float = 3.0e-4):
        self.lr = lr
        # >>> build your policy/optimizer here (you may reuse gym_dr.networks.DeepRacerCNN)

    def fit(self, env, ctx: TrainingContext) -> TrainResult:
        import numpy as np

        started = time.monotonic()
        rng = np.random.default_rng(ctx.seed or 0)

        def act(obs):
            # >>> replace with your policy: action = self.policy(obs)
            return env.action_space.sample()

        ctx.save_model(self._save, name="initial_model")
        total = ctx.training.total_timesteps
        eval_freq = max(1, ctx.training.eval_freq)
        ckpt_freq = max(1, ctx.training.checkpoint_freq)
        last_eval = float("nan")
        step = 0
        try:
            while step < total:
                obs, _ = env.reset()
                done = False
                info: dict = {}
                while not done and step < total:
                    obs, reward, term, trunc, info = env.step(act(obs))
                    step += 1
                    done = bool(term or trunc)
                    # >>> your update: store transition; periodically compute loss + optimizer.step()
                    if step % 2048 == 0:
                        ctx.log_metrics({"train/dummy_loss": float(rng.standard_normal())}, step)
                    if step % eval_freq == 0:
                        agg = ctx.evaluate(act, env, n_episodes=ctx.training.n_eval_episodes, step=step)
                        last_eval = agg.get("mean_reward", last_eval)
                    if step % ckpt_freq == 0:
                        ctx.save_checkpoint(self._save, step=step)
                    ctx.set_status("running", {"timesteps_completed": step})
                if done:
                    ctx.record_episode(info, step)
            ctx.save_model(self._save, name="final_model")
        finally:
            ctx.save_model(self._save, name="latest_model")
        return TrainResult(
            final_eval_reward=last_eval,
            extra={"timesteps_completed": step,
                   "elapsed_seconds": int(time.monotonic() - started)},
        )

    def _save(self, path) -> None:
        # >>> serialize your model, e.g. torch.save(self.policy.state_dict(), path)
        path.write_bytes(b"placeholder-model")
