"""Stable-Baselines3 trainer — the default implementation of `Trainer`.

`Sb3Trainer` is a frozen dataclass so it composes cleanly with the HPO
override mechanism (`cfg.with_overrides(**{"trainer.kwargs.learning_rate": ...})`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gym_dr.trainers.base import TrainingContext, TrainResult
from gym_dr.trainers.sb3.algorithms import load_model, make_model
from gym_dr.trainers.sb3.callbacks import (
    CtxCheckpointCallback,
    CtxEvalCallback,
    MlflowMirrorCallback,
    RewardMetricsCallback,
    StatusJsonCallback,
    WallClockLimitCallback,
    update_training_status,
)


@dataclass(frozen=True)
class Sb3Trainer:
    """Stable-Baselines3 trainer — the default ``Trainer`` implementation.

    Builds an SB3 model via the algorithm registry, wires per-chunk callbacks
    (status JSON, wall-clock limit, MLflow scalar mirroring, periodic
    checkpoint with metadata sidecar, eval-based MLflow logging + Optuna
    pruning), and calls ``model.learn``.

    Fields
    ------
    - ``name``    — algorithm key (``"ppo"``, ``"sac"``, ``"td3"``, ``"a2c"``,
      ``"ddpg"``).
    - ``policy``  — SB3 policy class name (string).
    - ``kwargs``  — algorithm hyperparameters; passed straight to the SB3
      algorithm constructor.
    - ``device``  — ``"cpu"`` or ``"cuda"`` (or ``"auto"``).
    """

    name: str = "ppo"
    """Which SB3 algorithm to instantiate. One of: ``ppo``, ``sac``, ``td3``,
    ``a2c``, ``ddpg``. PPO is the only one that works out-of-the-box with
    image-dict observations at the default ``buffer_size``; off-policy
    algorithms require an explicit small ``kwargs["buffer_size"]`` (e.g.
    ``50_000``) or they OOM on the camera obs."""

    policy: str = "MultiInputPolicy"
    """SB3 policy class. ``MultiInputPolicy`` is required for DeepRacer's
    dict observation space. ``CnnPolicy`` works for a single Box obs;
    ``MlpPolicy`` for flat vectors (not the DeepRacer default)."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Algorithm-specific hyperparameters. For PPO common keys are
    ``learning_rate``, ``n_steps``, ``batch_size``, ``ent_coef``, ``gamma``,
    ``gae_lambda``, ``clip_range``, ``n_epochs``, ``vf_coef``. HPO sweeps
    these via dotted overrides like ``trainer.kwargs.learning_rate``."""

    device: str = "cpu"
    """Torch device. ``"cpu"`` is the default for the simapp's CPU image.
    ``"cuda"`` requires the GPU base image (``bootstrap.sh -a gpu``)."""

    def fit(self, env: Any, ctx: TrainingContext) -> TrainResult:
        from stable_baselines3.common.callbacks import CallbackList

        run_dir = ctx.run_dir
        tensorboard_dir = run_dir / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_model_dir = run_dir / "best_model"
        eval_log_dir = run_dir / "eval"

        started_at = time.monotonic()
        wall_clock_callback: WallClockLimitCallback | None = None

        if ctx.training.resume_from:
            print(f"Resuming model from: {ctx.training.resume_from}", flush=True)
            model = load_model(
                ctx.training.resume_from,
                env,
                name=self.name,
                device=self.device,
                tensorboard_log=str(tensorboard_dir),
            )
        else:
            model = make_model(
                env,
                name=self.name,
                policy=self.policy,
                kwargs=dict(self.kwargs),
                device=self.device,
                tensorboard_log=str(tensorboard_dir),
            )

        ctx.save_model(
            lambda p: model.save(str(p.with_suffix(""))),
            name="initial_model",
        )
        update_training_status(run_dir, "running")

        callbacks = [
            CtxCheckpointCallback(
                save_freq=max(1, ctx.training.checkpoint_freq),
                save_path=str(checkpoints_dir),
                name_prefix=f"{self.name}_checkpoint",
                ctx=ctx,
            ),
            StatusJsonCallback(
                run_dir=run_dir,
                started_at=started_at,
                update_interval_steps=ctx.training.status_update_steps,
                update_interval_seconds=ctx.training.status_update_seconds,
                max_train_seconds=ctx.training.max_train_seconds,
            ),
            MlflowMirrorCallback(),
            RewardMetricsCallback(),
        ]
        if ctx.training.max_train_seconds is not None:
            wall_clock_callback = WallClockLimitCallback(
                run_dir=run_dir,
                started_at=started_at,
                max_train_seconds=ctx.training.max_train_seconds,
            )
            callbacks.append(wall_clock_callback)

        eval_callback = CtxEvalCallback(
            eval_env=env,
            ctx=ctx,
            best_model_save_path=str(best_model_dir),
            log_path=str(eval_log_dir),
            eval_freq=max(1, ctx.training.eval_freq),
            n_eval_episodes=ctx.training.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        try:
            model.learn(
                total_timesteps=ctx.training.total_timesteps,
                callback=CallbackList(callbacks),
                reset_num_timesteps=not bool(ctx.training.resume_from),
            )
        finally:
            ctx.save_model(
                lambda p: model.save(str(p.with_suffix(""))),
                name="latest_model",
            )

        final_path = ctx.save_model(
            lambda p: model.save(str(p.with_suffix(""))),
            name="final_model",
        )
        return TrainResult(
            final_eval_reward=float(eval_callback.last_mean_reward),
            final_model_path=final_path,
            extra={
                "time_limit_reached": bool(
                    wall_clock_callback and wall_clock_callback.time_limit_reached
                ),
                "elapsed_seconds": int(time.monotonic() - started_at),
                "timesteps_completed": int(model.num_timesteps),
            },
        )
