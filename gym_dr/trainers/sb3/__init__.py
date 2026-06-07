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

    frame_stack: int = 1
    """How many consecutive observations to stack along the channel/feature
    axis before the policy sees them. ``1`` = no stacking (raw env obs).
    ``> 1`` wraps the env in ``DummyVecEnv`` + ``VecFrameStack(n_stack=...)``
    so each step's observation includes the last N frames — gives the
    policy implicit temporal context (velocity, acceleration cues). For
    Dict obs (DeepRacer's ``FRONT_FACING_CAMERA``) SB3 stacks each key
    independently along its first axis. Typical sweep range: 1–4."""

    @staticmethod
    def _swap_world(model: Any, world: str) -> None:
        """Swap the live Gazebo track to *world* and re-sync SB3's rollout state.

        Calls ``DeepRacerEnv.set_world`` through the vec-env (``env_method``
        forwards the call down through every VecEnvWrapper and gym wrapper to
        the base env), then resets the env on the new world and points SB3's
        ``_last_obs`` / ``_last_episode_starts`` at that fresh observation so
        the next ``model.learn(reset_num_timesteps=False)`` rolls out on the
        new track instead of continuing from a stale frame.
        """
        import numpy as np

        vec = model.get_env()
        vec.env_method("set_world", world)
        model._last_obs = vec.reset()
        model._last_episode_starts = np.ones((vec.num_envs,), dtype=bool)

    def fit(self, env: Any, ctx: TrainingContext) -> TrainResult:
        from stable_baselines3.common.callbacks import CallbackList
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecEnv,
            VecFrameStack,
        )

        # GPU misconfig handling. If cuda was requested but torch can't see a
        # CUDA runtime, fall back to CPU with a loud warning rather than
        # failing every HPO trial — failing-fast killed entire studies when
        # users forgot to rebuild with `./bootstrap.sh -a gpu`. The trial
        # still runs (just slower); the WARNING in stdout tells the user to
        # rebuild if they care about GPU speed.
        device = self.device
        # Normalize "gpu" → "cuda". torch's device parser rejects "gpu" with
        # a verbose error listing every backend; users reasonably type "gpu".
        if device.lower() == "gpu":
            device = "cuda"
        if device.lower().startswith("cuda"):
            try:
                import torch

                cuda_ok = torch.cuda.is_available()
            except ImportError:
                cuda_ok = False
            if not cuda_ok:
                print(
                    "[Sb3Trainer] WARNING: device='cuda' requested but "
                    "torch.cuda.is_available() is False. Falling back to CPU "
                    "for this trial.\n"
                    "  To actually use GPU: (a) rebuild the image with "
                    "`./bootstrap.sh -a gpu`; (b) set ExperimentConfig.use_gpu=True "
                    "so `docker run` gets `--gpus all`; (c) ensure the host has "
                    "the NVIDIA Container Toolkit installed.",
                    flush=True,
                )
                device = "cpu"

        run_dir = ctx.run_dir
        tensorboard_dir = run_dir / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_model_dir = run_dir / "best_model"
        eval_log_dir = run_dir / "eval"

        # Frame stacking: wrap the env in VecFrameStack(n_stack) when requested.
        # Upstream DeepRacerEnv emits a single frame per step (verified by
        # reading deepracer_env/sensors/sensors_rollout.py + utils.py); the
        # policy gets temporal context only if we stack frames here.
        if self.frame_stack > 1:
            if not isinstance(env, VecEnv):
                env = DummyVecEnv([lambda env=env: env])
            env = VecFrameStack(env, n_stack=self.frame_stack)

        started_at = time.monotonic()
        wall_clock_callback: WallClockLimitCallback | None = None

        # Inject ctx.seed into algorithm kwargs unless the user set their own.
        # SB3 forwards `seed` to its torch RNG + the first `env.reset(seed=...)`.
        sb3_kwargs = dict(self.kwargs)
        if ctx.seed is not None and "seed" not in sb3_kwargs:
            sb3_kwargs["seed"] = int(ctx.seed)

        if ctx.training.resume_from:
            print(f"Resuming model from: {ctx.training.resume_from}", flush=True)
            model = load_model(
                ctx.training.resume_from,
                env,
                name=self.name,
                device=device,
                tensorboard_log=str(tensorboard_dir),
            )
        else:
            model = make_model(
                env,
                name=self.name,
                policy=self.policy,
                kwargs=sb3_kwargs,
                device=device,
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

        # Use model.get_env() so the eval env carries the same SB3-applied
        # wrappers (VecTransposeImage on top of our VecFrameStack/DummyVecEnv).
        # Without this, SB3 warns "Training and eval env are not of the same
        # type" because it transposes image obs for the training env but not
        # for an eval env passed in raw.
        eval_callback = CtxEvalCallback(
            eval_env=model.get_env(),
            ctx=ctx,
            best_model_save_path=str(best_model_dir),
            log_path=str(eval_log_dir),
            eval_freq=max(1, ctx.training.eval_freq),
            n_eval_episodes=ctx.training.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        # Unified naming: SB3's default TB subdir is "<AlgoClass>_<auto_idx>"
        # (e.g. PPO_1). Naming it after the run makes the TB sidebar legible
        # and matches the MLflow run name + the Optuna trial.user_attr.
        try:
            cb = CallbackList(callbacks)
            if ctx.world_plan:
                # In-container runtime rotation: train chunk_steps per world,
                # swapping the track between chunks WITHOUT restarting Gazebo.
                # The model (weights + PPO optimizer state) and the env persist
                # across swaps — that is the whole point of doing it in one
                # process instead of one container per (rotation, world).
                chunk_steps = ctx.chunk_steps or ctx.training.total_timesteps
                resumed = bool(ctx.training.resume_from)
                for i, world in enumerate(ctx.world_plan):
                    if i > 0:
                        self._swap_world(model, world)
                    print(
                        f"[Sb3Trainer] chunk {i + 1}/{len(ctx.world_plan)}: "
                        f"world={world!r} steps={chunk_steps}",
                        flush=True,
                    )
                    model.learn(
                        total_timesteps=chunk_steps,
                        callback=cb,
                        # Only the very first chunk may reset the step counter;
                        # later chunks accumulate so TB/eval curves stay
                        # continuous across worlds.
                        reset_num_timesteps=(i == 0 and not resumed),
                        tb_log_name=ctx.run_dir.name,
                    )
            else:
                model.learn(
                    total_timesteps=ctx.training.total_timesteps,
                    callback=cb,
                    reset_num_timesteps=not bool(ctx.training.resume_from),
                    tb_log_name=ctx.run_dir.name,
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
