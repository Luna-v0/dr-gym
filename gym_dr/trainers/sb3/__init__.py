"""Stable-Baselines3 trainer — the default implementation of `Trainer`.

`Sb3Trainer` is a frozen dataclass so it composes cleanly with the HPO
override mechanism (`cfg.with_overrides(**{"trainer.kwargs.learning_rate": ...})`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gym_dr.trainers.base import Trainer, TrainingContext, TrainResult
from gym_dr.trainers.sb3.algorithms import load_model, make_model
from gym_dr.trainers.sb3.callbacks import (
    CtxCheckpointCallback,
    CtxEvalCallback,
    HeartbeatCallback,
    MlflowKVWriter,
    MultiWorldEvalCallback,
    RewardMetricsCallback,
    StatusJsonCallback,
    WallClockLimitCallback,
    update_training_status,
)


@dataclass(frozen=True)
class Sb3Trainer(Trainer):
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

    # Process-global (NOT a dataclass field — no annotation, so it stays off
    # __init__/asdict): has any training chunk run in THIS worker container yet?
    # An HPO worker runs several trials back-to-back in one container and the
    # Gazebo world persists across them. The very first trial can trust the
    # container's boot WORLD_NAME and must NOT call set_world before the env's
    # first reset (upstream raises "only valid between episodes"). Every later
    # trial inherits whatever world the previous trial's rotation left loaded,
    # so it has to pin the track back to world_plan[0] before chunk 0. See fit().
    _boot_world_consumed = False

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

    @staticmethod
    def _write_rotation_resume(ctx: TrainingContext, start_index: int) -> None:
        """Persist the chunk index + checkpoint to resume from after a
        gzserver crash, so the host can relaunch the container and continue
        the rotation. The checkpoint itself is written by fit's ``finally``
        (``latest_model.zip``) as the exception unwinds."""
        import json

        state = {
            "start_index": int(start_index),
            "resume_from": str(ctx.run_dir / "latest_model.zip"),
        }
        try:
            (ctx.run_dir / "rotation_resume.json").write_text(
                json.dumps(state) + "\n", encoding="utf-8")
            print("[Sb3Trainer] gzserver died mid-swap; wrote "
                  "rotation_resume.json {}".format(state), flush=True)
        except Exception as ex:  # noqa: BLE001
            print("[Sb3Trainer] failed to write rotation_resume.json:", ex,
                  flush=True)

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

        # SB3 callbacks tick once per VecEnv step (not per env-timestep), so with
        # N cars a raw ``checkpoint_freq`` saves every freq*N timesteps — at n=12 the
        # first checkpoint wouldn't land until 1.2M steps. Divide by the env count so
        # ``checkpoint_freq`` always means TIMESTEPS, regardless of car count.
        _n_envs = int(getattr(env, "num_envs", 1) or 1)
        callbacks = [
            CtxCheckpointCallback(
                save_freq=max(1, ctx.training.checkpoint_freq // _n_envs),
                save_path=str(checkpoints_dir),
                name_prefix=f"{self.name}_checkpoint",
                ctx=ctx,
                keep_last=ctx.training.checkpoint_keep_last,
            ),
            StatusJsonCallback(
                run_dir=run_dir,
                started_at=started_at,
                update_interval_steps=ctx.training.status_update_steps,
                update_interval_seconds=ctx.training.status_update_seconds,
                max_train_seconds=ctx.training.max_train_seconds,
            ),
            RewardMetricsCallback(),
            # Touch $GYM_DR_HEARTBEAT so the host watchdog can distinguish a
            # wedged-but-alive sim (hang) from real progress (d3-hang-postmortem).
            HeartbeatCallback(),
        ]
        if ctx.training.max_train_seconds is not None:
            wall_clock_callback = WallClockLimitCallback(
                run_dir=run_dir,
                started_at=started_at,
                max_train_seconds=ctx.training.max_train_seconds,
            )
            callbacks.append(wall_clock_callback)

        # Evaluation. When the strategy supplies a held-out eval world list
        # (OrderedSplit), measure track generalisation across those worlds;
        # otherwise evaluate on the current training world via SB3's
        # EvalCallback. Both expose ``last_mean_reward`` for the TrainResult.
        # Same VecEnv-tick caveat as the checkpoint callback: eval_freq is in
        # callback ticks (VecEnv steps), so with N cars a raw eval_freq evaluates
        # every freq*N timesteps. At n_cars=2 with eval_freq == chunk length, eval
        # NEVER fired within a chunk (no held-out metrics, no eval-phase perception
        # frames, mastery early-stop never triggered). Divide by the env count so
        # eval_freq means TIMESTEPS.
        if ctx.eval_worlds:
            eval_callback: Any = MultiWorldEvalCallback(
                ctx=ctx,
                eval_worlds=ctx.eval_worlds,
                eval_freq=max(1, ctx.training.eval_freq // _n_envs),
                n_eval_episodes=ctx.training.n_eval_episodes,
                best_model_save_path=str(best_model_dir),
                deterministic=True,
            )
        else:
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
                eval_freq=max(1, ctx.training.eval_freq // _n_envs),
                n_eval_episodes=ctx.training.n_eval_episodes,
                deterministic=True,
                render=False,
            )
        callbacks.append(eval_callback)

        # Pre-create the SB3 logger ONCE, before any model.learn(). The rotation
        # path calls model.learn() per chunk; each call would otherwise re-run
        # SB3's configure(tensorboard_log, tb_log_name) and open a NEW
        # SummaryWriter at run_name_1, run_name_2, ... — TB then shows N
        # overlapping partial "runs" and no complete curve. Setting an explicit
        # logger flips _custom_logger=True so _setup_learn skips re-configuring:
        # every chunk shares ONE TensorBoard writer with a continuous step axis.
        # The MlflowKVWriter output mirrors EVERY dumped scalar (rollout/*,
        # train/*, dr/*, eval/*) to the active MLflow run at each dump() — the
        # old rollout-end mirror missed the train/* + rollout/* signals.
        from stable_baselines3.common.logger import configure as _configure

        run_logger = _configure(
            str(tensorboard_dir / run_dir.name), ["stdout", "tensorboard"]
        )
        run_logger.output_formats.append(MlflowKVWriter())
        model.set_logger(run_logger)

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
                # Reused worker container (2nd+ trial in this process): the live
                # Gazebo world is whatever the previous trial's rotation left.
                # Establish one episode so set_world's between-episodes contract
                # holds, then pin the track to world_plan[0] before chunk 0.
                # set_world is idempotent, so on the rare path where the stale
                # world already equals world_plan[0] this is just a clean
                # rebuild. The first trial skips this and trusts the boot world.
                reused_container = type(self)._boot_world_consumed
                # Mark up front: any later trial must treat the world as dirty
                # even if this trial crashes partway through the rotation.
                type(self)._boot_world_consumed = True
                # rotate_start_index > 0 means the host relaunched this
                # container to recover from a mid-rotation gzserver crash:
                # Gazebo booted on world_plan[start] and we resume from there,
                # skipping the already-completed chunks.
                start = max(0, min(int(ctx.rotate_start_index or 0),
                                   len(ctx.world_plan) - 1))
                if reused_container:
                    model._last_obs = model.get_env().reset()
                    self._swap_world(model, ctx.world_plan[start])
                for i in range(start, len(ctx.world_plan)):
                    world = ctx.world_plan[i]
                    # Swap on every chunk except the first one we execute — the
                    # boot/pinned world is already world_plan[start].
                    if i > start:
                        try:
                            self._swap_world(model, world)
                        except Exception as ex:  # noqa: BLE001
                            # gzserver segfaulted mid-swap (WorldSwapError):
                            # persist where to resume so the host can relaunch
                            # the container on this world from the checkpoint.
                            if type(ex).__name__ == "WorldSwapError":
                                self._write_rotation_resume(ctx, i)
                            raise
                    # Stamp the trace's world context so every step row written
                    # this chunk carries the right hot-swapped world + a chunk
                    # counter that distinguishes repeated rotations of the same
                    # world (docs/trace-contract.md §2).
                    if ctx.metrics_state is not None:
                        ctx.metrics_state.world_name = world
                        ctx.metrics_state.chunk_index = i
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
                # True if any chunk ended early because the configured
                # EarlyStopStrategy qualified during evaluation. See
                # TrainingConfig.early_stop.
                "early_stopped": bool(getattr(eval_callback, "early_stops", 0)),
                "elapsed_seconds": int(time.monotonic() - started_at),
                "timesteps_completed": int(model.num_timesteps),
            },
        )
