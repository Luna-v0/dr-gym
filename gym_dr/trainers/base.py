"""Trainer interface.

A `Trainer` is anything with a `fit(env, ctx)` method that returns a
`TrainResult`. The default implementation (`gym_dr.trainers.sb3.Sb3Trainer`)
wraps Stable-Baselines3, but users can drop in any implementation — a custom
PyTorch loop, RLlib, CleanRL, whatever — and the surrounding pipeline (MLflow,
artifact layout, DeepRacer-compatible per-checkpoint metadata, Optuna pruning)
keeps working as long as the trainer reports through the `TrainingContext`.

Minimal custom trainer:

    from gym_dr.trainers.base import Trainer, TrainingContext, TrainResult

    class MyTrainer(Trainer):
        def __init__(self, lr: float = 1e-3):
            self.lr = lr

        def fit(self, env, ctx: TrainingContext) -> TrainResult:
            for step in range(ctx.training.total_timesteps):
                ...  # your training step
                if step % ctx.training.eval_freq == 0:
                    mean = self._evaluate(env)
                    ctx.report_eval(mean, step=step)
                if step % ctx.training.checkpoint_freq == 0:
                    ctx.save_checkpoint(self._save, step=step)
            return TrainResult(final_eval_reward=mean)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gym_dr.action_space import ActionSpaceConfig
    from gym_dr.config import TrainingConfig


@dataclass
class TrainResult:
    """Return value from ``Trainer.fit``.

    ``extra`` is a free-form dict the orchestrator copies into
    ``training_status.json``. Conventional keys: ``elapsed_seconds``,
    ``timesteps_completed``, ``time_limit_reached``.
    """

    final_eval_reward: float = float("nan")
    final_model_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingContext:
    """Handles to the pipeline's shared services, passed to every trainer.

    The trainer never imports MLflow / Optuna / the metadata writer directly —
    it just calls these methods and the orchestrator wires the rest.
    """

    run_dir: Path
    action_space: "ActionSpaceConfig"
    training: "TrainingConfig"
    trial: Any | None = None
    name_prefix: str = "checkpoint"
    seed: int | None = None
    """Random seed plumbed from ``ExperimentConfig.seed``. Trainers should
    forward this to their RL library and to ``env.reset(seed=...)``."""

    def save_model(self, save_fn: Callable[[Path], None], *, name: str) -> Path:
        """Save a top-level model artifact with its DeepRacer metadata sidecar.

        `save_fn` is called with a `Path` and should write a `.zip` (or whatever
        format) at that path. The metadata sidecar is written next to it as
        `<name>.model_metadata.json`.
        """
        from gym_dr.action_space import write_model_metadata

        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / f"{name}.zip"
        save_fn(path)
        if path.exists():
            write_model_metadata(
                path.with_suffix(".model_metadata.json"), self.action_space
            )
        return path

    def save_checkpoint(
        self,
        save_fn: Callable[[Path], None],
        *,
        step: int,
        name_prefix: str | None = None,
    ) -> Path:
        """Save a periodic checkpoint with its DeepRacer metadata sidecar.

        Path is ``<run_dir>/checkpoints/<prefix>_<step>_steps.zip`` with a
        sibling ``.model_metadata.json``. Cherry-pick a single checkpoint
        from this dir and the metadata travels with it — required to ship
        the model to the physical car.
        """
        from gym_dr.action_space import write_model_metadata

        prefix = name_prefix or self.name_prefix
        checkpoints_dir = self.run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoints_dir / f"{prefix}_{step}_steps.zip"
        save_fn(path)
        if path.exists():
            write_model_metadata(
                path.with_suffix(".model_metadata.json"), self.action_space
            )
        return path

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a scalar to the active MLflow run if one is open. No-op otherwise."""
        try:
            import mlflow
        except ImportError:
            return
        if mlflow.active_run() is None:
            return
        try:
            mlflow.log_metric(name.replace("/", "_"), float(value), step=step)
        except (TypeError, ValueError):
            return

    def report_eval(self, mean_reward: float, step: int) -> None:
        """Log evaluation reward to MLflow and check Optuna for pruning.

        Always logs ``eval_mean_reward`` to MLflow at ``step``. When the
        trainer was invoked as part of an HPO trial (``ctx.trial`` is set),
        this also calls ``trial.report`` and raises ``optuna.TrialPruned``
        if the pruner decides this trial is unlikely to win.
        """
        self.log_metric("eval_mean_reward", mean_reward, step)
        if self.trial is not None:
            self.trial.report(float(mean_reward), step)
            if self.trial.should_prune():
                import optuna

                raise optuna.TrialPruned()


@runtime_checkable
class Trainer(Protocol):
    """Anything with this method shape is a Trainer.

    The contract: take a gym env and a TrainingContext, train, return a
    TrainResult. The orchestrator handles run-dir setup, MLflow lifecycle,
    artifact archival, and status-JSON updates around this call.
    """

    def fit(self, env: Any, ctx: TrainingContext) -> TrainResult:
        """Train against ``env`` until completion or pruning.

        Implementations should:

        - call ``ctx.save_model(fn, name="initial_model")`` before training,
        - call ``ctx.save_checkpoint(fn, step=N)`` periodically,
        - call ``ctx.report_eval(mean_reward, step=N)`` after evaluations
          (this is what feeds MLflow logs and Optuna pruning),
        - call ``ctx.save_model(fn, name="latest_model")`` in a ``finally``
          so resume targets exist even on crash,
        - call ``ctx.save_model(fn, name="final_model")`` on clean exit.
        """
        ...
