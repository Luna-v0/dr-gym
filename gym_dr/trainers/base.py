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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
    metrics_state: Any | None = None
    """Handle to ``gym_dr.metrics._EpisodeMetrics``, owned by the orchestrator.
    Trainers can flip ``metrics_state.use_eval_reward`` around evaluation
    episodes so SB3's ``EvalCallback.last_mean_reward`` (and hence the value
    forwarded to Optuna via ``report_eval``) reflects the configured
    ``eval_reward`` instead of the per-trial training reward."""

    world_plan: list[str] | None = None
    """Expanded, ordered sequence of worlds to train across in a *single*
    container via the env's runtime track swap (``DeepRacerEnv.set_world``).
    One entry per chunk: chunk ``i`` trains :attr:`chunk_steps` timesteps on
    ``world_plan[i]`` then the trainer swaps to ``world_plan[i+1]`` without
    restarting Gazebo. ``None`` (the default) means single-world training —
    the legacy one-``model.learn`` path. The first entry is the world Gazebo
    loaded at container startup (``WORLD_NAME``); the trainer trusts it for the
    first trial in a container, but an HPO worker that runs several trials
    back-to-back re-pins the track to ``world_plan[0]`` at the start of every
    later trial (the previous trial's rotation left a different world loaded)."""

    chunk_steps: int | None = None
    """Timesteps to train per world chunk before swapping. Only consulted when
    :attr:`world_plan` is set; falls back to ``training.total_timesteps``."""

    rotate_start_index: int = 0
    """Index into :attr:`world_plan` at which to (re)start the rotation. ``0``
    for a fresh run; ``> 0`` when the host relaunched the container to recover
    from a mid-rotation ``gzserver`` crash — the already-completed chunks are
    skipped and training resumes from this chunk's world (which Gazebo loaded
    at startup) using ``training.resume_from``."""

    eval_worlds: list[str] | None = None
    """Ordered held-out worlds to evaluate on (from
    ``WorldStrategy.evaluation_worlds``). When set, the trainer swaps the env to
    each of these worlds at evaluation time, measures the policy, then restores
    the current training world — giving a track-generalisation metric. ``None``
    (default) means evaluate on the current training world."""

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
            mlflow.log_metric(name, float(value), step=step)
        except (TypeError, ValueError):
            return

    def report_eval(self, mean_reward: float, step: int) -> None:
        """Log evaluation reward to MLflow and check Optuna for pruning.

        Always logs ``eval/mean_reward`` to MLflow at ``step`` (matching the
        SB3 TensorBoard key written by ``EvalCallback``). When the trainer
        was invoked as part of an HPO trial (``ctx.trial`` is set), this
        also calls ``trial.report`` and raises ``optuna.TrialPruned`` if the
        pruner decides this trial is unlikely to win.
        """
        self.log_metric("eval/mean_reward", mean_reward, step)
        if self.trial is not None:
            self.trial.report(float(mean_reward), step)
            if self.trial.should_prune():
                import optuna

                raise optuna.TrialPruned()

    # ----- framework-agnostic services for custom trainers ------------------ #
    # These let a non-SB3 algorithm reuse all the pipeline plumbing (logging,
    # episode metrics, world-swap, the held-out eval protocol) without importing
    # TensorBoard/MLflow/the metadata writer. Call them at rollout/episode/eval
    # boundaries — none belong in the per-env-step hot loop, so they add no
    # measurable training overhead.

    def set_status(self, status: str, extra: dict | None = None) -> None:
        """Write ``training_status.json``. The orchestrator sets the initial and
        terminal statuses; a long custom loop may call this periodically with
        progress (e.g. ``{"timesteps_completed": n}``)."""
        from gym_dr.trainers.sb3.callbacks import update_training_status

        update_training_status(self.run_dir, status, extra)

    def tb_writer(self):
        """Lazily-created TensorBoard ``SummaryWriter`` under
        ``run_dir/tensorboard/`` (the same place SB3 writes), cached across calls."""
        w = getattr(self, "_tb_writer", None)
        if w is None:
            from torch.utils.tensorboard import SummaryWriter

            w = SummaryWriter(log_dir=str(self.run_dir / "tensorboard" / f"{self.name_prefix}_1"))
            object.__setattr__(self, "_tb_writer", w)
        return w

    def log_metrics(self, metrics: "dict[str, float]", step: int) -> None:
        """Log scalars to **both TensorBoard and MLflow** — the single call a
        custom trainer needs for loss curves, ``dr/ep_*`` episode metrics, and
        eval. Cheap, but call it at rollout/episode/eval boundaries, not per
        env-step."""
        try:
            w = self.tb_writer()
            for k, v in metrics.items():
                try:
                    w.add_scalar(k, float(v), step)
                except (TypeError, ValueError):
                    continue
            w.flush()
        except Exception:  # noqa: BLE001 — TB must never break training
            pass
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def record_episode(self, info: Any, step: int):
        """Drain a finished episode's ``dr_episode`` summary from ``info`` (present
        when the env was built through the orchestrator's metrics wrapper) to
        TB+MLflow; returns it, or ``None`` if absent."""
        summary = info.get("dr_episode") if isinstance(info, dict) else None
        if not summary:
            return None
        self.log_metrics(summary, step)
        return summary

    def swap_world(self, env: Any, world: str) -> None:
        """Hot-swap the env's Gazebo track to ``world`` and reset. Works for a raw
        gymnasium env (forwarded through wrappers) or an SB3 ``VecEnv``."""
        if hasattr(env, "env_method"):  # SB3 VecEnv
            env.env_method("set_world", world)
            env.reset()
            return
        setter = getattr(env, "set_world", None) or getattr(
            getattr(env, "unwrapped", env), "set_world", None
        )
        if setter is None:
            raise AttributeError("env has no set_world(); expected a DeepRacerEnv")
        setter(world)
        env.reset()

    def evaluate(self, predict_fn: Callable[[Any], Any], env: Any, *,
                 n_episodes: int = 3, step: int = 0) -> "dict[str, float]":
        """Run the project's standard **held-out evaluation** for a raw gymnasium
        env + your ``predict_fn(obs) -> action``.

        Swaps to each world in :attr:`eval_worlds` (or the current world if
        unset), runs ``n_episodes`` each, aggregates the success-criterion metrics
        from the env's ``dr_episode`` summaries (clean-completion rate, completion
        rate, progress, eval reward, off-track rate), logs per-world + aggregate to
        TB+MLflow, and calls :meth:`report_eval` (MLflow + Optuna). Returns the
        aggregate. Flips ``metrics_state.use_eval_reward`` around the eval. For an
        SB3/VecEnv trainer, use SB3's own eval instead.
        """
        worlds = self.eval_worlds or [None]
        state = self.metrics_state
        if state is not None:
            state.use_eval_reward = True
        per_world: "dict[str, dict]" = {}
        try:
            for world in worlds:
                if world is not None:
                    self.swap_world(env, world)
                summaries = []
                while len(summaries) < n_episodes:
                    obs, _info = env.reset()
                    done, info = False, {}
                    while not done:
                        obs, _r, term, trunc, info = env.step(predict_fn(obs))
                        done = bool(term or trunc)
                    if isinstance(info, dict) and info.get("dr_episode"):
                        summaries.append(info["dr_episode"])
                per_world[world or "current"] = _agg_eval(summaries)
            train_world = getattr(state, "world_name", None)
            if worlds != [None] and train_world:
                self.swap_world(env, train_world)
        finally:
            if state is not None:
                state.use_eval_reward = False
        agg = _agg_over_worlds(per_world)
        flat = {f"eval/{w}_{k}": v for w, m in per_world.items() for k, v in m.items()}
        flat.update({f"eval/{k}": v for k, v in agg.items()})
        ctrl = getattr(env, "adr_controller", None)  # Automatic Domain Randomization
        if ctrl is not None:
            flat.update(ctrl.update(agg.get("clean_completion_rate", 0.0)))
        self.log_metrics(flat, step)
        self.report_eval(agg.get("mean_reward", float("nan")), step)
        return agg


class Trainer(ABC):
    """The abstract base every algorithm extends — the "bring your own algorithm"
    seam (no Stable-Baselines lock-in).

    Subclass it and implement :meth:`fit`: take a gym env and a
    :class:`TrainingContext`, train, return a :class:`TrainResult`. The
    orchestrator handles run-dir setup, MLflow lifecycle, artifact archival, and
    status-JSON updates around the call, and ``TrainingContext`` hands you the
    whole ecosystem (TensorBoard + MLflow logging, checkpointing with the
    DeepRacer metadata sidecar, the held-out eval protocol, Optuna pruning) so a
    custom loop reuses all of it — see ``experiments/custom_trainer_example.py``.

    ``Sb3Trainer`` (SB3) and ``FsrlTrainer`` (safe-RL) are the shipped adapters;
    a pure-PyTorch trainer that drives its own rollout loop via a
    ``gym_dr.pipeline.Stage`` is equally first-class.
    """

    @abstractmethod
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

        Or simply reuse the shared services on ``ctx``: ``log_metrics`` (TB+MLflow),
        ``record_episode`` (drains ``dr/ep_*``), ``swap_world`` (curriculum), and
        ``evaluate`` (the held-out clean-completion protocol) — see
        ``docs/trainer-contract.md`` and ``experiments/custom_trainer_example.py``.
        """
        ...


def _agg_eval(summaries: "list[dict]") -> "dict[str, float]":
    """Aggregate per-episode ``dr_episode`` summaries into the success-criterion
    eval metrics (used by :meth:`TrainingContext.evaluate`)."""
    n = max(len(summaries), 1)

    def mean(key: str) -> float:
        return sum(float(s.get(key, 0.0)) for s in summaries) / n

    return {
        "clean_completion_rate": mean("dr/ep_completed_clean"),
        "completion_rate": mean("dr/ep_completed"),
        "mean_progress": mean("dr/ep_max_progress"),
        "mean_reward": mean("dr/ep_eval_reward"),
        "offtrack_rate": mean("dr/ep_offtrack_rate"),
    }


def _agg_over_worlds(per_world: "dict[str, dict]") -> "dict[str, float]":
    """Mean of each metric across the held-out worlds."""
    if not per_world:
        return {}
    keys = list(next(iter(per_world.values())).keys())
    n = len(per_world)
    return {k: sum(w[k] for w in per_world.values()) / n for k in keys}
