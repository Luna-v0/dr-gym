from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import KVWriter


import os as _os


def _touch_heartbeat() -> None:
    """Touch the host-watchdog heartbeat (no-op if ``$GYM_DR_HEARTBEAT`` unset).
    Called from BOTH the training HeartbeatCallback and the eval step-callback, so
    a long eval phase (no training steps, several world swaps) doesn't look like a
    hang. Env read per-call (throttled callers) so it's robust to late setting.
    See docs/reports/d3-hang-postmortem.md."""
    path = _os.getenv("GYM_DR_HEARTBEAT")
    if not path:
        return
    try:
        Path(path).touch()
    except OSError:
        pass


def _make_eval_collector(
    capture_paths: bool = False,
) -> tuple[Any, dict[str, int], list[dict]]:
    """Build an ``evaluate_policy`` step-callback that mines finished episodes.

    ``evaluate_policy`` calls the callback once per env step with ``locals()``
    exposing the current ``done`` flag and ``info`` dict. On each *done* step we
    read what the metrics wrapper stamped into ``info``:

    - ``info["dr_episode"]["dr/ep_ended_offtrack"]`` — bumps ``count["n"]`` so it
      ends up as the number of eval episodes that terminated off-track.
    - ``info["dr_episode_path"]`` (only present when ``capture_path`` is on) — the
      episode's trajectory + skeleton, appended to ``episodes`` for plotting.

    Returns ``(callback, count, episodes)``; the caller reads them after
    ``evaluate_policy`` returns.
    """
    count = {"n": 0, "completed": 0, "clean": 0}
    episodes: list[dict] = []

    def _cb(locals_: dict, _globals: dict) -> None:
        _touch_heartbeat()  # keep the watchdog alive through long evals
        if not locals_.get("done"):
            return
        info = locals_.get("info") or {}
        if not isinstance(info, dict):
            return
        summary = info.get("dr_episode")
        if summary:
            if summary.get("dr/ep_ended_offtrack", 0.0):
                count["n"] += 1
            if summary.get("dr/ep_completed", 0.0):
                count["completed"] += 1
            if summary.get("dr/ep_completed_clean", 0.0):
                count["clean"] += 1
        if capture_paths:
            path = info.get("dr_episode_path")
            if path:
                episodes.append(path)

    return _cb, count, episodes


def _eval_policy(model, vec, *, n_eval_episodes: int, deterministic: bool, callback):
    """Evaluate ``model`` on ``vec`` for ``n_eval_episodes``, calling ``callback`` with
    the same ``{"done", "info"}`` locals contract as SB3's ``evaluate_policy``.

    For a RECURRENT policy (sb3-contrib ``RecurrentPPO``, detected via ``policy.lstm_actor``)
    SB3's ``evaluate_policy`` is WRONG — it calls ``predict(obs)`` without carrying the
    LSTM hidden state, so the recurrent net would be judged as if memoryless. Here we run
    the loop manually, threading ``state`` + ``episode_start`` so the hidden state persists
    within an episode and resets at its boundary. Non-recurrent models fall straight
    through to SB3's ``evaluate_policy`` (unchanged behaviour). Returns ``(mean, std)``."""
    import numpy as np

    is_recurrent = hasattr(getattr(model, "policy", None), "lstm_actor")
    if not is_recurrent:
        from stable_baselines3.common.evaluation import evaluate_policy

        return evaluate_policy(model, vec, n_eval_episodes=n_eval_episodes,
                               deterministic=deterministic, warn=False, callback=callback)

    n_envs = vec.num_envs
    obs = vec.reset()
    states = None
    episode_starts = np.ones((n_envs,), dtype=bool)
    cur = np.zeros((n_envs,), dtype=np.float64)
    ep_rewards: list[float] = []
    while len(ep_rewards) < n_eval_episodes:
        actions, states = model.predict(
            obs, state=states, episode_start=episode_starts, deterministic=deterministic)
        obs, rewards, dones, infos = vec.step(actions)
        cur += np.asarray(rewards, dtype=np.float64)
        episode_starts = np.asarray(dones, dtype=bool)   # reset LSTM state at episode end
        for i in range(n_envs):
            callback({"done": bool(dones[i]), "info": infos[i]}, {})
            if dones[i]:
                ep_rewards.append(float(cur[i])); cur[i] = 0.0
    sel = ep_rewards[:n_eval_episodes]
    return float(np.mean(sel)), float(np.std(sel))


def _log_eval_paths(logger: Any, world: str, timestep: int, episodes: list[dict]) -> None:
    """Render the eval trajectories for *world* and log them as TB images.

    One overlay figure (all episodes, colour + legend per episode) plus one
    figure per individual episode. No-op when there are no captured episodes.
    Matplotlib/plot imports are deferred here so they're only paid when
    ``TrainingConfig.eval_path_plots`` is enabled.
    """
    if not episodes:
        return
    from stable_baselines3.common.logger import Figure

    from gym_dr.trainers.sb3.plots import render_episode, render_overlay

    exclude = ("stdout", "log", "json", "csv")
    overlay = render_overlay(world, timestep, episodes)
    logger.record(f"eval_paths/{world}", Figure(overlay, close=True), exclude=exclude)
    for i, ep in enumerate(episodes):
        fig = render_episode(world, timestep, i, ep)
        logger.record(f"eval_paths/{world}/ep{i}", Figure(fig, close=True), exclude=exclude)


def update_training_status(run_dir: Path, status: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_status.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


class _EarlyStopMixin:
    """Pluggable early stop, shared by the eval callbacks.

    Each eval round produces aggregate metrics (off-track rate, clean-completion
    rate, mean reward, …). This mixin hands them to an
    :class:`gym_dr.early_stopping.EarlyStopController` built from
    ``ctx.training.early_stop``; when the strategy qualifies for its ``patience``
    consecutive rounds, the host ``_on_step`` returns ``False`` — the same
    early-exit SB3 honours for ``WallClockLimitCallback`` — which ends the chunk's
    ``model.learn`` and hands control back to ``Sb3Trainer.fit`` (advancing the
    rotation, or ending a single-track run). The controller's streak resets per
    chunk via ``_on_training_start`` so mastering one track never pre-credits the
    next. ``early_stop=None`` disables it (the controller no-ops).
    """

    def _es_init(self) -> None:
        self._es_ctrl = None  # (re)built per chunk in _on_training_start
        self.mastered = False
        self.early_stops = 0  # non-reset: how many chunks this instance stopped

    def _on_training_start(self) -> None:
        super()._on_training_start()  # type: ignore[misc]
        from gym_dr.early_stopping import EarlyStopController

        self._es_ctrl = EarlyStopController(
            self._ctx.training.early_stop  # type: ignore[attr-defined]
        )
        self.mastered = False

    def _apply_early_stop(self, metrics: "dict[str, float]") -> bool:
        """Fold this eval round's aggregate metrics into the strategy; return
        ``True`` when the chunk should stop now."""
        ctrl = getattr(self, "_es_ctrl", None)
        if ctrl is None or not ctrl.enabled:
            return False
        if not ctrl.update(metrics):
            return False
        self.mastered = True
        self.early_stops += 1
        steps = int(self.num_timesteps)  # type: ignore[attr-defined]
        print(
            f"[early-stop] {ctrl.strategy.describe()} met at {steps} steps; "
            f"ending chunk",
            flush=True,
        )
        logged = {
            k: float(metrics[k])
            for k in ("offtrack_rate", "clean_completion_rate", "mean_reward")
            if k in metrics
        }
        update_training_status(
            self._ctx.run_dir,  # type: ignore[attr-defined]
            "early_stopped",
            {"timesteps_completed": steps, **logged},
        )
        return True


class CtxCheckpointCallback(CheckpointCallback):
    """SB3 CheckpointCallback that routes saves through TrainingContext.

    Ensures every periodic checkpoint zip gets its `model_metadata.json`
    sibling, matching what `ctx.save_checkpoint` writes. When ``keep_last`` is
    set, prunes older checkpoints after each save so a long run doesn't fill
    the disk (``best_model``/``final_model``/``latest_model`` live elsewhere and
    are never touched).
    """

    def __init__(self, *args: Any, ctx, keep_last: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = ctx
        self._keep_last = keep_last

    def _on_step(self) -> bool:
        if self.save_freq > 0 and (self.n_calls + 1) % self.save_freq == 0:
            step = int(self.num_timesteps) + 1  # match super's post-save num_timesteps
            self._ctx.save_checkpoint(
                lambda p: self.model.save(str(p.with_suffix(""))),
                step=step,
                name_prefix=self.name_prefix,
            )
            self._prune_old_checkpoints()
            self.n_calls += 1
            return True
        proceed = super()._on_step()
        self._prune_old_checkpoints()
        return proceed

    def _prune_old_checkpoints(self) -> None:
        """Keep only the ``keep_last`` most recent checkpoints (by step number).

        Deletes each pruned ``<prefix>_<step>_steps.zip`` together with its
        ``.model_metadata.json`` sidecar. No-op when ``keep_last`` is ``None``."""
        if not self._keep_last or self._keep_last <= 0:
            return
        ckpt_dir = Path(self.save_path)
        zips = list(ckpt_dir.glob(f"{self.name_prefix}_*_steps.zip"))
        if len(zips) <= self._keep_last:
            return

        def _step(path: Path) -> int:
            try:
                return int(path.stem.split("_")[-2])  # <prefix>_<step>_steps
            except (ValueError, IndexError):
                return -1

        for old in sorted(zips, key=_step)[: -self._keep_last]:
            old.unlink(missing_ok=True)
            old.with_suffix(".model_metadata.json").unlink(missing_ok=True)


class CtxEvalCallback(_EarlyStopMixin, EvalCallback):
    """SB3 EvalCallback that calls ctx.report_eval after each evaluation.

    `ctx.report_eval` handles MLflow logging and Optuna pruning. We also write
    the metadata sidecar for `best_model.zip` whenever it gets saved.

    During evaluation we flip ``ctx.metrics_state.use_eval_reward`` so the env
    returns ``ExperimentConfig.eval_reward(params)`` instead of the training
    reward. Without this, ``last_mean_reward`` (and therefore the Optuna
    pruning signal) would be in units of the *training* reward — making
    cross-trial comparison meaningless whenever the HPO search sweeps the
    reward function. The flag is restored to False after the eval block so
    subsequent training rollouts go back to the training reward.
    """

    def __init__(self, *args: Any, ctx, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = ctx
        self._offtrack_resets = 0
        self._completed = 0
        self._clean = 0
        self._eval_episodes: list[dict] = []
        self._es_init()

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        # Keep SB3's success-rate bookkeeping, then additionally tally eval
        # episodes that ended with the car off-track (a track-out reset) and, if
        # path plots are enabled, collect each episode's trajectory.
        super()._log_success_callback(locals_, globals_)
        if not locals_.get("done"):
            return
        info = locals_.get("info") or {}
        if not isinstance(info, dict):
            return
        summary = info.get("dr_episode")
        if summary:
            if summary.get("dr/ep_ended_offtrack", 0.0):
                self._offtrack_resets += 1
            if summary.get("dr/ep_completed", 0.0):
                self._completed += 1
            if summary.get("dr/ep_completed_clean", 0.0):
                self._clean += 1
        path = info.get("dr_episode_path")
        if path:
            self._eval_episodes.append(path)

    def _on_step(self) -> bool:
        is_eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        state = getattr(self._ctx, "metrics_state", None)
        if is_eval_step and state is not None:
            state.use_eval_reward = True
        if is_eval_step:
            self._offtrack_resets = 0
            self._completed = 0
            self._clean = 0
            self._eval_episodes = []
        try:
            proceed = super()._on_step()
        finally:
            if state is not None:
                state.use_eval_reward = False
        if is_eval_step:
            # Track-out resets for this eval. Single-world path, so the per-track
            # series and the global series coincide; log both so the metric name
            # is consistent with the multi-world callback.
            self.logger.record("eval/offtrack_resets", float(self._offtrack_resets))
            n_eps = max(1, self.n_eval_episodes)
            self.logger.record("eval/completion_rate", self._completed / n_eps)
            self.logger.record("eval/clean_completion_rate", self._clean / n_eps)
            world = getattr(state, "world_name", None)
            if world:
                self.logger.record(
                    f"eval/{world}_offtrack_resets", float(self._offtrack_resets)
                )
                self.logger.record(
                    f"eval/{world}_completion_rate", self._completed / n_eps
                )
                self.logger.record(
                    f"eval/{world}_clean_completion_rate", self._clean / n_eps
                )
            if self._eval_episodes and world:
                _log_eval_paths(
                    self.logger, world, int(self.num_timesteps), self._eval_episodes
                )
            if self.best_model_save_path is not None:
                best_zip = Path(self.best_model_save_path) / "best_model.zip"
                if best_zip.exists():
                    from gym_dr.action_space import write_model_metadata

                    write_model_metadata(
                        best_zip.with_suffix(".model_metadata.json"),
                        self._ctx.action_space,
                    )
            self._ctx.report_eval(float(self.last_mean_reward), int(self.num_timesteps))
            # Pluggable early stop: end the chunk once the configured strategy
            # qualifies on this eval round (advances the rotation / ends the run).
            if self._apply_early_stop({
                "offtrack_rate": self._offtrack_resets / n_eps,
                "clean_completion_rate": self._clean / n_eps,
                "completion_rate": self._completed / n_eps,
                "mean_reward": float(self.last_mean_reward),
            }):
                return False
        return proceed


class MultiWorldEvalCallback(_EarlyStopMixin, BaseCallback):
    """Evaluate across an ordered, held-out set of worlds (track generalisation).

    Used by ``Sb3Trainer`` when ``ctx.eval_worlds`` is set (e.g. the
    :class:`~gym_dr.worlds.OrderedSplit` strategy). At each eval trigger it:

    1. flips ``metrics_state.use_eval_reward`` so the eval reward is measured,
    2. swaps the (shared) Gazebo env to each world in ``eval_worlds`` via
       ``set_world``, running ``n_eval_episodes`` per world,
    3. restores the world training was on (read from ``metrics_state.world_name``,
       which ``Sb3Trainer.fit`` keeps current per chunk),
    4. reports the mean-across-worlds as the eval metric (per-world means are
       logged as ``eval/<world>_mean_reward``), and saves ``best_model`` on
       improvement.

    Exposes ``last_mean_reward`` so the trainer can read the final eval score
    the same way it does for ``CtxEvalCallback``.
    """

    def __init__(
        self,
        *,
        ctx,
        eval_worlds: list[str],
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self._ctx = ctx
        self.eval_worlds = list(eval_worlds)
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic
        self.last_mean_reward = float("nan")
        self.best_mean_reward = -float("inf")
        self._es_init()

    def _can_set_world(self, vec) -> bool:
        """True if the (shared) env can actually hot-swap tracks. The multi-car
        backend (MultiAgentDeepRacerEnv) has no set_world, so env_method('set_world')
        is a silent no-op — iterating eval_worlds there would 'evaluate' every held-out
        world on the CURRENT training tracks and emit garbage per-world / gap metrics.
        MultiCarVecEnv advertises ``can_set_world=False``; single-car envs lack the attr
        (get_attr raises) and are assumed swappable."""
        try:
            return bool(vec.get_attr("can_set_world")[0])
        except Exception:  # noqa: BLE001 — attr absent (single-car) => assume swappable
            return True

    def _swap(self, vec, world: str) -> None:
        import numpy as np

        vec.env_method("set_world", world)
        self.model._last_obs = vec.reset()
        self.model._last_episode_starts = np.ones((vec.num_envs,), dtype=bool)

    def _resume_reset(self, vec, world) -> None:
        """Reset the shared env so training resumes on clean episodes. Single-car:
        swap back to the training world. Either way this runs AFTER the recorder phase
        is restored to 'train', so the post-eval in-progress episode is started fresh
        under 'train' (not stamped 'eval' and flushed to eval/ — phase contamination)."""
        import numpy as np

        if world is not None and self._can_set_world(vec):
            vec.env_method("set_world", world)
        self.model._last_obs = vec.reset()
        self.model._last_episode_starts = np.ones((vec.num_envs,), dtype=bool)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        import numpy as np

        state = getattr(self._ctx, "metrics_state", None)
        train_world = getattr(state, "world_name", None)
        vec = self.model.get_env()
        can_swap = self._can_set_world(vec)

        capture_paths = bool(getattr(self._ctx.training, "eval_path_plots", False))
        per_world: dict[str, float] = {}
        counts_per_world: dict[str, dict] = {}
        train_clean_rate = None
        if state is not None:
            state.use_eval_reward = True
        # Tag perception-dataset shards captured during eval as phase="eval" (no-op
        # when no recorder is attached); restored to "train" in the finally below.
        try:
            vec.env_method("set_recorder_phase", "eval")
            vec.env_method("set_metrics_eval_mode", True)   # per-car metrics -> eval reward
        except Exception:  # noqa: BLE001
            pass
        try:
            if can_swap:
                for world in self.eval_worlds:
                    self._swap(vec, world)
                    eval_cb, counts, episodes = _make_eval_collector(capture_paths)
                    mean_reward, _ = _eval_policy(
                        self.model, vec, n_eval_episodes=self.n_eval_episodes,
                        deterministic=self.deterministic, callback=eval_cb)
                    per_world[world] = float(mean_reward)
                    counts_per_world[world] = counts
                    if capture_paths:
                        _log_eval_paths(self.logger, world, int(self.num_timesteps), episodes)
                # Held-in train-world score for a live generalization gap.
                if train_world is not None and train_world not in per_world:
                    self._swap(vec, train_world)
                    eval_cb, train_counts, _eps = _make_eval_collector(False)
                    _eval_policy(
                        self.model, vec, n_eval_episodes=self.n_eval_episodes,
                        deterministic=self.deterministic, callback=eval_cb)
                    train_clean_rate = train_counts["clean"] / max(1, self.n_eval_episodes)
            else:
                # MULTI-CAR: no set_world. Evaluate ONCE on the CURRENT training tracks
                # (deterministic policy) — an honest in-distribution eval that drives the
                # per-chunk early-stop and captures eval-phase dataset frames. We do NOT
                # iterate eval_worlds (that would fake per-held-out-world metrics + a ~0
                # generalization gap). True held-out generalization + held-out dataset
                # frames are a separate single-car pass (scripts/eval_physical_tracks.py,
                # scripts/perception_capture_heldout.py).
                eval_cb, counts, episodes = _make_eval_collector(capture_paths)
                mean_reward, _ = _eval_policy(
                    self.model, vec, n_eval_episodes=self.n_eval_episodes,
                    deterministic=self.deterministic, callback=eval_cb)
                per_world["current_tracks"] = float(mean_reward)
                counts_per_world["current_tracks"] = counts
                if capture_paths:
                    _log_eval_paths(self.logger, "current_tracks",
                                    int(self.num_timesteps), episodes)
        finally:
            if state is not None:
                state.use_eval_reward = False
            # Restore phase to "train" BEFORE the resume reset (fixes eval->train
            # contamination: else the post-eval episode is stamped "eval" and flushed
            # to eval/).
            try:
                vec.env_method("set_recorder_phase", "train")
                vec.env_method("set_metrics_eval_mode", False)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._resume_reset(vec, train_world)
            except Exception:  # noqa: BLE001
                pass

        agg = float(np.mean(list(per_world.values()))) if per_world else float("nan")
        self.last_mean_reward = agg
        # Per-world reward only when the worlds are REAL (held-out swap). Multi-car's
        # single "current_tracks" entry is logged via the counts block below as honest
        # in-distribution eval, not as a held-out world.
        if can_swap:
            for world, m in per_world.items():
                self.logger.record(f"eval/{world}_mean_reward", m)
        self.logger.record("eval/mean_reward", agg)
        self.logger.record("eval/n_worlds", float(len(per_world)))
        # Track-out resets + completion rates, per-track and aggregated. The
        # (clean-)completion rate is the success-criterion yardstick: the fraction
        # of held-out eval episodes that finished the lap (cleanly, with no
        # off-track step). See docs/eval-protocol.md.
        n_eps = max(1, self.n_eval_episodes)
        for world, c in counts_per_world.items():
            self.logger.record(f"eval/{world}_offtrack_resets", float(c["n"]))
            self.logger.record(f"eval/{world}_completion_rate", c["completed"] / n_eps)
            self.logger.record(f"eval/{world}_clean_completion_rate", c["clean"] / n_eps)
        self.logger.record(
            "eval/offtrack_resets", float(sum(c["n"] for c in counts_per_world.values()))
        )
        if counts_per_world:
            self.logger.record(
                "eval/completion_rate",
                float(np.mean([c["completed"] / n_eps for c in counts_per_world.values()])),
            )
            clean_rate = float(np.mean([c["clean"] / n_eps for c in counts_per_world.values()]))
            self.logger.record("eval/clean_completion_rate", clean_rate)
            # Live generalization gap: how much worse the held-out tracks are than
            # the track currently being trained. The success-criterion headline —
            # a policy that drives held-out tracks as well as trained ones has
            # gap ≈ 0. (docs/reports/q1-generalization.md, docs/eval-protocol.md)
            if train_clean_rate is not None:
                self.logger.record("eval/train_clean_completion_rate", train_clean_rate)
                self.logger.record("eval/generalization_gap", train_clean_rate - clean_rate)
            # Automatic Domain Randomization: grow/shrink DR ranges on this signal.
            try:
                ctrl = vec.get_attr("adr_controller")[0]
            except Exception:  # noqa: BLE001 — env has no ADR controller
                ctrl = None
            if ctrl is not None:
                for k, v in ctrl.update(clean_rate).items():
                    self.logger.record(k, v)

        if per_world and agg > self.best_mean_reward:
            self.best_mean_reward = agg
            self._save_best()

        self._ctx.report_eval(agg, int(self.num_timesteps))
        # Pluggable early stop, measured across what was actually evaluated this
        # round (held-out worlds when swappable; the single current-tracks eval for
        # multi-car) — NOT len(eval_worlds), which multi-car never iterates.
        total_eps = self.n_eval_episodes * max(1, len(counts_per_world))
        total_off = sum(c["n"] for c in counts_per_world.values())
        es_metrics: "dict[str, float]" = {
            "offtrack_rate": (total_off / total_eps) if total_eps else 1.0,
            "mean_reward": agg,
        }
        if counts_per_world:
            es_metrics["clean_completion_rate"] = clean_rate
            es_metrics["completion_rate"] = float(
                np.mean([c["completed"] / n_eps for c in counts_per_world.values()])
            )
        if self._apply_early_stop(es_metrics):
            return False
        return True

    def _save_best(self) -> None:
        if not self.best_model_save_path:
            return
        from gym_dr.action_space import write_model_metadata

        best_dir = Path(self.best_model_save_path)
        best_dir.mkdir(parents=True, exist_ok=True)
        best_zip = best_dir / "best_model.zip"
        self.model.save(str(best_zip.with_suffix("")))
        write_model_metadata(
            best_zip.with_suffix(".model_metadata.json"), self._ctx.action_space
        )


class StatusJsonCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        started_at: float,
        update_interval_steps: int,
        update_interval_seconds: int,
        max_train_seconds: int | None,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.started_at = started_at
        self.update_interval_steps = max(1, update_interval_steps)
        self.update_interval_seconds = max(1, update_interval_seconds)
        self.max_train_seconds = max_train_seconds
        self._last_report_step = 0
        self._last_report_time = started_at

    def _on_step(self) -> bool:
        now = time.monotonic()
        if (
            self.num_timesteps - self._last_report_step < self.update_interval_steps
            and now - self._last_report_time < self.update_interval_seconds
        ):
            return True
        elapsed = int(now - self.started_at)
        extra: dict[str, Any] = {
            "timesteps_completed": self.num_timesteps,
            "elapsed_seconds": elapsed,
        }
        if self.max_train_seconds is not None:
            extra["time_limit_seconds"] = self.max_train_seconds
            extra["time_remaining_seconds"] = max(0, self.max_train_seconds - elapsed)
        update_training_status(self.run_dir, "running", extra)
        self._last_report_step = self.num_timesteps
        self._last_report_time = now
        return True


class WallClockLimitCallback(BaseCallback):
    def __init__(self, run_dir: Path, started_at: float, max_train_seconds: int) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.started_at = started_at
        self.max_train_seconds = max_train_seconds
        self.time_limit_reached = False

    def _on_step(self) -> bool:
        elapsed = int(time.monotonic() - self.started_at)
        if elapsed < self.max_train_seconds:
            return True
        self.time_limit_reached = True
        update_training_status(
            self.run_dir,
            "time_limit_reached",
            {
                "timesteps_completed": self.num_timesteps,
                "elapsed_seconds": elapsed,
                "time_limit_seconds": self.max_train_seconds,
            },
        )
        print(
            f"Wall-clock training limit reached after {elapsed}s at {self.num_timesteps} timesteps",
            flush=True,
        )
        return False


class MlflowKVWriter(KVWriter):
    """SB3 logger output that mirrors EVERY dumped scalar to the active MLflow run.

    Registered in the model's logger (via ``model.set_logger``), so it fires at
    every ``logger.dump()`` with the COMPLETE ``name_to_value`` snapshot —
    ``rollout/*``, ``train/*``, ``dr/*`` and ``eval/*`` all reach MLflow. The
    previous rollout-end mirror read ``name_to_value`` inside ``collect_rollouts``,
    *before* ``rollout/ep_rew_mean`` / ``time/fps`` / all ``train/*`` (policy loss,
    entropy, clip fraction, explained variance) had been recorded — so those core
    signals never reached the MLflow UI. A ``KVWriter`` fires after ``dump()`` with
    everything present, and is version-stable across SB3 v1/v2.
    """

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        try:
            import mlflow
        except ImportError:
            return
        if mlflow.active_run() is None:
            return
        for key, value in key_values.items():
            # Scalar metrics only — skip images/figures/text/HParam and bools.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            try:
                mlflow.log_metric(key, float(value), step=int(step))
            except (TypeError, ValueError):
                continue

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


class RewardMetricsCallback(BaseCallback):
    """Drain per-episode DeepRacer metrics into the SB3 logger, with trend lines.

    The orchestrator wraps the env so every terminal step's ``info`` dict
    carries a ``dr_episode`` summary (see ``gym_dr/metrics.py``). This callback
    inspects ``self.locals["infos"]`` each step, picks up any finalized
    summaries, and pushes their keys/values into the logger. Three views of each
    ``dr/ep_*`` metric are logged (all reach TensorBoard + MLflow):

    - the raw per-rollout mean (``record_mean``) — noisy, one bad rollout (e.g.
      right after a track swap) is indistinguishable from a real regression;
    - ``<key>_ema`` — an exponential moving average across ALL episodes so far
      (``alpha`` close to 1 = smooth secular trend); one float, ~zero overhead;
    - ``<key>_win<W>`` — the mean over the last ``W`` episodes (local trend).

    The EMA/window state lives on the callback instance, which the trainer
    creates once and reuses across every chunk of a rotation — so the trend lines
    are continuous across track swaps, not reset per chunk.
    """

    def __init__(self, window: int = 100, ema_alpha: float = 0.99) -> None:
        super().__init__()
        self._window = max(1, int(window))
        self._alpha = min(max(float(ema_alpha), 0.0), 1.0)
        self._ema: "dict[str, float]" = {}
        self._buffers: "dict[str, deque]" = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            summary = info.get("dr_episode") if isinstance(info, dict) else None
            if not summary:
                continue
            for key, value in summary.items():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                self.logger.record_mean(key, v)
                # Cross-rollout trend lines (persist across chunks).
                prev = self._ema.get(key)
                self._ema[key] = v if prev is None else self._alpha * prev + (1.0 - self._alpha) * v
                buf = self._buffers.get(key)
                if buf is None:
                    buf = self._buffers[key] = deque(maxlen=self._window)
                buf.append(v)
                self.logger.record(f"{key}_ema", self._ema[key])
                self.logger.record(f"{key}_win{self._window}", sum(buf) / len(buf))
        return True


class HeartbeatCallback(BaseCallback):
    """Touch a heartbeat file periodically so the HOST can tell training is making
    progress (vs. a wedged-but-alive gzserver, which hangs silently at high CPU —
    see docs/reports/d3-hang-postmortem.md). The path comes from
    ``$GYM_DR_HEARTBEAT``; if unset the callback is a no-op. Touched every
    ``interval_steps`` env steps (cheap), and once on training start so the
    host's boot grace can end as soon as the first rollout begins."""

    def __init__(self, interval_steps: int = 256) -> None:
        super().__init__()
        self._interval = max(1, interval_steps)
        self._last = 0

    def _on_training_start(self) -> None:
        _touch_heartbeat()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self._interval:
            _touch_heartbeat()
            self._last = self.num_timesteps
        return True
