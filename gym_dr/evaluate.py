"""Run a trained model in the simulator for inspection — no training.

This is "view mode": load a saved SB3 model, drive the DeepRacer env with
``model.predict(deterministic=True)``, and print per-step + per-episode
detail. Pair it with ``ExperimentConfig.enable_gui=True`` (forced on by
``scripts/evaluate.py``) to watch the car over VNC while the numbers stream
to stdout.

``run_evaluation`` runs *inside the container*. The host-side launcher is
``scripts/evaluate.py``, which spawns the Docker container with the GUI
enabled and the right env vars — mirroring how ``app.py`` dispatches
host-vs-container for training.

Frame stacking: a model trained with ``Sb3Trainer(frame_stack=N>1)`` expects
stacked observations. ``run_evaluation`` reads ``trainer.frame_stack`` from
the model's sibling ``run_config.json`` (written by every training run) and
re-applies the same ``VecFrameStack`` so the observation shape matches what
the policy was trained on. Pass ``frame_stack=`` explicitly to override.
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

from gym_dr.config import ExperimentConfig

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Resolving which experiment to evaluate a model under
# --------------------------------------------------------------------------- #

def experiment_for_model(
    model_path: Path, app_path: Path | None = None
) -> ExperimentConfig:
    """Get the ``ExperimentConfig`` to evaluate ``model_path`` under.

    - With ``app_path``: load that experiment script directly. Use this if
      the run used callables defined *inline* in the script (which can't be
      resolved by import path).
    - Without ``app_path`` (the default): reconstruct from the model's
      sibling ``run_config.json``. Every training run writes that file with
      the fully-resolved config — ``env_factory`` and ``reward`` as dotted
      import paths, ``action_space`` as a dict, ``trainer.frame_stack`` —
      which is everything evaluation needs. No ``app.py`` required.
    """
    model_path = Path(model_path)
    if app_path is not None:
        from gym_dr.config import load_config

        return load_config(app_path)

    run_config = _load_run_config(model_path)
    if run_config is None:
        raise FileNotFoundError(
            f"no run_config.json next to {model_path} — pass --app explicitly "
            "(the run dir is the directory the model .zip lives in)"
        )
    return _reconstruct_experiment(run_config)


def _load_run_config(model_path: Path) -> dict | None:
    for cfg_path in (
        model_path.parent / "run_config.json",
        model_path.with_suffix(".run_config.json"),
    ):
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
    return None


def _resolve_dotted(path: str):
    """Resolve a ``module.qualname`` string back to the live object."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"not a dotted import path: {path!r}")
    mod = importlib.import_module(module_path)
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _rebuild_action_space(d: dict):
    from gym_dr.action_space import (
        ContinuousActionSpaceConfig,
        DiscreteAction,
        DiscreteActionSpaceConfig,
    )

    d = dict(d)
    kind = d.pop("action_space_type", "continuous")
    if kind == "discrete":
        actions = [DiscreteAction(**a) for a in d.pop("actions", [])]
        return DiscreteActionSpaceConfig(actions=actions, **d)
    d.pop("actions", None)  # not present for continuous, but be defensive
    return ContinuousActionSpaceConfig(**d)


def _reconstruct_experiment(rc: dict) -> ExperimentConfig:
    """Rebuild a minimal ExperimentConfig from a serialized run_config.json.

    Only the pieces evaluation actually needs are reconstructed faithfully:
    ``env_factory`` + ``reward`` (resolved from their dotted paths),
    ``action_space``, and ``trainer.frame_stack``. The policy architecture
    is *not* rebuilt from config — SB3 loads it whole from the model zip —
    so ``trainer.kwargs`` is intentionally left empty.
    """
    from gym_dr.config import TrackingConfig, TrainingConfig, WorldsConfig
    from gym_dr.trainers import Sb3Trainer

    trainer_d = rc.get("trainer", {}) or {}
    worlds_d = rc.get("worlds", {}) or {}

    return ExperimentConfig(
        name=rc.get("name", "eval"),
        env_factory=_resolve_dotted(rc["env_factory"]),
        reward=_resolve_dotted(rc["reward"]),
        action_space=_rebuild_action_space(rc["action_space"]),
        # Only frame_stack matters for eval — SB3 loads the policy from the
        # zip, so the rest of the trainer config is irrelevant here.
        trainer=Sb3Trainer(frame_stack=int(trainer_d.get("frame_stack", 1))),
        worlds=WorldsConfig(
            names=list(worlds_d.get("names", ["reinvent_base"])),
            chunk_steps=int(worlds_d.get("chunk_steps", 50_000)),
            rotations=int(worlds_d.get("rotations", 1)),
        ),
        training=TrainingConfig(),
        tracking=TrackingConfig(),
    )


def run_evaluation(
    experiment: ExperimentConfig,
    model_path: Path,
    *,
    n_episodes: int = 5,
    loop: bool = False,
    frame_stack: int | None = None,
    step_log_every: int = 20,
) -> list[dict[str, Any]]:
    """Drive ``model_path`` through the env for inspection. Returns per-episode summaries.

    Args:
        experiment: supplies ``env_factory``, ``reward``, ``action_space``.
        model_path: a saved SB3 ``.zip``.
        n_episodes: how many episodes to run (ignored if ``loop`` is True).
        loop: run forever until interrupted — handy for just watching.
        frame_stack: override the stack depth. Default: read from the
            model's sibling ``run_config.json``, else 1.
        step_log_every: print a compact per-step line every N steps (long
            episodes would otherwise flood stdout). Per-episode summaries
            always print.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from gym_dr.export import load_sb3_zip
    from gym_dr.metrics import install_metrics

    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if frame_stack is None:
        frame_stack = _frame_stack_from_run_config(model_path)
    LOG.info("evaluating %s (frame_stack=%d)", model_path, frame_stack)

    # Metrics wrapper gives us info["dr_episode"] summaries for free.
    wrapped_experiment, env_wrapper = install_metrics(experiment)
    base_env = env_wrapper(wrapped_experiment.env_factory(wrapped_experiment))

    venv = DummyVecEnv([lambda: base_env])
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)

    model = load_sb3_zip(model_path)

    summaries: list[dict[str, Any]] = []
    episode = 0
    try:
        obs = venv.reset()
        step = 0
        while loop or episode < n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            step += 1

            if step % max(1, step_log_every) == 0:
                LOG.info(
                    "ep %d step %d  reward=%.4f%s",
                    episode,
                    step,
                    float(rewards[0]),
                    _step_detail(infos[0]),
                )

            if bool(dones[0]):
                summary = infos[0].get("dr_episode", {})
                summaries.append(summary)
                LOG.info(
                    "── episode %d done in %d steps ──\n%s",
                    episode,
                    step,
                    _format_summary(summary),
                )
                episode += 1
                step = 0
                # DummyVecEnv auto-resets; obs already holds the new episode's
                # first observation.
    except KeyboardInterrupt:
        LOG.info("evaluation interrupted after %d episode(s)", episode)
    finally:
        venv.close()

    return summaries


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _frame_stack_from_run_config(model_path: Path) -> int:
    """Read ``trainer.frame_stack`` from the model's run dir, default 1.

    Every training run writes ``run_config.json`` into the run dir alongside
    the model zips. We look there first; if it's missing or doesn't carry
    the field (e.g. a model produced before frame_stack existed), default 1.
    """
    candidates = [
        model_path.parent / "run_config.json",
        model_path.with_suffix(".run_config.json"),
    ]
    for cfg_path in candidates:
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                fs = cfg.get("trainer", {}).get("frame_stack")
                if isinstance(fs, int) and fs >= 1:
                    LOG.info("frame_stack=%d (from %s)", fs, cfg_path.name)
                    return fs
            except (json.JSONDecodeError, OSError):
                pass
    LOG.info("frame_stack not found in run_config — defaulting to 1")
    return 1


def _step_detail(info: dict) -> str:
    """Compact tail of interesting per-step keys, if the env's info has them.

    The real DeepRacer env returns its agents-info map as ``info``; the stub
    env returns ``{}``. Either way this stays terse.
    """
    keys = ("speed", "steering_angle", "progress", "is_offtrack", "all_wheels_on_track")
    bits = [f"{k}={info[k]}" for k in keys if k in info]
    return ("  " + "  ".join(bits)) if bits else ""


def _format_summary(summary: dict) -> str:
    if not summary:
        return "  (no dr_episode summary — env didn't emit metrics)"
    order = [
        "dr/ep_reward",
        "dr/ep_length",
        "dr/ep_max_progress",
        "dr/ep_offtrack_count",
        "dr/ep_offtrack_rate",
        "dr/ep_crash_count",
        "dr/ep_mean_speed",
        "dr/ep_mean_steering_abs",
    ]
    lines = []
    for key in order:
        if key in summary:
            lines.append(f"  {key:28s} {summary[key]:.4f}")
    return "\n".join(lines)
