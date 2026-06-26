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
    model_path: Path,
    app_path: Path | None = None,
    run_config_path: Path | None = None,
) -> ExperimentConfig:
    """Get the ``ExperimentConfig`` to evaluate ``model_path`` under.

    - With ``app_path``: load that experiment script directly. Use this if
      the run used callables defined *inline* in the script (which can't be
      resolved by import path).
    - With ``run_config_path``: reconstruct from that explicit
      ``run_config.json`` (the ``--run-config`` override).
    - Otherwise (the default): reconstruct from the run_config.json found
      next to the model, or — if the model sits in a subdir like
      ``best_model/`` — in the nearest ancestor directory. Every training
      run writes that file with the fully-resolved config (``env_factory``
      and ``reward`` as dotted import paths, ``action_space`` as a dict,
      ``trainer.frame_stack``), which is everything evaluation needs.
    """
    model_path = Path(model_path)
    if app_path is not None:
        from gym_dr.config import load_config

        return load_config(app_path)

    run_config = _load_run_config(model_path, run_config_path)
    if run_config is None:
        if run_config_path is not None:
            raise FileNotFoundError(
                f"--run-config {run_config_path} not found or not valid JSON"
            )
        raise FileNotFoundError(
            f"no run_config.json found next to {model_path} or in any parent "
            "directory — pass --run-config PATH (or --app) explicitly"
        )
    return _reconstruct_experiment(run_config)


def _find_run_config(model_path: Path, explicit: Path | None = None) -> Path | None:
    """Locate the ``run_config.json`` governing ``model_path``.

    Search order:
      1. ``explicit`` if given (the ``--run-config`` override).
      2. ``run_config.json`` next to the model, or ``<model>.run_config.json``.
      3. ``run_config.json`` in the nearest ancestor directory — models are
         often nested in a ``best_model/`` or ``checkpoints/`` subdir of the
         run dir, with the config one level up.
    """
    if explicit is not None:
        return explicit if explicit.exists() else None
    model_path = Path(model_path)
    siblings = (
        model_path.parent / "run_config.json",
        model_path.with_suffix(".run_config.json"),
    )
    for cfg_path in siblings:
        if cfg_path.exists():
            return cfg_path
    for ancestor in model_path.parent.parents:
        cfg_path = ancestor / "run_config.json"
        if cfg_path.exists():
            return cfg_path
    return None


def _load_run_config(model_path: Path, explicit: Path | None = None) -> dict | None:
    cfg_path = _find_run_config(model_path, explicit)
    if cfg_path is None:
        return None
    try:
        return json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError):
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
    ``action_space``, ``trainer.frame_stack``, and ``use_gpu`` (so the host
    launcher selects the matching image arch). The policy architecture is
    *not* rebuilt from config — SB3 loads it whole from the model zip — so
    ``trainer.kwargs`` is intentionally left empty.
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
        # ``build_env`` dispatches on (n_cars, camera_obs): a feature-obs model
        # (camera_obs=False) needs the feature path, else the env yields a camera
        # DICT obs that mismatches the policy's Box feature space. These default to
        # (1, True) on ExperimentConfig, so they MUST be carried from the run_config
        # or a feature model fails to evaluate. (DR is intentionally NOT carried —
        # eval runs the policy's clean deterministic behaviour, no injected noise /
        # random start, which is what view-mode is for.)
        camera_obs=bool(rc.get("camera_obs", True)),
        n_cars=int(rc.get("n_cars", 1)),
        # Carry through how the model was trained so the host launcher picks
        # the matching image arch (gpu vs cpu) and GPU access.
        use_gpu=bool(rc.get("use_gpu", False)),
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
    run_config_path: Path | None = None,
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
        frame_stack = _frame_stack_from_run_config(model_path, run_config_path)
    LOG.info("evaluating %s (frame_stack=%d)", model_path, frame_stack)

    # Metrics wrapper gives us info["dr_episode"] summaries for free.
    wrapped_experiment, env_wrapper, _metrics_state = install_metrics(experiment)
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


def evaluate_on_tracks(
    experiment: ExperimentConfig,
    model_path: Path,
    tracks: list[str],
    *,
    n_episodes: int = 5,
    frame_stack: int | None = None,
    run_config_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Score a trained model on each of ``tracks`` OUT OF THE TRAINING LOOP.

    For each track: hot-swap the env to it, run ``n_episodes`` with
    ``deterministic=True``, and aggregate the success-criterion metrics from the
    per-episode ``dr_episode`` summaries:

      - ``clean_completion_rate`` — fraction of episodes that finished the lap
        with zero off-track steps (the headline);
      - ``completion_rate`` — finished the lap (off-track allowed);
      - ``mean_max_progress`` / ``mean_speed`` / ``mean_offtrack_rate``.

    Used by ``scripts/eval_physical_tracks.py`` to report sim-to-real transfer on
    the maintainer's physical tracks (``reInvent2019_track``, ``Oval_track``),
    which are reserved out of training/eval (see ``docs/eval-protocol.md``).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from gym_dr.export import load_sb3_zip
    from gym_dr.metrics import install_metrics

    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if frame_stack is None:
        frame_stack = _frame_stack_from_run_config(model_path, run_config_path)

    wrapped_experiment, env_wrapper, _state = install_metrics(experiment)
    base_env = env_wrapper(wrapped_experiment.env_factory(wrapped_experiment))
    venv = DummyVecEnv([lambda: base_env])
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    model = load_sb3_zip(model_path)

    def _mean(summaries: list[dict], key: str) -> float:
        vals = [float(s.get(key, 0.0)) for s in summaries]
        return sum(vals) / max(1, len(vals))

    results: dict[str, dict[str, float]] = {}
    try:
        for track in tracks:
            try:
                venv.env_method("set_world", track)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("could not set_world(%s): %s — skipping", track, exc)
                continue
            obs = venv.reset()
            summaries: list[dict] = []
            while len(summaries) < n_episodes:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, dones, infos = venv.step(action)
                if bool(dones[0]):
                    summaries.append(infos[0].get("dr_episode", {}) or {})
            results[track] = {
                "clean_completion_rate": _mean(summaries, "dr/ep_completed_clean"),
                "completion_rate": _mean(summaries, "dr/ep_completed"),
                "mean_max_progress": _mean(summaries, "dr/ep_max_progress"),
                "mean_speed": _mean(summaries, "dr/ep_mean_speed"),
                "mean_offtrack_rate": _mean(summaries, "dr/ep_offtrack_rate"),
                "n_episodes": float(len(summaries)),
            }
            r = results[track]
            LOG.info(
                "[%s] clean_completion=%.2f  completion=%.2f  progress=%.1f%%  "
                "speed=%.2f  offtrack_rate=%.2f  (n=%d)",
                track, r["clean_completion_rate"], r["completion_rate"],
                r["mean_max_progress"], r["mean_speed"], r["mean_offtrack_rate"],
                int(r["n_episodes"]),
            )
    finally:
        venv.close()
    return results


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _frame_stack_from_run_config(
    model_path: Path, explicit: Path | None = None
) -> int:
    """Read ``trainer.frame_stack`` from the model's run_config.json, default 1.

    Uses the same lookup as the experiment reconstruction (explicit override,
    then sibling, then nearest ancestor). If the config is missing or doesn't
    carry the field (e.g. a model produced before frame_stack existed),
    default 1.
    """
    cfg = _load_run_config(model_path, explicit)
    if cfg is not None:
        fs = cfg.get("trainer", {}).get("frame_stack")
        if isinstance(fs, int) and fs >= 1:
            LOG.info("frame_stack=%d (from run_config)", fs)
            return fs
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
