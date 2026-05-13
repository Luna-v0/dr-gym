"""Typed configuration dataclasses.

``ExperimentConfig`` is the single object the user composes in ``app.py``.
It carries everything ``gym_dr.train(experiment)`` needs to run a training:
which env to build, which trainer to use, which reward function, which
action space, which world(s), how long to train, and where to log.

All dataclasses are ``frozen=True`` so they hash; HPO mutates them through
``with_overrides(**flat_dotted_keys)`` which uses ``dataclasses.replace``
to return a new instance.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from gym_dr.action_space import ActionSpaceConfig, ContinuousActionSpaceConfig

if TYPE_CHECKING:
    from gym_dr.trainers.base import Trainer


@dataclass(frozen=True)
class TrainingConfig:
    """Per-chunk training control.

    A *chunk* is one ``model.learn`` call: one container, one ``WORLD_NAME``.
    Multi-world runs string several chunks together — see ``WorldsConfig``.
    """

    total_timesteps: int = 500_000
    checkpoint_freq: int = 1_000
    max_train_seconds: int | None = None
    status_update_steps: int = 1_000
    status_update_seconds: int = 30
    resume_from: str | None = None
    rtf_override: int | None = None
    eval_freq: int = 5_000
    n_eval_episodes: int = 3


@dataclass(frozen=True)
class TrackingConfig:
    """MLflow + TensorBoard settings.

    The default ``mlflow_tracking_uri`` is a **relative** file URI so it
    resolves consistently on both sides of the host/container boundary:

    - On the host, ``python app.py`` runs from the project dir; ``./mlruns``
      lands at ``<project_dir>/mlruns``.
    - Inside the container, the Dockerfile CMD does ``cd /workspace`` first,
      so ``./mlruns`` resolves to ``/workspace/mlruns`` — the same dir, via
      the ``-v <project_dir>/mlruns:/workspace/mlruns`` bind mount.

    Override this only if you want a remote MLflow server.
    """

    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "gym-dr"
    tensorboard: bool = True
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldsConfig:
    """Worlds to rotate through during a single training run.

    Multi-world runs use *sequential rotation with shared policy*: the
    orchestrator trains for ``chunk_steps`` timesteps on the first world,
    saves a checkpoint, restarts the container with the next ``WORLD_NAME``
    and ``RESUME_FROM`` pointing at that checkpoint, and continues. The
    optimizer state and weights persist across switches (off-policy replay
    buffers, if any, are lost — PPO has none).

    Example::

        worlds = WorldsConfig(
            names=["reinvent_base", "Bowtie_track"],
            chunk_steps=20_000,
            rotations=3,
        )

    runs 6 chunks of 20k timesteps each:
    reinvent_base -> Bowtie_track -> reinvent_base -> ... -> Bowtie_track.

    For valid world names see ``.deepracer-env-upstream/tracks.txt``. The
    current upstream simapp loads the world at container startup and cannot
    switch at runtime; see the README's "Future work" section for the
    runtime-switch design.
    """

    names: list[str] = field(default_factory=lambda: ["reinvent_base"])
    chunk_steps: int = 50_000
    rotations: int = 1


def _default_env_factory():
    from gym_dr.envs import time_trial

    return time_trial


def _default_trainer():
    from gym_dr.trainers import Sb3Trainer

    return Sb3Trainer()


def _default_reward():
    from gym_dr.rewards import center_line

    return center_line


@dataclass(frozen=True)
class ExperimentConfig:
    """A full training experiment definition.

    Compose one of these in your ``app.py``, then call
    ``gym_dr.train(experiment)``. The orchestrator handles host-vs-container
    mode dispatch, multi-world rotation, MLflow tracking, and artifact
    layout — your code only has to declare *what* to train.

    Plug-in points
    --------------
    - ``env_factory``: swap the env. Default ``gym_dr.envs.time_trial`` builds
      a single-agent time-trial ``DeepRacerEnv``. To use a different race
      type (object avoidance etc.) or a future env version, write a sibling
      factory under ``gym_dr/envs/`` and reference it here.
    - ``trainer``: swap the RL algorithm/library. Default
      ``gym_dr.trainers.Sb3Trainer()`` wraps SB3 PPO/SAC/TD3/A2C/DDPG. Any
      object with ``fit(env, ctx) -> TrainResult`` satisfies the protocol.
    - ``reward``: plain ``(params: dict) -> float`` callable. Receives the
      upstream DeepRacer reward params dict (see ``gym_dr/rewards.py`` for
      the key list and example functions).
    - ``action_space``: continuous bounds or a discrete action list.
    - ``worlds``: list of world names to rotate through.
    """

    name: str
    env_factory: Callable[["ExperimentConfig"], Any] = field(default_factory=_default_env_factory)
    trainer: "Trainer" = field(default_factory=_default_trainer)
    reward: Callable[[dict], float] = field(default_factory=_default_reward)
    action_space: ActionSpaceConfig = field(default_factory=ContinuousActionSpaceConfig)
    worlds: WorldsConfig = field(default_factory=WorldsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON dump / MLflow logging.

        Callables serialize as ``module.qualname`` strings. The trainer is
        special-cased: if it's a dataclass we ``asdict()`` it so its kwargs
        survive the round-trip.
        """
        return {
            "name": self.name,
            "env_factory": _describe_callable(self.env_factory),
            "trainer": _describe(self.trainer),
            "reward": _describe_callable(self.reward),
            "action_space": {
                **dataclasses.asdict(self.action_space),
                "action_space_type": self.action_space.action_space_type,
            },
            "worlds": dataclasses.asdict(self.worlds),
            "training": dataclasses.asdict(self.training),
            "tracking": dataclasses.asdict(self.tracking),
        }

    def flat_params(self) -> dict[str, Any]:
        """Flatten ``to_dict()`` into dotted keys for ``mlflow.log_params``."""
        flat: dict[str, Any] = {}

        def walk(prefix: str, val: Any) -> None:
            if isinstance(val, dict):
                if not val:
                    flat[prefix] = "{}"
                    return
                for k, v in val.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(val, (list, tuple)):
                flat[prefix] = json.dumps(val)
            elif val is None:
                flat[prefix] = ""
            else:
                flat[prefix] = val

        walk("", self.to_dict())
        return flat

    def with_overrides(self, **overrides: Any) -> ExperimentConfig:
        """Return a new ExperimentConfig with dotted-key overrides applied.

        Walks dataclass fields and dict-typed fields. Examples::

            cfg.with_overrides(name="trial_3")
            cfg.with_overrides(**{"trainer.kwargs.learning_rate": 1e-4})

        Used by HPO to mutate a base experiment per trial, and by the
        in-container chunk dispatcher to apply per-chunk env-var overrides.
        """
        return _apply_overrides(self, overrides)


def _describe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        d["__class__"] = f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"
        return d
    return _describe_callable(obj)


def _describe_callable(obj: Any) -> str:
    mod = getattr(obj, "__module__", "?")
    name = getattr(obj, "__qualname__", repr(obj))
    return f"{mod}.{name}"


def _apply_overrides(obj: Any, overrides: dict[str, Any]) -> Any:
    grouped: dict[str, dict[str, Any]] = {}
    leaves: dict[str, Any] = {}
    for key, val in overrides.items():
        if "." in key:
            top, rest = key.split(".", 1)
            grouped.setdefault(top, {})[rest] = val
        else:
            leaves[key] = val

    replacements: dict[str, Any] = dict(leaves)
    for top, sub in grouped.items():
        current = getattr(obj, top)
        if dataclasses.is_dataclass(current):
            replacements[top] = _apply_overrides(current, sub)
        elif isinstance(current, dict):
            new_dict = dict(current)
            for sub_key, sub_val in sub.items():
                _set_nested(new_dict, sub_key.split("."), sub_val)
            replacements[top] = new_dict
        else:
            raise ValueError(
                f"Cannot apply nested override {top}.{next(iter(sub))} to non-dataclass field"
            )
    return dataclasses.replace(obj, **replacements)


def _set_nested(d: dict, path: list[str], value: Any) -> None:
    cursor = d
    for key in path[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    cursor[path[-1]] = value


def load_config(path: str | Path) -> ExperimentConfig:
    """Import a Python file and return its ``experiment`` module attribute.

    Used by the in-container worker to load the same script the host ran.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "experiment"):
        raise ValueError(f"{p} must export `experiment: ExperimentConfig`")
    cfg = module.experiment
    if not isinstance(cfg, ExperimentConfig):
        raise TypeError(f"{p} `experiment` must be ExperimentConfig, got {type(cfg)}")
    return cfg


def load_search_space(path: str | Path):
    """Import a Python file and return its ``search_space`` module attribute.

    The function should take an Optuna trial and return a flat dotted-key
    overrides dict consumable by ``ExperimentConfig.with_overrides``.
    """
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "search_space", None)
    if fn is None:
        raise ValueError(f"{p} must export `search_space(trial) -> dict` for HPO")
    return fn
