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
class RewardConfig:
    factory: str = "center_line"
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
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
    mlflow_tracking_uri: str = "file:///workspace/mlruns"
    mlflow_experiment: str = "gym-dr"
    tensorboard: bool = True
    tags: dict[str, str] = field(default_factory=dict)


def _default_env_factory():
    from gym_dr.envs import deepracer_env_v1

    return deepracer_env_v1


def _default_trainer():
    from gym_dr.trainers import Sb3Trainer

    return Sb3Trainer()


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    world_name: str = "reinvent_base"
    env_factory: Callable[["ExperimentConfig"], Any] = field(default_factory=_default_env_factory)
    trainer: "Trainer" = field(default_factory=_default_trainer)
    reward: RewardConfig = field(default_factory=RewardConfig)
    action_space: ActionSpaceConfig = field(default_factory=ContinuousActionSpaceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "world_name": self.world_name,
            "env_factory": _describe_callable(self.env_factory),
            "trainer": _describe(self.trainer),
            "reward": dataclasses.asdict(self.reward),
            "action_space": {
                **dataclasses.asdict(self.action_space),
                "action_space_type": self.action_space.action_space_type,
            },
            "training": dataclasses.asdict(self.training),
            "tracking": dataclasses.asdict(self.tracking),
        }
        return d

    def flat_params(self) -> dict[str, Any]:
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
