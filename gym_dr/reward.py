from __future__ import annotations

import inspect
import json
from typing import Callable

from gym_dr.config import RewardConfig

RewardFn = Callable[[dict], float]
RewardFactory = Callable[[dict], RewardFn]

_REGISTRY: dict[str, RewardFactory] = {}


def register(name: str):
    def decorator(fn: RewardFactory) -> RewardFactory:
        if name in _REGISTRY:
            raise ValueError(f"reward factory {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def make_reward(cfg: RewardConfig) -> RewardFn:
    if cfg.factory not in _REGISTRY:
        raise KeyError(
            f"unknown reward factory {cfg.factory!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[cfg.factory](cfg.params)


def factory_source(name: str) -> str:
    return inspect.getsource(_REGISTRY[name])


def render_reward_source(cfg: RewardConfig) -> str:
    header = (
        f"# Auto-generated for this run.\n"
        f"# factory = {cfg.factory!r}\n"
        f"# params  = {json.dumps(cfg.params, indent=2)}\n\n"
    )
    return header + factory_source(cfg.factory)


CENTER_LINE_DEFAULTS: dict[str, float] = {
    "marker_1_frac": 0.1,
    "marker_2_frac": 0.25,
    "marker_3_frac": 0.5,
    "reward_center": 100.0,
    "reward_mid": 0.5,
    "reward_outer": 0.1,
    "reward_off": 1e-3,
}


@register("center_line")
def center_line(params: dict) -> RewardFn:
    p = {**CENTER_LINE_DEFAULTS, **params}
    marker_1_frac = float(p["marker_1_frac"])
    marker_2_frac = float(p["marker_2_frac"])
    marker_3_frac = float(p["marker_3_frac"])
    reward_center = float(p["reward_center"])
    reward_mid = float(p["reward_mid"])
    reward_outer = float(p["reward_outer"])
    reward_off = float(p["reward_off"])

    def reward_function(params: dict) -> float:
        track_width = params["track_width"]
        distance_from_center = params["distance_from_center"]
        if distance_from_center <= marker_1_frac * track_width:
            return reward_center
        if distance_from_center <= marker_2_frac * track_width:
            return reward_mid
        if distance_from_center <= marker_3_frac * track_width:
            return reward_outer
        return reward_off

    return reward_function
