"""Env-factory dispatch over ``(n_cars, camera_obs)`` — the composable 2x2.

The default ``ExperimentConfig.env_factory`` routes here, so a single config
selects single/multi-car × camera/feature without changing experiment code:

    n_cars=1, camera_obs=True   -> time_trial            (classic single-car camera)
    n_cars=1, camera_obs=False  -> feature_time_trial    (single-car feature obs, camera-off)
    n_cars>1, camera_obs=True   -> multi_car (camera)     (N cars, each a camera sub-env)
    n_cars>1, camera_obs=False  -> multi_car (feature)    (N feature cars, no rendering)

Everything else (reward, curriculum, DR, trace, action norm) is shared across the
four. See ``docs/reports/multi-car.md``.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from gym_dr.envs.feature_obs import FeatureObsWrapper
from gym_dr.envs.time_trial import time_trial

if TYPE_CHECKING:
    from gym_dr.config import ExperimentConfig


def feature_time_trial(experiment: "ExperimentConfig") -> Any:
    """Single-car env whose observation is the ``ALL_FEATURES`` vector built from
    ``reward_params`` (camera-off path). Wraps the standard ``time_trial`` env: a
    reward tap captures each step's params (reusing the collector pattern), and
    ``FeatureObsWrapper`` turns them into the observation. The policy's reward is
    unchanged (the tap forwards to it), so a reward transfers to/from camera obs.
    """
    captured: dict = {}
    inner_reward = experiment.reward

    def _tap(params: dict) -> float:
        captured.clear()
        captured.update(params)
        return inner_reward(params)

    # Feature obs reads reward_params (pose/track position), NOT pixels, so drop
    # the camera sensor: the base env's CompositeSensor then returns {} (no
    # blocking image read at reset) and nothing renders — mirrors the multi-car
    # feature path (gym_dr/envs/multi_car.py:432).
    cam_free = dataclasses.replace(experiment.action_space, sensor=[])
    env = time_trial(dataclasses.replace(experiment, reward=_tap, action_space=cam_free))
    # GYM_DR_FEATURE_SET=actor_extended -> the 11-feature actor vector (9 ⊕
    # curvature_ahead, nearest_object_dist); default keeps the validated 9.
    # feature_noise (DR) perturbs the actor's vector; GYM_DR_ASYM_CRITIC=1 makes the
    # obs a Dict{actor:noised, critic:true} for the asymmetric value net.
    import os
    from gym_dr.randomization import spec_bounds
    dr = getattr(experiment, "domain_randomization", None)
    fnoise = spec_bounds(getattr(dr, "feature_noise", 0.0))[1] if dr is not None else 0.0
    asym = os.getenv("GYM_DR_ASYM_CRITIC") == "1"
    seed = getattr(dr, "seed", None) if dr is not None else None
    if os.getenv("GYM_DR_FEATURE_SET") == "actor_extended":
        from gym_dr.perception import ACTOR_FEATURES, actor_targets
        return FeatureObsWrapper(env, lambda: captured,
                                 features=ACTOR_FEATURES, targets_fn=actor_targets,
                                 feature_noise=fnoise, asymmetric=asym, seed=seed)
    return FeatureObsWrapper(env, lambda: captured,
                             feature_noise=fnoise, asymmetric=asym, seed=seed)


def build_env(experiment: "ExperimentConfig") -> Any:
    """Dispatch to the right env for ``(n_cars, camera_obs)``."""
    n_cars = int(getattr(experiment, "n_cars", 1) or 1)
    camera = bool(getattr(experiment, "camera_obs", True))
    if n_cars <= 1:
        return time_trial(experiment) if camera else feature_time_trial(experiment)
    from gym_dr.envs.multi_car import multi_car  # lazy: multi-agent backend

    return multi_car(experiment)
