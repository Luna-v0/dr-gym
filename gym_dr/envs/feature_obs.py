"""Feature-vector observation (camera-off path).

When ``ExperimentConfig.camera_obs=False`` the policy observes a low-dim vector of
frame-local driving features built from the privileged ``reward_params``
(``gym_dr.perception.all_targets``) instead of the camera. Two payoffs: the
policy learns *control* in far fewer steps than a CNN learning perception+control
jointly, and — paired with the sim camera-off toggle — no rendering is needed, so
each sim step is much cheaper (``docs/reports/multi-car.md``).

This wrapper only does the **observation transform**; the ``reward_params`` are
supplied by a ``params_source`` callable (the env factory wires a reward tap that
captures them, the same pattern as ``scripts/collect_perception_data.py``). The
reward the policy receives is unchanged — it still operates on ``reward_params``,
so a reward/policy transfers between the camera and feature observation spaces.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from gym_dr.perception import ALL_FEATURES, all_targets


class FeatureObsWrapper(gym.Wrapper):
    """Replace the env's observation with the ``ALL_FEATURES`` vector built from
    the latest ``reward_params`` (provided by ``params_source``).

    The observation space becomes ``Box(-1, 1, (len(ALL_FEATURES),))`` — every
    feature is already normalised to ``[-1,1]`` (signed) or ``[0,1]`` (bounded).
    Tracks the previous params so the temporal/dynamic features (accel, etc.)
    finite-difference correctly; resets that history on ``reset``.
    """

    def __init__(self, env: gym.Env, params_source: Callable[[], Optional[dict]],
                 features: tuple = ALL_FEATURES, targets_fn: Callable = all_targets) -> None:
        super().__init__(env)
        self._params_source = params_source
        self._prev: Optional[dict] = None
        self._targets_fn = targets_fn           # all_targets (9) or actor_targets (11)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(len(features),), dtype=np.float32
        )

    def _features(self) -> np.ndarray:
        params = self._params_source() or {}
        feat = self._targets_fn(params, self._prev).astype(np.float32)
        if params:
            self._prev = dict(params)
        return feat

    def reset(self, **kwargs: Any):
        _obs, info = self.env.reset(**kwargs)
        self._prev = None
        return self._features(), info

    def step(self, action: Any):
        _obs, reward, terminated, truncated, info = self.env.step(action)
        return self._features(), reward, terminated, truncated, info
