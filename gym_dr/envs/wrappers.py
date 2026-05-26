"""Observation wrappers for the DeepRacer env.

``GrayscaleObs`` converts the camera observation from RGB to single-channel
grayscale, matching what the physical AWS DeepRacer car feeds its model.

Why this lives in the *env pipeline* (not inside the network): the AWS car's
inference node does the RGB/BGR -> gray conversion before the model
(``aws-deepracer-inference-pkg/.../image_process.cpp``). Doing it here means
the observation *space* itself becomes 1-channel — so frame-stacking stacks
grayscale frames (as AWS does), and the ONNX/.pb export's input is grayscale,
matching what the car will actually feed it. If the conversion lived in the
network, the exported model would still expect RGB.

Conversion: ITU-R BT.601 luma weights ``0.299 R + 0.587 G + 0.114 B`` — the
same weights ``cv2.COLOR_RGB2GRAY`` uses, and what AWS's training filter
(``ObservationRGBToYFilter``) and on-device ``cv2.COLOR_BGR2GRAY`` both use.
The community ``seresheim/deepracer-env`` sim emits RGB frames, so we use the
RGB ordering. Output stays uint8 [0,255] — AWS does **not** normalize before
the network (and neither should we, if the exported model is to match).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

# ITU-R BT.601 luma weights (R, G, B).
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class ActionBounds(gym.ActionWrapper):
    """Re-bound the env's continuous action space to ``[(s_lo, v_lo), (s_hi, v_hi)]``.

    The upstream ``DeepRacerEnv`` always exposes ``Box([-30, 0.1], [30, 4.0])``
    and ``rollout_agent_ctrl.py`` hardcodes ``MIN_SPEED=0.1`` — so passing a
    tighter ``ContinuousActionSpaceConfig`` to the gym factory has no effect
    on what speeds the policy can command (or what the env will execute)
    unless we also enforce it here.

    Concretely: this wrapper reports the tighter Box to PPO (so its Gaussian
    is parametrised over the right range) AND clips every commanded action
    before it reaches the inner env (so the upstream MIN_SPEED clip becomes
    a no-op for actions that are already above our floor).

    Use this when you need a hard minimum speed — e.g. to stop the policy
    from "winning" by crawling at 0.1 m/s.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        steering_low: float,
        steering_high: float,
        speed_low: float,
        speed_high: float,
    ) -> None:
        super().__init__(env)
        self._low = np.array([steering_low, speed_low], dtype=np.float32)
        self._high = np.array([steering_high, speed_high], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=self._low, high=self._high, dtype=np.float32
        )

    def action(self, action):
        a = np.asarray(action, dtype=np.float32)
        return np.clip(a, self._low, self._high)


class GrayscaleObs(gym.ObservationWrapper):
    """Convert RGB image keys in a Dict observation to single-channel gray.

    Image keys with a trailing dim of 3 (``(H, W, 3)``) become ``(H, W, 1)``
    uint8. Non-image keys and already-grayscale keys pass through untouched.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError(
                f"GrayscaleObs expects a Dict observation space, got "
                f"{type(env.observation_space).__name__}"
            )
        new_spaces: dict[str, gym.Space] = {}
        self._gray_keys: list[str] = []
        for key, space in env.observation_space.spaces.items():
            if _is_rgb(space):
                self._gray_keys.append(key)
                h, w, _ = space.shape
                new_spaces[key] = gym.spaces.Box(
                    low=0, high=255, shape=(h, w, 1), dtype=np.uint8
                )
            else:
                new_spaces[key] = space
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, observation: dict) -> dict:
        out = dict(observation)
        for key in self._gray_keys:
            rgb = np.asarray(observation[key], dtype=np.float32)
            gray = rgb @ _LUMA  # (H, W, 3) . (3,) -> (H, W)
            out[key] = np.clip(gray, 0, 255).astype(np.uint8)[..., None]  # (H, W, 1)
        return out


def _is_rgb(space: Any) -> bool:
    return (
        isinstance(space, gym.spaces.Box)
        and len(space.shape) == 3
        and space.shape[-1] == 3
    )
