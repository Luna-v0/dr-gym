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


class NormalizeActions(gym.ActionWrapper):
    """Present a symmetric ``[-1, 1]`` action space to the policy and map it
    linearly onto the inner env's (engineering-unit) Box.

    PPO's diagonal-Gaussian policy initializes ``log_std = 0`` (std ≈ 1.0) *per
    action dimension, in the action space's own units*. Against the raw
    DeepRacer Box ``[-30,30] × [speed_low, speed_high]`` that means steering
    explores only ~±1° (≈1.7% of its range) while speed explores ±1 m/s
    (≈33%) — steering is barely explored, so the policy struggles to learn to
    corner (see ``docs/reports/q1-generalization.md``). Rescaling every
    dimension to ``[-1, 1]`` makes the unit Gaussian explore each dimension
    comparably.

    The inner env — and therefore the simapp, ``model_metadata.json`` and the
    ONNX export — keeps operating in engineering units; only the action space
    the *policy* sees changes. Wrap this OUTSIDE ``ActionBounds`` so ``[-1, 1]``
    maps onto the configured ``[speed_low, speed_high]`` (etc.) range.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        inner = env.action_space
        if not isinstance(inner, gym.spaces.Box):
            raise TypeError(
                f"NormalizeActions expects a Box action space, got "
                f"{type(inner).__name__}"
            )
        self._low = np.asarray(inner.low, dtype=np.float32)
        self._high = np.asarray(inner.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=inner.shape, dtype=np.float32
        )

    def action(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # [-1, 1] -> [low, high]
        return self._low + (a + 1.0) * 0.5 * (self._high - self._low)


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


class ActuatorNoise(gym.ActionWrapper):
    """Add Gaussian noise to the commanded action — actuator/calibration drift DR.

    Noise is in **engineering units** (deg, m/s). Wrap it OUTSIDE ``ActionBounds``
    (so the inner clip re-bounds the noisy command into the valid range) and
    INSIDE ``NormalizeActions`` (so the [-1,1] policy action is mapped to eng
    units before the noise is added). Std 0 ⇒ identity.
    """

    def __init__(self, env: gym.Env, *, steering_std: float = 0.0,
                 speed_std: float = 0.0, steering_bias_max: float = 0.0,
                 speed_bias_max: float = 0.0, seed: int | None = None,
                 adr_state=None) -> None:
        super().__init__(env)
        self._std = np.array([steering_std, speed_std], dtype=np.float32)
        # Per-EPISODE constant lean (a miscalibrated actuator: steering trim off /
        # motor offset), resampled at reset ~U[-max, max]; distinct from the per-STEP
        # symmetric jitter above. Mirrors the multi-car path so both share the model.
        self._bias_max = np.array([steering_bias_max, speed_bias_max], dtype=np.float32)
        self._bias = np.zeros(2, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._adr = adr_state  # if set, read the (live, ADR-controlled) std each step

    def reset(self, **kwargs):
        if np.any(self._bias_max > 0):
            self._bias = self._rng.uniform(-self._bias_max, self._bias_max).astype(np.float32)
        return self.env.reset(**kwargs)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32)
        if self._adr is not None:
            std = np.array([self._adr.steering_noise, self._adr.speed_noise],
                           dtype=np.float32)
        else:
            std = self._std
        if a.shape[-1] == 2:
            if np.any(self._bias_max > 0):       # constant per-episode lean
                a = a + self._bias
            if np.any(std > 0):                  # per-step symmetric jitter
                a = a + self._rng.normal(0.0, 1.0, size=a.shape).astype(np.float32) * std
        return a


class DragRandomization(gym.ActionWrapper):
    """Per-episode throttle→speed effectiveness ("drag") randomization for sim2real.

    At each ``reset`` draw a factor ``~U[drag_min, 1.0]`` and multiply the commanded
    speed (``action[1]``, engineering units) by it for the whole episode — so a given
    throttle reaches different speeds across episodes, exactly the sim-vs-real
    mismatch (motor/drag/battery/surface). Wrap like ``ActuatorNoise`` (outside
    ``ActionBounds`` so the inner clip re-bounds, inside ``NormalizeActions``). The
    policy observes the *achieved* speed (raw m/s feature) and learns to react to it
    instead of assuming a fixed throttle→speed map. ``drag_min=1.0`` ⇒ identity.
    """

    def __init__(self, env: gym.Env, *, drag, seed: int | None = None) -> None:
        super().__init__(env)
        from gym_dr.randomization import sample_spec
        self._drag = drag                      # ParamSpec (Range/Choice/scalar)
        self._sample = sample_spec
        self._rng = np.random.default_rng(seed)
        self._factor = 1.0

    def reset(self, **kwargs):
        self._factor = float(self._sample(self._drag, self._rng))
        return self.env.reset(**kwargs)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32).copy()
        if a.shape[-1] == 2:
            a[1] = a[1] * self._factor          # scale commanded speed (eng units)
        return a


def apply_image_jitter(img, rng, *, gaussian_std: float = 0.0, brightness: float = 0.0,
                       contrast: float = 0.0, gamma: float = 0.0):
    """Photometric DR on a uint8 image — brightness, contrast, gamma, Gaussian.

    All knobs are symmetric magnitudes (0 => identity). Order: brightness
    (multiplicative), contrast (around mid-gray), gamma (luminance curve), then
    additive Gaussian sensor noise; clipped back to the input dtype's [0,255].
    Meaningful on grayscale (where "colour" reduces to lighting/contrast) so it's
    shared by the single-car ``ObservationNoise`` wrapper and the multi-car VecEnv.
    """
    if gaussian_std <= 0 and brightness <= 0 and contrast <= 0 and gamma <= 0:
        return img
    x = np.asarray(img, dtype=np.float32)
    if brightness > 0:
        x = x * (1.0 + rng.uniform(-brightness, brightness))
    if contrast > 0:
        x = (x - 128.0) * (1.0 + rng.uniform(-contrast, contrast)) + 128.0
    if gamma > 0:
        g = float(np.exp(rng.uniform(-gamma, gamma)))   # gamma in (e^-gamma, e^+gamma)
        x = 255.0 * np.clip(x / 255.0, 0.0, 1.0) ** g
    if gaussian_std > 0:
        x = x + rng.normal(0.0, gaussian_std, size=x.shape)
    return np.clip(x, 0, 255).astype(np.asarray(img).dtype)


class ObservationNoise(gym.ObservationWrapper):
    """Perturb image observations — observation-noise / lighting DR.

    For each uint8 image key in a Dict obs: brightness, contrast, gamma, then
    additive Gaussian noise (:func:`apply_image_jitter`). Apply OUTSIDE
    ``GrayscaleObs`` so it perturbs exactly what the policy sees. All knobs 0 =>
    identity. gaussian/brightness can be ADR-controlled; contrast/gamma are static.
    """

    def __init__(self, env: gym.Env, *, gaussian_std: float = 0.0,
                 brightness_jitter: float = 0.0, contrast: float = 0.0,
                 gamma: float = 0.0, seed: int | None = None, adr_state=None) -> None:
        super().__init__(env)
        self._std = float(gaussian_std)
        self._bj = float(brightness_jitter)
        self._contrast = float(contrast)
        self._gamma = float(gamma)
        self._rng = np.random.default_rng(seed)
        self._adr = adr_state  # if set, read the (live, ADR-controlled) std/jitter each step
        self._img_keys: list[str] = []
        if isinstance(env.observation_space, gym.spaces.Dict):
            for key, sp in env.observation_space.spaces.items():
                if isinstance(sp, gym.spaces.Box) and sp.dtype == np.uint8 and len(sp.shape) == 3:
                    self._img_keys.append(key)

    def observation(self, observation: dict) -> dict:
        std = self._adr.obs_gaussian if self._adr is not None else self._std
        bj = self._adr.obs_brightness if self._adr is not None else self._bj
        if (std <= 0 and bj <= 0 and self._contrast <= 0 and self._gamma <= 0) or not self._img_keys:
            return observation
        out = dict(observation)
        for key in self._img_keys:
            out[key] = apply_image_jitter(
                observation[key], self._rng, gaussian_std=std, brightness=bj,
                contrast=self._contrast, gamma=self._gamma)
        return out


class CostInfoWrapper(gym.Wrapper):
    """Surface the graded-risk **cost** as ``info["cost"]`` for safe-RL backends
    (FSRL / Tianshou, OmniSafe) that read the constraint signal from there.

    The cost is already computed by the metrics tap: ``install_metrics`` wires a
    ``cost_fn`` into ``_EpisodeMetrics`` (default ``cost_near_edge``), which runs it
    on the reward-params each step and stores ``last_cost``. Build the env through
    ``gym_dr.metrics.install_metrics`` and pass the resulting ``metrics_state``
    here; this wrapper publishes ``state.last_cost`` as ``info["cost"]`` per step
    (0.0 on reset). No deepracer-env change needed — it reuses the existing tap.
    """

    def __init__(self, env: gym.Env, metrics_state: Any) -> None:
        super().__init__(env)
        self._state = metrics_state

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info = dict(info) if isinstance(info, dict) else {}
        info["cost"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {}
        info["cost"] = float(getattr(self._state, "last_cost", 0.0))
        return obs, reward, terminated, truncated, info
