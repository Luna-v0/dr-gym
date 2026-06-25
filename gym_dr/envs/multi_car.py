"""Multi-car (N agents in one Gazebo world) -> SB3 ``VecEnv``.

ONE Gazebo world steps physics once for all N namespaced racecars, so the
per-step sim cost amortizes across agents and PPO gets decorrelated parallel
samples. The raw N-agent orchestration is ``deepracer_env``'s
``MultiAgentDeepRacerEnv`` (free-running step, per-car reset); this module adapts
it to the SB3 ``VecEnv`` interface (``num_envs = n_cars``), applying per-car:

  * **action transform** — policy acts in ``[-1,1]`` (when ``normalize_actions``)
    mapped to engineering units and clipped to the action bounds, mirroring the
    single-car ``NormalizeActions`` + ``ActionBounds`` wrappers;
  * **observation transform** — grayscale camera frame (``camera_obs=True``) or
    the ``ALL_FEATURES`` vector from ``reward_params`` (``camera_obs=False``);
  * **auto-reset** — a car whose episode ended is reset on its own (terminal obs
    in ``info['terminal_observation']``), VecEnv convention, others keep driving.

Frame stacking stays the trainer's job (``VecFrameStack`` wraps this VecEnv). The
``backend`` is injectable so the orchestration logic is unit-tested against a mock
Gazebo (``tests/test_multi_car_vecenv.py``). See ``docs/reports/multi-car.md``.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from gym_dr.action_space import ContinuousActionSpaceConfig
from gym_dr.perception import ALL_FEATURES, all_targets

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # BT.601, matches GrayscaleObs


def _find_image_key(space: gym.spaces.Dict) -> Optional[str]:
    for key, sub in space.spaces.items():
        if isinstance(sub, gym.spaces.Box) and len(sub.shape) == 3:
            return key
    return None


class MultiCarVecEnv(VecEnv):
    """SB3 ``VecEnv`` over N cars sharing one Gazebo world."""

    def __init__(self, backend, *, camera_obs: bool,
                 action_cfg: ContinuousActionSpaceConfig,
                 actuator_steering_std: float = 0.0,
                 actuator_speed_std: float = 0.0,
                 noise_seed: Optional[int] = None) -> None:
        self._backend = backend
        self._camera = camera_obs
        self._cfg = action_cfg
        self._normalize = bool(getattr(action_cfg, "normalize_actions", True))
        n = backend.n_cars
        # Actuator-noise DR (engineering units) applied per-car in the action
        # transform — the single-car ActuatorNoise wrapper can't reach this VecEnv
        # (metrics.wrap passes it through), so we replicate it here. 0 => off.
        self._act_noise = (
            np.array([actuator_steering_std, actuator_speed_std], dtype=np.float32)
            if (actuator_steering_std or actuator_speed_std) else None)
        self._noise_rng = np.random.default_rng(noise_seed)

        # engineering-unit action bounds (what the sim executes)
        self._low = np.array([action_cfg.steering_low, action_cfg.speed_low], dtype=np.float32)
        self._high = np.array([action_cfg.steering_high, action_cfg.speed_high], dtype=np.float32)
        # the action space the POLICY sees: [-1,1] when normalizing, else eng units
        if self._normalize:
            action_space: gym.spaces.Space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        else:
            action_space = gym.spaces.Box(self._low, self._high, dtype=np.float32)

        if camera_obs:
            self._image_key = _find_image_key(backend.single_observation_space)
            if self._image_key is None:
                raise ValueError("camera_obs=True but backend obs has no image space")
            h, w, _ = backend.single_observation_space.spaces[self._image_key].shape
            self._hw = (h, w)
            # Dict obs (grayscale single frame) — matches the single-car GrayscaleObs
            # output so MultiInputPolicy + DeepRacerCNN + VecFrameStack work unchanged.
            obs_space: gym.spaces.Space = gym.spaces.Dict(
                {self._image_key: gym.spaces.Box(0, 255, (h, w, 1), np.uint8)})
        else:
            self._image_key = None
            obs_space = gym.spaces.Box(-1.0, 1.0, (len(ALL_FEATURES),), np.float32)

        super().__init__(num_envs=n, observation_space=obs_space, action_space=action_space)
        self._actions: Optional[np.ndarray] = None
        self._prev_params: List[Optional[dict]] = [None] * n

    # ---- action / obs transforms ------------------------------------- #
    def _to_engineering(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32)
        if self._normalize:  # [-1,1] -> [low, high]
            a = self._low + (a + 1.0) * 0.5 * (self._high - self._low)
        if self._act_noise is not None:  # additive Gaussian in engineering units
            a = a + self._act_noise * self._noise_rng.standard_normal(2).astype(np.float32)
        return np.clip(a, self._low, self._high)

    def _obs_from(self, raw_obs, info, car: int):
        if self._camera:
            img = np.asarray(raw_obs[self._image_key], dtype=np.uint8)
            gray = (img[..., :3].astype(np.float32) @ _LUMA).astype(np.uint8)
            return {self._image_key: gray[..., None]}  # Dict{key: (H, W, 1)}
        params = (info or {}).get("reward_params", {}) if info else {}
        feat = all_targets(params, self._prev_params[car]).astype(np.float32)
        if params:
            self._prev_params[car] = dict(params)
        return feat

    def _stack(self, obs_list: Sequence[Any]):
        """Batch per-car obs to num_envs-first. Dict (camera) stacks per key;
        array (feature) stacks directly."""
        if self._camera:
            return {self._image_key: np.stack([o[self._image_key] for o in obs_list], axis=0)}
        return np.stack(obs_list, axis=0)

    # ---- VecEnv interface -------------------------------------------- #
    def reset(self) -> np.ndarray:
        self._prev_params = [None] * self.num_envs
        raw = self._backend.reset()
        return self._stack([self._obs_from(raw[i], None, i) for i in range(self.num_envs)])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        eng = [self._to_engineering(self._actions[i]) for i in range(self.num_envs)]
        raw_obs, rewards, dones, infos = self._backend.step(eng)
        obs_out, info_out = [], []
        for i in range(self.num_envs):
            info = dict(infos[i]) if infos[i] else {}
            o = self._obs_from(raw_obs[i], infos[i], i)
            if dones[i]:
                info["terminal_observation"] = o
                self._prev_params[i] = None
                reset_raw = self._backend.reset_one(i)
                o = self._obs_from(reset_raw, None, i)
            obs_out.append(o)
            info_out.append(info)
        return (self._stack(obs_out), np.asarray(rewards, np.float32),
                np.asarray(dones, dtype=bool), info_out)

    def close(self) -> None:
        self._backend.close()

    # SB3 plumbing (single-process VecEnv; these touch the shared backend) ----
    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [getattr(self, attr_name, getattr(self._backend, attr_name, None))
                for _ in self._idx(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
        fn = getattr(self._backend, method_name, None)
        return [fn(*args, **kwargs) if callable(fn) else None for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False for _ in self._idx(indices)]

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices


def multi_car(experiment) -> MultiCarVecEnv:
    """Build the N-car VecEnv for ``experiment`` (n_cars > 1)."""
    from deepracer_env.environments.multi_agent_env import MultiAgentDeepRacerEnv

    cfg = experiment.action_space
    if not isinstance(cfg, ContinuousActionSpaceConfig):
        raise TypeError("multi-car requires a ContinuousActionSpaceConfig action space")
    import os
    # Gazebo Classic renders only 2 camera sensors per world: at n>=3 the extra
    # cameras advertise their topic but never publish a frame (verified: racecar_0
    # & racecar_1 at 15Hz, racecar_2 at 0Hz), and the blocking sensor read then
    # log_and_exit()s the whole run ~120s in. Fail fast with the explanation rather
    # than that cryptic crash. Feature obs (camera_obs=False) scales to n=8; for >2
    # camera cars use separate processes. Override only if your renderer supports it.
    if bool(experiment.camera_obs) and int(experiment.n_cars) > 2 \
            and os.getenv("GYM_DR_ALLOW_CAMERA_NCARS") != "1":
        raise ValueError(
            f"camera_obs multi-car is capped at n_cars=2 on Gazebo Classic (only 2 "
            f"camera sensors render per world); got n_cars={experiment.n_cars}. Use "
            f"camera_obs=False (feature obs scales to n=8), run >2 camera cars as "
            f"separate processes, or set GYM_DR_ALLOW_CAMERA_NCARS=1 to override.")
    # Per-car track list (the generalization engine): GYM_DR_DEMO_WORLDS is a
    # comma-separated list of track names, one per car — each car drives its own
    # track instance, so N cars train across N different tracks in one world.
    # Shorter-than-n_cars lists cycle; empty falls back to the experiment world
    # (all cars same track = parallel sampling). Forwarded by app.py.
    worlds_env = os.getenv("GYM_DR_DEMO_WORLDS", "").strip()
    worlds = None
    if worlds_env:
        names = [w.strip() for w in worlds_env.split(",") if w.strip()]
        n = int(experiment.n_cars)
        worlds = [names[i % len(names)] for i in range(n)]
    # Feature obs reads reward_params (car pose / track position), NOT pixels, so
    # the camera sensor is dead weight — and worse, each agent's camera sensor
    # blocks on a Gazebo image at reset; with N cars a camera can miss its frame and
    # DoubleBuffer.get() calls log_and_exit, killing the whole run (seen at n>=4).
    # Empty sensor list => CompositeSensor returns {} (no blocking) and nothing to
    # render. Camera obs keeps the camera sensor.
    sensors = list(cfg.sensor) if bool(experiment.camera_obs) else []
    # Domain randomization for the multi-car VecEnv (mirrors the single-car
    # time_trial wiring): random_start/random_direction are deepracer-env reset
    # modes passed through the controller config to every car; actuator noise is
    # applied per-car inside MultiCarVecEnv (the wrapper stack can't reach it).
    dr = getattr(experiment, "domain_randomization", None)
    reset_config: dict = {}
    if dr is not None and getattr(dr, "random_start", False):
        reset_config["random_start"] = True
    if dr is not None and getattr(dr, "random_direction", False):
        reset_config["random_direction"] = True
    backend = MultiAgentDeepRacerEnv(
        n_cars=int(experiment.n_cars),
        reward_fn=experiment.reward,
        sensors=sensors,
        worlds=worlds,
        config=reset_config or None,
        # metres between separated track instances. Default 300 (cars can't see
        # each other); set GYM_DR_DEMO_SPACING small (e.g. 50) to fit both in one
        # VNC view for visual validation. Forwarded into the container by app.py.
        spacing=float(os.getenv("GYM_DR_DEMO_SPACING", "300")),
    )
    from gym_dr.randomization import spec_bounds
    return MultiCarVecEnv(
        backend, camera_obs=bool(experiment.camera_obs), action_cfg=cfg,
        actuator_steering_std=spec_bounds(dr.steering_noise)[1] if dr else 0.0,
        actuator_speed_std=spec_bounds(dr.speed_noise)[1] if dr else 0.0,
        noise_seed=getattr(dr, "seed", None) if dr else None,
    )
