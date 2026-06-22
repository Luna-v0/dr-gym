"""Tests for the W-dr noise wrappers."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from gym_dr.domain_randomization import ADRController, ADRState, DomainRandomizationConfig
from gym_dr.envs.wrappers import ActuatorNoise, ObservationNoise


class _ActEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-30.0, 1.0], dtype=np.float32),
            high=np.array([30.0, 4.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        self.last_action = None

    def reset(self, **kw):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


def test_actuator_noise_zero_is_identity():
    env = _ActEnv()
    w = ActuatorNoise(env, steering_std=0.0, speed_std=0.0)
    w.step(np.array([5.0, 2.0], dtype=np.float32))
    assert np.allclose(env.last_action, [5.0, 2.0])


def test_actuator_noise_perturbs_and_is_seeded():
    a = ActuatorNoise(_ActEnv(), steering_std=3.0, speed_std=0.5, seed=0)
    b = ActuatorNoise(_ActEnv(), steering_std=3.0, speed_std=0.5, seed=0)
    base = np.array([5.0, 2.0], dtype=np.float32)
    a.step(base.copy()); b.step(base.copy())
    assert not np.allclose(a.env.last_action, base)          # noise applied
    assert np.allclose(a.env.last_action, b.env.last_action)  # same seed ⇒ same noise


class _ObsEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "FRONT_FACING_CAMERA": gym.spaces.Box(0, 255, (8, 8, 1), dtype=np.uint8)})
        self.action_space = gym.spaces.Discrete(2)
        self._img = np.full((8, 8, 1), 128, dtype=np.uint8)

    def reset(self, **kw):
        return {"FRONT_FACING_CAMERA": self._img.copy()}, {}

    def step(self, a):
        return {"FRONT_FACING_CAMERA": self._img.copy()}, 0.0, True, False, {}


def test_observation_noise_zero_is_identity():
    w = ObservationNoise(_ObsEnv(), gaussian_std=0.0, brightness_jitter=0.0)
    obs, _ = w.reset()
    assert np.array_equal(obs["FRONT_FACING_CAMERA"], np.full((8, 8, 1), 128, dtype=np.uint8))


def test_observation_noise_perturbs_stays_uint8():
    w = ObservationNoise(_ObsEnv(), gaussian_std=20.0, brightness_jitter=0.1, seed=1)
    obs, _ = w.reset()
    img = obs["FRONT_FACING_CAMERA"]
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255
    assert not np.array_equal(img, np.full((8, 8, 1), 128, dtype=np.uint8))


def test_adr_controller_grows_shrinks_clamps():
    cfg = DomainRandomizationConfig(actuator_steering_std=10.0, obs_gaussian_std=20.0,
                                    adr=True, adr_step=0.5, adr_promote=0.7, adr_demote=0.3)
    st = ADRState()
    ctrl = ADRController(cfg, st)
    assert st.actuator_steering_std == 0.0
    ctrl.update(0.9)                              # promote: +0.5*10 = 5
    assert st.actuator_steering_std == 5.0 and st.obs_gaussian_std == 10.0
    ctrl.update(0.9)                              # -> ceiling 10
    assert st.actuator_steering_std == 10.0
    ctrl.update(0.9)                              # clamp at ceiling
    assert st.actuator_steering_std == 10.0
    ctrl.update(0.1)                              # demote: -5 -> 5
    assert st.actuator_steering_std == 5.0
    r = ctrl.update(0.5)                          # in between: no change
    assert st.actuator_steering_std == 5.0
    assert r["adr/actuator_steering_std"] == 5.0


def test_actuator_noise_reads_adr_state_live():
    st = ADRState()
    env = _ActEnv()
    w = ActuatorNoise(env, seed=0, adr_state=st)
    w.step(np.array([5.0, 2.0], dtype=np.float32))
    assert np.allclose(env.last_action, [5.0, 2.0])   # ranges 0 -> no noise
    st.actuator_steering_std = 5.0
    st.actuator_speed_std = 0.5
    w.step(np.array([5.0, 2.0], dtype=np.float32))
    assert not np.allclose(env.last_action, [5.0, 2.0])  # live ranges -> noise now


def test_config_adr_enables_wrappers():
    assert DomainRandomizationConfig(adr=True).has_action_noise
    assert DomainRandomizationConfig(adr=True).has_obs_noise
    assert not DomainRandomizationConfig().has_action_noise
