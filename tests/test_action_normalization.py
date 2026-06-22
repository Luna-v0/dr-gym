"""Tests for the P3 NormalizeActions wrapper.

It presents a symmetric [-1, 1] action space to the policy and maps linearly
onto the inner env's engineering-unit Box, so PPO's unit Gaussian explores
every dimension comparably while the env / ONNX / on-car interface stays in
engineering units.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from gym_dr.envs.wrappers import NormalizeActions


class _BoxEnv(gym.Env):
    """Minimal env that just records the action its inner step received."""

    def __init__(self, low, high):
        self.action_space = gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.last_action = None

    def reset(self, **kw):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


def _wrapped():
    inner = _BoxEnv([-30.0, 1.0], [30.0, 4.0])  # DeepRacer-like steering, speed
    return inner, NormalizeActions(inner)


def test_action_space_is_symmetric_unit():
    _, w = _wrapped()
    assert np.allclose(w.action_space.low, [-1.0, -1.0])
    assert np.allclose(w.action_space.high, [1.0, 1.0])


@pytest.mark.parametrize(
    "norm,expected",
    [
        ([-1.0, -1.0], [-30.0, 1.0]),   # min -> low
        ([1.0, 1.0], [30.0, 4.0]),      # max -> high
        ([0.0, 0.0], [0.0, 2.5]),       # mid -> midpoint
    ],
)
def test_maps_unit_to_engineering(norm, expected):
    inner, w = _wrapped()
    w.step(np.array(norm, dtype=np.float32))
    assert np.allclose(inner.last_action, expected, atol=1e-5)


def test_clips_out_of_range():
    inner, w = _wrapped()
    w.step(np.array([5.0, -5.0], dtype=np.float32))  # outside [-1,1]
    assert np.allclose(inner.last_action, [30.0, 1.0], atol=1e-5)


def test_rejects_non_box():
    env = _BoxEnv([-1.0], [1.0])
    env.action_space = gym.spaces.Discrete(3)
    with pytest.raises(TypeError):
        NormalizeActions(env)
