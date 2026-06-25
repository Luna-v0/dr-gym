"""MC-3 host tests: the MultiCarVecEnv orchestration logic against a MOCK Gazebo
backend (no sim). Validates action batching/transform, the camera+feature obs
transforms, batched shapes, and per-car auto-reset bookkeeping — the parts that
don't need ROS. The live multi-agent ROS integration is validated in VNC."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from gym_dr.action_space import ContinuousActionSpaceConfig
from gym_dr.envs.multi_car import MultiCarVecEnv
from gym_dr.perception import ALL_FEATURES, all_targets


def _params(progress=10.0, speed=2.0, dist=0.05, off=False):
    return {"track_width": 1.0, "distance_from_center": dist, "is_left_of_center": False,
            "heading": 0.0, "speed": speed, "progress": progress, "steps": 1,
            "steering_angle": 0.0, "all_wheels_on_track": not off, "is_offtrack": off,
            "waypoints": [(float(i), 0.0) for i in range(10)], "closest_waypoints": [1, 2]}


class _MockBackend:
    """Mimics MultiAgentDeepRacerEnv: N cars, camera Dict obs, scripted dones."""
    def __init__(self, n_cars=3, h=8, w=12, done_on_step=None):
        self.n_cars = n_cars
        self.single_observation_space = gym.spaces.Dict(
            {"FRONT_FACING_CAMERA": gym.spaces.Box(0, 255, (h, w, 3), np.uint8)})
        self._h, self._w = h, w
        self.last_actions = None
        self.reset_calls = []
        self._done_on_step = done_on_step or {}   # {car_idx: True}

    def _obs(self, fill):
        return {"FRONT_FACING_CAMERA": np.full((self._h, self._w, 3), fill, np.uint8)}

    def reset(self):
        self.reset_calls.append("all")
        return [self._obs(i) for i in range(self.n_cars)]

    def reset_one(self, i):
        self.reset_calls.append(i)
        return self._obs(100 + i)

    def step(self, actions):
        self.last_actions = [np.asarray(a, np.float32) for a in actions]
        obs = [self._obs(i) for i in range(self.n_cars)]
        rewards = [float(i) for i in range(self.n_cars)]
        dones = [bool(self._done_on_step.get(i, False)) for i in range(self.n_cars)]
        infos = [{"reward_params": _params(progress=10.0 + i, off=dones[i])} for i in range(self.n_cars)]
        return obs, rewards, dones, infos

    def close(self):
        pass


def _cfg(normalize=True):
    return ContinuousActionSpaceConfig(steering_low=-30.0, steering_high=30.0,
                                       speed_low=1.0, speed_high=4.0,
                                       normalize_actions=normalize)


# --------------------------------------------------------------------------- #
_CAM = "FRONT_FACING_CAMERA"


def test_camera_vecenv_spaces_and_reset():
    ve = MultiCarVecEnv(_MockBackend(n_cars=3), camera_obs=True, action_cfg=_cfg())
    assert ve.num_envs == 3
    assert ve.observation_space[_CAM].shape == (8, 12, 1)  # Dict, grayscale single frame
    assert ve.action_space.shape == (2,)
    obs = ve.reset()
    assert obs[_CAM].shape == (3, 8, 12, 1) and obs[_CAM].dtype == np.uint8


def test_feature_vecenv_spaces_and_reset():
    ve = MultiCarVecEnv(_MockBackend(n_cars=2), camera_obs=False, action_cfg=_cfg())
    assert ve.observation_space.shape == (len(ALL_FEATURES),)
    obs = ve.reset()
    assert obs.shape == (2, len(ALL_FEATURES))


def test_action_normalization_to_engineering_units():
    be = _MockBackend(n_cars=1)
    ve = MultiCarVecEnv(be, camera_obs=True, action_cfg=_cfg(normalize=True))
    ve.reset()
    ve.step_async(np.array([[1.0, 1.0]], np.float32))      # max in [-1,1]
    ve.step_wait()
    assert np.allclose(be.last_actions[0], [30.0, 4.0])    # -> [steer_hi, speed_hi]
    ve.step_async(np.array([[-1.0, -1.0]], np.float32))    # min
    ve.step_wait()
    assert np.allclose(be.last_actions[0], [-30.0, 1.0])


def test_no_normalization_clips_to_bounds():
    be = _MockBackend(n_cars=1)
    ve = MultiCarVecEnv(be, camera_obs=True, action_cfg=_cfg(normalize=False))
    ve.reset()
    ve.step_async(np.array([[999.0, 999.0]], np.float32))
    ve.step_wait()
    assert np.allclose(be.last_actions[0], [30.0, 4.0])    # clipped


def test_feature_obs_matches_all_targets():
    be = _MockBackend(n_cars=2)
    ve = MultiCarVecEnv(be, camera_obs=False, action_cfg=_cfg())
    ve.reset()
    ve.step_async(np.zeros((2, 2), np.float32))
    obs, rew, dones, infos = ve.step_wait()
    # car 0's feature obs built from its reward_params (prev was the reset, i.e. None)
    expected0 = all_targets(_params(progress=10.0), None)
    assert np.allclose(obs[0], expected0)
    assert rew.shape == (2,) and dones.shape == (2,)


def test_per_car_auto_reset_on_done():
    be = _MockBackend(n_cars=3, done_on_step={1: True})     # car 1 finishes
    ve = MultiCarVecEnv(be, camera_obs=True, action_cfg=_cfg())
    ve.reset()
    be.reset_calls.clear()
    ve.step_async(np.zeros((3, 2), np.float32))
    obs, rew, dones, infos = ve.step_wait()
    assert dones.tolist() == [False, True, False]
    # only car 1 was reset, and its terminal obs is in info
    assert be.reset_calls == [1]
    assert "terminal_observation" in infos[1]
    assert "terminal_observation" not in infos[0]
    # car 1's returned obs is the RESET obs (fill 100+1=101), not the terminal one
    assert int(obs[_CAM][1, 0, 0, 0]) != int(infos[1]["terminal_observation"][_CAM][0, 0, 0])


def test_step_returns_batched_consistently():
    ve = MultiCarVecEnv(_MockBackend(n_cars=4), camera_obs=True, action_cfg=_cfg())
    ve.reset()
    obs, rew, dones, infos = ve.step(np.zeros((4, 2), np.float32))
    assert obs[_CAM].shape == (4, 8, 12, 1)
    assert rew.shape == (4,) and dones.shape == (4,) and len(infos) == 4
