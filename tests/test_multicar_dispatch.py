"""MC-1 host tests (no sim): the (n_cars, camera_obs) env-factory dispatch and
the feature-observation wrapper. The sim-backed factories (time_trial / multi_car
backend) are validated live; the routing and the obs transform are not, and they
are the load-bearing new logic."""
from __future__ import annotations

import numpy as np
import pytest

import gym_dr.envs.dispatch as dispatch
from gym_dr.perception import ALL_FEATURES, all_targets


# --------------------------------------------------------------------------- #
# Dispatch routing
# --------------------------------------------------------------------------- #
class _Exp:
    """Minimal stand-in for ExperimentConfig (dispatch only reads two attrs)."""
    def __init__(self, n_cars=1, camera_obs=True):
        self.n_cars = n_cars
        self.camera_obs = camera_obs


def _patch_factories(monkeypatch):
    calls = {}

    def _mk(key, sentinel):
        def _f(e):
            calls[key] = e
            return sentinel
        return _f

    monkeypatch.setattr(dispatch, "time_trial", _mk("time_trial", "TT"))
    monkeypatch.setattr(dispatch, "feature_time_trial", _mk("feat", "FEAT"))
    import gym_dr.envs.multi_car as mc
    monkeypatch.setattr(mc, "multi_car", _mk("multi", "MULTI"))
    return calls


def test_dispatch_single_camera(monkeypatch):
    _patch_factories(monkeypatch)
    assert dispatch.build_env(_Exp(n_cars=1, camera_obs=True)) == "TT"


def test_dispatch_single_feature(monkeypatch):
    _patch_factories(monkeypatch)
    assert dispatch.build_env(_Exp(n_cars=1, camera_obs=False)) == "FEAT"


def test_dispatch_multi_camera(monkeypatch):
    _patch_factories(monkeypatch)
    assert dispatch.build_env(_Exp(n_cars=4, camera_obs=True)) == "MULTI"


def test_dispatch_multi_feature(monkeypatch):
    _patch_factories(monkeypatch)
    assert dispatch.build_env(_Exp(n_cars=4, camera_obs=False)) == "MULTI"


def test_dispatch_defaults_to_single_camera(monkeypatch):
    _patch_factories(monkeypatch)

    class _Bare:  # no attrs -> getattr defaults (1, True)
        pass
    assert dispatch.build_env(_Bare()) == "TT"


def test_multi_car_factory_importable():
    # multi_car is now implemented (MC-3); building the backend needs the sim
    # (deepracer_env), so off-sim it raises ImportError/TypeError — but the
    # factory + VecEnv import cleanly and the dispatcher routes to it.
    from gym_dr.envs.multi_car import MultiCarVecEnv, multi_car
    assert callable(multi_car) and MultiCarVecEnv is not None


# --------------------------------------------------------------------------- #
# FeatureObsWrapper
# --------------------------------------------------------------------------- #
import gymnasium as gym  # noqa: E402

from gym_dr.envs.feature_obs import FeatureObsWrapper  # noqa: E402


class _MockEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (4, 8, 8), np.uint8)
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, **kw):
        return np.zeros((4, 8, 8), np.uint8), {}

    def step(self, a):
        return np.zeros((4, 8, 8), np.uint8), 1.0, False, False, {}


def _params(progress=10.0, speed=2.0, **kw):
    return {"track_width": 1.0, "distance_from_center": 0.05, "is_left_of_center": False,
            "heading": 0.0, "speed": speed, "progress": progress, "steps": 1,
            "steering_angle": 0.0, "all_wheels_on_track": True, "is_offtrack": False,
            "waypoints": [(float(i), 0.0) for i in range(10)], "closest_waypoints": [1, 2], **kw}


def test_feature_obs_space_and_shape():
    src = _params()
    w = FeatureObsWrapper(_MockEnv(), lambda: src)
    assert w.observation_space.shape == (len(ALL_FEATURES),)
    obs, _ = w.reset()
    assert obs.shape == (len(ALL_FEATURES),) and obs.dtype == np.float32


def test_feature_obs_matches_all_targets():
    holder = {"p": _params(progress=5.0)}
    w = FeatureObsWrapper(_MockEnv(), lambda: holder["p"])
    obs0, _ = w.reset()                       # first step: no prev
    assert np.allclose(obs0, all_targets(holder["p"], None))
    # advance: prev is now the reset params, so dynamic features finite-diff
    holder["p"] = _params(progress=8.0, speed=2.5)
    obs1, _r, _t, _tr, _i = w.step(np.zeros(2, np.float32))
    assert np.allclose(obs1, all_targets(holder["p"], _params(progress=5.0)))


def test_feature_obs_resets_prev_history():
    holder = {"p": _params(progress=90.0, speed=3.0)}
    w = FeatureObsWrapper(_MockEnv(), lambda: holder["p"])
    w.reset(); w.step(np.zeros(2, np.float32))
    # new episode: reset clears prev, so dynamic features compute against None (0)
    holder["p"] = _params(progress=1.0, speed=1.0)
    obs, _ = w.reset()
    assert np.allclose(obs, all_targets(holder["p"], None))


def test_feature_obs_handles_empty_params():
    w = FeatureObsWrapper(_MockEnv(), lambda: {})
    obs, _ = w.reset()
    assert obs.shape == (len(ALL_FEATURES),) and np.all(np.isfinite(obs))


def test_config_fields_serialize():
    from gym_dr.config import ExperimentConfig
    e = ExperimentConfig(name="t", n_cars=4, camera_obs=False)
    d = e.to_dict()
    assert d["n_cars"] == 4 and d["camera_obs"] is False
    # default factory is the dispatcher
    assert "build_env" in d["env_factory"] or "dispatch" in d["env_factory"]
