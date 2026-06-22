"""Env-contract checks (W1).

Two layers:
  * Offline (always runs): the pure-pursuit control law from
    ``scripts/scripted_baseline.py`` — pure function, no sim.
  * Live (sim-gated): contract checks against a real ``DeepRacerEnv``. These
    skip unless ``deepracer_env`` imports AND a Gazebo sim is reachable, so they
    run inside the container and are skipped on a dev box.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

# --- load the control law from scripts/ without needing scripts to be a package
_SB = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "scripted_baseline.py"
_spec = importlib.util.spec_from_file_location("scripted_baseline", _SB)
scripted_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scripted_baseline)
pure_pursuit_action = scripted_baseline.pure_pursuit_action


def _params(x=0.0, y=0.0, heading=0.0, wps=None, nxt=1):
    return {
        "x": x, "y": y, "heading": heading,
        "waypoints": wps or [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
                             (4.0, 0.0), (5.0, 0.0)],
        "closest_waypoints": [nxt - 1, nxt],
    }


def _act(p, **kw):
    kw.setdefault("lookahead", 2)
    kw.setdefault("speed", 1.8)
    kw.setdefault("steer_sign", 1.0)
    kw.setdefault("steer_gain", 1.0)
    return pure_pursuit_action(p, **kw)


def test_pure_pursuit_straight_goes_straight():
    steer, speed = _act(_params())  # car at origin facing +x, track straight ahead
    assert abs(steer) < 1e-6
    assert speed == 1.8


def test_pure_pursuit_opposite_turns_have_opposite_sign():
    left = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]    # turn toward +y
    right = [(0.0, 0.0), (1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]  # toward -y
    s_left, _ = _act(_params(wps=left))
    s_right, _ = _act(_params(wps=right))
    assert s_left * s_right < 0, "left/right targets must steer opposite ways"


def test_pure_pursuit_clamps_to_limit():
    sharp = [(0.0, 0.0), (0.0, 5.0)]  # 90deg to the side, big error
    steer, _ = _act(_params(wps=sharp), steer_gain=10.0)
    assert -30.0 <= steer <= 30.0


def test_pure_pursuit_handles_missing_waypoints():
    steer, speed = _act({"x": 0.0, "y": 0.0, "heading": 0.0})
    assert math.isfinite(steer) and speed == 1.8


# --------------------------------------------------------------------------- #
# Live env contract — sim-gated.
# --------------------------------------------------------------------------- #

@pytest.fixture
def live_env():
    pytest.importorskip("deepracer_env", reason="sim package not installed")
    try:
        from deepracer_env.environments.deepracer_env import DeepRacerEnv
        env = DeepRacerEnv(reward_fn=lambda p: 0.0, sensors=["FRONT_FACING_CAMERA"])
    except Exception as exc:  # noqa: BLE001 — no live Gazebo on a dev box
        pytest.skip(f"no live sim: {exc}")
    yield env
    env.close()


def test_action_space_is_engineering_units(live_env):
    import numpy as np
    sp = live_env.action_space
    assert sp.shape == (2,)
    # steering deg, speed m/s — NOT normalized [-1,1]
    assert sp.low[0] <= -29 and sp.high[0] >= 29
    assert sp.high[1] <= 4.0 + 1e-3 and sp.low[1] >= 0.0
    assert np.issubdtype(sp.dtype, np.floating)


def test_observation_is_camera_dict(live_env):
    import gymnasium as gym
    import numpy as np
    obs_sp = live_env.observation_space
    assert isinstance(obs_sp, gym.spaces.Dict)
    cam = obs_sp.spaces["FRONT_FACING_CAMERA"]
    assert cam.shape == (120, 160, 3) and cam.dtype == np.uint8


def test_reset_and_step_contract(live_env):
    obs, info = live_env.reset(seed=0)
    assert isinstance(obs, dict) and isinstance(info, dict)
    out = live_env.step(live_env.action_space.sample())
    assert len(out) == 5  # gymnasium 5-tuple
    _o, r, terminated, truncated, _i = out
    assert isinstance(float(r), float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
