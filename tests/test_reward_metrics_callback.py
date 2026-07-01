"""Tests for RewardMetricsCallback EMA + sliding-window trend metrics (Task 4)."""
from __future__ import annotations

from gym_dr.trainers.sb3.callbacks import RewardMetricsCallback


class FakeLogger:
    """Captures SB3 logger record / record_mean calls."""

    def __init__(self):
        self.recorded = {}          # key -> last record() value
        self.means = {}             # key -> list of record_mean() values

    def record(self, key, value, exclude=None):
        self.recorded[key] = value

    def record_mean(self, key, value):
        self.means.setdefault(key, []).append(value)


class FakeModel:
    """SB3 BaseCallback.logger is a property returning self.model.logger."""

    def __init__(self, logger):
        self.logger = logger


def _cb(**kw):
    cb = RewardMetricsCallback(**kw)
    cb.model = FakeModel(FakeLogger())
    return cb


def _feed(cb, **metrics):
    cb.locals = {"infos": [{"dr_episode": dict(metrics)}]}
    return cb._on_step()


def test_ema_and_window_track_across_episodes():
    cb = _cb(window=3, ema_alpha=0.9)

    _feed(cb, **{"dr/ep_reward": 10.0})
    assert cb.logger.recorded["dr/ep_reward_ema"] == 10.0     # first sample seeds EMA
    assert cb.logger.recorded["dr/ep_reward_win3"] == 10.0

    _feed(cb, **{"dr/ep_reward": 20.0})
    assert abs(cb.logger.recorded["dr/ep_reward_ema"] - 11.0) < 1e-9   # .9*10 + .1*20
    assert cb.logger.recorded["dr/ep_reward_win3"] == 15.0            # mean(10, 20)

    _feed(cb, **{"dr/ep_reward": 30.0})
    _feed(cb, **{"dr/ep_reward": 40.0})
    # window maxlen=3 keeps the last three: mean(20, 30, 40) == 30
    assert cb.logger.recorded["dr/ep_reward_win3"] == 30.0
    # the raw per-rollout mean is still recorded every episode
    assert cb.logger.means["dr/ep_reward"] == [10.0, 20.0, 30.0, 40.0]


def test_state_persists_across_calls():
    # Trend state lives on the instance (persists across chunks in a rotation).
    cb = _cb(window=100, ema_alpha=0.5)
    _feed(cb, **{"dr/ep_offtrack_rate": 1.0})
    _feed(cb, **{"dr/ep_offtrack_rate": 0.0})
    assert cb.logger.recorded["dr/ep_offtrack_rate_ema"] == 0.5


def test_non_numeric_values_skipped():
    cb = _cb()
    _feed(cb, **{"dr/ep_reward": "oops", "dr/ep_progress": 42.0})
    assert "dr/ep_reward_ema" not in cb.logger.recorded
    assert cb.logger.recorded["dr/ep_progress_ema"] == 42.0


def test_missing_summary_is_ignored():
    cb = _cb()
    cb.locals = {"infos": [{}, {"other": 1}, "notadict"]}
    assert cb._on_step() is True
    assert cb.logger.recorded == {}


def test_alpha_clamped_and_window_floored():
    cb = RewardMetricsCallback(window=0, ema_alpha=5.0)
    assert cb._window == 1
    assert cb._alpha == 1.0
