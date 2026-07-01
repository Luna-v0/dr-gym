"""Tests for pluggable early-stopping strategies (gym_dr.early_stopping)."""
from __future__ import annotations

import pytest

from gym_dr.early_stopping import (
    AllOf,
    AnyOf,
    CleanCompletion,
    EarlyStopController,
    MetricThreshold,
    OfftrackRate,
    RewardThreshold,
)


# --------------------------------------------------------------- strategies

def test_offtrack_rate_reproduces_historical_default():
    # Old strict behaviour: stop when NO eval episode left the track.
    strat = OfftrackRate(max_offtrack_rate=0.0, patience=1)
    assert strat.met({"offtrack_rate": 0.0}) is True
    assert strat.met({"offtrack_rate": 0.1}) is False
    # Missing key is treated as "not mastered" (rate 1.0), never a false stop.
    assert strat.met({}) is False


def test_offtrack_rate_tolerant_threshold():
    strat = OfftrackRate(max_offtrack_rate=0.5)
    assert strat.met({"offtrack_rate": 0.5}) is True
    assert strat.met({"offtrack_rate": 0.6}) is False


def test_metric_threshold_max_mode():
    strat = MetricThreshold(metric="mean_reward", threshold=50.0, mode="max")
    assert strat.met({"mean_reward": 50.0}) is True
    assert strat.met({"mean_reward": 51.0}) is True
    assert strat.met({"mean_reward": 49.9}) is False
    assert strat.met({}) is False  # missing metric -> never qualifies


def test_metric_threshold_min_mode():
    strat = MetricThreshold(metric="mean_cost", threshold=10.0, mode="min")
    assert strat.met({"mean_cost": 10.0}) is True
    assert strat.met({"mean_cost": 5.0}) is True
    assert strat.met({"mean_cost": 10.1}) is False


def test_metric_threshold_rejects_bad_mode():
    with pytest.raises(ValueError):
        MetricThreshold(metric="x", threshold=1.0, mode="sideways")


def test_reward_threshold():
    strat = RewardThreshold(min_reward=100.0)
    assert strat.met({"mean_reward": 100.0}) is True
    assert strat.met({"mean_reward": 99.0}) is False
    assert strat.met({}) is False


def test_clean_completion():
    strat = CleanCompletion(min_rate=1.0)
    assert strat.met({"clean_completion_rate": 1.0}) is True
    assert strat.met({"clean_completion_rate": 0.9}) is False
    assert strat.met({}) is False


def test_all_of_requires_every_child():
    strat = AllOf((CleanCompletion(1.0), MetricThreshold("mean_cost", 10.0, "min")))
    assert strat.met({"clean_completion_rate": 1.0, "mean_cost": 8.0}) is True
    assert strat.met({"clean_completion_rate": 1.0, "mean_cost": 12.0}) is False
    assert strat.met({"clean_completion_rate": 0.5, "mean_cost": 8.0}) is False


def test_any_of_requires_one_child():
    strat = AnyOf((CleanCompletion(1.0), RewardThreshold(200.0)))
    assert strat.met({"clean_completion_rate": 1.0, "mean_reward": 0.0}) is True
    assert strat.met({"clean_completion_rate": 0.0, "mean_reward": 250.0}) is True
    assert strat.met({"clean_completion_rate": 0.0, "mean_reward": 0.0}) is False


def test_strategies_are_frozen_and_hashable():
    # Frozen dataclasses must hash so they serialise/sweep like the rest of config.
    s = {OfftrackRate(0.0), OfftrackRate(0.0), CleanCompletion(1.0)}
    assert len(s) == 2  # the two identical OfftrackRate collapse
    with pytest.raises(Exception):
        OfftrackRate(0.0).max_offtrack_rate = 0.5  # type: ignore[misc]


def test_describe_is_human_readable():
    assert "OfftrackRate" in OfftrackRate(0.0).describe()
    assert "<=" in MetricThreshold("mean_cost", 10.0, "min").describe()
    assert ">=" in MetricThreshold("mean_reward", 50.0, "max").describe()


# --------------------------------------------------------------- controller

def test_controller_none_strategy_never_stops():
    ctrl = EarlyStopController(None)
    assert ctrl.enabled is False
    assert ctrl.update({"offtrack_rate": 0.0}) is False


def test_controller_patience_streak():
    ctrl = EarlyStopController(OfftrackRate(max_offtrack_rate=0.0, patience=2))
    # First qualifying round: streak 1, not yet at patience.
    assert ctrl.update({"offtrack_rate": 0.0}) is False
    assert ctrl.streak == 1
    # Second consecutive qualifying round: streak 2 == patience -> stop.
    assert ctrl.update({"offtrack_rate": 0.0}) is True
    assert ctrl.streak == 2


def test_controller_streak_resets_on_failing_round():
    ctrl = EarlyStopController(OfftrackRate(max_offtrack_rate=0.0, patience=2))
    assert ctrl.update({"offtrack_rate": 0.0}) is False  # streak 1
    assert ctrl.update({"offtrack_rate": 0.3}) is False  # fails -> streak 0
    assert ctrl.streak == 0
    assert ctrl.update({"offtrack_rate": 0.0}) is False  # streak 1 again
    assert ctrl.update({"offtrack_rate": 0.0}) is True   # streak 2 -> stop


def test_controller_reset_zeroes_streak_between_chunks():
    ctrl = EarlyStopController(CleanCompletion(min_rate=1.0, patience=2))
    ctrl.update({"clean_completion_rate": 1.0})  # streak 1
    ctrl.reset()  # new chunk: mastering one track must not pre-credit the next
    assert ctrl.streak == 0
    assert ctrl.update({"clean_completion_rate": 1.0}) is False  # streak 1, not 2


def test_controller_patience_one_stops_immediately():
    ctrl = EarlyStopController(OfftrackRate(0.0, patience=1))
    assert ctrl.update({"offtrack_rate": 0.0}) is True
