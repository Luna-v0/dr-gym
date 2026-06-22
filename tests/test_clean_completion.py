"""Tests for the P1 clean-completion eval reward + episode metrics.

The success criterion: finish every held-out track *without leaving it*, at a
reasonable (non-minimum) speed. ``clean_completion`` is the per-step proxy;
``dr/ep_completed`` / ``dr/ep_completed_clean`` are the episode metrics.
"""
from __future__ import annotations

import math

from gym_dr.rewards import (
    CLEAN_OFFTRACK_PENALTY,
    COMPLETION_BONUS,
    REWARD_VARIANTS,
    clean_completion,
)
from gym_dr.metrics import _EpisodeMetrics


def _params(progress=50.0, steps=200, speed=2.5, offtrack=False):
    return {
        "progress": progress,
        "steps": steps,
        "speed": speed,
        "steering_angle": 0.0,
        "is_offtrack": offtrack,
        "all_wheels_on_track": not offtrack,
    }


def test_clean_completion_on_track_positive():
    r = clean_completion(_params())
    assert isinstance(r, float) and math.isfinite(r) and r > 0


def test_clean_completion_offtrack_penalty():
    assert clean_completion(_params(offtrack=True)) == CLEAN_OFFTRACK_PENALTY
    assert CLEAN_OFFTRACK_PENALTY < 0
    # Off-track must be strictly worse than any on-track step.
    assert clean_completion(_params(offtrack=True)) < clean_completion(_params())


def test_clean_completion_rewards_finishing():
    done = clean_completion(_params(progress=100.0, steps=200, speed=2.5))
    almost = clean_completion(_params(progress=99.0, steps=200, speed=2.5))
    assert done - almost >= COMPLETION_BONUS - 1.0  # the one-off bonus kicks in


def test_clean_completion_prefers_faster_clean_lap():
    fast = clean_completion(_params(progress=50.0, steps=100, speed=3.0))
    slow = clean_completion(_params(progress=50.0, steps=400, speed=1.0))
    assert fast > slow  # better pace AND faster, both clean


def test_clean_completion_is_eval_only():
    assert "clean_completion" not in REWARD_VARIANTS
    assert clean_completion not in REWARD_VARIANTS.values()


def _run_episode(progresses, offtrack_steps=()):
    m = _EpisodeMetrics()
    for i, pr in enumerate(progresses):
        m.record_step(_params(progress=pr, offtrack=(i in offtrack_steps)), reward=1.0)
    return m.summary()


def test_completed_clean_metric():
    s = _run_episode([10, 40, 70, 100])
    assert s["dr/ep_completed"] == 1.0
    assert s["dr/ep_completed_clean"] == 1.0


def test_completed_but_dirty_metric():
    # An off-track step on the way to a finish ⇒ completed but not clean.
    s = _run_episode([20, 50, 80, 100], offtrack_steps={1})
    assert s["dr/ep_completed"] == 1.0
    assert s["dr/ep_completed_clean"] == 0.0


def test_incomplete_metric():
    s = _run_episode([10, 30, 55])
    assert s["dr/ep_completed"] == 0.0
    assert s["dr/ep_completed_clean"] == 0.0
