"""Unit tests for the reward function variants.

Each variant must:
  - return a finite float
  - never return 0 (we floor to 1e-3 so PPO doesn't see exact-zero rewards)
  - be robust to partial params dicts (the env doesn't always supply
    every key — e.g. on the first step of an episode some fields are
    defaulted)
"""
from __future__ import annotations

import math

import pytest

from gym_dr.rewards import (
    OFFTRACK_PENALTY,
    REWARD_VARIANTS,
    anti_zigzag,
    center_line,
    centerline_quadratic,
    progress_and_speed,
    progress_per_step,
    progress_safe,
    waypoint_anticipation,
)


def _full_params(**overrides):
    """A reasonable DeepRacer params dict; override fields per-test."""
    base = {
        "track_width": 1.0,
        "distance_from_center": 0.05,
        "x": 0.0,
        "y": 0.0,
        "heading": 0.0,
        "progress": 50.0,
        "steps": 200,
        "speed": 2.5,
        "steering_angle": 5.0,
        "track_length": 17.6,
        "waypoints": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0),
                      (5.0, 0.0), (6.0, 0.0), (7.0, 0.0)],
        "closest_waypoints": [0, 1],
        "all_wheels_on_track": True,
        "is_offtrack": False,
        "is_crashed": False,
    }
    base.update(overrides)
    return base


VARIANTS = list(REWARD_VARIANTS.items())


@pytest.mark.parametrize("name,fn", VARIANTS, ids=[n for n, _ in VARIANTS])
def test_variant_returns_finite_float(name, fn):
    r = fn(_full_params())
    assert isinstance(r, float)
    assert math.isfinite(r)
    assert r > 0.0   # on-track reward is positive


@pytest.mark.parametrize("name,fn", VARIANTS, ids=[n for n, _ in VARIANTS])
def test_variant_handles_offtrack(name, fn):
    """Off-track / wheels-off must yield a *finite, negative* reward —
    actively punishing excursion, since the upstream env doesn't terminate
    on off-track. Exact magnitude varies by variant."""
    r_on = fn(_full_params(all_wheels_on_track=True, is_offtrack=False))
    r_off = fn(_full_params(all_wheels_on_track=False, is_offtrack=True))
    assert math.isfinite(r_off)
    assert r_off < 0, f"{name}: off-track reward must be negative, got {r_off}"
    # On-track must strictly dominate off-track for every variant (no
    # perverse incentive to leave the road).
    assert r_on > r_off, f"{name}: on={r_on} off={r_off}"


@pytest.mark.parametrize(
    "name,fn",
    [(n, f) for n, f in VARIANTS if n != "center_line"],  # center_line is the user's custom; skip
    ids=[n for n, _ in VARIANTS if n != "center_line"],
)
def test_variant_robust_to_missing_keys(name, fn):
    """A skeletal params dict (only the most-common keys present) must still
    yield a finite float — tests defensive .get() defaults."""
    skeletal = {"track_width": 1.0, "distance_from_center": 0.1}
    r = fn(skeletal)
    assert isinstance(r, float)
    assert math.isfinite(r)


def test_progress_per_step_monotonic_in_pace():
    """progress_per_step should reward finishing more progress in fewer steps."""
    slow = progress_per_step(_full_params(progress=50.0, steps=400, speed=1.0))
    fast = progress_per_step(_full_params(progress=50.0, steps=100, speed=1.0))
    assert fast > slow


def test_anti_zigzag_penalizes_sharp_steering():
    smooth = anti_zigzag(_full_params(steering_angle=5.0, distance_from_center=0.05))
    sharp = anti_zigzag(_full_params(steering_angle=20.0, distance_from_center=0.05))
    assert smooth > sharp


def test_centerline_quadratic_peaks_at_center():
    centered = centerline_quadratic(_full_params(distance_from_center=0.0))
    near_edge = centerline_quadratic(_full_params(distance_from_center=0.4))
    assert centered > near_edge


def test_progress_safe_offtrack_penalty():
    """progress_safe is the eval-only reward. Off-track must produce a
    negative per-step value strictly worse than any on-track value so a
    cleaner lap always ranks above a dirtier one of the same length."""
    on = progress_safe(_full_params(all_wheels_on_track=True, is_offtrack=False))
    off_wheels = progress_safe(_full_params(all_wheels_on_track=False))
    off_flag = progress_safe(_full_params(is_offtrack=True))
    assert on > 0
    assert off_wheels == OFFTRACK_PENALTY
    assert off_flag == OFFTRACK_PENALTY
    assert OFFTRACK_PENALTY < 0
    # Off-track must strictly worsen episode total per step.
    assert off_wheels < on


def test_progress_safe_not_in_training_variants():
    """progress_safe is eval-only — it must NOT be sampled as a training
    reward by HPO (the large negative would destabilise PPO gradients)."""
    assert "progress_safe" not in REWARD_VARIANTS
    assert progress_safe not in REWARD_VARIANTS.values()


def test_progress_safe_is_default_eval_reward():
    """ExperimentConfig.eval_reward defaults to progress_safe so trials
    sweeping different training rewards rank on the same axis."""
    from gym_dr.config import ExperimentConfig
    exp = ExperimentConfig(name="t")
    assert exp.eval_reward is progress_safe


def test_waypoint_anticipation_uses_track_geometry():
    """When the upcoming waypoints curve sharply, slow speed should reward
    more than high speed."""
    # Build a hard-right turn waypoint sequence ahead of index 0.
    wps = [(0.0, 0.0), (1.0, 0.0), (1.5, 1.0), (1.5, 2.0),
           (1.5, 3.0), (1.5, 4.0), (1.5, 5.0), (1.5, 6.0)]
    fast = waypoint_anticipation(_full_params(
        waypoints=wps, closest_waypoints=[0, 1], heading=0.0, speed=3.0,
    ))
    slow = waypoint_anticipation(_full_params(
        waypoints=wps, closest_waypoints=[0, 1], heading=0.0, speed=1.0,
    ))
    # On a turn ahead, slowing down should be rewarded.
    assert slow > fast
