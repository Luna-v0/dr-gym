"""Tests for the W-saferl cost functions — graded *risk* (proximity to a bad
state), in [0,1], rising as the car nears the boundary. Terminal off-track/crash
are NOT costs."""
from __future__ import annotations

import pytest

from gym_dr.costs import (
    COST_VARIANTS,
    cost_near_collision,
    cost_near_edge,
    make_composite_cost,
)


def _p(distance_from_center=0.0, track_width=1.0, **kw):
    return {"distance_from_center": distance_from_center, "track_width": track_width, **kw}


def test_near_edge_zero_at_center_one_at_edge():
    assert cost_near_edge(_p(0.0)) == 0.0          # centre
    assert cost_near_edge(_p(0.5)) == 1.0          # edge (track_width/2)
    assert cost_near_edge(_p(0.6)) == 1.0          # past the edge ⇒ saturates
    assert cost_near_edge({}) == 0.0               # no track_width


def test_near_edge_is_graded_and_monotone():
    # onset=0.5 ⇒ 0 in the inner half, ramps to 1 at the edge.
    assert cost_near_edge(_p(0.2)) == 0.0          # f=0.4 < onset
    assert cost_near_edge(_p(0.375)) == pytest.approx(0.5)  # f=0.75
    assert cost_near_edge(_p(0.4)) > cost_near_edge(_p(0.3))  # rises toward the edge


def test_near_collision_graded():
    assert cost_near_collision({}) == 0.0                       # no objects
    assert cost_near_collision(_p(objects_distance=[1.0])) == 0.0  # beyond threshold
    assert cost_near_collision(_p(objects_distance=[0.375]), threshold_m=0.75) == pytest.approx(0.5)
    assert cost_near_collision(_p(objects_distance=[0.0])) == 1.0  # contact
    assert cost_near_collision(_p(objects_distance=[2.0, 0.3]), threshold_m=0.75) == pytest.approx(0.6)


def test_composite_weights_validation():
    with pytest.raises(ValueError):
        make_composite_cost({"near_edge": -1.0})
    with pytest.raises(ValueError):
        make_composite_cost({"is_offtrack": 1.0})  # terminal flag is NOT a valid cost term


def test_composite_near_edge_term():
    c = make_composite_cost({"near_edge": 1.0})
    assert c(_p(0.0)) == 0.0
    assert c(_p(0.5)) == pytest.approx(1.0)


def test_composite_steering_jerk_needs_two_steps():
    c = make_composite_cost({"steering_jerk": 1.0})
    assert c({"steering_angle": 0.0}) == 0.0              # no previous steering
    assert c({"steering_angle": 30.0}) == pytest.approx(0.5)  # 30/60 swing


def test_cost_variants_are_graded_risk():
    assert set(COST_VARIANTS) == {"near_edge", "near_collision"}


def test_episode_metrics_logs_cost():
    from gym_dr.metrics import _EpisodeMetrics
    m = _EpisodeMetrics()
    m.cost_fn = cost_near_edge
    edge = {"track_width": 1.0, "distance_from_center": 0.5, "speed": 1.0,
            "steering_angle": 0.0, "progress": 10, "is_offtrack": False,
            "all_wheels_on_track": True}
    center = {**edge, "distance_from_center": 0.0, "progress": 20}
    m.record_step(edge, reward=1.0)    # at edge -> cost 1.0
    m.record_step(center, reward=1.0)  # centre -> cost 0.0
    s = m.summary()
    assert s["dr/ep_mean_cost"] == 0.5
    assert s["dr/ep_max_cost"] == 1.0


def test_config_cost_default_none_and_serializes():
    from gym_dr.config import ExperimentConfig
    e = ExperimentConfig(name="t")
    assert e.cost is None
    assert e.to_dict()["cost"] is None
    e2 = ExperimentConfig(name="t", cost=cost_near_edge)
    assert "cost_near_edge" in e2.to_dict()["cost"]
