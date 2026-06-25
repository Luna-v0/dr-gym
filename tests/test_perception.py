"""Tests for the W-perception supervised head — the target builder
(`perception_targets`) maps `reward_params` to frame-local labels correctly, and
the net produces in-range outputs of the right shape. No sim required."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gym_dr.perception import (
    ALL_FEATURES,
    DYNAMIC_FEATURES,
    PERCEPTION_FEATURES,
    PRIVILEGED_EXTRA_FEATURES,
    SIGNED_FEATURES,
    all_targets,
    critic_state,
    dynamic_targets,
    enrich_reward_params,
    perception_targets,
    privileged_state,
    signed_indices_for,
)


def _straight_track_params(**kw):
    """A simple straight track heading east (+x): tangent = 0 deg."""
    p = {
        "track_width": 1.0,
        "distance_from_center": 0.0,
        "is_left_of_center": False,
        "heading": 0.0,
        "speed": 0.0,
        "waypoints": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        "closest_waypoints": [1, 2],
    }
    p.update(kw)
    return p


def test_feature_vector_shape_and_order():
    t = perception_targets(_straight_track_params())
    assert t.shape == (len(PERCEPTION_FEATURES),)
    assert t.dtype == np.float32


def test_centerline_is_zero_offset_and_equal_edges():
    t = perception_targets(_straight_track_params())
    f = dict(zip(PERCEPTION_FEATURES, t))
    assert f["lateral_offset"] == pytest.approx(0.0)
    # at centre, each edge is half the track width away -> 0.5 of full width
    assert f["dist_left_edge"] == pytest.approx(0.5)
    assert f["dist_right_edge"] == pytest.approx(0.5)


def test_lateral_offset_sign_and_saturation():
    # right of centre -> +; at the edge (half width) -> +1
    right_edge = perception_targets(
        _straight_track_params(distance_from_center=0.5, is_left_of_center=False)
    )
    left_edge = perception_targets(
        _straight_track_params(distance_from_center=0.5, is_left_of_center=True)
    )
    fr = dict(zip(PERCEPTION_FEATURES, right_edge))
    fl = dict(zip(PERCEPTION_FEATURES, left_edge))
    assert fr["lateral_offset"] == pytest.approx(1.0)
    assert fl["lateral_offset"] == pytest.approx(-1.0)
    # at the right edge: dist to right edge ~0, dist to left edge ~full width
    assert fr["dist_right_edge"] == pytest.approx(0.0)
    assert fr["dist_left_edge"] == pytest.approx(1.0)


def test_edges_sum_to_one():
    for d, left in [(0.2, False), (0.3, True), (0.0, False)]:
        t = perception_targets(
            _straight_track_params(distance_from_center=d, is_left_of_center=left)
        )
        f = dict(zip(PERCEPTION_FEATURES, t))
        assert f["dist_left_edge"] + f["dist_right_edge"] == pytest.approx(1.0)


def test_heading_error_against_tangent():
    # car points 45 deg left of an east-bound track -> heading_error = 45/180
    t = perception_targets(_straight_track_params(heading=45.0))
    f = dict(zip(PERCEPTION_FEATURES, t))
    assert f["heading_error"] == pytest.approx(45.0 / 180.0)


def test_heading_error_wraps():
    # heading 170, tangent 0 -> +170/180, NOT -190
    t = perception_targets(_straight_track_params(heading=170.0))
    f = dict(zip(PERCEPTION_FEATURES, t))
    assert f["heading_error"] == pytest.approx(170.0 / 180.0)


def test_speed_is_raw_mps():
    # speed_mps is the RAW speed in m/s (un-normalised — sim2real-stable), not /max.
    t = perception_targets(_straight_track_params(speed=4.0))
    f = dict(zip(PERCEPTION_FEATURES, t))
    assert f["speed_mps"] == pytest.approx(4.0)
    t2 = perception_targets(_straight_track_params(speed=2.0))
    assert dict(zip(PERCEPTION_FEATURES, t2))["speed_mps"] == pytest.approx(2.0)


def test_yaw_rate_finite_difference():
    prev = _straight_track_params(heading=0.0)
    cur = _straight_track_params(heading=15.0)
    t = perception_targets(cur, prev)
    f = dict(zip(PERCEPTION_FEATURES, t))
    assert f["yaw_rate"] == pytest.approx(15.0 / 30.0)
    # no previous step -> 0
    t0 = perception_targets(cur, None)
    assert dict(zip(PERCEPTION_FEATURES, t0))["yaw_rate"] == 0.0


def test_missing_keys_dont_raise():
    t = perception_targets({})
    assert t.shape == (len(PERCEPTION_FEATURES),)
    assert np.all(np.isfinite(t))


def test_no_waypoints_zero_heading_error():
    p = _straight_track_params(heading=90.0)
    del p["waypoints"]
    t = perception_targets(p)
    assert dict(zip(PERCEPTION_FEATURES, t))["heading_error"] == 0.0


def test_privileged_state_shape_and_flags():
    p = _straight_track_params(progress=0.42, is_offtrack=True,
                               all_wheels_on_track=False, is_crashed=False)
    t = privileged_state(p)
    assert t.shape == (len(PRIVILEGED_EXTRA_FEATURES),)
    f = dict(zip(PRIVILEGED_EXTRA_FEATURES, t))
    assert f["progress_frac"] == pytest.approx(0.42)
    assert f["offtrack"] == 1.0
    assert f["wheels_on_track"] == 0.0
    assert f["crashed"] == 0.0


def test_privileged_progress_accepts_0_100():
    # tolerate 0-100 progress as well as 0-1
    assert dict(zip(PRIVILEGED_EXTRA_FEATURES, privileged_state(
        _straight_track_params(progress=50.0))))["progress_frac"] == pytest.approx(0.5)


def test_privileged_nearest_object():
    far = privileged_state(_straight_track_params())  # no objects -> 1.0
    near = privileged_state(_straight_track_params(objects_distance=[0.0, 4.0]))
    assert dict(zip(PRIVILEGED_EXTRA_FEATURES, far))["nearest_object_dist"] == pytest.approx(1.0)
    assert dict(zip(PRIVILEGED_EXTRA_FEATURES, near))["nearest_object_dist"] == pytest.approx(0.0)


def test_curvature_ahead_zero_on_straight_signed_on_turn():
    # a long straight (enough waypoints that the K-ahead lookahead doesn't wrap)
    straight = {
        **_straight_track_params(),
        "waypoints": [(float(i), 0.0) for i in range(12)],
        "closest_waypoints": [1, 2],
    }
    assert dict(zip(PRIVILEGED_EXTRA_FEATURES, privileged_state(straight)))[
        "curvature_ahead"] == pytest.approx(0.0)
    # a left-bending waypoint sequence -> positive curvature
    left_turn = {
        **_straight_track_params(),
        "waypoints": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.5), (2.5, 1.5), (2.7, 2.7),
                      (2.7, 3.9), (2.5, 5.1), (2.0, 6.1)],
        "closest_waypoints": [0, 1],
    }
    c = dict(zip(PRIVILEGED_EXTRA_FEATURES, privileged_state(left_turn)))["curvature_ahead"]
    assert c > 0.0


def test_critic_state_is_concat_superset():
    p = _straight_track_params(progress=0.3, distance_from_center=0.2)
    cs = critic_state(p)
    assert cs.shape == (len(PERCEPTION_FEATURES) + len(PRIVILEGED_EXTRA_FEATURES),)
    # first block == actor features, second block == privileged extras
    assert np.allclose(cs[: len(PERCEPTION_FEATURES)], perception_targets(p))
    assert np.allclose(cs[len(PERCEPTION_FEATURES):], privileged_state(p))


def test_dynamic_targets_zero_without_prev():
    t = dynamic_targets(_straight_track_params(), None)
    assert t.shape == (len(DYNAMIC_FEATURES),)
    assert np.all(t == 0.0)


def test_dynamic_long_accel_sign():
    prev = _straight_track_params(speed=1.0)
    cur = _straight_track_params(speed=1.4)   # speeding up -> + accel
    f = dict(zip(DYNAMIC_FEATURES, dynamic_targets(cur, prev)))
    assert f["long_accel"] > 0.0
    # slowing down -> negative
    f2 = dict(zip(DYNAMIC_FEATURES, dynamic_targets(_straight_track_params(speed=0.6), prev)))
    assert f2["long_accel"] < 0.0


def test_dynamic_lateral_velocity_sign():
    # moving from center toward the right edge -> + lateral_velocity
    prev = _straight_track_params(distance_from_center=0.0)
    cur = _straight_track_params(distance_from_center=0.2, is_left_of_center=False)
    f = dict(zip(DYNAMIC_FEATURES, dynamic_targets(cur, prev)))
    assert f["lateral_velocity"] > 0.0


def test_dynamic_edge_closing_rate_positive_when_approaching():
    # nearer the edge than last step -> approaching -> + closing rate
    prev = _straight_track_params(distance_from_center=0.1, is_left_of_center=False)
    cur = _straight_track_params(distance_from_center=0.3, is_left_of_center=False)
    f = dict(zip(DYNAMIC_FEATURES, dynamic_targets(cur, prev)))
    assert f["edge_closing_rate"] > 0.0


def test_all_targets_concat_and_names():
    assert ALL_FEATURES == PERCEPTION_FEATURES + DYNAMIC_FEATURES
    p = _straight_track_params(speed=1.0)
    c = _straight_track_params(speed=1.5, distance_from_center=0.1)
    at = all_targets(c, p)
    assert at.shape == (len(ALL_FEATURES),)
    assert np.allclose(at[: len(PERCEPTION_FEATURES)], perception_targets(c, p))
    assert np.allclose(at[len(PERCEPTION_FEATURES):], dynamic_targets(c, p))


def test_signed_indices_for_core_and_extended():
    # core six: lateral_offset(0), heading_error(1), yaw_rate(5)
    assert signed_indices_for(PERCEPTION_FEATURES) == (0, 1, 5)
    # extended: dynamic derivatives are all signed too
    idx = signed_indices_for(ALL_FEATURES)
    for name in ("long_accel", "lateral_velocity", "edge_closing_rate"):
        assert ALL_FEATURES.index(name) in idx
    # edge distances / speed are NOT signed
    assert ALL_FEATURES.index("dist_left_edge") not in idx
    assert ALL_FEATURES.index("speed_mps") not in idx


def test_enrich_reward_params_adds_features_keeps_originals():
    p = _straight_track_params(speed=2.0, distance_from_center=0.1)
    prev = _straight_track_params(speed=1.0, distance_from_center=0.0)
    e = enrich_reward_params(p, prev)
    # originals preserved
    assert e["speed"] == 2.0 and e["distance_from_center"] == 0.1
    # derived feature keys present and matching all_targets
    at = dict(zip(ALL_FEATURES, all_targets(p, prev)))
    for name in ALL_FEATURES:
        assert e[name] == pytest.approx(at[name])
    # a reward function could now read e.g. edge_closing_rate
    assert "edge_closing_rate" in e and "long_accel" in e


def test_net_extended_output_ranges():
    import torch

    from gym_dr.perception import PerceptionNet

    net = PerceptionNet(in_channels=4, input_hw=(60, 80), n_outputs=len(ALL_FEATURES),
                        signed_indices=signed_indices_for(ALL_FEATURES))
    x = torch.randint(0, 256, (2, 4, 60, 80), dtype=torch.float32)
    y = net(x)
    assert y.shape == (2, len(ALL_FEATURES))
    for name in SIGNED_FEATURES:
        if name in ALL_FEATURES:
            col = ALL_FEATURES.index(name)
            assert torch.all(y[:, col] >= -1.0) and torch.all(y[:, col] <= 1.0)


def test_net_forward_shape_and_range():
    import torch

    from gym_dr.perception import PerceptionNet

    net = PerceptionNet(in_channels=4, input_hw=(120, 160))
    x = torch.randint(0, 256, (3, 4, 120, 160), dtype=torch.float32)
    y = net(x)
    assert y.shape == (3, len(PERCEPTION_FEATURES))
    # signed channels in [-1,1], bounded channels in [0,1]
    signed_idx = [0, 1, 5]
    bounded_idx = [2, 3, 4]
    assert torch.all(y[:, signed_idx] >= -1.0) and torch.all(y[:, signed_idx] <= 1.0)
    assert torch.all(y[:, bounded_idx] >= 0.0) and torch.all(y[:, bounded_idx] <= 1.0)


def test_net_can_overfit_one_batch():
    """A sanity check that the net + targets are wired for learning: it should
    drive the loss down on a tiny fixed batch."""
    import torch

    from gym_dr.perception import PerceptionNet

    torch.manual_seed(0)
    net = PerceptionNet(in_channels=4, input_hw=(60, 80))
    x = torch.randint(0, 256, (8, 4, 60, 80), dtype=torch.float32)
    y = torch.rand(8, len(PERCEPTION_FEATURES))
    y[:, [0, 1, 5]] = y[:, [0, 1, 5]] * 2 - 1  # signed targets into [-1,1]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first
