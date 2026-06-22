"""Fast tests for the speed-coloured eval charts + path-speed capture (no sim/PPO)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from gym_dr.metrics import _EpisodeMetrics
from gym_dr.trainers.sb3.plots import render_episode, render_overlay


def _ep():
    return {
        "x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 0.0, 1.0, 1.0],
        "speed": [1.0, 2.0, 3.0, 2.5],
        "wp_x": [0.0, 1.0, 2.0, 3.0, 4.0], "wp_y": [0.0, 0.0, 0.0, 0.0, 0.0],
        "track_width": 1.0, "status": "lap-complete", "progress": 100.0,
    }


def test_render_episode_speed_coloured():
    fig = render_episode("Spain_track", 1000, 0, _ep())
    assert fig is not None
    assert len(fig.axes) >= 2  # main axes + the speed colourbar


def test_render_episode_without_speed_falls_back():
    ep = _ep(); ep.pop("speed")
    fig = render_episode("Spain_track", 1000, 0, ep)
    assert fig is not None


def test_render_overlay_returns_figure():
    fig = render_overlay("Spain_track", 1000, [_ep(), _ep()])
    assert fig is not None


def test_metrics_path_payload_includes_aligned_speed():
    m = _EpisodeMetrics()
    m.capture_path = True
    for i in range(3):
        m.record_step(
            {"x": float(i), "y": 0.0, "speed": float(i) + 1.0, "progress": float(i * 10),
             "is_offtrack": False, "all_wheels_on_track": True},
            reward=1.0,
        )
    p = m.path_payload()
    assert p["speed"] == [1.0, 2.0, 3.0]
    assert len(p["speed"]) == len(p["x"]) == len(p["y"])
