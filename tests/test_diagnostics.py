"""Tests for the diagnostic quality metric (gym_dr.analysis.diagnostics)."""
from __future__ import annotations

import pandas as pd
import pytest

from gym_dr.analysis.diagnostics import (
    episode_diagnostics,
    failure_modes,
    quality_score,
    run_diagnostics,
    summarize_diagnostics,
)


def _episode(speed, progress, offtrack, *, track_width=1.0, dist=0.0, crashed=False):
    """Build a one-episode trace slice from per-step lists (or scalars)."""
    n = len(speed)

    def col(v):
        return v if isinstance(v, list) else [v] * n

    return pd.DataFrame({
        "episode": [0] * n,
        "speed": col(speed),
        "progress": col(progress),
        "is_offtrack": col(offtrack),
        "is_crashed": col(crashed),
        "track_width": col(track_width),
        "distance_from_center": col(dist),
        "phase": ["eval"] * n,
    })


# ----------------------------------------------------- episode_diagnostics

def test_clean_lap_scores_one():
    # Finishes the lap (progress hits 100), never off-track, drives above crawl.
    df = _episode(speed=[3.0, 3.5, 4.0], progress=[10.0, 60.0, 100.0], offtrack=False)
    d = episode_diagnostics(df, min_speed=1.0)
    assert d["clean_completed"] is True
    assert d["offtrack_rate"] == 0.0
    assert d["speed_below_min_rate"] == 0.0
    assert quality_score(d) == 1.0
    assert failure_modes(d) == []


def test_crawler_is_flagged_and_scores_low():
    # Creeps at the minimum speed the whole time — the failure a speed reward hides.
    df = _episode(speed=[1.0, 1.05, 1.0, 1.1], progress=[5.0, 8.0, 10.0, 12.0], offtrack=False)
    d = episode_diagnostics(df, min_speed=1.0)
    assert d["speed_below_min_rate"] == 1.0
    assert d["clean_completed"] is False
    assert quality_score(d) == 0.0            # driving factor is 0
    assert "crawl" in failure_modes(d)


def test_offtrack_is_flagged():
    df = _episode(speed=[3.0] * 4, progress=[10.0, 20.0, 30.0, 40.0],
                  offtrack=[False, True, True, True])
    d = episode_diagnostics(df)
    assert d["offtrack_rate"] == 0.75
    assert d["clean_completed"] is False
    assert "offtrack" in failure_modes(d)
    assert quality_score(d) < 0.2


def test_completed_but_left_track_is_not_clean():
    df = _episode(speed=[3.0, 3.0], progress=[50.0, 100.0], offtrack=[True, False])
    d = episode_diagnostics(df)
    assert d["progress_reached"] == 100.0
    assert d["clean_completed"] is False      # completion requires zero off-track


def test_normalized_distance_from_center():
    # dist 0.5 on track_width 2 -> half-width 1 -> normalized 0.5
    df = _episode(speed=[3.0], progress=[100.0], offtrack=False, track_width=2.0, dist=0.5)
    d = episode_diagnostics(df)
    assert abs(d["mean_dist_normalized"] - 0.5) < 1e-9


def test_crashed_flag_and_mode():
    df = _episode(speed=[3.0, 3.0], progress=[30.0, 35.0], offtrack=False, crashed=[False, True])
    d = episode_diagnostics(df)
    assert d["crashed"] is True
    assert "crashed" in failure_modes(d)


def test_incomplete_only_mode_when_no_other_failure():
    df = _episode(speed=[3.0, 3.0], progress=[40.0, 55.0], offtrack=False)
    d = episode_diagnostics(df)
    assert failure_modes(d) == ["incomplete"]


def test_empty_episode_safe():
    d = episode_diagnostics(pd.DataFrame(columns=["speed", "progress"]))
    assert d["n_steps"] == 0
    assert quality_score(d) == 0.0


# ---------------------------------------------------------- run_diagnostics

def test_run_diagnostics_reads_trace(tmp_path):
    # Write two eval-phase episode shards + one train-phase shard; expect eval only.
    steps_dir = tmp_path / "trace" / "steps"
    steps_dir.mkdir(parents=True)
    clean = _episode(speed=[3.0, 4.0], progress=[50.0, 100.0], offtrack=False)
    crawl = _episode(speed=[1.0, 1.0], progress=[5.0, 6.0], offtrack=False)
    crawl["episode"] = 1
    train = _episode(speed=[3.0], progress=[100.0], offtrack=False)
    train["episode"] = 2
    train["phase"] = "train"
    clean.to_parquet(steps_dir / "ep_000000.parquet")
    crawl.to_parquet(steps_dir / "ep_000001.parquet")
    train.to_parquet(steps_dir / "ep_000002.parquet")

    diag = run_diagnostics(tmp_path, phase="eval")
    assert len(diag) == 2                      # train episode filtered out
    by_ep = {int(r.episode): r for r in diag.itertuples()}
    assert by_ep[0].clean_completed is True
    assert "crawl" in by_ep[1].failure_modes

    summary = summarize_diagnostics(diag)
    assert summary["n_episodes"] == 2
    assert summary["clean_completion_rate"] == 0.5
    assert summary["dominant_failure"] == "crawl"


def test_run_diagnostics_empty_when_no_trace(tmp_path):
    diag = run_diagnostics(tmp_path)
    assert len(diag) == 0
    assert summarize_diagnostics(diag)["n_episodes"] == 0
