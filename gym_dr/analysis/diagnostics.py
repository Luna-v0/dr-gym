"""Diagnostic quality metric — the "reward-as-metric" for post-hoc analysis.

The maintainer's need: a single interpretable signal that says *how well* a policy
actually drives — one that recognises the two dominant failure modes a plain mean
reward hides:

- **crawl**: the car creeps at (near) the minimum speed — technically "safe" but
  useless; a speed-weighted training reward can even reward it;
- **off-track**: the car leaves the track (and/or crashes).

These functions are **pure** (a slice of the Tier-1 trace in, a dict/DataFrame out),
so they are reproducible and unit-testable on synthetic episodes, and they are used
ONLY for analysis — never as a training reward.

Trace columns consumed (see ``gym_dr.trace.STEP_COLUMNS``): ``episode``, ``speed``,
``progress`` (0–100), ``is_offtrack``, ``is_crashed``, ``distance_from_center``,
``track_width``, ``phase`` (``train``/``eval``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# Failure-mode thresholds (tunable; the defaults match the maintainer's intent).
CRAWL_RATE = 0.5        # > this fraction of steps at min speed ⇒ "crawl"
OFFTRACK_RATE = 0.3     # > this fraction of steps off-track ⇒ "offtrack"
COMPLETE_PROGRESS = 100.0


def episode_diagnostics(
    df_episode: "pd.DataFrame",
    *,
    min_speed: float = 1.0,
    speed_eps: float = 0.1,
) -> "dict[str, Any]":
    """Diagnostics for ONE episode (a single-episode slice of the trace).

    Returns a dict with:

    - ``clean_completed`` — finished the lap (progress ≥ 100) with zero off-track steps;
    - ``progress_reached`` — max progress (0–100);
    - ``offtrack_rate`` — fraction of steps off the track;
    - ``speed_below_min_rate`` — fraction of steps at/near the minimum speed
      (``speed <= min_speed + speed_eps``) — the crawl detector;
    - ``mean_speed``;
    - ``mean_dist_normalized`` — mean |distance-from-centre| / (track_width/2);
    - ``crashed``; ``n_steps``.
    """
    n = int(len(df_episode))
    if n == 0:
        return {
            "clean_completed": False, "progress_reached": 0.0, "offtrack_rate": 0.0,
            "speed_below_min_rate": 0.0, "mean_speed": 0.0, "mean_dist_normalized": 0.0,
            "crashed": False, "n_steps": 0,
        }
    import numpy as np

    def col(name, default=0.0):
        return df_episode[name] if name in df_episode.columns else _const(df_episode, default)

    progress = float(np.nanmax(col("progress").to_numpy(dtype=float)))
    offtrack = col("is_offtrack", False).fillna(False).astype(bool)
    offtrack_rate = float(offtrack.mean())
    speed = col("speed").astype(float)
    speed_below_min_rate = float((speed <= (min_speed + speed_eps)).mean())
    mean_speed = float(speed.mean())

    half_w = (col("track_width", 1.0).astype(float) / 2.0).replace(0.0, np.nan)
    dist = col("distance_from_center").astype(float).abs()
    mean_dist_norm = float((dist / half_w).mean())

    crashed = bool(col("is_crashed", False).fillna(False).astype(bool).any())
    clean_completed = bool(progress >= COMPLETE_PROGRESS and int(offtrack.sum()) == 0)

    return {
        "clean_completed": clean_completed,
        "progress_reached": progress,
        "offtrack_rate": offtrack_rate,
        "speed_below_min_rate": speed_below_min_rate,
        "mean_speed": mean_speed,
        "mean_dist_normalized": (0.0 if np.isnan(mean_dist_norm) else mean_dist_norm),
        "crashed": crashed,
        "n_steps": n,
    }


def _const(df, value):
    import pandas as pd

    return pd.Series([value] * len(df), index=df.index)


def quality_score(d: "Mapping[str, Any]") -> float:
    """A single diagnostic quality score in ``[0, 1]`` (NOT a training reward).

    A clean lap driven at a real (non-crawl) speed scores ``1.0``. Otherwise the
    score is the product of three [0,1] factors — progress made, fraction on-track,
    and fraction driven above the crawl speed — so a policy is rewarded only when it
    makes progress *and* stays on track *and* actually drives. A pure crawler
    (``speed_below_min_rate ≈ 1``) or a lap-leaver scores near ``0`` even if a
    speed-weighted training reward looked fine.
    """
    if bool(d.get("clean_completed")) and float(d.get("speed_below_min_rate", 1.0)) < CRAWL_RATE:
        return 1.0
    progress = float(d.get("progress_reached", 0.0)) / 100.0
    on_track = 1.0 - float(d.get("offtrack_rate", 1.0))
    driving = 1.0 - float(d.get("speed_below_min_rate", 1.0))
    return max(0.0, min(1.0, progress * on_track * driving))


def failure_modes(d: "Mapping[str, Any]") -> "List[str]":
    """Human-readable failure labels for an episode (empty ⇒ clean lap).

    ``crawl`` (mostly min-speed), ``offtrack`` (leaves the track a lot),
    ``crashed``, ``incomplete`` (never finished the lap).
    """
    modes: List[str] = []
    if float(d.get("speed_below_min_rate", 0.0)) > CRAWL_RATE:
        modes.append("crawl")
    if float(d.get("offtrack_rate", 0.0)) > OFFTRACK_RATE:
        modes.append("offtrack")
    if bool(d.get("crashed")):
        modes.append("crashed")
    if float(d.get("progress_reached", 0.0)) < COMPLETE_PROGRESS and not modes:
        modes.append("incomplete")
    return modes


def run_diagnostics(
    run_dir,
    *,
    phase: "str | None" = "eval",
    min_speed: float = 1.0,
) -> "pd.DataFrame":
    """Per-episode diagnostics for a whole run directory (reads the Tier-1 trace).

    Loads ``run_dir/trace/steps/*.parquet`` via ``gym_dr.trace.load_steps``,
    optionally filters to ``phase`` (``eval`` by default; ``None`` = all), groups by
    ``episode`` and stacks :func:`episode_diagnostics` + :func:`quality_score` +
    :func:`failure_modes` into a DataFrame (one row per episode). Requires the run to
    have been recorded with ``trace.enabled=True``.
    """
    import pandas as pd

    from gym_dr.trace import load_steps

    df = load_steps(run_dir)
    if len(df) and phase and "phase" in df.columns:
        df = df[df["phase"] == phase]
    rows: List[dict] = []
    if len(df) and "episode" in df.columns:
        for episode, g in df.groupby("episode"):
            d = episode_diagnostics(g, min_speed=min_speed)
            d["episode"] = episode
            d["quality_score"] = quality_score(d)
            d["failure_modes"] = ",".join(failure_modes(d))
            rows.append(d)
    return pd.DataFrame(rows)


def aggregate_runs(
    run_dirs: "List[Any]",
    *,
    phase: "str | None" = "eval",
    min_speed: float = 1.0,
) -> "dict[str, Any]":
    """Diagnose several run dirs (e.g. one per seed / config) and aggregate them.

    Returns ``{"per_run": [summary, ...], "overall": {...}}`` where each per-run
    summary is :func:`summarize_diagnostics` plus its ``run`` path, and ``overall``
    averages ``mean_quality`` / ``clean_completion_rate`` across the runs that
    produced episodes. This is the unit for cross-seed comparison (feed the per-run
    ``mean_quality`` to rliable for IQM + CIs when ≥3 seeds are available).
    """
    per_run: List[dict] = []
    for rd in run_dirs:
        summary = summarize_diagnostics(run_diagnostics(rd, phase=phase, min_speed=min_speed))
        summary["run"] = str(rd)
        per_run.append(summary)
    scored = [r for r in per_run if r["n_episodes"] > 0]
    n = len(scored)
    overall = {
        "n_runs": len(per_run),
        "n_scored_runs": n,
        "mean_quality": (sum(r["mean_quality"] for r in scored) / n) if n else 0.0,
        "mean_clean_completion": (
            sum(r["clean_completion_rate"] for r in scored) / n) if n else 0.0,
    }
    return {"per_run": per_run, "overall": overall}


def summarize_diagnostics(diag: "pd.DataFrame") -> "dict[str, Any]":
    """Roll a per-episode diagnostics DataFrame up to a run-level verdict."""
    n = int(len(diag))
    if n == 0:
        return {"n_episodes": 0, "mean_quality": 0.0, "clean_completion_rate": 0.0,
                "crawl_rate": 0.0, "offtrack_episode_rate": 0.0, "dominant_failure": "none"}
    from collections import Counter

    modes = Counter()
    for cell in diag["failure_modes"]:
        for m in (cell.split(",") if cell else []):
            modes[m] += 1
    dominant = modes.most_common(1)[0][0] if modes else "none"
    return {
        "n_episodes": n,
        "mean_quality": float(diag["quality_score"].mean()),
        "clean_completion_rate": float(diag["clean_completed"].mean()),
        "crawl_rate": float((diag["speed_below_min_rate"] > CRAWL_RATE).mean()),
        "offtrack_episode_rate": float((diag["offtrack_rate"] > OFFTRACK_RATE).mean()),
        "dominant_failure": dominant,
    }
