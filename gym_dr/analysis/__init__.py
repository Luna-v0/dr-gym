"""Post-hoc analysis of the Tier-1 trace — diagnostic metrics + robust aggregation.

Pure, testable consumers of the ``gym_dr.trace`` Parquet shards (no simulator).
See :mod:`gym_dr.analysis.diagnostics` for the per-episode diagnostic quality
metric (the "reward-as-metric" that scores *how well* a policy drives, used for
analysis — never for the agent to learn).
"""
from gym_dr.analysis.diagnostics import (
    aggregate_runs,
    episode_diagnostics,
    failure_modes,
    quality_score,
    run_diagnostics,
    summarize_diagnostics,
)

__all__ = [
    "aggregate_runs",
    "episode_diagnostics",
    "failure_modes",
    "quality_score",
    "run_diagnostics",
    "summarize_diagnostics",
]
