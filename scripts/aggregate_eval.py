#!/usr/bin/env python
"""Aggregate the diagnostic quality metric across runs / seeds (Task 3).

Reads the Tier-1 trace of one or more run directories (recorded with
``trace.enabled=True``), scores each with the diagnostic quality metric
(``gym_dr.analysis``), and prints a per-run + overall summary — the failure-mode
verdict (crawl / off-track / crashed / incomplete) plus the mean quality score.

With ``--rliable`` and the ``analysis`` dependency group installed
(``uv sync --group analysis`` — brings deepracer-utils[rliable]), it also reports
the robust **IQM** of the per-run mean quality with 95% bootstrap CIs (the
statistically sound way to compare seed groups; needs ≥3 runs).

Examples
--------
    uv run python scripts/aggregate_eval.py --run artifacts/run_a --run artifacts/run_b
    uv run python scripts/aggregate_eval.py --run artifacts/oracle_rep* --phase eval --json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym_dr.analysis import aggregate_runs  # noqa: E402


def _expand(patterns):
    dirs = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        dirs.extend(matches if matches else [p])
    return [Path(d) for d in dirs]


def _rliable_iqm(scores):
    """IQM + 95% CI of a 1-D score array via the vendored rliable wrapper.
    Returns None if deepracer-utils/rliable are not installed."""
    try:
        import numpy as np
        from deepracer.logs.rliable_utils import aggregate_metrics  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        # aggregate_metrics expects a {method: (runs, tasks)} score dict.
        return aggregate_metrics({"policy": np.asarray(scores).reshape(-1, 1)})
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="DIR",
                    help="a run directory (repeatable; globs allowed)")
    ap.add_argument("--phase", default="eval", help="eval | train | all")
    ap.add_argument("--min-speed", type=float, default=1.0)
    ap.add_argument("--rliable", action="store_true", help="add IQM + 95% CIs")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    if not args.run:
        ap.error("pass at least one --run DIR")
    run_dirs = _expand(args.run)
    phase = None if args.phase == "all" else args.phase
    result = aggregate_runs(run_dirs, phase=phase, min_speed=args.min_speed)

    if args.rliable:
        scores = [r["mean_quality"] for r in result["per_run"] if r["n_episodes"] > 0]
        iqm = _rliable_iqm(scores) if len(scores) >= 3 else None
        result["overall"]["rliable"] = iqm or "unavailable (need >=3 runs + analysis deps)"

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    print(f"{'run':40s} {'eps':>5s} {'quality':>8s} {'clean%':>7s} {'failure':>12s}")
    print("-" * 76)
    for r in result["per_run"]:
        run = Path(r["run"]).name[:40]
        print(f"{run:40s} {r['n_episodes']:5d} {r['mean_quality']:8.3f} "
              f"{r['clean_completion_rate'] * 100:6.1f}% {r['dominant_failure']:>12s}")
    o = result["overall"]
    print("-" * 76)
    print(f"OVERALL  runs={o['n_runs']} scored={o['n_scored_runs']}  "
          f"mean_quality={o['mean_quality']:.3f}  mean_clean={o['mean_clean_completion'] * 100:.1f}%")
    if args.rliable:
        print(f"rliable IQM: {result['overall']['rliable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
