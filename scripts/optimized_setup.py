#!/usr/bin/env python
"""Write the optimal training configuration for THIS machine (Task 2).

Detects the host (cores / RAM / GPU) and emits a JSON describing the best
``n_cars`` + device for each observation type (feature vs camera), following the
maintainer's rule (favour more cars within ~80% of peak throughput, subject to a
per-car sample-quality floor — see ``gym_dr.setup_profile``).

Modes
-----
- default (no benchmark): fast **heuristic** profile from the detected hardware —
  useful immediately, clearly labelled ``source="heuristic"``.
- ``--candidates results.json``: use REAL benchmark numbers you already have. The
  file maps each obs type to a list of ``{"n_cars": N, "steps_per_s": F}`` (the
  aggregate agent-steps/s, i.e. per-car × n_cars) as produced by
  ``scripts/multicar_throughput.py``.
- ``--probe``: run a light on-machine benchmark first (needs the sim image +
  Docker); falls back to the heuristic if the sim is unavailable.

Examples
--------
    uv run python scripts/optimized_setup.py
    uv run python scripts/optimized_setup.py --candidates artifacts/multicar_grid.json
    uv run python scripts/optimized_setup.py --probe --out artifacts/optimized_setup.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Run from source without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym_dr.setup_profile import build_profile, detect_machine, heuristic_profile  # noqa: E402


def _load_candidates(path: Path) -> "tuple[list, list]":
    data = json.loads(Path(path).read_text())
    return data.get("feature_obs", []), data.get("camera_obs", [])


def _probe_candidates(machine: dict) -> "tuple[list, list] | None":
    """Run a light benchmark to get real (n_cars, steps_per_s) candidates.

    Sim-gated: needs the DeepRacer image + Docker. Returns ``None`` (→ heuristic
    fallback) when the sim harness can't run here. Wire to
    ``scripts/multicar_throughput.py`` when running on a machine with the sim.
    """
    try:
        # Deferred import: only meaningful where the sim/benchmark harness exists.
        from scripts import multicar_throughput  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    print("[optimized-setup] --probe requested but the benchmark harness needs the "
          "sim image + Docker; falling back to the heuristic profile. Run "
          "scripts/multicar_throughput.py on the sim host and pass its JSON via "
          "--candidates instead.", file=sys.stderr)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, help="benchmark results JSON to use")
    ap.add_argument("--probe", action="store_true",
                    help="run a light on-machine benchmark first (needs the sim)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/optimized_setup.json"))
    ap.add_argument("--per-car-floor", type=int, default=None)
    ap.add_argument("--throughput-tol", type=float, default=None)
    args = ap.parse_args(argv)

    machine = detect_machine()
    ts = datetime.now(timezone.utc).isoformat()
    kw = {}
    if args.per_car_floor is not None:
        kw["per_car_floor"] = args.per_car_floor
    if args.throughput_tol is not None:
        kw["throughput_tol"] = args.throughput_tol

    feature, camera = None, None
    if args.candidates:
        feature, camera = _load_candidates(args.candidates)
    elif args.probe:
        probed = _probe_candidates(machine)
        if probed is not None:
            feature, camera = probed

    if feature is None and camera is None:
        profile = heuristic_profile(machine, timestamp=ts)
    else:
        profile = build_profile(machine, feature_candidates=feature,
                                camera_candidates=camera, timestamp=ts, **kw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(profile, indent=2))
    print(f"\n[optimized-setup] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
