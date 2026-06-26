#!/usr/bin/env python3
"""Track geometry analyzer — the **wobble × tightness** difficulty/diversity map.

Turns each DeepRacer track centreline (a closed polygon of waypoints) into two
named, physically-meaningful axes:

  * **tightness** — how sharp the corners get. p95 of |curvature| along the lap
    (robust "max"), i.e. how small the tightest corner radius is. The *physical*
    "how hard / how much must the car slow" axis.
  * **wobble** — how much the track changes direction. Sign-changes of curvature
    per metre (left<->right switches). The *rhythm* "zig-zag vs flowing" axis.

Curvature is the signed "hand rule": the z-component of the cross product of
successive tangent vectors (the convex/reflex test underlying ear-clipping),
turn-angle per arc-length, computed on the centreline resampled to uniform
arc-length (waypoints are unevenly spaced). Curvature is rotation/translation
invariant, so the same shape placed anywhere scores identically. Scale is kept in
1/m on purpose — a 0.5 m hairpin is hard regardless of overall track length.

Both axes are **z-scored** across the analyzed set (robust to the one-hairpin-
waypoint outliers that would wreck min-max). Outputs:

  1. a sorted difficulty table (CSV) with raw sub-numbers + z-scores + a single
     difficulty score (mean of z-scores; magnitude/max variants alongside),
  2. the 2-D **wobble x tightness** scatter map, with the oracle study's train /
     held-out / physical tracks colour-coded — so you can SEE where the Oval fell
     into a hole and pick a distribution-spanning training set.

    uv run --no-sync python scripts/track_geometry.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# --- the oracle study's track sets (experiments/oracle_feature_study.py) --------
TRAIN = {"Spain_track", "Monaco", "Austin", "arctic_pro", "caecer_gp"}
HELDOUT = {"Bowtie_track", "jyllandsringen_pro", "penbay_pro"}
PHYSICAL = {"reInvent2019_track", "Oval_track"}

_ROUTES = Path("/home/lunav0/Projects/deepracer-env/simulation/routes")
_VARIANT = re.compile(r"_(cw|ccw|mirrored)$")  # direction/mirror = same shape -> dedupe

# Tunables (scale-aware; DeepRacer tracks are metres, corners ~0.5-5 m radius).
_DS = 0.15            # resample arc-length step (m)
_STRAIGHT_KAPPA = 0.10   # |kappa| below this (radius > 10 m) counts as "straight"
_TURN_KAPPA = 0.15       # |kappa| above this counts as a real turn (sign-change deadband)


def load_centerline(npy_path: Path) -> np.ndarray:
    """Centreline (x, y) from a route file (cols 0,1 = centre; 2,3 inner; 4,5 outer)."""
    return np.load(npy_path, allow_pickle=True)[:, 0:2].astype(float)


def resample_closed(xy: np.ndarray, ds: float = _DS) -> tuple[np.ndarray, float]:
    """Resample a closed centreline to uniform arc-length ``ds``; return (pts, length).

    Waypoints are unevenly spaced, so a fixed *count* window would mix long and
    short segments — uniform arc-length fixes that. The loop is closed (last->first).
    """
    pts = np.vstack([xy, xy[:1]])
    seg = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    length = float(s[-1])
    n = max(32, int(round(length / ds)))
    su = np.linspace(0.0, length, n, endpoint=False)
    return np.column_stack([np.interp(su, s, pts[:, 0]),
                            np.interp(su, s, pts[:, 1])]), length


def signed_curvature(loop: np.ndarray, ds: float) -> np.ndarray:
    """Signed curvature (1/m) at each uniform-arc-length point: the hand rule.

    theta_i = signed turn between successive tangents (cross/dot -> atan2, scale-
    free); kappa = theta / ds. +left (CCW), -right (CW)."""
    t1 = loop - np.roll(loop, 1, axis=0)
    t2 = np.roll(loop, -1, axis=0) - loop
    cross = t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0]
    dot = (t1 * t2).sum(axis=1)
    return np.arctan2(cross, dot) / ds


def polygon_area(xy: np.ndarray) -> float:
    """Shoelace area of the centreline polygon (abs)."""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convexity(xy: np.ndarray) -> float:
    """Polygon area / convex-hull area: 1.0 = convex (oval), lower = more dents."""
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xy)
        return float(polygon_area(xy) / hull.volume)  # 2-D hull.volume == area
    except Exception:  # noqa: BLE001 — degenerate track => report 1.0
        return 1.0


def sign_changes_per_m(kappa: np.ndarray, length: float) -> tuple[float, int]:
    """Curvature left<->right switches per metre (the wobble primitive) + reflex count.

    Only |kappa| > _TURN_KAPPA counts (deadband kills straight-line noise). Reflex =
    turns opposite the track's majority direction (the non-ear vertices)."""
    turning = kappa[np.abs(kappa) > _TURN_KAPPA]
    if turning.size < 2:
        return 0.0, 0
    sign = np.sign(turning)
    changes = int(np.count_nonzero(np.diff(sign) != 0))
    majority = np.sign(turning.sum()) or 1.0
    reflex = int(np.count_nonzero(sign == -majority))
    return changes / length, reflex


def analyze(npy_path: Path) -> dict:
    xy = load_centerline(npy_path)
    loop, length = resample_closed(xy)
    kappa = signed_curvature(loop, _DS)
    absk = np.abs(kappa)
    wobble, reflex = sign_changes_per_m(kappa, length)
    p95 = float(np.percentile(absk, 95))
    kmax = float(absk.max())
    return {
        "track": npy_path.stem,
        "n_waypoints": int(xy.shape[0]),
        "length_m": round(length, 1),
        "tightness_raw": round(p95, 4),            # p95 |kappa| -> the tightness axis
        "min_radius_m": round(1.0 / kmax, 2) if kmax > 1e-6 else float("inf"),
        "mean_curv": round(float(absk.mean()), 4),
        "wobble_raw": round(wobble, 4),            # sign-changes / m -> the wobble axis
        "reflex_verts": reflex,
        "convexity": round(convexity(xy), 3),
        "pct_straight": round(float((absk < _STRAIGHT_KAPPA).mean()), 3),
    }


def _set_of(track: str) -> str:
    base = _VARIANT.sub("", track)
    if base in PHYSICAL:
        return "physical"
    if base in TRAIN:
        return "train"
    if base in HELDOUT:
        return "held-out"
    return "other"


def _zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v)


def _percentile_rank(v: np.ndarray) -> np.ndarray:
    order = v.argsort().argsort().astype(float)
    return order / (len(v) - 1) if len(v) > 1 else np.zeros_like(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", type=Path, default=_ROUTES)
    ap.add_argument("--out", type=Path, default=Path("artifacts/track_geometry"))
    ap.add_argument("--all-variants", action="store_true",
                    help="keep _cw/_ccw/_mirrored (default: dedupe to one shape each)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    files = sorted(args.routes.glob("*.npy"))
    if not args.all_variants:  # one representative per geometry (variants are same shape)
        seen, keep = set(), []
        for f in files:
            base = _VARIANT.sub("", f.stem)
            if base not in seen:
                seen.add(base)
                keep.append(f)
        files = keep

    rows = []
    for f in files:
        try:
            rows.append(analyze(f))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {f.stem}: {exc}")
    df = pd.DataFrame(rows)
    df["set"] = df["track"].map(_set_of)

    # z-score the two axes (user's choice; robust to one-hairpin-waypoint outliers).
    df["z_wobble"] = _zscore(df["wobble_raw"].to_numpy())
    df["z_tightness"] = _zscore(df["tightness_raw"].to_numpy())
    # Single difficulty score = mean of z-scores (primary). Magnitude/max use a
    # [0,1] percentile scaling so "0 = easy" (z-magnitude would call below-average
    # tracks "extreme"); kept alongside for comparison.
    pw = _percentile_rank(df["wobble_raw"].to_numpy())
    pt = _percentile_rank(df["tightness_raw"].to_numpy())
    df["difficulty_avg"] = (df["z_wobble"] + df["z_tightness"]) / 2.0
    df["difficulty_mag"] = np.hypot(pw, pt) / np.sqrt(2)
    df["difficulty_max"] = np.maximum(pw, pt)

    df = df.sort_values("difficulty_avg", ascending=False).reset_index(drop=True)
    csv = args.out / "track_geometry.csv"
    df.to_csv(csv, index=False)

    cols = ["track", "set", "wobble_raw", "tightness_raw", "min_radius_m",
            "convexity", "pct_straight", "difficulty_avg"]
    print(f"\nanalyzed {len(df)} tracks -> {csv}\n")
    print(df[cols].head(12).to_string(index=False))
    print("  ...")
    print(df[cols].tail(6).to_string(index=False))
    print("\n=== the oracle study's tracks ===")
    print(df[df["set"] != "other"][cols].to_string(index=False))

    _plot(df, args.out / "track_map.png")
    print(f"\nmap -> {args.out / 'track_map.png'}")
    return 0


def _plot(df, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"other": "0.75", "train": "tab:blue", "held-out": "tab:green",
               "physical": "tab:red"}
    zorder = {"other": 1, "train": 2, "held-out": 2, "physical": 3}
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    for s in ["other", "train", "held-out", "physical"]:
        sub = df[df["set"] == s]
        ax.scatter(sub["z_wobble"], sub["z_tightness"], s=46 if s != "other" else 22,
                   c=colours[s], label=f"{s} ({len(sub)})", zorder=zorder[s],
                   edgecolors="k" if s != "other" else "none", linewidths=0.5)
    # label the named sets + the wobbliest/tightest extremes
    to_label = df[df["set"] != "other"]
    extremes = pd.concat([df.nlargest(2, "z_wobble"), df.nlargest(2, "z_tightness")])
    for _, r in pd.concat([to_label, extremes]).drop_duplicates("track").iterrows():
        ax.annotate(r["track"], (r["z_wobble"], r["z_tightness"]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.axvline(0, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel("wobble  (curvature sign-changes / m)  — flowing  →  zig-zag")
    ax.set_ylabel("tightness  (p95 |curvature|, 1/m)  — open  →  hairpins")
    ax.set_title("Track difficulty/diversity map — wobble × tightness (z-scored)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import pandas as pd  # noqa: E402  (used in _plot label concat)
    raise SystemExit(main())
