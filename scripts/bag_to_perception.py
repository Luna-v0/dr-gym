"""Build a perception dataset from a TRAINING ROSBAG + the run's Parquet trace
(W-perception, the rosbag data source — `docs/reports/perception.md`).

Idea (maintainer): record the camera topic during a normal training run; the
LABELS come free from the trace we already write. Join the two **on `sim_time`**
(the trace column reserved for exactly this "bag→trace path", `gym_dr/trace.py`)
→ `(4-frame stack, ALL_FEATURES label)` → `npz` for `experiments/train_perception.py`.

    record (in the sim):  rosbag record /<racecar>/camera/zed/rgb/image_rect_color /clock
    build (host):         python scripts/bag_to_perception.py run.bag \
                              --trace artifacts/<run>/trace --out artifacts/perception/from_bag.npz

PRECONDITION (not yet met — see the note at the bottom): the Parquet trace must
carry the derived `ALL_FEATURES` columns. Today's trace stores raw geometry but
NOT `is_left_of_center` or `waypoints`, so `lateral_offset`'s sign and
`heading_error` can't be recomputed from it — the trace needs the feature columns
added at write time (`gym_dr/metrics.py`). The pure join/stack core below is
ready and tested; the bag reader needs `pip install rosbags` and a real bag.
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Pure, testable core — no ROS / no I/O.
# --------------------------------------------------------------------------- #
def nearest_label_indices(
    frame_times: Sequence[float],
    label_times: Sequence[float],
    max_dt: float = 0.05,
) -> List[Optional[int]]:
    """For each frame timestamp, the index of the nearest label row by time, or
    ``None`` if the closest label is farther than ``max_dt`` seconds (a dropped /
    unmatched frame). Both series are in the SAME clock (``sim_time``). O(n+m) via
    a merge walk; assumes ``label_times`` is sorted ascending."""
    lt = np.asarray(label_times, dtype=np.float64)
    out: List[Optional[int]] = []
    if lt.size == 0:
        return [None] * len(frame_times)
    for ft in frame_times:
        j = int(np.searchsorted(lt, ft))
        # candidate nearest is j-1 or j
        best, best_dt = None, None
        for k in (j - 1, j):
            if 0 <= k < lt.size:
                dt = abs(float(lt[k]) - float(ft))
                if best_dt is None or dt < best_dt:
                    best, best_dt = k, dt
        out.append(best if (best_dt is not None and best_dt <= max_dt) else None)
    return out


def stack_frames(frames: Sequence[np.ndarray], idx: int, k: int = 4) -> np.ndarray:
    """Reconstruct a ``k``-frame stack ending at ``frames[idx]`` (channels-first
    ``(k, H, W)`` uint8). Pads by repeating the earliest available frame at the
    start of an episode (matching the collector / VecFrameStack reset behaviour)."""
    start = idx - k + 1
    sel = [frames[i if i >= 0 else 0] for i in range(start, idx + 1)]
    return np.stack(sel, axis=0).astype(np.uint8)


def build_dataset(
    frame_times: Sequence[float],
    frames: Sequence[np.ndarray],
    label_times: Sequence[float],
    labels: np.ndarray,
    *,
    frame_stack: int = 4,
    max_dt: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Join frames↔labels on time and emit ``(obs (N,k,H,W) uint8, targets (N,F))``.
    Frames with no label within ``max_dt`` are dropped."""
    matches = nearest_label_indices(frame_times, label_times, max_dt=max_dt)
    obs, tgt = [], []
    for fi, li in enumerate(matches):
        if li is None:
            continue
        obs.append(stack_frames(frames, fi, frame_stack))
        tgt.append(labels[li])
    if not obs:
        return (np.empty((0, frame_stack, 1, 1), np.uint8), np.empty((0, labels.shape[1]), np.float32))
    return np.stack(obs).astype(np.uint8), np.stack(tgt).astype(np.float32)


# --------------------------------------------------------------------------- #
# Bag reader + trace loader (needs `rosbags` + a real bag / trace).
# --------------------------------------------------------------------------- #
def _read_camera_bag(bag_path: str, topic: str) -> Tuple[List[float], List[np.ndarray]]:
    """Read grayscale frames + their /clock-derived sim_time from a ROS1 bag.
    Lazy `rosbags` import so the pure core above stays dependency-free."""
    from rosbags.highlevel import AnyReader  # pip install rosbags
    from pathlib import Path

    _LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # BT.601, matches GrayscaleObs
    times: List[float] = []
    frames: List[np.ndarray] = []
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise SystemExit(f"topic {topic} not in bag; have: {[c.topic for c in reader.connections]}")
        for conn, _t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            gray = (img[..., :3].astype(np.float32) @ _LUMA).astype(np.uint8)
            # header stamp is the sim clock under use_sim_time (the join key)
            times.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            frames.append(gray)
    return times, frames


def _load_trace_labels(trace_dir: str):
    """Load (sim_time, ALL_FEATURES) from the run's Parquet trace shards."""
    from gym_dr.perception import ALL_FEATURES
    from gym_dr.trace import load_steps  # concatenates per-episode shards

    df = load_steps(trace_dir)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(
            "trace is missing the derived feature columns "
            f"{missing} — extend gym_dr/metrics.py to write ALL_FEATURES into the "
            "trace before using the rosbag path (see this script's docstring).")
    df = df.dropna(subset=["sim_time"]).sort_values("sim_time")
    return df["sim_time"].to_numpy(), df[list(ALL_FEATURES)].to_numpy(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("--trace", required=True, help="run trace dir (has steps/*.parquet)")
    ap.add_argument("--topic", default="/racecar/camera/zed/rgb/image_rect_color")
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--max-dt", type=float, default=0.05)
    ap.add_argument("--out", default="artifacts/perception/from_bag.npz")
    args = ap.parse_args()

    from gym_dr.perception import ALL_FEATURES

    frame_times, frames = _read_camera_bag(args.bag, args.topic)
    label_times, labels = _load_trace_labels(args.trace)
    obs, tgt = build_dataset(frame_times, frames, label_times, labels,
                             frame_stack=args.frame_stack, max_dt=args.max_dt)
    if obs.shape[0] == 0:
        print("[bag] no frame matched a trace row within --max-dt; nothing written")
        return 1
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, obs=obs, targets=tgt, features=np.array(ALL_FEATURES))
    print(f"[bag] wrote {obs.shape[0]} samples -> {args.out} (obs {obs.shape}, targets {tgt.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
