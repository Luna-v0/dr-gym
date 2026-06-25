"""Tests for the pure join/stack core of the rosbag→perception path
(`scripts/bag_to_perception.py`). The bag reader + trace loader need ROS/a real
bag; the time-join and frame-stacking logic do not, and they're the tricky part."""
from __future__ import annotations

import numpy as np

from scripts.bag_to_perception import (
    build_dataset,
    nearest_label_indices,
    stack_frames,
)


def test_nearest_label_indices_exact():
    # frames land exactly on labels
    fr = [0.0, 1.0, 2.0]
    lab = [0.0, 1.0, 2.0]
    assert nearest_label_indices(fr, lab, max_dt=0.1) == [0, 1, 2]


def test_nearest_label_indices_offset_within_tol():
    fr = [0.04, 0.96]
    lab = [0.0, 1.0]
    assert nearest_label_indices(fr, lab, max_dt=0.05) == [0, 1]


def test_nearest_label_indices_drops_far_frames():
    fr = [0.0, 0.5]     # 0.5 is 0.5s from the nearest label (0.0) -> dropped
    lab = [0.0, 2.0]
    assert nearest_label_indices(fr, lab, max_dt=0.05) == [0, None]


def test_nearest_label_indices_empty_labels():
    assert nearest_label_indices([0.0, 1.0], [], max_dt=1.0) == [None, None]


def test_stack_frames_pads_at_episode_start():
    frames = [np.full((2, 2), i, np.uint8) for i in range(5)]
    # idx 0 -> all four frames are frame 0 (padding)
    s0 = stack_frames(frames, 0, k=4)
    assert s0.shape == (4, 2, 2)
    assert np.all(s0[0] == 0) and np.all(s0[-1] == 0)
    # idx 3 -> frames 0,1,2,3 in order
    s3 = stack_frames(frames, 3, k=4)
    assert [int(s3[i, 0, 0]) for i in range(4)] == [0, 1, 2, 3]


def test_build_dataset_joins_and_stacks():
    H, W, F = 3, 3, 9
    frames = [np.full((H, W), i, np.uint8) for i in range(4)]
    frame_times = [0.0, 0.1, 0.2, 0.3]
    label_times = [0.0, 0.1, 0.2, 0.3]
    labels = np.arange(4 * F, dtype=np.float32).reshape(4, F)
    obs, tgt = build_dataset(frame_times, frames, label_times, labels, frame_stack=4, max_dt=0.05)
    assert obs.shape == (4, 4, H, W)
    assert tgt.shape == (4, F)
    # last sample's newest frame is frame 3, label row 3
    assert int(obs[-1, -1, 0, 0]) == 3
    assert np.allclose(tgt[-1], labels[3])


def test_build_dataset_drops_unmatched():
    frames = [np.zeros((2, 2), np.uint8) for _ in range(3)]
    frame_times = [0.0, 5.0, 0.2]        # middle frame has no nearby label
    label_times = [0.0, 0.2]
    labels = np.zeros((2, 4), np.float32)
    obs, tgt = build_dataset(frame_times, frames, label_times, labels, frame_stack=2, max_dt=0.05)
    assert obs.shape[0] == 2 and tgt.shape[0] == 2  # the 5.0s frame dropped
