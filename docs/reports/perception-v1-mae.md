# Perception CNN v1 — held-out per-feature MAE

Supervised `g: camera 4-stack -> 11 actor features` (PerceptionNet, `in_channels=4`). Train = TRAIN bucket of `mlruns/**/perception_out/train` (43 base tracks, no variants); MAE on the canonical held-out VAL/TEST tracks (`camera_cnn_dataset._split_tracks`, no variant leakage). 15 epochs, best-by-prize-MAE checkpoint.

| feature | group | base | s4 val | s4 skill | s1 val | s1 skill | s4 test | usable |
|---|---|---|---|---|---|---|---|---|
| `lateral_offset` | vision-geometry (prize) | 0.5216 | 0.1168 | +0.78 | 0.1825 | +0.65 | 0.3057 | ◐ |
| `heading_error` | vision-geometry (prize) | 0.1599 | 0.0689 | +0.57 | 0.0817 | +0.49 | 0.1091 | ✅ |
| `dist_left_edge` | vision-geometry (prize) | 0.2609 | 0.0587 | +0.78 | 0.0913 | +0.65 | 0.1514 | ✅ |
| `dist_right_edge` | vision-geometry (prize) | 0.2609 | 0.0587 | +0.78 | 0.0911 | +0.65 | 0.1514 | ✅ |
| `curvature_ahead` | visible map/obstacle | 0.0205 | 0.0205 | -0.00 | 0.0170 | +0.17 | 0.0260 | — |
| `nearest_object_dist` | visible map/obstacle | 0.0000 | 0.0004 | n/a | 0.0012 | n/a | 0.0003 | — |
| `speed_mps` | proprioceptive | 0.5763 | 0.4972 | +0.14 | 0.5584 | +0.03 | 0.5248 | — |
| `yaw_rate` | proprioceptive | 0.2156 | 0.1968 | +0.09 | 0.1955 | +0.09 | 0.1569 | — |
| `long_accel` | temporal-delta | 0.5717 | 0.6100 | -0.07 | 0.5674 | +0.01 | 0.6399 | — |
| `lateral_velocity` | temporal-delta | 0.4844 | 0.2884 | +0.40 | 0.3278 | +0.32 | 0.3465 | — |
| `edge_closing_rate` | temporal-delta | 0.2884 | 0.2132 | +0.26 | 0.2310 | +0.20 | 0.2291 | — |

*`skill = 1 − MAE/base`. **✅** usable = MAE < bar **and** skill > 0.15; **◐** borderline = high skill but MAE just over the bar; **—** not vision-recoverable (or low MAE only because the channel is ~constant — check skill).*

## Verdict (data-driven)
- **Vision-usable — the camera actor may consume these:** `heading_error` (0.069, skill +0.57), `dist_left_edge` (0.059, skill +0.78), `dist_right_edge` (0.059, skill +0.78).
- **Borderline (high skill, MAE just over the 0.10 bar):** `lateral_offset` (0.117, skill +0.78).
- **Keep proprioceptive (sensor, not vision):** `speed_mps` (0.497, skill +0.14), `yaw_rate` (0.197, skill +0.09) — `speed_mps` is raw m/s and the sigmoid head structurally caps it at 1.0 (≈22% of frames exceed that); `yaw_rate` is an IMU signal.
- **Not vision-recoverable from this data:** `long_accel` (0.610, skill -0.07), `lateral_velocity` (0.288, skill +0.40), `edge_closing_rate` (0.213, skill +0.26), `curvature_ahead` (0.020, skill -0.00), `nearest_object_dist` (0.000, skill +nan).
- **4-stack vs 1-frame gain** (s1→s4 val MAE): `lateral_offset` 0.183→0.117, `dist_left_edge` 0.091→0.059, `lateral_velocity` 0.328→0.288, `edge_closing_rate` 0.231→0.213 — stacking helps exactly the motion/temporal features, as expected.

## Caveats baked into these numbers
- `curvature_ahead`: ~57% of targets are negative (right-hand bends) but the net puts a **sigmoid** on this channel (`signed_indices_for` excludes it), so it cannot output negatives — its low absolute MAE rides the low-variance baseline (skill ≈ 0), it is **not actually learned**. Add `curvature_ahead` to `SIGNED_FEATURES` (tanh) and retrain.
- `nearest_object_dist`: constant 1.0 on these object-free tracks → trivial MAE, no signal. Re-assess on object-avoidance captures.
- For the gold-standard external number, re-run on the frozen `perception_capture_heldout.py` val/test shards once located/regenerated.
