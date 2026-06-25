# Multi-car throughput — comprehensive grid (this machine: 8 cores, 1 GPU 16GB)

`scripts/multicar_throughput.py`. Data: `artifacts/multicar_throughput_camerafree.json`
(feature scaling) + `artifacts/multicar_grid.json` (camera 2x2 + rtf sweep).
Metric: SB3 aggregate agent-steps/s (`fps`) + effective RTF from `/clock`.
deepracer-env overlay: NUMBER_OF_RESETS=0, wheel mu=1.5, zero-quaternion guard;
dr-gym: feature obs uses an EMPTY sensor list (no camera).

## Two env types (both supported)
- **Feature (camera-free)**: obs = ALL_FEATURES from reward_params, no camera sensor,
  nothing rendered. Fast path.
- **Camera/lidar**: keeps the full sensor stack (vision-dependent tasks).

## Feature env — scales with cars-in-one-sim
| n_cars | steps/s (total) | per-car |
|---|---|---|
| 1 | 47 | 47 |
| 2 | 76 | 38 |
| 4 | 95 | 24 |
| 6 | 106 | 18 |
| 8 | 114 | 14 |

Aggregate throughput **rises 2.4x** from n1→n8; per-car falls (shared physics step).
Diminishing returns past n=4–6 (n6→n8 is only +8). **Sweet spot n=4–6** (good aggregate,
still enough samples/car for PPO). CPU inference (small MLP); GPU not needed.

## Camera env — 2x2 (render x inference), caps at n=2
| n | render | inference | steps/s |
|---|---|---|---|
| 2 | GPU | GPU | **51** |
| 2 | SW (cpu) | GPU | 39 |
| 2 | GPU | CPU | 17 |
| 2 | SW (cpu) | CPU | 17 |
| 3–8 | any | any | **no rollout completes** (render saturation) |

- **GPU inference is the big lever** (51 vs 17 = 3x) — the CNN dominates.
- **GPU render** adds ~30% over software (51 vs 39) at n=2.
- **Camera mode hard-caps at n=2 — a Gazebo Classic rendering limit (root-caused).**
  At n=3 all three camera topics advertise, but only TWO publish frames:
  racecar_0=15.7Hz, racecar_1=15.4Hz, **racecar_2=0Hz** (never publishes). Gazebo
  Classic renders only 2 camera sensors per world; the 3rd starves, the blocking
  sensor read times out and `log_and_exit()`s the whole run ~120s in. NOT fixable by
  rtf (n=3 @ rtf=3 still failed) or timeout. `multi_car` now raises a clear error at
  camera n>2 (override GYM_DR_ALLOW_CAMERA_NCARS=1). For >2 camera cars: separate
  processes, or feature obs (scales to n=8). Deeper fix = Gz Sim/ROS2 (big port).

## rtf_override — non-binding
feature n=4 across rtf {5,10,40,80} → 65,66,63,65 steps/s (flat). The machine is
physics/render bound: effective RTF stays ~1.0–1.7 regardless of the override. So
**rtf_override needs no tuning** — set it high as a non-binding ceiling. (It only
matters as a crash trigger if set absurdly high relative to what the sim sustains.)

## fps ≠ sim speed ≠ wall-clock throughput (don't confuse them)
Three different rates, repeatedly conflated:
- **SB3 `fps`** = agent *training* steps/s (control steps that land in the rollout buffer). Steady ~64 for
  single-car feature obs here.
- **effective RTF** = sim-seconds / wall-second (from `/clock`). Measured ~1.5–9× — the **sim is not slow**.
- **wall-clock to N steps** ≠ `N/fps`: it also includes evals, world swaps, gradient updates, container idle.

**Eval is the real wall-clock sink, and it scales with policy quality.** On `oracle_feature_study`: 13 wall-h
produced only ~198k *counted* training steps, but the sim clock had advanced ~118 sim-hours and fps was a
steady 64. The 13h breakdown: ~1h of actual training (4×50k-step chunks), the rest **held-out evaluation** —
one eval *world* took ~60–68 min (e.g. jyllandsringen→penbay swap gap), because as the policy stopped
crashing the eval episodes ran very long (eval `mean_reward` ~2.9e5 over long laps) with no tight episode cap.
**Fix:** cap eval episode length (step/time limit) and/or drop `n_eval_episodes` 3→1–2 / eval less often —
several-fold wall-clock savings, training is healthy.

## Multi-car shares ONE rollout buffer
`MultiCarVecEnv` sets `num_envs = n_cars` and `step_wait` returns `(n_cars, …)`-stacked arrays, so SB3 PPO
fills a single `RolloutBuffer` of shape `(n_steps, n_cars, …)` and updates on the flattened
`n_steps × n_cars` batch. The N cars (each its own track + DR) contribute **decorrelated** samples to one
gradient update — the VecEnv payoff that isolated containers can't give.

## Optimal parameters (this machine)
- **Feature training: 4–6 cars / sim, CPU inference, rtf any.** ~95–106 steps/s, one
  process → central DR + curriculum + per-car tracks (the VecEnv win).
- **Camera training: 2 cars / sim (max), GPU render + GPU inference.** ~51 steps/s.

## Caveat
The app watchdog restarted the paused single-car training mid-grid, so some grid
points ran with background contention. Qualitative conclusions are robust (camera
n=2 GPU/GPU = 51 matches the earlier clean run; camera n>=3 unusable; feature scaling
from the contention-free run; rtf flat); absolute numbers on contended points ±15%.
