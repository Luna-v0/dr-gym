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
- **Camera mode caps at n=2 — but this is a LAUNCH/CONFIG limit, NOT a rendering
  limit. CORRECTION (2026-06-28):** the original "render limit" claim below was a
  misdiagnosis (see [status-2026-06-28](status-2026-06-28.md)). The launch only spawns
  **2 car bodies** (`racetrack_with_racecar.launch` has hardcoded `racecar_0/1`
  `<include>`s + `car_node.py args="2"`); nothing spawns a 3rd body. So at n=3
  racecar_2's camera topic advertises (rospy subscriber registers the name) but has
  **no publisher** → 0 Hz → the blocking `DoubleBuffer.get(timeout=120.0)` starves and
  `log_and_exit()`s ~120s in. The measurement below was "3 agents, 2 cars" — it never
  placed a 3rd camera in the world, so it tested the launch limit, not the renderer.
  rtf=3 "still failing" is consistent (no publisher regardless of RTF).
  - To raise it: generate `racecar_2..N` `<include>` blocks + parameterize `car_node`.
    The *real* residual limit then appears — Gazebo Classic serialises camera rendering
    on one OGRE thread (n=2 already saturates render at ~51 steps/s / RTF~1.0), so 3+
    real cameras **degrade fps/RTF gracefully**, not the current hard failure.
  - `multi_car` still raises a clear error at camera n>2 (override
    GYM_DR_ALLOW_CAMERA_NCARS=1) because the launch isn't generalized yet. Scale-out
    without touching the launch: separate Gazebo processes (1–2 camera cars each), or
    feature obs (n=8 — phantom agents tick because missing-model *state* reads don't
    block). Original (incorrect) note retained below for history:
  - _orig:_ "Gazebo Classic renders only 2 camera sensors per world; the 3rd starves…
    NOT fixable by rtf (n=3 @ rtf=3 still failed) or timeout. Deeper fix = Gz Sim/ROS2."

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
