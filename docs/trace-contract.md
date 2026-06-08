# Trace Contract & Pipeline Schematics

The single data contract that joins the **RL world** (dr-gym: SB3 + MLflow +
TensorBoard + Optuna) to the **robotics world** (deepracer-env: ROS1 + Gazebo),
and lets us run analysis that **supersedes** `deepracer-utils` instead of
obeying its folder/S3 layout.

> **Why this exists.** deepracer-env *computes* the per-step DeepRacer record
> but persists none of it: the metrics writer is a no-op
> (`deepracer_env/environments/deepracer_env.py:126`, wired at `:176`), and the
> gym `step()` `info` only surfaces `agents_info_map` + four OA flags
> (`:327-334`). The full record reaches exactly one place — the **reward
> callback** — which dr-gym already taps (`gym_dr/metrics.py:111`). Camera and
> LiDAR live in `obs`, which the reward callback never sees. This contract
> defines what each side emits and how the streams reconcile.

---

## 1. Principles

1. **One join key: `sim_time`.** Sim-clock seconds (`/clock`), never wall clock.
   Gazebo real-time-factor drift makes wall clock unsafe for alignment. Wall
   clock is carried as a secondary column only.
2. **Three tiers, separated by bandwidth.** Scalars (cheap, always on),
   high-bandwidth sensors (video/LiDAR, referenced not inlined), and aggregates
   (one row per episode). They reconcile on `sim_time` via `merge_asof`.
3. **Track identity is a per-step fact, not a per-run constant.** With runtime
   `DeepRacerEnv.set_world` hot-swaps (`environments/world_swap.py`), the world —
   and therefore `track_len` and the waypoint geometry — changes *inside* a
   single run. Every row carries its world. (This breaks the old
   deepracer-utils assumption that `track_len` is constant within a run.)
4. **Column names match the `deepracer-utils` *internal* DataFrame** (see
   `deepracer-utils/docs/output-format.md`) so its analysis runs on our trace
   unmodified — but we never import its loaders (`handler.py`, `LogFolderType`,
   `TrainingMetrics`, S3 keys). The contract is *ours*; compatibility is a
   convenience, not a dependency.
5. **Cross-repo coupling is this schema and nothing else.** dr-gym never imports
   deepracer-env analysis; the bag converter never imports dr-gym training.

---

## 2. Tier 1 — Scalar step trace (the simtrace-equivalent)

One row per env step. Replaces the AWS simtrace CSV. Tapped from the reward
callback in-process (dr-gym) and/or from a real `MetricsInterface` in
deepracer-env.

### Identity & sync keys
| Field | Type | Notes |
|---|---|---|
| `run_id` | str | MLflow run id of the training chunk (Optuna trial = tag, §6). |
| `episode` | int | Episode index within the chunk. |
| `steps` | int | 1-based step within the episode. |
| `sim_time` | float | **Primary join key.** Sim-clock seconds from `/clock`. |
| `wall_time` | float | Wall-clock unix seconds. Secondary; never join on this. |

### Track / world (hot-swap aware)
| Field | Type | Notes |
|---|---|---|
| `world_name` | str | Canonical world id (`reinvent_base`, `Bowtie_track`, … per `gym_dr/tracks.py`). |
| `world_label` | str? | Human label via `gym_dr.tracks.display_name`. |
| `chunk_index` | int | Monotonic 0-based counter per `set_world` segment in the run. **Distinguishes repeated rotations of the same world** (`WorldsConfig.rotations`). |
| `rotation` | int? | Which rotation pass produced this segment (optional convenience). |

> An **episode is normally single-world**; a swap happens at a chunk boundary.
> If `world_name` ever changes *within* one `(run_id, episode)` that's an
> anomaly worth flagging — the swap event stream (§3.4) is authoritative.

### Pose / kinematics
| Field | Type | CSV name | Notes |
|---|---|---|---|
| `x` | float | `X` | metres, track frame |
| `y` | float | `Y` | metres |
| `yaw` | float | `yaw` | degrees |
| `steering_angle` | float | `steer` | degrees, +left |
| `speed` | float | `throttle` | m/s commanded |
| `action` | int | `action` | `-1` for continuous spaces |

### Progress / track geometry
| Field | Type | CSV name | Notes |
|---|---|---|---|
| `progress` | float | `progress` | 0–100 |
| `closest_waypoint` | int | `closest_waypoint` | indexes the **current world's** waypoints |
| `track_len` | float | `track_len` | metres — **constant per `chunk_index`, NOT per run** |
| `on_track` | bool | `all_wheels_on_track` | |

### Outcome
| Field | Type | Notes |
|---|---|---|
| `reward` | float | Training reward returned this step. |
| `eval_reward` | float? | dr-gym parallel eval-reward (`metrics.py:117`); null in pure ROS producer. |
| `done` | bool | |
| `episode_status` | str | enum from `deepracer_env/metrics/constants.py:85`. |
| `pause_duration` | float | cumulative pause seconds; null on old producers. |

### Object-avoidance extension (present only when OA is enabled)
| Field | Type | Notes |
|---|---|---|
| `oa_enabled` | bool | False ⇒ all OA fields null. |
| `is_crashed` | bool | collided with an object this step (`deepracer_env.py:332`). |
| `is_offtrack` | bool | `deepracer_env.py:333`. |
| `closest_objects` | list[int] | `[prev_idx, next_idx]` nearest object waypoints (`:334`). |
| `n_objects` | int | active object count. |
| `closest_object_distance` | float | metres to nearest object. |
| `objects_location_ref` | str? | key into Tier-2 object table (§3.3) for dynamic OA. |

> Static-box OA → object set is constant *within a world*; store it in the
> per-segment metadata. Dynamic/bot OA → positions stream to Tier 2.3.

---

## 3. Tier 2 — High-bandwidth & event streams (never inlined)

Stored separately, joined to Tier 1 on `sim_time` on demand. Source of truth =
the ROS bag in deepracer-env. This is where video and LiDAR — invisible to the
scalar sink — live.

### 3.1 Camera (FRONT_FACING / STEREO / LEFT)
mp4 per episode + frame-index table `frame_idx, sim_time, episode, steps`.

### 3.2 LiDAR (LIDAR / SECTOR_LIDAR / DISCRETIZED_SECTOR_LIDAR)
`sim_time, episode, steps, ranges[64]` (+ sector encoding). Specs:
`deepracer_env/sensors/constants.py:55` (64 samples, ±2.618 rad, 0.15–12 m).

### 3.3 Object positions over time (dynamic OA)
`sim_time, object_id, x, y, yaw, type`. Tier 1's `objects_location_ref` slices
this for the step. Enables near-miss / clearance / overtake analysis.

### 3.4 World-swap events (authoritative track timeline)
`sim_time, chunk_index, from_world, to_world, rotation, track_len`. One row per
`set_world` call. This is the ground truth for *which geometry applies at a
given `sim_time`* — bag-derived sensor rows resolve their world through this
table, and trajectory plots load `routes/<to_world>.npy` accordingly.

**Rule:** Tier 2 is *referenced*, never merged into Tier 1 at write time.

---

## 4. Tier 3 — Episode / run aggregates

One row per episode, keyed by `(run_id, world_name, chunk_index, episode)` so
rotations don't collapse together. What dr-gym already emits (`dr/ep_*`,
`metrics.py:80`) and ships to TB/MLflow; the Optuna objective surface. Extend
with OA aggregates (`ep_crash_count`, `ep_object_passes`, `ep_min_clearance`,
`ep_near_miss_count`) and per-world rollups for cross-track generalisation.

---

## 5. Storage layout (extends docs/artifact-layout.md)

```
artifacts/<run_name>/
├── run_config.json            # already written (trainer.py:149)
├── model_metadata.json        # already written
├── reward_function.py         # already written
├── trace/
│   ├── step.parquet           # Tier 1  (+ world_name, chunk_index, …)
│   ├── episode.parquet        # Tier 3
│   ├── swaps.parquet          # Tier 2.4 world-swap timeline
│   ├── objects.parquet        # Tier 2.3 (dynamic OA only)
│   └── sensors/
│       ├── camera/<ep>.mp4 + frames.parquet   # Tier 2.1
│       └── lidar.parquet                       # Tier 2.2
└── bags/<ep>.bag              # raw ROS bag (optional, recorded by deepracer-env)
```

The trainer already uploads the whole run dir to MLflow (`mlflow_utils.py:70` ←
`trainer.py:207`), so the trace travels with the run. MLflow is the **index**.

---

## 6. Producer matrix — who emits what, and where it's tapped

| Stream | Producer | Tap point |
|---|---|---|
| Tier 1 scalars + track cols | **dr-gym in-process sink** | reward callback, full `params` (`metrics.py:111`); `world_name`/`chunk_index` from the trainer's world plan (`trainers/base.py:78`) |
| Tier 1 (alt) | deepracer-env | real `MetricsInterface` replacing `_NoopMetrics` (`deepracer_env.py:126`) |
| Tier 2 camera/LiDAR | **deepracer-env rosbag** | `rosbag record` of sensor topics + `/clock` |
| Tier 2 (offline decode) | **bag→trace converter** | `rosbags` (pure-Python) in the dr-gym venv — no live ROS |
| Tier 2.4 swap events | dr-gym trainer | emit a row each `set_world` call (it already drives the rotation) |
| Tier 3 aggregates | dr-gym | `RewardMetricsCallback` (`callbacks.py:185`) |
| Optuna trial ↔ trace | dr-gym | trial-number MLflow tag beside `run_group` (`mlflow_utils.py:37`) |

**Clock discipline:** every producer stamps `sim_time` from `/clock`; the bag
records `/clock`. If sim time is unavailable the producer falls back to wall and
sets `sim_time` null so the join layer knows not to trust it.

---

## 7. Analysis layer — an *iteration over* deepracer-utils

We do **not** depend on `deepracer-utils`; we lift its DataFrame-only analysis
into `gym_dr/analysis/` and extend it. Vendor (strip `boto3`/handler imports):

- `logs/stability.py` — `SimtraceStabilityAnalyzer`, `episode_stats`.
- `logs/log_utils.py` — `PlottingUtils`, `AnalysisUtils`, `ActionBreakdownUtils`,
  `NewRewardUtils`.
- `model/visualization.py` — track heatmaps (now keyed per `world_name`).
- `tracks/track_utils.py` — waypoint geometry; **resolve the per-world `.npy`
  via the swap timeline (§3.4)**, since a run now spans multiple tracks.

Drop entirely: `handler.py`, `misc.py` (`LogFolderType`), `metrics.py`
(`TrainingMetrics.json`), all S3/robomaker-log parsing.

**New analysis beyond utils (the reason to iterate, not reuse):**
- **OA tracking** — object-clearance time series, near-miss/overtake detection,
  crash attribution by object id. Built on Tier 1 OA fields + Tier 2.3.
- **Cross-track generalisation** — same policy across hot-swapped worlds;
  stability/completion per `world_name`. Impossible in the old world (one model
  folder = one track).
- **Multimodal alignment** — overlay LiDAR/camera on the trajectory at a chosen
  `sim_time`, against the correct per-world geometry.
- **Cross-trial** — stability/OA metrics indexed by Optuna trial, fed back into
  the study.

All consume the canonical Tier-1 DataFrame — identical column names to utils —
so vendored functions work without a rewrite.

---

## 8. Schematics

```
        ROBOTICS WORLD (ROS1 / Gazebo)              RL WORLD (dr-gym, py3.8)
        deepracer-env venv                          SB3 · MLflow · TB · Optuna
 ┌──────────────────────────────────────┐   ┌──────────────────────────────────┐
 │ DeepRacerEnv.step()                   │   │ trainer.run_training              │
 │   obs (camera, LiDAR) ───────┐        │   │   install_metrics (metrics.py)    │
 │   reward_params (full dict) ──┼──────────────► reward callback _wrap_reward   │
 │   info (4 OA flags only)      │        │   │     │  record_step(params)        │
 │                               │        │   │     ▼   + world_name/chunk_index  │
 │ set_world() ──► swap event ───┼──────────────► swaps.parquet (Tier 2.4)        │
 │  /clock /scan /camera /tf     │        │   │  Tier1 step.parquet  Tier3 ep.*   │
 │      │  rosbag record         │        │   │     │         │                   │
 └──────┼────────────────────────┘        │   └─────┼─────────┼───────────────────┘
        │ bags/<ep>.bag                    │         │         ▼
        ▼                                  │         │   TB scalars + MLflow metrics
 ┌──────────────────────┐  rosbags (pure-py)│        │         │
 │ bag → trace converter│──────────────────┐│        │         ▼
 │  (runs in dr-gym venv)│ Tier2 camera/lidar/objects │   MLflow run = INDEX
 └──────────────────────┘                  ││        │         │
                                           ▼▼        ▼         ▼
                         ┌───────────────────────────────────────────────┐
                         │  join on sim_time (merge_asof);                │
                         │  resolve world via swaps.parquet               │
                         │  Tier1 ⨝ Tier2 ⨝ Tier3                         │
                         └───────────────────────────────┬───────────────┘
                                                          ▼
                         ┌───────────────────────────────────────────────┐
                         │  gym_dr/analysis  (iteration over dr-utils)    │
                         │  vendored: stability · plotting · track viz    │
                         │  new: OA clearance/near-miss · cross-track ·   │
                         │       multimodal · cross-trial (Optuna)        │
                         └───────────────────────────────────────────────┘
```

Two clocks, one key (`sim_time`). Track identity per row + a swap timeline. Two
scalar producers, one schema. Three dashboards, one index (MLflow). One analysis
package, superseding dr-utils.
