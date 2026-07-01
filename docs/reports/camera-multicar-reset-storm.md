# Camera multi-car reset-storm at n=8 — investigation

_Investigation run 2026-07-01. Context: the camera→feature dataset collector (`experiments/camera_cnn_dataset.py`) stalls into a "reset storm" and its container is SIGKILLed (`rc=137`) at n=8 camera cars, while n=4 runs cleanly. This report establishes the actual cause from code and logs, records two bugs fixed this session, and frames the one open decision._

> **Note on `rc=137`.** `rc=137` = 128+9 = SIGKILL. In these runs it is **not** an OOM: the box has 61 GiB RAM (44 GiB free) and the n=8 container peaked <2 GiB, and earlier n=8 diagnostics found no memory leak. The SIGKILL comes from the **watchdog** (`docker_runner`, on a stalled/heartbeat-starved container) or from the **test harness** itself (`docker rm -f` / `timeout` in the bisect + manual scripts). Read every `rc=137` below as "the stalled container was force-killed," not "ran out of memory."

## TL;DR

- The n=8 camera multi-car storm is **oversubscription on an 8-core box** — controller-manager bringup contention, a pose/TF flood, and CPU starvation — after which the stalled container is **force-killed (`rc=137` SIGKILL)** by the watchdog/test-harness (not an OOM — see note above). It is **not** an XML car cap and **not** crowding (arenas are 300 m apart).
- **n=4 is the reliable maximum camera car count on this hardware.** The works/storms boundary is sharp: n=4 works, n≥5 storms. `bisect_result.txt` records `n=5: BRINGUP-FAILED (no training start in 320s)`; `bisect_n5.log`/`bisect_n6.log` both end `rc=137`.
- The real discriminator is **shards-per-reset**, not reset count. n=8 yields ~0.36–0.48 shards/reset (~20–24 shards/min); n=4 sustains ~1.0 shard/reset (350–540 shards/min). n=4 actually has *more* resets than n=8 — its resets simply produce usable data.
- Controller-manager spawner failures are a **bringup-only handful** (2 and 4 events on the two n=8 runs) and do **not** distinguish n=8 from n=4 (n=4 shows 10). They are a symptom of the dogpile, not the bulk of the storm.
- Two real bugs were fixed this session: an `os._exit(0)` hard-teardown to dodge an rclpy-finalize segfault (`f0eb830`), and forwarding of the `GYM_DR_CAM_*` env knobs into the worker container (`db03372`).
- The launch XML **no longer caps car count**: the old hardcoded 2-car limit was generalized to 8 spawn blocks driven by comma-separated `simapp_versions`.
- A hard 8-car ceiling remains regardless of the storm — there are only `racecar_0..racecar_7` include blocks (and a matching one-byte collide-bitmask allocation `0x01..0x80`).
- The pose-settle angle (`refresh_state(force=True)` re-serving a stale pre-teleport pose) is **real but not the whole story**; a wait-for-fresh fix was tried and reverted (see below).

## How the car count is actually set

The number of DeepRacer cars that spawn is driven by the launch file — specifically by how many comma-separated `simapp_versions` entries are passed. The old hardcoded 2-car cap has been generalized to 8 spawn blocks. A separate `camera_obs` Python guard still fails-fast at 2 (deliberate, override-able), and an implicit 8-body ceiling remains because only `racecar_0..racecar_7` blocks exist.

**Launch mechanism** — `simulation/src/deepracer_simulation_environment/launch/racetrack_with_racecar.launch`:

- Single-car body: `racecar` include gated `unless="$(arg multicar)"`, bitmask `0x01` (lines 20–39; bitmask line 22).
- First two multi-car bodies gated **only** on `if="$(arg multicar)"` (NOT on `simapp_versions` length):
  - `racecar_0` — include line 41, bitmask `0x01` (line 43).
  - `racecar_1` — include line 62, bitmask `0x02` (line 64).
- Generalized bodies `racecar_2..racecar_7`, each gated on the comma-count of `simapp_versions`:
  - `racecar_2` — `if="$(eval len(str(simapp_versions).split(',')) > 2)"`, bitmask `0x04` (lines 86, 88).
  - `racecar_3` — `... > 3`, bitmask `0x08` (lines 109, 111).
  - `racecar_4` — `... > 4`, bitmask `0x10` (lines 132, 134).
  - `racecar_5` — `... > 5`, bitmask `0x20` (lines 155, 157).
  - `racecar_6` — `... > 6`, bitmask `0x40` (lines 178, 180).
  - `racecar_7` — `... > 7`, bitmask `0x80` (lines 201, 203).
- The car-reset node count is likewise driven by `simapp_versions`: `car_node.py args="$(eval len(str(simapp_versions).split(',')))"` under `multicar` (line 228; single-car path `args="1"` line 227).
- In-file comments (lines 83–85, 106–108, …) state this explicitly: *"Generalises the old hardcoded 2-car cap (a LAUNCH limit, not a render limit). car_node.py / get_racecar_names already handle N; only these blocks did not."*

So with `multicar=true`, passing `simapp_versions` with N comma-separated entries spawns **min(N, 8)** bodies: the first two always spawn (gated only on `multicar`), and bodies 3–8 each spawn once the comma-count exceeds 2..7. `get_racecar_names(N)` (`deepracer_env/utils.py:152`) and `car_node.py:157` already generate names for arbitrary N, so the launch blocks were the sole thing pinning the old 2-cap.

> Nuance: passing `multicar=true` with fewer than 2 entries would `IndexError` (e.g. `simapp_versions.split(',')[1]` at line 66). The `racecars_with_cameras` default (line 10) is `racecar,racecar_0,...,racecar_7` — **9** comma-separated entries (the single-car name plus 8 multi-car bodies), not 8.

**Bitmask flow (corroborates the 8-body byte boundary).** Each block's `racecar_bitmask` is forwarded through `racecar.launch` (arg declared line 8 `default="0x01"`, passed to xacro line 33) into `configure_collide_bitmask`, which emits `<collide_bitmask>${bitmask}</collide_bitmask>` (`simulation/urdf/deepracer/macros.xacro:44`; arg declared `simulation/urdf/deepracer/racecar.xacro:9`, applied to the 4 wheels + body at lines 87–91). The launch assigns exactly `0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80` — 8 distinct single-bit masks = one full byte. Note `<collide_bitmask>` is a 16-bit field, so `0x80` is not a technical cap; the **true** hard ceiling is that only 8 include blocks exist (no `racecar_8`). The URDF/xacro lives under `simulation/urdf/<variant>/`, not under `simulation/src/deepracer_simulation_environment/urdf/`.

**Python fail-fast guard** — `gym_dr/envs/multi_car.py:408-416`:

```python
if bool(experiment.camera_obs) and int(experiment.n_cars) > 2 \
        and os.getenv("GYM_DR_ALLOW_CAMERA_NCARS") != "1":
    raise ValueError(
        f"camera_obs multi-car is capped at n_cars=2: ...
        f"GYM_DR_ALLOW_CAMERA_NCARS=1 to override.")
```

The env-var override is checked at line 409 (also line 416); the message is at line 411. The preceding comment (lines 398–407) documents this as a launch/config limit (the camera topic has no publisher for un-spawned bodies → blocking read aborts ~120 s), NOT a "Gazebo renders only 2 cameras" limit — feature obs (`camera_obs=False`) scales to n=8.

## Each car gets its own decoupled arena

N namespaced `racecar_{i}` agents share one Gazebo world, but each owns its own track mesh, its own offset `TrackData`, and its own blocking, per-entity reset. Spacing — not de-crowding — guarantees no cross-arena sight or collision.

**(a) Per-car names, per-car track instance, per-car offset TrackData, per-car Agent** — `deepracer_env/environments/multi_agent_env.py` (class `MultiAgentDeepRacerEnv`):

- Car names — line 60: `self.car_names = [f"racecar_{i}" for i in range(self.n_cars)]`.
- Per-car track spawn (n_cars > 1) — lines 69–76: `WorldSwapper().spawn_track_instance(self.worlds[i], f"racetrack_{i}", (ox, oy))`. **Car 0 is skipped** (it reuses the origin `.world` track). `WorldSwapper.spawn_track_instance(world_name, model_name, offset=(0.0,0.0))` is at `deepracer_env/environments/world_swap.py:273`.
- Per-car offset `TrackData` — line 81: `track_data = TrackData.create(self.worlds[i], offset=self.offsets[i])` (`track_geom/track_data.py:181`).
- `track_data` bound per Agent — lines 82–85 via `build_agent(..., track_data=track_data)`; `build_agent` (`environments/deepracer_env.py:205-215`) forwards it (`extra['track_data'] = track_data`), so each car computes progress/off-track/reset against its own separated track, not the shared singleton.
- Grid offsets from `grid_offsets(n_cars, spacing)` (lines 25–29): a `ceil(sqrt(n)) x ceil(sqrt(n))` row-major grid with car 0 at `(0,0)`.

**(b) Spacing (not crowding) prevents collision/sight** — `deepracer_env/sim_control/arena.py`:

- `DEFAULT_ARENA_SPACING_M = 300.0` (line 60). Docstrings (lines 16–18, 57–59): arenas are far enough apart that "cars can neither see nor collide with each other" — "large enough that the biggest shipped track plus its camera far-clip cannot reach a neighbour." Crowding is explicitly **not** the failure mode.
- `arena.py` is **pure geometry/bookkeeping** (lines 28–34) — imports no ROS and no simulator; a `SimControl` backend consumes `Arena` objects. `ArenaLayout.to_local`/`to_world` (lines 204–234) is the read-time offset transform.

> Two parallel 300.0 values: `DEFAULT_ARENA_SPACING_M` in `arena.py` (used by the newer pure-geometry `ArenaLayout`) is separate from the hardcoded `spacing: float = 300.0` constructor default in `MultiAgentDeepRacerEnv.__init__` (`multi_agent_env.py:53`). The legacy env does not import `DEFAULT_ARENA_SPACING_M`; both just happen to be 300.0. The two modules also disagree on car naming (`arena.py` `ArenaLayout` defaults `car_{index}` → `car_0`; the legacy env uses `racecar_{i}`); track-entity naming agrees (`racetrack_{index}`).

**(c) reset() is sequential; each reset does a blocking set + blocking get:**

- Sequential reset — `multi_agent_env.py:117`: `return [agent.reset_agent() for agent in self._agents]` (one car at a time, index order). `reset_one(i)` (lines 119–122) resets just one car for VecEnv auto-reset.
- Chain: `Agent.reset_agent()` (`agents/agent.py:74-86`) resets sensor buffers then calls `self._ctrl_.reset_agent()` (line 85). `RolloutCtrl.reset_agent()` (`agent_ctrl/rollout_agent_ctrl.py:326`), lines 359–360:
  ```python
  SetModelStateTracker.get_instance().set_model_state(start_model_state, blocking=True)
  GetModelStateTracker.get_instance().get_model_state(self._agent_name_, '', blocking=True)
  ```
  — a blocking set followed by a blocking get (comment lines 353–358: the get avoids an outdated async position).
- The blocking set is per-entity through the SimControl seam: `SetModelStateTracker.set_model_state(..., blocking=True)` calls `self._sim.set_entity_state(model_state.model_name, ...)` (`gazebo_tracker/trackers/set_model_state_tracker.py:44-64`); the module docstring (lines 8–11) calls this "per-entity — the basis for decoupled multi-arena resets." Each car's reset teleports only its own entity.

**Bottom line.** The decoupling is real and code-backed: N `racecar_{i}` agents share one world but each owns (i) its own `racetrack_{i}` mesh at a grid offset, (ii) its own offset `TrackData` bound into its controller, and (iii) its own blocking, per-entity reset. 300 m spacing — not de-crowding — guarantees no cross-arena sight/collision. Keep straight that `arena.py`'s `ArenaLayout`/`Arena` is the newer pure-geometry representation, distinct from the live legacy `MultiAgentDeepRacerEnv` that drives training today.

## Why n=8 storms: the evidence

All facts below were re-derived from code and logs, not from framing. Several earlier quantitative claims were wrong and are corrected here.

**(a) `refresh_state(force=...)` ignores `force` on the subscription path — CONFIRMED.** `deepracer_env/sim_control/backends/ros_gz_backend.py`, `refresh_state` (lines 261–289). The bridged-subscription fast path returns before `force` is read:

```
269	    self._ensure_pose_sub()
270	    if self._pose_sub is not None and self._sub_cache:
271	        snapshot = self._sub_cache  # atomic ref (callback rebinds, never mutates)
272	        if snapshot is not self._pose_cache:
273	            self._prev_pose_cache = self._pose_cache
274	            self._pose_cache = snapshot
275	        return
276	
277	    now = time.monotonic()
278	    if not force and (now - self._last_refresh_t) < self._refresh_min_interval_s:
279	        return
```

`force` is consulted **only at line 278**, reached only when `self._pose_sub is None` or `self._sub_cache` is empty (the gz-CLI fallback, lines 281–284). So with the bridge live, `get_entity_state`'s lazy `refresh_state(force=True)` (line 293) does not trigger a fresh read — it re-serves whatever the background `_on_pose_tf` callback last cached.

**(b) n=8 runs — actual observed counts** (grep `-cE "process has died.*spawner|Failed to activate controller|Failed to load controller"`):

| log (n=8) | contention | "Reset agent (count: 0)" lines | reset *starts* | in-window shards | shards/reset | container exit |
|---|---|---|---|---|---|---|
| `/tmp/dr_drive/cam_n8_fix.log` | **2** | 593 | 297 | **144** | 0.48 | `rc=137` (1 chunk, force-killed) |
| `/tmp/dr_drive/cam_real_collect.log` | **4** | 1725 | 863 | **312** | 0.36 | 6× `rc=137`, 5× `rc=125` |

- The 593/1725 counts **double-count** (the grep matches both the `(count: 0)` start line and the `(count: 0) finished` line); true reset starts are 297 and 863. Still "hundreds."
- Resets come in tight back-to-back bursts (e.g. `cam_n8_fix.log:1296–1299` = reset/finished/reset/finished on consecutive lines) — the storm cadence is real.
- Shards are **not ~0**: 144 and 312 npz files were produced in-window. The real storm signal is **low yield** (~0.36–0.48 shards/reset, ~20–24 shards/min): roughly half of n=8 episodes reset without ever emitting a usable perception shard, and the run's stalled container is ultimately force-killed (`rc=137`).
- Shard mtime histogram, n=8 window `cam_n8_fix` (11:43→11:49, then SIGKILL): `3, 37, 28, 24, 24, 24, 4` per minute.

**(c) n=4 contrast — resets are NOT fewer; shards FLOW:**

| log (n=4) | contention | reset *starts* | in-window shards | shards/reset | notes |
|---|---|---|---|---|---|
| `/tmp/dr_drive/cam_n4_collect.log` | **10** | 1945 | 1932 | 0.99 | 22 chunks, still hit 13× `rc=137` |
| `/tmp/dr_drive/cam_n4_collect2.log` | **2** | 6666 | 6651 | ~1.0 | clean working run, no kill in window |

n=4 shard histogram peaks at **377–540 shards/min** (11:33–11:37 for `cam_n4_collect`; 11:51–12:28 sustained 350–540/min for `cam_n4_collect2`) vs n=8's ~24/min. The discriminator between "works" and "storms" is **shards-per-reset (~1.0 for n=4 vs ~0.4 for n=8)** and shards/min — **not** reset count.

`/tmp/dr_drive/bisect_result.txt` (verbatim):
```
n=5: BRINGUP-FAILED (no training start in 320s)
```
`bisect_n5.log` and `bisect_n6.log` each end `container gym-dr-camera_cnn_dataset exited rc=137; aborting` — note the bisect harness itself force-kills the container (`docker rm -f`) after its detection window, so this `rc=137` is partly harness-induced; the point that stands is that n≥5 never reached a healthy shard-producing steady state. The boundary is **n=4 works, n≥5 storms**.

**(d) Reset hot-path is NATIVE — CONFIRMED.** In `ros_gz_backend.py`:

- Teleport (reset) `set_entity_state` (lines 364–398) uses the bridged rclpy `SetEntityPose` client `self._set_pose_client` (`_ensure_set_pose_client`, 320–362; `ros_gz_interfaces/srv/SetEntityPose` on `{prefix}/set_pose`). The gz-CLI `_service("set_pose", …)` (394–398) is **fallback only** (client absent or call timed out at 391–392).
- Pose read `refresh_state` (261–289) is served from the bridged `dynamic_pose/info` TF subscription (`_ensure_pose_sub` 221–247, `_on_pose_tf` 249–259 → `self._sub_cache`). The gz-CLI snapshot (281–284) is **fallback only**.
- All gz-CLI subprocess calls (`self._run`/`self._service`) are cold-path: `spawn_entity`→`create` (196), `delete_entity`→`remove` (204), `list_entities` (211), `step`/`pause`/`unpause` (450/458/462), `set_visual_color`/`set_light` (479/495), and `_gz_alive` (172). The per-reset hot path never forks a gz CLI when the bridge is up.

**Bottom line.** The n=8 storm is a **reset-yield collapse** whose stalled container is eventually force-killed (`rc=137`, not OOM), driven by oversubscription on an 8-core box (controller-manager bringup contention + pose/TF flood + CPU starvation), not by a controller-manager contention flood per se. Spawner failures are a bringup-only handful (2–4 events; n=4 shows 10) and do not distinguish n=8 from n=4. What distinguishes n=8 is that ~half its many resets produce no shard and the container is SIGKILLed; n=4 sustains ~1 shard/reset and hundreds of shards/min.

## The pose-settle angle (tried, reverted)

`refresh_state(force=True)` ignores freshness on the subscription path (section (a) above): with the bridge live it returns the last-cached pose, which can be the **stale pre-teleport pose** immediately after a reset teleports the entity. A wait-for-fresh-message fix (block until `_on_pose_tf` delivers a post-teleport TF sample before serving `get_entity_state`) was tried and **reverted**, because:

- its test was **confounded by a controller death** — the run it was validated against also lost a controller-manager spawner, so the observed improvement/regression could not be cleanly attributed to the pose-settle change; and
- it touches the **shared reset path** (`RolloutCtrl.reset_agent` → `GetModelStateTracker` → `refresh_state`) that the **laptop HPO** also exercises, so a regression there would silently corrupt the oracle/HPO runs.

The stale-pose effect is **real but not the whole story** — it does not by itself explain the reset-yield collapse. It is worth revisiting once isolated from controller deaths and validated against the HPO path.

## Fixes shipped this session

Both commits are in `/home/lunav0/Projects/dr-gym` (HEAD = `db03372`, HEAD~1 = `f0eb830`), authored by Eduardo Dantas Luna on Wed Jul 1 2026, each `Co-Authored-By: Claude Opus 4.8`.

**(a) `f0eb830` — hard-exit a single-chunk train container to dodge the rclpy-finalize segfault.**
Full hash `f0eb830cf40e2972303a1ef55ef76a3da165e59b`. Stat: `gym_dr/app.py | 15 ++++++++++++++-` (+14 / -1).
In `train()` (`@@ -87,7 +87,20 @@`), the in-container branch changed from a direct return into capture-then-hard-exit:

- Before: `if os.getenv("GYM_DR_IN_CONTAINER"): return _train_one_chunk(experiment)`
- After: store `result = _train_one_chunk(experiment)`, then
  ```python
  if not os.getenv("GYM_DR_WORKER"):
      import sys
      sys.stdout.flush()
      sys.stderr.flush()
      os._exit(0)
  return result
  ```

`os._exit(0)` fires right after `_train_one_chunk` returns, gated to the in-container single-chunk path (`GYM_DR_IN_CONTAINER` set AND `GYM_DR_WORKER` not set), so the HPO worker loop is excluded. Purpose: bypass a native rclpy/DDS context-finalize teardown segfault (`rc=139`, no Python traceback) after a multi-car camera run, since data + checkpoints are already flushed. `stdout`/`stderr` are flushed first.

**(b) `db03372` — forward camera-run knobs into the worker container.**
Full hash `db03372c97480c6941171f09a0bd6f1a1b7eb177`. Stat: `experiments/camera_cnn_dataset.py | 7 +++++--` and `gym_dr/app.py | 8 +++++++-` (+12 / -3).

`gym_dr/app.py`, `_train_host()` (`@@ -245,9 +245,15 @@`): the host→container env-forward allowlist tuple gained four vars, now ending:
```python
"GYM_DR_ALLOW_CAMERA_NCARS", "GYM_DR_CAM_SMOKE", "GYM_DR_CAM_NCARS",
"GYM_DR_CAM_CHUNK_STEPS", "GYM_DR_GZ_DR_INTERVAL_S"
```

`experiments/camera_cnn_dataset.py` (`@@ -46,14 +46,17 @@`): two module-level constants made env-overridable:
- `CHUNK_STEPS = int(os.getenv("GYM_DR_CAM_CHUNK_STEPS", "2000" if SMOKE else "60000"))` (was `2_000 if SMOKE else 60_000`).
- `N_CARS = int(os.getenv("GYM_DR_CAM_NCARS", "2" if SMOKE else "8"))` (was `2 if SMOKE else int(os.getenv("GYM_DR_CAM_NCARS", "8"))`).

Latent bug fixed: the spawned worker re-imports the experiment module and rebuilds config from these module-level constants; without forwarding, a `GYM_DR_CAM_SMOKE=1` run built a smoke-sized host plan while the container (SMOKE unset) trained a full 60k-step chunk. `GYM_DR_GZ_DR_INTERVAL_S` is read sim-side inside the container.

## Open decision

**Maximum camera car count on this hardware.** Two options:

1. **Accept n=4 as the stable max.** n=4 is empirically reliable (~1.0 shard/reset, 350–540 shards/min, clean runs); n≥5 storms and never reaches a healthy steady state (its stalled container is force-killed, `rc=137`). No further engineering; the dataset collector caps at n=4.
2. **Invest in staggered per-arena controller bringup** — spawn arenas sequentially so the ~24 controller spawners (roughly 6 per car × 4+ cars) don't dogpile the CPU at bringup, aiming to reach n=8. This targets the oversubscription root cause directly.

Either way, a **hard 8-car ceiling** remains regardless — there are only `racecar_0..racecar_7` include blocks and a matching one-byte collide-bitmask allocation (`0x01..0x80`), so n>8 is not reachable without new launch blocks.

See the appended `D10` entry in `docs/questions-for-maintainer.md`.

## Cross-references

- `docs/reports/multicar_throughput.md` — camera n=2/3/4 table + the `DoubleBuffer`-starvation / launch-limit correction (direct parent; extend, don't duplicate).
- `docs/reports/status-2026-06-28.md` — "Camera system" section: the definitive launch-vs-render diagnosis and the generalized-launch benchmark table.
- `docs/deepracer-env-upgrade-handoff.md` — §2 (8-car launch ceiling root cause), §4 (real single-thread ODE/OGRE RTF cliff), §5 (phantom cars + the `gym_dr/envs/multi_car.py` camera cap).
- `docs/reports/session-state-2026-06-29.md` — 8-car launch-XML copy-paste ceiling + phantom-car facts.
- `docs/reports/throughput.md` — RTF ~4.5× cap + single-thread OGRE render context.
- `docs/reports/domain-randomization.md` — reset/friction DR knobs (`NUMBER_OF_RESETS`).
- `docs/arch-robustness-study-design.md` — the max-parallelism big-rollout study this reset-storm constrains.
- Code anchors: `gym_dr/envs/multi_car.py:408-416` (camera cap), `deepracer_env/sim_control/backends/ros_gz_backend.py:261-289` (`refresh_state` stale-pose), `simulation/src/deepracer_simulation_environment/launch/racetrack_with_racecar.launch` (car blocks).
