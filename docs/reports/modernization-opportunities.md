# Modernization opportunities — ROS 2 Lyrical / Gazebo Jetty (gz-sim 10.x)

_Task 9 "Modernizer". Authored 2026-07-01. Audience: the maintainer, feeding real
engineering. Every recommendation is grounded in the actual deepracer-env code
(cited `file:line`) and cross-checked against current gz-sim / ROS 2 docs (URLs
inline). Scope: newer ROS 2 / Gazebo primitives the ROS1→ROS2 port under-used or
skipped, prioritized by leverage on the **camera reset-storm**, **multi-camera
throughput**, **determinism**, and **multi-robot spawn** goals._

Companion to `docs/reports/camera-multicar-reset-storm.md` (root-cause of the
n≥5 storm) — this report proposes the concrete gz/ROS 2 fixes. Read that first
for the failure evidence; this for what to build.

---

## Top 3 highest-leverage

1. **Batch the per-car reset into ONE `set_pose_vector` call (gz.msgs.Pose_V).**
   The reset is today N *sequential blocking* round-trips (`multi_agent_env.py:117`
   → `rollout_agent_ctrl.py:359-360`). gz-sim exposes a native
   `/world/<w>/set_pose_vector` service that teleports **many entities in one
   call, applied on one gz update tick**. Collapsing N teleports into 1 attacks
   the reset-storm directly (the storm is N resets clustering in a window and
   serializing against the single-thread sim/render). **Effort: M.** This is #1
   exactly as expected.

2. **Activate the already-authored `simulation_interfaces` backend by launching
   gz through the `ros_gz_sim` `gzserver` component.** The repo says "no process
   serves the services yet" (`simulation_interfaces_backend.py:16-20`) — that is
   now **stale**. Gazebo Jetty ships a `simulation_interfaces` server *inside the
   `ros_gz_sim` gzserver node* (`/gzserver/set_entity_state`,
   `/gzserver/get_entities_states`, `/gzserver/step_simulation`, …). The launch
   currently starts the **raw** `gz sim -s` binary (`multi_arena.launch.py:160-165`),
   which does **not** expose those services. Switching to the `gzserver`
   component flips on: a **batched multi-entity state read** (`GetEntitiesStates`),
   **atomic pose+twist** reset (`SetEntityState`), and **deterministic**
   `StepSimulation` — all through the dormant, already-written
   `SimulationInterfacesBackend`. **Effort: M** (mostly launch + a one-line
   factory default + verification; the backend code exists).

3. **Replace the `gz service` / `gz topic` CLI subprocess forks with in-process
   `gz-transport13` Python bindings.** Every cold-path control op in the active
   backend shells out (`ros_gz_backend.py:128-158`, `_run`/`_service`): spawn,
   remove, step, pause, and — per **episode, per car** — visual recolour and
   lighting DR (`set_visual_color`/`set_light`, lines 467-496). Under a reset
   storm those forks pile onto an already CPU-starved 8-core box. `gz.transport13`
   has Python `node.request(...)` bindings (sync, in-process) that eliminate the
   fork and can be issued concurrently from a thread pool. **Effort: M.**

Together these turn "N sequential subprocess-y round-trips per reset window" into
"one or two in-process batched requests" — the core of the storm.

---

## Priority 1 — reset-storm: parallelize / batch the reset seam

### 1.1 Batch teleport via `set_pose_vector` (gz.msgs.Pose_V)
- **What.** gz-sim's `UserCommands` system serves `/world/<w>/set_pose_vector`
  taking a `gz.msgs.Pose_V` (a vector of named poses) and applying them together
  on the next update tick — the batch analogue of the single `set_pose` the
  backend uses today.
- **Why here.** `MultiAgentDeepRacerEnv.reset()` does
  `[agent.reset_agent() for agent in self._agents]` (`multi_agent_env.py:117`) and
  each `reset_agent()` blocks on `set_model_state(..., blocking=True)` then
  `get_model_state(..., blocking=True)` (`rollout_agent_ctrl.py:359-360`). With N
  cars these serialize into the exact window the reset-storm report identifies as
  the failure. One `set_pose_vector` for all cars removes the per-car
  serialization and the per-car round-trip latency.
- **API / docs.** `gz.msgs.Pose_V` request, `gz.msgs.Boolean` reply;
  `UserCommands` set-pose services:
  https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1UserCommands.html ,
  entity/pose service list https://gazebosim.org/api/sim/9/entity_creation.html
- **Where.** Add `set_entity_states(list[(name, EntityState)])` to `SimControl`
  (`sim_control/interface.py`) + a `RosGzBackend` impl building one `Pose_V`
  (`sim_control/backends/ros_gz_backend.py`); add a batched `reset_all()` to
  `MultiAgentDeepRacerEnv` that computes all start states then issues one call
  (bypassing the per-car `SetModelStateTracker`). **Effort: M.**

### 1.2 Batched pose read via `GetEntitiesStates` (removes the finite-diff twist hack)
- **What.** `simulation_interfaces/GetEntitiesStates` returns pose **and twist**
  for many entities (optionally name/tag-filtered) in one call.
- **Why here.** The active backend has **no** real velocity: it finite-differences
  twist from two pose snapshots (`ros_gz_backend.py:300-318`) — noisy, and a known
  source of the stale-pose-after-teleport effect (`camera-multicar-reset-storm.md`
  §(a)). A true batched state read gives every car's pose+twist consistently in one
  round-trip and drops ~30 lines of estimation.
- **API / docs.** `/gzserver/get_entities_states`, service
  `GetEntitiesStates` — Gazebo Jetty:
  https://gazebosim.org/docs/latest/ros2_sim_interfaces/ ; interface API:
  https://docs.ros.org/en/ros2_packages/jazzy/api/simulation_interfaces/
- **Where.** `SimulationInterfacesBackend` (already wires `GetEntityState`; add the
  plural), consumed by `GetModelStateTracker.update_tracker`
  (`gazebo_tracker/trackers/get_model_state_tracker.py:67`). **Effort: S–M** once
  the backend is live (see 1.3).

### 1.3 Atomic pose+twist reset via `SetEntityState`
- **What.** `simulation_interfaces/SetEntityState` sets pose **and** twist in one
  request (`set_pose`/`set_twist` flags).
- **Why here.** `RosGzBackend.set_entity_state` sets **pose only** and explicitly
  relies on "physics + the zeroed wheel commands" to re-settle velocity
  (`ros_gz_backend.py:364-366`). A car reset while sliding keeps residual velocity
  into the next episode — a subtle non-determinism the atomic twist-zero removes.
  The `SimulationInterfacesBackend` already builds the twist
  (`simulation_interfaces_backend.py:167-175`); it just needs to be the live
  backend.
- **API / docs.** Jetty ros2_sim_interfaces (above). Note: `SetEntityState`
  assigns *both* pose and twist from the message.
- **Where.** Flip the factory default (`sim_control/factory.py:58`) once the server
  is up (1.4). **Effort: S.**

### 1.4 Launch gz via the `ros_gz_sim` `gzserver` (composable) node → unlock all of the above
- **What.** The `simulation_interfaces` server is a component **inside the
  `ros_gz_sim` gzserver node** (`GzServer` registers a `gz_simulation_interfaces`
  member). The raw `gz sim -s` binary does not run it.
- **Why here.** This is the single switch that makes 1.2/1.3 (and deterministic
  stepping, §3) real, and it retires the CLI-subprocess control plane in favour of
  proper ROS 2 services. It also enables **composition** (§2.2).
- **API / docs.** gzserver source (registers simulation interfaces + is a
  composable component): https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_sim/src/gzserver.cpp ;
  launch: https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_sim/launch/gz_server.launch.py ;
  statically-composable gzserver tracking issue:
  https://github.com/gazebosim/ros_gz/issues/631 ; Jetty ros2_sim_interfaces:
  https://gazebosim.org/docs/latest/ros2_sim_interfaces/
- **Where.** `simulation/src/deepracer_simulation_environment/launch/multi_arena.launch.py:160-165`
  (swap the `ExecuteProcess(["gz","sim","-s",...])` for the `GzServer` action /
  composable node); set `SimulationInterfacesBackend(service_namespace="/gzserver")`.
  Verify `is_available()` + the service-presence probe already in
  `factory.py:91-100` then auto-selects it. **Effort: M** (launch + in-stack
  verification; the storm-verification loop is the real cost).

---

## Priority 2 — multi-camera render throughput

### 2.1 Set realistic expectations: gz-sensors renders cameras SEQUENTIALLY
- **What / why here.** The n=4-camera ceiling is **not** a bug you can code around
  cheaply: `gz-sensors` renders every camera **sequentially on one render thread**,
  a long-standing known bottleneck (multi-GPU/multi-threaded rendering is an open
  feature request, not shipped). This matches the project's own "single-thread
  OGRE render degradation" note. So throughput work should target **fewer/cheaper
  render events during resets** and **GPU**, not "make OGRE parallel."
- **Docs.** Multi-threaded rendering request (gz-sensors#81):
  https://github.com/gazebosim/gz-sensors/issues/81 ; camera-rate-vs-GUI
  bottleneck (gz-sensors#332): https://github.com/gazebosim/gz-sensors/issues/332
- **Where.** N/A (context for the ceiling in
  `camera-multicar-reset-storm.md` "open decision"). **Effort: n/a.**

### 2.2 Composable nodes + intra-process zero-copy for the image/pose bridges
- **What.** Load `gzserver` and the `ros_gz_bridge`/`ros_gz_image` bridges as
  **composable nodes in one container**; on Jazzy-era ros_gz this makes
  gz↔bridge traffic **intra-process with zero-copy** for images (and the image
  bridge already `memcpy`s instead of `std::copy`).
- **Why here.** The launch spawns **one `image_bridge` process per car**
  (`multi_arena.launch.py:281-286`) plus a separate `parameter_bridge`
  (lines 173-184). At n≥5 the report pins the storm on "pose/TF flood + CPU
  starvation" on an 8-core box — every camera frame and pose sample crosses a
  process boundary and is copied. Intra-process zero-copy collapses that per-frame
  copy + serialization cost, freeing cores for the render/reset path.
- **API / docs.** ros_gz composition + intra-process:
  https://gazebosim.org/docs/latest/ros2_overview/ ; launch-from-ROS composition:
  https://gazebosim.org/docs/latest/ros2_launch_gazebo/ ; bridge overview:
  https://index.ros.org/p/ros_gz_bridge/
- **Where.** `multi_arena.launch.py` — replace the per-car `Node(...image_bridge)`
  and the `parameter_bridge` with `ComposableNodeContainer` +
  `ComposableNode`s (and the `GzServer` composable from 1.4) in one container.
  **Effort: M.**

### 2.3 Throttle/skip camera rendering during the reset window
- **What.** Camera `<update_rate>15</update_rate>` is static in the URDF
  (`deepracer_gz.urdf.xacro`, `zed_camera` sensor). gz-sim can change a sensor's
  rate at runtime and lets sensors be enabled/`always_on` toggled; the reset burst
  doesn't need fresh frames.
- **Why here.** During a reset storm the env clears and re-fills camera buffers
  anyway (`agent.reset_agent` → `sensor.reset()`), so frames rendered mid-teleport
  are discarded — pure waste on the single render thread. Lowering the effective
  rate (or gating rendering) while cars are mid-reset gives the render thread back
  to the cars that are actually stepping.
- **Docs.** gz sensor config / rates: https://gazebosim.org/libs/sensors/ ;
  Sensors system: https://gazebosim.org/libs/sim/
- **Where.** URDF sensor block (static lower bound) and/or a runtime rate service
  toggled from `MultiAgentDeepRacerEnv.reset()`. **Effort: M** (runtime path) /
  **S** (static rate drop).

### 2.4 Keep headless EGL/GPU rendering first-class
- **What / why here.** The render path already gates on `GYM_DR_RENDER` +
  `--headless-rendering` (`multi_arena.launch.py:160-163`) and the project has a
  render-gpu image (memory: "gz Jetty GPU rendering"). This is the right modern
  primitive (OGRE2 + EGL, no X); the recommendation is to make it the **default**
  for any camera run and document the EGL ICD gotcha, since software GL cannot
  render OGRE2 at all. **Effort: S** (config/docs).

---

## Priority 3 — determinism & reproducibility

### 3.1 Deterministic lock-step via `StepSimulation` (trade-off, opt-in)
- **What.** `simulation_interfaces/StepSimulation` (and gz's `WorldControl
  multi_step`) advance an exact integer number of ticks and return paused —
  race-free and reproducible given a seed.
- **Why here.** The multi-car step is **free-running**, paced by *wall-polling the
  `/clock`* (`multi_agent_env.py:94-111`, `_pace_to_sim_dt`) — reproducible only up
  to timing jitter, and it "outruns the sim under load." A deterministic
  per-env-step advance is the correct primitive for the dissertation's repro/ADR
  claims and for A/B-ing the decoupled-observation architecture. The backend
  already advertises `Capability.DETERMINISTIC_STEP` and implements it
  (`ros_gz_backend.py:447-455`, `simulation_interfaces_backend.py:179-183`); it is
  simply **not used** in the free-running loop.
- **Trade-off.** Lock-step throttles throughput (must wait each tick) — keep it a
  config flag: `deterministic` for evaluation/repro runs, `free-run` for
  max-throughput DR rollouts. Gate on `SimControl.supports(DETERMINISTIC_STEP)`.
- **Docs.** Jetty StepSimulation (`STEP_SIMULATION_SINGLE`/`_MULTIPLE`):
  https://gazebosim.org/docs/latest/ros2_sim_interfaces/ ;
  https://docs.ros.org/en/ros2_packages/jazzy/api/simulation_interfaces/
- **Where.** A stepping strategy in the env step (`multi_agent_env.py:124-145`)
  choosing pace-vs-`sim.step(n)`. **Effort: L** (behavior change → needs a parity
  gate vs the current free-run).

### 3.2 Do NOT adopt global `ResetSimulation` — it breaks decoupling
- **What / why here.** Jetty exposes `/gzserver/reset_simulation`, but a global
  reset clobbers **every** arena at once — antithetical to the tiled multi-arena
  design (`sim_control/interface.py:279-290` deliberately disables world reset).
  Flagging this so it is not mistaken for a reset-storm fix: **per-entity
  `set_pose_vector` (§1.1) is the correct primitive, not world reset.**
  **Effort: n/a** (guardrail).

---

## Priority 4 — multi-robot spawning & bringup

### 4.1 Stagger / reduce the controller_manager spawner dogpile
- **What.** Each car gets its own namespaced `gz_ros2_control` controller_manager
  plus 3 spawners (`multi_arena.launch.py:220-276`), all racing at bringup with a
  120 s timeout (line 209). The report ties n≥5 bringup failures to this dogpile +
  CPU starvation.
- **Why here / options.** (a) **Stagger** arena bringup with `TimerAction` /
  chained `OnProcessExit` so the ~3N spawners don't all hit the CPU at once (this
  is the report's "open decision" option 2). (b) The spawners are already the
  modern `controller_manager/spawner` with `--param-file` — good; the win is
  ordering, not the tool.
- **Docs.** gz_ros2_control (Jazzy): https://control.ros.org/jazzy/doc/gz_ros2_control/doc/index.html
- **Where.** `multi_arena.launch.py` per-arena action emission. **Effort: S–M.**

### 4.2 Batch track spawns via `create_multiple` (gz.msgs.EntityFactory_V)
- **What.** gz-sim serves `/world/<w>/create_multiple` to insert several entities
  in one request (batch analogue of `create`).
- **Why here.** Extra track instances are spawned **one `ros_gz_sim create`
  process per arena** (`multi_arena.launch.py:226-237`) and, on the Python side,
  one `spawn_entity`→`create` CLI call each (`ros_gz_backend.py:181-201` via
  `WorldSwapper.spawn_track_instance`). Batch spawn cuts bringup latency and
  process count for large N. Cold path, so lower priority than the reset seam.
- **Docs.** https://gazebosim.org/api/sim/9/entity_creation.html (`create_multiple`).
- **Where.** `RosGzBackend.spawn_entity` (add a plural), `world_swap.py` spawn
  loop, and/or the launch. **Effort: M.**

---

## What the port KEPT as a ROS1-era pattern (modern primitive available)

| ROS1-era pattern still in the port | Where | Modern gz/ROS 2 primitive | This report |
|---|---|---|---|
| **Sequential per-car blocking `SetModelState`** teleport | `multi_agent_env.py:117`, `rollout_agent_ctrl.py:359` | `set_pose_vector` (batch, one tick) | §1.1 |
| **`GetModelState` per-tick snapshot + finite-diff twist** | `ros_gz_backend.py:261-318` | `GetEntitiesStates` (batch pose+twist) | §1.2 |
| **Pose-only reset; velocity "re-settles"** | `ros_gz_backend.py:364-366` | `SetEntityState` (atomic pose+twist) | §1.3 |
| **Control plane via `gz service`/`gz topic` CLI forks** | `ros_gz_backend.py:128-158,467-496` | `gz.transport13` Python `node.request` | Top-3 #3 |
| **Raw `gz sim -s` binary (no ROS sim-control services)** | `multi_arena.launch.py:160-165` | `ros_gz_sim` `gzserver` component (`simulation_interfaces`) | §1.4 |
| **One bridge process per car, cross-process image copy** | `multi_arena.launch.py:173-184,281-286` | composable nodes + intra-process zero-copy | §2.2 |
| **Free-running step paced by wall-polling `/clock`** | `multi_agent_env.py:94-111` | deterministic `StepSimulation` (opt-in) | §3.1 |
| **Static camera `update_rate` (no reset-window throttle)** | `deepracer_gz.urdf.xacro` `zed_camera` | runtime sensor-rate / enable toggling | §2.3 |

Already modernized well (no action): batched pose **subscription** via the bridged
`dynamic_pose/info` TF stream (`ros_gz_backend.py:221-259`), per-entity decoupled
seam design (`sim_control/interface.py`), native gz `visual_config`/`light_config`
DR replacing the custom C++ plugin, and headless EGL scaffolding.

---

## Suggested sequencing (dependency-ordered)

1. **§1.4** launch `gzserver` component → unlocks §1.2, §1.3, §3.1 (the enabling change).
2. **§1.1** `set_pose_vector` batch reset (biggest single storm lever; works on the
   current `ros_gz` backend too, so it is **not** blocked on §1.4).
3. **§1.2 / §1.3** flip to `SimulationInterfacesBackend` (batch read + atomic twist).
4. **§2.2** composable/zero-copy bridges (frees cores the storm needs).
5. **§4.1** staggered controller bringup; **§2.3** reset-window render throttle.
6. **Top-3 #3** gz-transport13 Python (retire CLI forks) — orthogonal, do anytime.
7. **§3.1** deterministic stepping behind a config flag (needs a parity gate).

Each lands with tests + a smoke pre-flight per the project's Definition of Done;
§1.1 and §2.2 should be re-verified against the n=5/6/8 storm bisect in
`camera-multicar-reset-storm.md` (the pass/fail oracle already exists).

---

## Sources

- Gazebo Jetty — ROS 2 Simulation Interfaces: https://gazebosim.org/docs/latest/ros2_sim_interfaces/
- `simulation_interfaces` package overview: https://index.ros.org/p/simulation_interfaces/
- `simulation_interfaces` API (Jazzy): https://docs.ros.org/en/ros2_packages/jazzy/api/simulation_interfaces/
- ros_gz `gzserver` (simulation-interfaces component, composable): https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_sim/src/gzserver.cpp
- ros_gz `gz_server.launch.py`: https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_sim/launch/gz_server.launch.py
- Statically-composable gzserver (issue #631): https://github.com/gazebosim/ros_gz/issues/631
- gz-sim entity creation / pose services (`create`, `create_multiple`, `set_pose`, `set_pose_vector`): https://gazebosim.org/api/sim/9/entity_creation.html
- gz-sim `UserCommands` (set-pose services): https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1UserCommands.html
- gz-transport Python bindings (gz.transport13): https://gazebosim.org/api/transport/13/python.html
- gz-sensors sequential/multi-threaded rendering (issue #81): https://github.com/gazebosim/gz-sensors/issues/81
- gz-sensors camera-rate bottleneck (issue #332): https://github.com/gazebosim/gz-sensors/issues/332
- gz-sensors library docs: https://gazebosim.org/libs/sensors/
- ros_gz ROS 2 integration overview (composition, intra-process): https://gazebosim.org/docs/latest/ros2_overview/
- Launch Gazebo from ROS 2 (composition): https://gazebosim.org/docs/latest/ros2_launch_gazebo/
- ros_gz_bridge overview: https://index.ros.org/p/ros_gz_bridge/
- gz_ros2_control (Jazzy): https://control.ros.org/jazzy/doc/gz_ros2_control/doc/index.html
</content>
</invoke>
