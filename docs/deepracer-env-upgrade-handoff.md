# deepracer-env upgrade handoff — uncapping cars + (optional) ROS2/Gazebo migration

> **Audience:** the agent doing the deepracer-env upgrade. This is a hard, exhaustive
> dump of everything learned in the 2026-06-29 session. The maintainer will refine it.
> Written by Claude (Opus 4.8). Treat versions/paths as verified-this-session facts;
> re-verify anything before relying on it (the stack changes).

---

## 0. Mission (in priority order)

1. **Break the 8-car ceiling** so `n_cars` is a single parameter, not a hand-duplicated
   XML count. This is the immediate blocker for the "max-parallelism big-rollout DR"
   study design (see `docs/arch-robustness-study-design.md`).
2. **(Strategic, optional)** Migrate ROS Noetic + Gazebo 11 Classic → ROS2 + Gazebo Sim
   (Ignition). Bigger payoff (native launch loops, newer physics, off-EOL software) but
   a multi-month project. Do NOT do this *just* to get a loop — option 1 below gets that today.

---

## 1. Ground-truth current versions (verified 2026-06-29)

- **ROS Noetic** (ROS1's final LTS, EOL May 2025). `FROM ros:noetic-robot`,
  `ENV ROS_DISTRO=noetic`. Python 3.8 in-container.
- **Gazebo 11 Classic** (final Gazebo Classic, EOL Jan 2025). Repo also references
  `Gazebo9` (legacy strings) but the live image is `gazebo-11` / `ros-noetic-gazebo-*`.
- Launch files are **pure XML `.launch`** (ROS1 roslaunch). 8 of them. No `.launch.py`.
- Repo root: `/home/lunav0/Projects/deepracer-env` (also on the laptop at
  `/home/eduardoluna/Repos/dissertation/deepracer-env`). **Not a git repo on the main PC**
  (`git: false`); sync between machines is via `scp`/`rsync`, not push (push to main is
  blocked by an auto-mode classifier — see session notes).

### Packages (catkin, under `simulation/src/`)
| Package | Language | ROS2-port difficulty |
|---|---|---|
| `deepracer_simulation_environment` | Python nodes + XML launch + xacro/urdf | medium (launch rewrite, node API) |
| `deepracer_msgs` | msg/srv definitions | low-medium (msg/srv mostly port cleanly) |
| `deepracer_gazebo_system_plugin` | **C++ Gazebo system plugin** | **HARD** — Gazebo Classic plugin API → Gazebo Sim (Ignition) is a full rewrite |

The C++ plugin is the long pole for any ROS2/Gazebo-Sim migration:
`simulation/src/deepracer_gazebo_system_plugin/src/deepracer_gazebo_system_plugin.cpp`
(+ header). It hooks Gazebo Classic's model/physics API directly.

---

## 2. The 8-car ceiling — exact root cause

**The cap is `n=8` purely because the launch XML hand-lists 8 car blocks.** Nothing else.

File: `simulation/src/deepracer_simulation_environment/launch/racetrack_with_racecar.launch`
(233 lines). It contains 8 blocks `racecar_0` … `racecar_7`, each a ~23-line
`<include file=".../racecar.launch" ns="/racecar_N">` gated on
`if="$(eval len(str(simapp_versions).split(',')) > N)"`. The last block's own comment says it:

> *"Generalises the old hardcoded 2-car cap (a LAUNCH limit, not a render limit).
> car_node.py / get_racecar_names already handle N; only these blocks did not."*

### Why hand-duplicated: ROS1 roslaunch XML has NO loop construct
Confirmed via ROS Answers (sources below). You cannot `for i in range(N)` in a `.launch`.
So each car body is a literal copy-paste with the index hardcoded in ~15 places per block
(`split(',')[7]`, `racecar_bitmask 0x80`, xacro variant lines, kinesis, camera/lidar flags).
Whoever extended it from the old 2-car cap stopped at index 7.

### Crucial: the per-car logic is ALREADY a parameterized template
`racecar.launch` (sibling file) is the real per-car bringup (xacro→robot_description,
controllers, namespace, friction, sensors). Its args (verified):
`racecar_name`, `racecar_bitmask`, `simapp_version`, `racecar_xacro_file`,
`body_shell_type`, `friction_mu` (default 1.5 — per-spawn friction DR knob),
`include_camera` (false = feature-obs/camera-off path), `include_lidar_sensor`,
`include_second_camera`, the lidar_360_* params, kinesis_*.
**The parent file does nothing but call this template 8×.** A "loop" only needs to call
it N times — it does NOT need to reimplement controllers/bringup.

### What already handles arbitrary N (blast radius is SMALL — just the launch)
- `scripts/car_node.py:157` → `utils.get_racecar_names(RACER_NUM)` — name generation is
  N-parameterized via the `RACER_NUM` param.
- `get_racecar_names(RACER_NUM)` (markov.utils) generates N namespaced names.
- mp4/video/camera-topic test nodes all take `racecar_names` lists.
- dr-gym side (`gym_dr/envs/multi_car.py`) is a generic N-car VecEnv (`num_envs = n_cars`).

So: **fix the launch, and N just works downstream.**

---

## 3. Options to uncap N (researched 2026-06-29, sources at bottom)

Ordered by scope. Recommendation: **Option 1** for the immediate unblock.

### Option 1 — Generator script (RECOMMENDED, ROS1-native, contained)
A small Python/Jinja2/empy step that reads `N` and **emits the parent launch** with N
`<include>` blocks (index `i`, bitmask `1<<i`, `simapp_version[i]`, etc.) before
`roslaunch` runs — exactly how xacro generates URDF. Reuses `racecar.launch` untouched.
- **Pros:** clean, robust, no migration, `N` = one number, low risk.
- **Cons:** adds a pre-launch codegen step in the container entrypoint.
- **Watch-out:** the per-car bitmask is `1<<i` (`racecar_7 = 0x80`). Past 8 cars it spills
  out of one byte → verify nothing downstream assumes a byte-wide mask (grep `bitmask`).

### Option 2 — Self-recursive launch (pure-XML loop, ROS1)
Parent includes itself with `index-1`, emitting one `racecar.launch` per level.
- **Pros:** no codegen step.
- **Cons:** finicky — roslaunch `eval` can't use `<`/`<=`, so you hack it with `==`
  conditionals; recursion depth + readability suffer. Not recommended.

### Option 3 — Runtime spawner node (ROS1, most flexible, most work)
A Python node that spawns each car at runtime by calling the Gazebo
`/gazebo/spawn_urdf_model` service (`rosrun gazebo_ros spawn_model -model <name>`) in a
loop, then spawns each car's controllers via `controller_manager`.
- **Pros:** N at runtime, dynamic add/remove cars live.
- **Cons:** reimplements per-car controller bringup that `racecar.launch` already does.
- The existing launch already calls `/gazebo/spawn_urdf_model` per car (seen in sim boot
  logs: *"Calling service /gazebo/spawn_urdf_model"* → *"Successfully spawned entity"*).

### Option 4 — ROS2 + Gazebo Sim migration (the "newer version" answer)
**ROS2 launch files are Python**, so spawning N namespaced robots is a native
`for`/list-comprehension (`gen_robots_list`-style) — idiomatic, no codegen hack.
- **Pros:** native loops; newer physics (DART/bullet in Gazebo Sim); off-EOL; better
  multi-robot namespacing; long-term maintainability.
- **Cons:** **multi-month migration of the whole stack.** All 3 catkin packages → ament;
  the **C++ system plugin → Gazebo Sim plugin API rewrite** (hardest); `deepracer_msgs`
  → ROS2 msg/srv; controllers → `ros2_control`; every node's rospy→rclpy; the dr-gym
  side (`docker_runner.py`, env vars, the ROS1 service calls in `multi_car`/backend) all
  need rewiring. Justify it as a **platform investment**, not a loop workaround.

---

## 4. The REAL ceiling underneath the launch (must MEASURE, not read off a file)

Even with N uncapped, one Gazebo world runs a **single-threaded ODE physics step**. Pile
in more bodies → per-car RTF falls off a cliff. **OpenAI's "1024 envs" were 1024 separate
lightweight sims, not 1024 bodies in one world** — you cannot replicate that in one Gazebo
world. Realistic single-world ceiling is ~8–16 cars before per-car RTF is unusable.

**Action for the upgrade agent:** after uncapping, run an **RTF-vs-n_cars sweep** for the
feature-obs (camera-off) path and report the cliff. That number — not the launch — is the
true `top-N` for one sim. (Camera-on is far lower; OGRE render is single-threaded too.)

To exceed one-world limits you'd need a **multi-sim VecEnv**: N containers × M cars each,
all feeding ONE PPO learner (one trial). dr-gym does NOT have this aggregation layer today
(today multiple sims = multiple Optuna *trials*, not one big rollout). That's the real
path to OpenAI-scale env parallelism and is a separate, large dr-gym build.

---

## 5. Multi-car gotchas the upgrade must preserve/fix

1. **No `set_world` in multi-car.** `MultiCarVecEnv.can_set_world = hasattr(backend,
   "set_world")` is False for the multi-agent backend — each car's track is fixed at
   launch, no in-process hot-swap. This is *why* the single-car path exists (it CAN
   `set_world` for in-loop held-out eval). If the upgrade adds per-car world-swap, the
   eval architecture simplifies a lot (`gym_dr/trainers/sb3/callbacks.py` ~L466-502).
2. **Phantom cars when `n_cars` > launch bodies.** If the env requests more cars than the
   launch spawns, the extra namespaces advertise topics with **no body/publisher**.
   - Camera path: a 3rd camera car's blocking sensor read `log_and_exit()`s ~120s in →
     hard cap at 2, guarded in `gym_dr/envs/multi_car.py:406` (raises unless
     `GYM_DR_ALLOW_CAMERA_NCARS=1`).
   - Feature path: a missing model's STATE read does NOT block, so it "works" silently
     with phantom agents — **dangerous**: the n=12 oracle (`experiments/oracle_asym_multicar.py`,
     `N_CARS=12`) requests MORE than the 8-body launch, so cars 8–11 were almost certainly
     phantom. **VERIFY this before trusting any oracle result at n>8.** After uncapping,
     add an assert that `n_cars <= spawned_bodies`.
3. **Friction DR is sim-side** (`friction_mu` per spawn, default 1.5). Wheel ODE μ is set
   per-episode via a Gazebo service (`FrictionRandomizer` in the reset path). Preserve
   this through any migration — it's a key DR knob. See `docs/domain-randomization.md`.
4. **2-camera "limit" was a misdiagnosis** earlier — it's a launch-block limit, NOT
   "Gazebo renders only 2 cameras." Real camera residual is single-thread OGRE render
   degradation (graceful fps drop, not a crash). See `docs/reports/status-2026-06-28.md`.

---

## 6. dr-gym ↔ deepracer-env interface (what the upgrade must not break)

- dr-gym spawns the sim via `gym_dr/docker_runner.py` (docker run with a big env-var set:
  `WORLD_NAME`, `GYM_DR_N_CARS`, `GYM_DR_CAMERAS`, `GYM_DR_FRICTION_MU`, `RTF_OVERRIDE`,
  `GYM_DR_ROTATE`, `SEED`, …). deepracer-env is bind-mounted into the container at
  `/usr/local/lib/python3.8/dist-packages/deepracer_env` (+ launch + urdf overlays).
- The env factory `gym_dr/envs/dispatch.py::build_env` dispatches on observation type ×
  n_cars. Single-car → `time_trial.py`; multi-car → `multi_car.py::MultiCarVecEnv`.
- The ROS1 service calls (set_world for single-car, spawn, reset, friction) live in the
  deepracer-env backend that dr-gym imports. A ROS2 migration must keep this Python
  surface (or dr-gym's `docker_runner`/`multi_car`/backend calls all need rewiring).
- Sim package overlays (camera-off toggle, friction, generalized racecar blocks) are
  bind-mounted from deepracer-env over the in-image package — see `docker_runner.py`
  mount list. Any new generated-launch file must be on that mount path.

---

## 7. Verification approach for the upgrade

1. **Smoke:** boot the sim with `n_cars` = 1, 2, 4, 8, 12, 16; confirm each car has a real
   body (no phantom — check `/gazebo/model_states` has N models) and nonzero per-car
   sensor/state Hz.
2. **RTF sweep:** measure aggregate env-steps/sec and per-car RTF vs n_cars (feature-obs).
   Report the cliff. Compare to the pre-upgrade `docs/multicar_throughput.md` baseline.
3. **DR integrity:** confirm per-car friction μ varies per episode (`/gazebo/get_*`), and
   per-car different-track assignment still works (the diversity engine).
4. **dr-gym end-to-end:** run a short `experiments/oracle_asym_multicar.py` at the new N;
   confirm nonzero `dr/ep_max_progress` per car (the phantom-car / reward-clobber trap —
   see memory `dr-gym-postinit-reward-clobber-bug`).
5. **Parity (if migrating):** same policy, same seed, compare trajectories ROS1 vs ROS2 —
   physics engines differ (ODE vs DART), so expect drift; quantify it before trusting.

---

## 8. Key file paths

**deepracer-env**
- Parent launch (the cap): `simulation/src/deepracer_simulation_environment/launch/racetrack_with_racecar.launch`
- Per-car template: `.../launch/racecar.launch`
- Car node (handles N): `.../scripts/car_node.py` (`RACER_NUM`, `get_racecar_names`)
- C++ Gazebo plugin (ROS2 long pole): `simulation/src/deepracer_gazebo_system_plugin/src/deepracer_gazebo_system_plugin.cpp`
- msgs/srvs: `simulation/src/deepracer_msgs/`

**dr-gym**
- Sim launcher: `gym_dr/docker_runner.py`
- Env factory: `gym_dr/envs/dispatch.py::build_env`
- Multi-car VecEnv + phantom-cap: `gym_dr/envs/multi_car.py` (cap at `:406`)
- Single-car (set_world): `gym_dr/envs/time_trial.py`
- Eval callback (set_world loop / multi-car branch): `gym_dr/trainers/sb3/callbacks.py:442-518`
- Existing context docs: `docs/deepracer-env-review.md`, `docs/domain-randomization.md`,
  `docs/multicar_throughput.md`, `docs/code-map.md`, `docs/reports/status-2026-06-28.md`

---

## 9. Sources (web research, 2026-06-29)

- ROS1 launch has no native loop / workarounds: https://answers.ros.org/question/334057/
- ROS1 variable number of robots: https://answers.ros.org/question/353817/
- Gazebo Classic roslaunch + `spawn_model` service: https://classic.gazebosim.org/tutorials?tut=ros_roslaunch
- ROS2 multi-robot (Python launch loops): https://osrf.github.io/ros2multirobotbook/simulation.html
- ROS2 spawn multiple robots in a loop: https://www.theconstruct.ai/spawning-multiple-robots-in-gazebo-with-ros2/
