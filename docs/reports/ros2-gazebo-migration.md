# ROS2 + newer Gazebo (gz-sim) migration — performance & scope · `[REAL]` · 2026-06-22

## Bottom line
Migrating ROS1 Noetic + **Gazebo Classic** → ROS2 + **gz-sim** (Harmonic/Ionic) is **not a throughput win.**
Published comparisons show Gazebo Classic often *matches or beats* Ignition/gz-sim on real-time factor (one
test: Classic ~0.3–0.99 RTF vs Ignition ~0.2). Migrate for **Gazebo Classic EOL (Jan 2025)**, **ROS2
alignment with the car's own stack** (`[REAL]`), modular architecture, and better rendering/sensors —
**not** raw speed. The device sweep already showed our ceiling is the **physics + ROS step loop**
(`docs/reports/throughput.md`); gz-sim won't necessarily raise it. So **don't migrate to chase throughput**
— the throughput levers are N-cars-in-one-world + sample efficiency.

## Scope of the change (what actually has to be ported)
- **Transport/bridge:** `ros_gz` replaces `gazebo_ros_pkgs` (ROS2 ↔ gz transport).
- **SDF / worlds / models:** gz-sim SDF (tag changes); **plugins rewritten** as gz-sim *system* plugins —
  the racecar joint controllers, the **ZED camera sensor**, and the **`WorldSwapper` hot-swap** all need
  porting.
- **ROS1 → ROS2 code:** `rospy`→`rclpy`, message/service types, launch (XML → Python launch), `use_sim_time`
  / `/clock`. `deepracer-env` is heavily ROS1 (`rollout_agent_ctrl`, `sensors`, `reset/rules`, `world_swap`,
  object-avoidance) — a substantial port.
- **dr-gym side:** mostly unaffected — it consumes the env through Docker and the coupling is schema-only;
  only the env image + any ROS1 assumptions change.

## Effort / risk
**Large** (a multi-week-to-month effort): the entire sim stack (controllers, camera, reset rules, hot-swap,
OA) is ROS1 + Classic. High regression risk → the **scripted-baseline + `tests/test_env_contract.py` (W1)**
are the safety net to port behaviour against.

## Recommendation — phased, and reframed
1. **Treat it as a `[REAL]`/maintainability project, not a performance one.** EOL + car-stack alignment
   justify it; throughput does not.
2. If pursued: stand up a **minimal gz-sim + ROS2 racecar** (camera + drive + one track) behind the *same*
   gym `DeepRacerEnv` interface (so dr-gym is unchanged), validate with the scripted baseline + contract
   tests, then port world-swap / reset-rules / OA.
3. **Re-benchmark RTF on current gz-sim (Harmonic) before committing** — it may have improved since the
   cited tests; verify empirically rather than assume (the device sweep harness can be pointed at it).

## Sources
- [Ignition vs Gazebo (RTF comparison)](https://www.allisonthackston.com/articles/ignition-vs-gazebo.html)
- [Migration Gazebo Classic → Ignition with ROS 2 (ros_gz bridge, plugins)](https://www.blackcoffeerobotics.com/blog/migration-from-gazebo-classic-to-ignition-with-ros-2)
- [Migrating from Gazebo Classic to Gazebo Sim — practical guide](https://ibrahimmansur4.medium.com/migrating-from-gazebo-classic-to-gazebo-sim-a-practical-guide-804af2011011)
