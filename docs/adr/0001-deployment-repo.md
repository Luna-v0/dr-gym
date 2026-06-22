# ADR-0001 — On-car deployment lives in a new `deepracer-deploy` repo

- **Status:** Proposed (awaiting maintainer — decision D6)
- **Date:** 2026-06-21 · **Tags:** `[REAL]`

## Context
The on-car path (ROS node loading an OpenVINO IR + the perception net, publishing `ServoCtrlMsg`, with a
watchdog) does not exist yet — only `gym_dr/export.py` + `gym_dr/optimize.py` (ONNX→IR, 2 passing gates)
and `docs/physical-car-integration-notes.md`. It needs a home. The car runtime is **ROS 2 + OpenVINO on
the physical car**, a different stack/cadence from `dr-gym` (py3.8 training) and `deepracer-env` (ROS1
sim). The training↔deploy seam is already just two artifacts: the **IR model** and `model_metadata.json`.

## Decision
Create a new lightweight repo **`deepracer-deploy`** for the on-car inference node, the `ServoCtrlMsg`
rescale, the perception-net inference, and the watchdog. It depends only on the contract artifacts — **not**
on `dr-gym` or `deepracer-env` at runtime.

## Consequences
- (+) Clean lifecycle/dependency separation; the training stack stays lean; the car repo stays minimal.
- (+) Forces the action/units + IR-I/O **contract** to be explicit (ADR-0003).
- (−) A fourth repo to maintain; the rescale logic must be kept in sync via the shared contract (R1).
- Alternatives rejected: into `deepracer-env` (couples ROS1 sim with ROS2 car) or `dr-gym` (couples
  training deps with the car runtime).
