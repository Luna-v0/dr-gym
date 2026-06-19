# ONNX → OpenVINO IR Support — Status Report

_Last updated: 2026-06-19_

Status of the two-gate ONNX-support work for the AWS DeepRacer model-optimizer pipeline.
Full design: `~/.claude/plans/tingly-mixing-plum.md`. Key code lives in
`gym_dr/optimize.py` and `scripts/smoke_test_*.py`.

## Context

dr-gym trains continuous-PPO policies (SB3 / PyTorch). The car deploys OpenVINO **IR**
(`.xml`/`.bin`). This work proves the `.onnx → IR` path is numerically faithful (Gate 1)
and that the real SB3 policy survives PyTorch → ONNX → IR with its deterministic action
preserved (Gate 2). Two key environment realities shaped the implementation:

- The AWS device-stack `model_optimizer_node` is **not** in this checkout, so Gate 1 was
  landed as a standalone, ROS-free module (`gym_dr/optimize.py`) that ports to the real
  node later.
- Nothing was installed; runtimes were added via uv into **two dedicated py3.8 venvs**
  (modern OpenVINO 2024.4 + TF/ORT/tf2onnx; legacy OpenVINO 2021.4.2 = device `mo` + IE),
  kept separate because legacy OpenVINO's NumPy pin conflicts with TensorFlow.

## ✅ Delivered (plan core — done & passing)

| Plan item | Status | Notes |
|---|---|---|
| Stage 0 — two uv venvs | ✅ | `.venv-ov-modern`, `.venv-ov-legacy` (setup documented in `pyproject.toml` `optimize` group) |
| Stage 1a — `gym_dr/optimize.py` | ✅ | `onnx_to_ir`, `run_ir`, legacy/modern backends, `force_fp32`, CLI |
| Stage 1b — Smoke Test 1 (TF→ONNX→IR) | ✅ **PASS** | All 4 runtimes agree ~1e-8; argmax identical (`scripts/smoke_test_1_pipeline.py`) |
| Stage 2 — Smoke Test 2 (SB3→ONNX→IR) | ✅ **PASS** | Action-mean parity ≤1.7e-4 on real model + real sim frames; level (i) raw mean + level (ii) post-processed (`scripts/smoke_test_2_parity.py`) |
| Tests — `tests/test_optimize.py` | ✅ | `importorskip`-guarded: runs in OV venvs, skips cleanly in dr-gym venv |
| pyproject `[optimize]` group | ✅ | Dependency + two-venv setup documented |
| Bonus — `scripts/collect_sim_obs.py` | ✅ | Captures real Gazebo camera frames for Gate 2 |

### Key finding: OpenVINO auto-bf16
On AVX512_BF16 CPUs, OpenVINO silently runs an FP32 IR in **bf16** (~3e-4 to 1e-2 error),
and *both* 2021.4 and 2024 do it — which masquerades as a "toolchain precision floor."
`run_ir(force_fp32=True)` (default) disables it (`INFERENCE_PRECISION_HINT=f32` modern /
`ENFORCE_BF16=NO` legacy); with it off, all runtimes match PyTorch to ~1e-8.
**Deployment implication:** set inference precision deliberately on-device (FP32 on the
Atom CPU; FP16 on the Gen9 iGPU) — do not trust the default.

## 🔧 In-flight, blocked

**Closed-loop precision experiment** — `scripts/precision_experiment.py` (host/container
dispatch) is written but **not run**. It drives the trained oval policy through the sim
under three precisions (PyTorch-fp32 / onnxruntime-fp32 / onnxruntime-fp16) and compares
**task metrics** (`dr/ep_max_progress`, off-track rate, reward) — the only way to answer
whether FP16 *inference* error compounds in the closed loop (per-step parity is open-loop
and can't).

- **Blocker:** the FP16 ONNX export hits an `onnxconverter-common` bug on the SB3 graph's
  leading uint8→float `Cast` (`Type Error … float16 vs float`). Fix: exclude that Cast via
  `op_block_list`, or use onnxruntime's float16 conversion tool.
- **Then:** run the experiment and record results.

## ⏸️ Deferred (intentionally, per plan)

- **Stage 3 — `device` param (CPU/GPU/MYRIAD) + `CACHE_DIR`** in the inference path, an
  **FP16 IR deployment variant**, and an **on-device CPU-vs-GPU benchmark**.
- **On-car `customrl` inference node** + ROS panic-stop / watchdog
  (see `docs/physical-car-integration-notes.md`).
- **Action-units → `ServoCtrlMsg` mapping code** — Smoke Test 2 prints the required
  rescale (engineering units deg·m/s → angle [-1,1], throttle [0,1]) but no on-car mapping
  code exists yet.

## ⚠️ Open gaps surfaced during this work

1. **Legacy version mismatch** — used pip's OpenVINO **2021.4.2**; the device runs
   **2021.1.110**. Close enough for the gate; final validation should use the device image.
2. **FP16 / GPU activation-headroom check** — unrun. `normalize_images=False` (raw 0–255
   input) makes early activations large; on the Gen9 iGPU's native-fp16 compute they could
   approach the fp16 ceiling (65504). Worth measuring before committing to a GPU/FP16 path
   (CPU/FP16 IR is safe — weights fp16, compute fp32). Fix if tight: add `/255` or stay on
   the FP32-CPU path.
3. **Porting `optimize.py` into the real `model_optimizer_node`** — still open; the module
   was deliberately built ROS-free so this is a clean lift later.

## Suggested next step

Fix the FP16 `Cast` export bug and run the closed-loop precision experiment — it closes the
one in-flight item and produces real data on the FP16-inference compounding question.
