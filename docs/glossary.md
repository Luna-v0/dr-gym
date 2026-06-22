# Glossary

Project-specific terms first, then the RL / safe-RL vocabulary used across the docs and reports.

## Project terms

| Term | Meaning |
|---|---|
| **chunk / `chunk_steps`** | A contiguous block of training on one world before the next track swap. |
| **rotation** | One full pass through the world list (`SequentialRotation`). |
| **world / track** | A Gazebo world. The first is loaded at container startup via `WORLD_NAME`; later ones via `DeepRacerEnv.set_world` (hot-swap, no restart). |
| **world strategy** | The schedule object (`gym_dr/worlds.py`): `SequentialRotation` or `OrderedSplit`. |
| **held-out world** | A world in `OrderedSplit.eval_worlds` not in `train_worlds`; used to measure generalization. |
| **`phase`** | Trace column: `train` vs `eval` episode. Lets analysis separate the two. |
| **`sim_time`** | The simulation clock (from `/clock`). The trace **join key**, not wall time (RTF drift desyncs wall clocks). |
| **trace tiers** | Tier-1 scalar per-step rows (the simtrace replacement); Tier-2 high-bandwidth/event streams (camera, lidar, swaps); Tier-3 episode aggregates. See `docs/trace-contract.md`. |
| **`reward_params`** | The 26-key dict deepracer-env passes (deep-copied) to the reward callback each step. The actor never sees most of it — only the camera obs. |
| **reward vs `eval_reward`** | `reward` is the (HPO-swept) training reward; `eval_reward` (default `progress_safe`) is a fixed yardstick used only during eval episodes. |
| **frame stack** | `VecFrameStack(4)` — the policy sees the last 4 grayscale frames, `Box(4,120,160) uint8`. |
| **action units** | Engineering units: steering in **degrees**, speed in **m/s** — *not* normalized `[-1,1]`. The on-car servo path must rescale (see `docs/physical-car-integration-notes.md`). |
| **`ActionBounds` / `GrayscaleObs`** | dr-gym env wrappers (`gym_dr/envs/wrappers.py`): clip the action box; convert RGB→grayscale. |
| **RTF** | Real-time factor — Gazebo sim speed multiplier (`training.rtf_override`). |
| **`run_group`** | MLflow tag grouping the chunks/trials of one experiment or study. |
| **`artifacts/<name>/`** | All output for a run (models, checkpoints, `run_config.json`, TB, trace). See `docs/artifact-layout.md`. |
| **IR** | OpenVINO Intermediate Representation (`.xml` + `.bin`) — the on-device inference format. |
| **`force_fp32`** | `run_ir` flag that disables OpenVINO's silent bf16 auto-cast on AVX512 CPUs (the precision-floor gotcha). |

## RL / safe-RL vocabulary

| Term | Meaning |
|---|---|
| **PPO** | Proximal Policy Optimization — the on-policy actor-critic algorithm in use (SB3). |
| **GAE (`gae_lambda`)** | Generalized Advantage Estimation — bias/variance knob for advantages. |
| **explained variance** | How well the value function predicts returns; ≈0 or negative ⇒ a broken/uninformative critic. |
| **approx KL / clip fraction** | PPO update diagnostics — policy step size and how often the ratio is clipped. |
| **entropy / entropy collapse** | Action-distribution spread; premature collapse ⇒ under-exploration. |
| **CMDP** | Constrained Markov Decision Process — maximize reward subject to expected **cost** ≤ limit. |
| **cost signal** | The constrained quantity (candidates: off-track, crash, jerk, near-edge time). As load-bearing as the reward. |
| **PPO-Lagrangian / PID-Lagrangian** | Constrained-PPO methods: a Lagrange multiplier on the cost, updated by dual ascent (PPO-Lag) or a PID controller (PID-Lag, damps oscillation). |
| **curriculum learning** | Ordering tasks/tracks by difficulty (automatic: success-gated, or Prioritized Level Replay). Addresses **task/track generalization**. |
| **domain randomization (DR) / ADR** | Randomizing sim parameters (noise, lighting, actuator drift); ADR expands the ranges as the agent succeeds. Addresses **environmental robustness**. |
| **generalization gap** | mean train-track performance − mean held-out performance. The headline generalization metric. |
| **task vs environmental robustness** | Two *separate* axes: track generalization (curriculum) vs sim-shift robustness (DR). Separate knobs, separate evaluation. |
| **privileged / asymmetric actor-critic** | The critic may consume full sim state during training; the actor consumes only deployable features. Bridge to the real car. |
| **perception net** | A supervised net mapping camera (or a stack) → "friendly" features (lateral offset, heading error, edge distances, speed), trained on sim ground truth. |
| **sim-to-real** | Transferring a sim-trained policy to the physical car; matters for `[REAL]`, not `[DISS]`. |
