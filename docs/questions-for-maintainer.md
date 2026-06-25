# Questions / decisions for the maintainer (batched)

Working rule: I do all autonomous work first and batch decisions here. **All 7 opening decisions were
answered 2026-06-21** — current state below. New questions get added as I hit them.

## Answered (2026-06-21)

- **D1 ✅ — clean-completion is now the DEFAULT `eval_reward`.** Done in code (`gym_dr/config.py`); was
  opt-in, now replaces `progress_safe` (still importable). Protocol: `docs/eval-protocol.md`.
- **D2 ✅ — `normalize_actions` now defaults `True`.** Done (`gym_dr/action_space.py`). The policy acts in
  `[-1,1]`; sim still gets engineering units, but **exported ONNX now outputs `[-1,1]`** → on-car rescale
  note in `docs/physical-car-integration-notes.md`.
- **D3 ✅ — training run approved, scheduled LAST** (it saturates the machine). ~3–5M steps, sim-only split
  (physical tracks reserved, see D7). Script: `experiments/p1p3_validation.py`.
- **D4 ✅ — throughput: run for a while + watch for throttling.** Maintainer note: the fps/rtf values
  *fluctuate for up to ~30 min* before settling, so benchmark over a long window; and **record whether the
  set `rtf_override` is actually achieved or the sim throttles to a lower effective rate**. Prefer
  one-world/N-cars; spec before building.
  **Result (`docs/reports/throughput.md`):** the sim **throttles to ~4.5× regardless of `rtf_override`** (10
  and 160 → identical ~4.5× / ~54 steps·s⁻¹); separate parallel containers **don't scale** on one GPU (1→54,
  4→57 aggregate, 7→thrash 0.8) ⇒ **1 instance is optimal** today. Next: benchmark **N-cars-in-one-world**.
- **D5 ✅ — stochastic spaced-repetition curriculum.** Implemented as `StochasticCurriculum` (newer tracks
  favoured, older always revisited). Mastery-gated unlocking is a documented v2 (needs runtime feedback).
- **D6 ✅ — stock car + a custom car.** RPi reachable at `ssh deepracer@192.168.15.5` (motors off). Benchmark
  on-car inference engine: **memory, latency, thermals** (thermals after a cooling upgrade). On-car code →
  new `deepracer-deploy` repo (ADR-0001).
- **D7 ✅ — reserve the physical tracks for an out-of-loop evaluator.** `reInvent2019_track` + `Oval_track`
  (and similar) never go in sim train/eval; a hardcoded evaluator scores models on them outside the loop:
  `scripts/eval_physical_tracks.py`.

## Queued / in-flight (no decision needed — executing in order, D3 LAST)

- [x] **Car SSH baseline (D6):** done — Pi4 aarch64, Ubuntu 24.04, no inference runtime
  (`docs/reports/car-baseline.md`). On-car benchmarking is blocked on **D8** + confirm-before-install.
- [x] **Safety-Gym experiments (new):** FSRL PPO-Lag **validated** on `SafetyPointGoal1` (reward↑ to 17.5,
  cost↓ to 9 ≤ limit 10) — `docs/reports/safe-rl-backend.md`. DeepRacer constrained run gated on D3's budget.
- [ ] **D7 evaluator run:** needs a trained model → after D3.
- [ ] **D4 throughput sweep + N-cars-in-one-world spec.**
- [ ] **D3 validation training run (LAST):** launch + monitor real rtf/fps vs set; ≥ ~30 min before reading.

## New open questions
### D8 ✅ `[REAL]` Inference engine on the aarch64 custom car → **onnxruntime**
**Resolved (2026-06-22):** Pi benchmark — onnxruntime **11.7 ms** vs OpenVINO-ARM 13.6 ms
(`docs/reports/oncar-engine-comparison.md`); ~80 Hz max ≫ 15 Hz control loop, even on the heavy net.
onnxruntime chosen. int8 quantization + the 2-target (Pi vs stock x86) comparison tracked in that report.
The custom car is a Pi4 **aarch64** with **no runtime installed**, and our OpenVINO IR pipeline is
**x86-validated** (`docs/reports/car-baseline.md`). aarch64 options: **onnxruntime** (best ARM support),
**OpenVINO ARM CPU plugin**, or **TFLite**. **Decision:** which to target first? Then I install it on the
Pi (a system change — I'll confirm first), push the exported ONNX, and benchmark latency/mem/thermals.
**Update:** the comparison scripts + the exported `agent.onnx` are now on the Pi at `~/oncar_bench/`. Run
(you offered): `ssh deepracer@192.168.15.5` → `cd ~/oncar_bench && bash install_engines.sh && source
venv/bin/activate && python bench_engines.py --cooldown 60 --temp-limit 70`. It compares **onnxruntime +
OpenVINO** now (both run the ONNX), is thermally paced (cooldown + temp guard between engines), writes
`engine_benchmark.json`. **TFLite/ExecuTorch** need converted models — I'll generate + push `agent.tflite` /
`agent.pte` next so they join the comparison.

### D9 ✅ `[DISS]` Safe-RL backend → **adopt FSRL `PPOLagAgent`** (PID-Lagrangian PPO) — **VALIDATED**
**Resolved (2026-06-22):** maintainer chose FSRL (PID + PPO-Lagrangian joined, turnkey). Built:
`CostInfoWrapper` (cost→`info["cost"]`), `FsrlTrainer` scaffold, `scripts/validate_fsrl_safetygym.py`.
**Validated (2026-06-22):** FSRL PPO-Lag runs end-to-end on `SafetyPointGoal1-v0` in a separate
`.venv-safe` (**Python 3.10** — 3.11 can't build safety-gymnasium's pinned pygame 2.1.0) and shows correct
constrained behaviour (cost driven under the limit while reward re-balances). Fixed one bug: Safety-Gym's
CMDP **6-tuple** → a `_CostToInfo` wrapper (`info["cost"]`, same contract as our `CostInfoWrapper`). Full
write-up: `docs/reports/safe-rl-backend.md`. Next: finalize the Tianshou camera CNN + asymmetric cost-critic
→ DeepRacer constrained run with `cost_limit` from empirical `dr/ep_mean_cost` (D3 logging it now).
Original analysis ↓
**Recommendation (2026-06-22, `docs/reports/safe-rl-backend.md`):** OmniSafe would NOT make our custom-CNN
architecture changes cleaner — it'd re-port the whole stack into its abstractions. Proposed **hybrid**:
validate the algorithm on **Safety-Gymnasium with OmniSafe** (turnkey, trustworthy) + build an **SB3
PID-Lagrangian `Trainer`** for DeepRacer (reuses DeepRacerCNN / curriculum / trace; full architecture
control; adds a separate cost-critic + dual update). **Decide:** approve the hybrid, or go full OmniSafe?
(You already OK'd installing `safety_gymnasium` + rebasing the branch.)
Original question below ↓
The branch adds `SafetyDeepRacerEnv` (a CMDP 6-tuple with a `cost`) + a `safety_gymnasium` registry. SB3 PPO
(dr-gym's current trainer) ignores `cost`, so a real constrained run needs a safe-RL trainer. **Decision:**
adopt **OmniSafe** (PPO/PID-Lagrangian, validate on Safety-Gymnasium first — recommended) vs hand-roll
Lagrangian-PPO. Also: OK to install `safety_gymnasium` (+ the branch image) to validate?

## Resolved
See "Answered (2026-06-21)" above.
