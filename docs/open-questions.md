# Open questions & decision log

A living log. **Read it at the start of every session; update it as you learn.** This is cross-session
memory for humans and agents. Each entry has a status, the question, what we know, and the next step.
Tags: `[DISS]` (simulator-only dissertation) · `[REAL]` (physical car) · `[BOTH]`.

Status key: 🔴 open · 🟡 in progress · 🟢 resolved (keep, with the answer).

**Maintainer's success criterion (the bar everything is measured against):** a policy that reaches the
**end of every held-out eval track without leaving the track**, at a **reasonable (non-minimum) speed**.

The full roadmap this log serves lives in the approved plan
`~/.claude/plans/based-on-this-plan-compressed-fox.md` (workstreams W0–W3, W-arch, W-curriculum, W-saferl,
W-dr, W-perception, W-tooling, W-dash, W-throughput, W-deploy, W-extensibility, W-ros2).

---

## Status & next-actions (updated 2026-06-22)
**Shipped on `main`:** P1 clean-completion eval · P3 action-normalization · D5 `StochasticCurriculum` ·
safe-RL graded *risk* costs (`gym_dr/costs.py`) · static DR noise wrappers · speed-coloured eval charts ·
**framework-agnostic trainer contract** (author your own algorithm — `docs/trainer-contract.md`) · env
baselines + contract tests · ONNX→IR deploy + on-car engine/model-size studies · throughput/device-sweep
tooling. Reports in `docs/reports/`.

**Key empirical findings:** RTF caps ~4.5–5× (`rtf_override` ignored) and separate-container parallelism
doesn't scale; training is **inference-device-bound, not render-bound** (CUDA ≈5× CPU; software-render =
same speed ⇒ GPU passthrough for rendering is unnecessary); the **Pi runs even a 24 M-param net in ~35 ms,
<200 MB** — memory is not the constraint, latency is; **onnxruntime** beats OpenVINO-ARM on the Pi.

**Maintainer-set objective (2026-06-22):** the first real target is an **end-to-end PURE-PPO** run
(architecture + Domain Randomization + curriculum, **no Lagrangian**) — but **train the feature-distillation
perception model first**, supervised on sim data. Sequencing (maintainer chose **let D3 finish first**):
1. **D3 runs to completion** = the pure-PPO + curriculum **no-DR baseline** (the clean ablation point).
2. **Phase 1** — collect sim data (`scripts/collect_perception_all.sh`, 8 worlds, DR-perturbed inputs +
   ground-truth labels) → fit `PerceptionNet` (`experiments/train_perception.py`) → per-feature MAE table.
3. **Phase 2** — `experiments/end_to_end_ppo.py` (D3 config + `DomainRandomizationConfig(adr=True)`),
   optionally with the perception net as the actor front-end (decided from the MAE). **All scripts READY.**

**Feature-based-policy decomposition (maintainer, 2026-06-22 — `docs/reports/feature-based-policy.md`):** split
the system at the feature vector — `π(features)` is state-based (sim==real by construction), `g(camera)→features`
is the **only** sim2real-sensitive part. Two added tests (backlog, gated on Phase 1):
**Test 1** oracle-feature PPO (train on ground-truth `ALL_FEATURES` → is the feature set sufficient? + the
teacher; train π with feature-noise calibrated to g's MAE), **Test 2** `π(g(camera))` = the perception
penalty. Enabling helper `enrich_reward_params` (derived features as reward-fn args / feature obs) built+tested.

**Running (2026-06-23):** the **reward search** (`experiments/reward_search.py`, Optuna, 16 trials × 500k
steps, 2 workers, ~20 h) — protected by the new **liveness watchdog** (`docs/reports/d3-hang-postmortem.md`,
built + tested; D3's hang can no longer stall a run). Searches the progress-normalized reward families
(offline filter narrowed the space — `docs/reports/reward-search.md`).

**Planned next experiment (chained):** when the search finishes, run **Phase 2**
(`experiments/phase2_from_search.py`) — it loads the **best reward** from the study and runs the full 4M-step
end-to-end PPO with it **+ DR/ADR + random valid-start** (the patched sim via `GYM_DR_DEEPRACER_ENV_SRC`).
Rationale: the offline filter showed the fast-crash is partly an *optimization* problem, so the fix is the
*combination* — best reward (search) **plus** DR/random-start (state coverage to learn cornering). Script is
built + import-verified; launches on one command when the search completes.

**D3 outcome (closed):** ran to 3.5M/4M then **hung** (gzserver wedged during the held-out eval world-swap;
recovery only caught *crashes*, not *hangs* — now fixed by the watchdog). Baseline verdict conclusive: pure
PPO + curriculum, no DR ⇒ fast-crash (~28% progress, 0 completions).

**Built while D3 runs (2026-06-22):**
- **deepracer-env random valid-start + random direction** (`RANDOM_START`/`RANDOM_DIRECTION` reset modes) —
  env code + `time_trial.py` wiring + enabled in `end_to_end_ppo.py`. **Deployed via bind-mount**
  (`GYM_DR_DEEPRACER_ENV_SRC`, `gym_dr/docker_runner.py`) — no image rebuild needed, validated in the base
  image. (Multi-view/contrastive SSL dropped — maintainer chose supervised-only.)
- **Rosbag→perception join core** (`scripts/bag_to_perception.py` + 7 tests). Precondition found: the trace
  must be extended with the derived `ALL_FEATURES` columns (task #18) since it lacks `is_left_of_center`/
  `waypoints` to recompute labels offline.

**Done today (2026-06-22):**
- **FSRL PPO-Lag VALIDATED on Safety-Gymnasium** (`.venv-safe`, Py3.10, 20 epochs): reward↑ to **17.5** with
  cost↓ to **9.0 ≤ limit 10** — the ideal CMDP outcome ⇒ D9 backend trusted. Fixed the integration bug:
  Safety-Gym's CMDP **6-tuple** → a `_CostToInfo` wrapper (the **same `info["cost"]` contract** as our
  DeepRacer `CostInfoWrapper`). `FsrlTrainer` finalized with verified kwargs + a real CNN camera path
  (`PPOLagrangian` policy + Tianshou CNN `preprocess_net` + separate reward/cost critics).
- **Software-render multi-instance throughput sweep:** **2-worker sweet spot ~83 steps/s (+50%)** with
  sw-render+GPU-inference; 4 workers collapse (CPU oversubscription). First lever to beat the single-instance
  ceiling — `docs/reports/throughput.md`.
- **W-perception built** (`gym_dr/perception.py` net + `perception_targets`, `scripts/collect_perception_data.py`,
  `experiments/train_perception.py`, `tests/test_perception.py` 17✅) — `docs/reports/perception.md`.
- **Asymmetric architecture feature study** (`docs/reports/asymmetric-architecture.md`): the
  deployable/privileged partition is now concrete code — `perception_targets` (actor, 6) vs `privileged_state`
  (critic-only extras: progress, curvature-ahead, object/contact flags, 6) vs `critic_state` (concat, 12);
  distinguishes asymmetric-AC (Pinto) from teacher→student distillation (Learning-by-Cheating); notes
  speed/yaw are *proprioceptive* (stay on the actor).
- **W-dash:** track-overlay images already render to TB with **true route `.npy` borders**
  (`test_eval_path_plots_logged_as_tb_images` ✅); added the live **`eval/generalization_gap`** +
  `eval/train_clean_completion_rate` scalars in `MultiWorldEvalCallback` (one extra train-world eval).

**Queue (no input needed):** **ADR ✅ built**; **cost-logging ✅ built**; **trainer contract ✅ built**;
**deepracer-env edits** (`sim_time` exposure, random-start/direction, episode-lifecycle — signed off);
finalize the **FSRL Tianshou CNN** for camera obs + DeepRacer constrained run (gated on D3's `dr/ep_mean_cost`);
**deepracer-utils** compat + chart port; TFLite/ExecuTorch + **int8 quant** on the Pi; **N-cars-in-one-world**
throughput sweep; **perception data collection + supervised fit** (need a free Gazebo); the
**`deepracer-deploy`** repo (on-car node + ServoCtrlMsg rescale + watchdog, ADR-0001).

**Open decision:** none blocking — **D9 resolved** (FSRL PPO-Lag, validated). Next maintainer touchpoint is
after D3 + FSRL Safety-Gym finish (read the generalization gap + the cost budget).

## Active questions

### Q1 🟡 `[DISS]` Why doesn't a PPO policy generalize across tracks? (highest priority)
- **Known:** `time_trail_hard_track_trial_18` trained on **`Oval_track` only** (legacy `worlds` config,
  `rotations: 1`, no `world_strategy`), so it had **no held-out eval worlds** — generalization was never
  trained for nor measured. HPO sampled a very low `learning_rate` (2.05e-5) over only 250k steps;
  `speed_low` was 1.0 (crawling already ruled out). The held-out infra (`OrderedSplit` +
  `MultiWorldEvalCallback`) exists but this run bypassed it.
- **W2 finding (2026-06-21):** trial_18 is **not really a generalization failure** — it's an
  **under-trained actor** (HPO `lr=2.05e-5`; `clip_fraction≈0`, `approx_kl≈1e-4`, per-dim `std` frozen ≈1.0)
  with **un-normalized actions** (steering explores only ~±1° over ±30°) scored by a **misleading eval
  metric** (`progress_safe` dominated by `speed²`; off-track penalty is -1.0, not the -100 its docstring
  claims). It never mastered even the training Oval (~60% progress, ~32% off-track) and **no held-out was
  ever measured** (legacy `worlds`, `rotations:1`). Full report: `docs/reports/q1-generalization.md`.
- **Highest-leverage fix (sign-off-gated):** normalize action space to [-1,1] + sane LR + fix the eval
  metric, then train `OrderedSplit` multi-track with held-out eval and report the generalization gap (W3).
- **Scope-review update (2026-06-21):** curriculum + multi-track + LR-sweep + a 16M-step run were *already
  tried* and didn't yield reliable generalization — so "no variety" is **not** the cause. The 21-trial
  `tt_multiworld` HPO was under-budgeted (600k/trial ⇒ all under-trained, prog <50%); high LR destabilized
  (`std`→63, prog→0.2), the curriculum collapsed (`std`→0.03, stuck 28%), and the **held-out eval metric
  barely discriminates** (~2–3k reward at both 0.2% and 50% progress). Re-scoped to: **P1 trustworthy
  eval/metric → P2 affordable learning (throughput/sample-efficiency) → P3 optimization robustness (action
  normalization, collapse guards) → P4 generalization levers (curriculum/DR/perception)**. Full report:
  `docs/reports/scope-review.md`.
- **Next:** confirm the re-prioritization with the maintainer; then P1 (eval metric + held-out protocol),
  with W1 scripted baseline as a cheap parallel env-soundness gate.

### Q3 🔴 `[DISS]` What actually terminates an episode, and is SB3 truncation-bootstrapping correct?
- **Known:** `gym_dr/envs/time_trial.py` builds `DeepRacerEnv` with **no `config` dict** and dr-gym adds
  **no `TimeLimit`** wrapper, so episode length is governed entirely by deepracer-env reset rules and their
  defaults (`is_continuous`, `number_of_trials`, `MAX_STEPS`, `CHANGE_START`). Unverified empirically.
- **Next:** instrument episode endings in W1/W2; confirm `terminated` vs `truncated` semantics flow
  correctly into SB3's value bootstrapping.
- **2026-06-23 update:** the D3 hang (`docs/reports/d3-hang-postmortem.md`) shows the *cost* of having no
  explicit episode cap — a never-terminating eval episode on a long track can stall the run. Defaults are
  `MAX_STEPS=10000` (deepracer-env); add an explicit, smaller **eval step cap** as defense in depth (tracked
  with the liveness-watchdog fix).

### Q4 🔴 `[DISS]` Safe-RL cost signal definition
- **Known:** the 26-key `reward_params` (`deepracer-env .../agent_ctrl/constants.py:RewardParam`) offers
  candidate cost terms: off-track, crash, steering/jerk, near-edge time.
- **Built:** cost is graded *risk* (`gym_dr/costs.py`) + logged (`dr/ep_mean_cost`) so the limit can be set
  empirically from an unconstrained run; backend = FSRL `PPOLagAgent` (D9 ✅).

### Q5 🟡 `[BOTH]` Throughput architecture (= P2)
- **Known:** camera-observation RL is rendering-bound; one Gazebo world steps physics once for all robots
  and lighting/background are global. dr-gym currently trains a **single env** (`DummyVecEnv([one])`).
- **Maintainer measurements (2026-06-21):** 1 instance @ rtf 160× → ~43 fps; 7 instances @ 10× → ~3 fps
  total ⇒ the sweet spot lies between. Maintainer prefers **one Gazebo world with N cars** (multi-robot,
  namespaced) over 7 full Gazebo stacks, and asks whether a **newer Gazebo on ROS 2** would improve
  throughput / determinism / multi-robot scaling.
- **Next:** benchmark single-env RTF curve vs N-cars-in-one-world vs a few processes; find the sweet spot;
  evaluate the ROS2/Gazebo question (W-throughput + W-ros2). Open: does multi-robot break per-agent DR
  (world-global lighting) and high-RTF determinism?

### Q6 🔴 `[REAL]` Where should perception + on-car deployment live?
- **Known:** ONNX→OpenVINO IR pipeline exists in dr-gym (`gym_dr/export.py`, `gym_dr/optimize.py`, 2
  passing smoke gates), but the on-car ROS inference node + `ServoCtrlMsg` rescaling + watchdog do not.
  Cross-repo coupling today is deliberately **schema-only**.
- **Next:** W-arch decides: a new lightweight `deepracer-deploy` repo vs into `deepracer-env` vs `dr-gym`,
  and the minimal versioned shared-contract surface between repos.

### Q7 🔴 `[DISS]` Does the curriculum suffer catastrophic forgetting?
- **Known (maintainer hypothesis):** the sequential curriculum (`tt_curriculum_earlystop`) likely forgets
  earlier tracks as it trains later ones — a known failure of naive sequential multi-task RL. Its `std` also
  collapsed to 0.03 (premature determinism), stuck at ~28% progress.
- **Next (P4):** measure retention on earlier tracks during/after the curriculum; try interleaved/replayed
  track sampling rather than pure easy→hard ordering; add entropy/std-collapse guards.

---

## Resolved

### Q2 🟢 Is `_app.py` the intended default entry?
- **Answer (maintainer):** `_app.py` is an intentional, temporary rename — the maintainer removed `app.py`
  to confirm the stack runs *without* a root `app.py` present. The `app.py` convention in
  `docs/configuration.md` still stands; restore `app.py` when you want the one-command `python app.py`
  entry back. Not a bug, not drift.
