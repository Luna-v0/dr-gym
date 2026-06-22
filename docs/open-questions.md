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

**Running:** D3 held-out validation (GPU/CUDA, ~20 h).

**Queue (no input needed):** build **ADR** (auto DR — `docs/reports/domain-randomization.md`); the
**deepracer-env edits** (`sim_time` exposure, random-start/direction, episode-lifecycle config — signed
off); **SB3 PID-Lagrangian trainer** (after D9, against the new contract); **deepracer-utils** compat +
chart port; TFLite/ExecuTorch + **int8 quant** on the Pi; **software-render multi-instance + N-cars**
throughput sweeps (need a free GPU); **perception net** + asymmetric critic (W-perception); the
**`deepracer-deploy`** repo (on-car node + ServoCtrlMsg rescale + watchdog, ADR-0001).

**Open decision:** **D9** — safe-RL backend (hybrid vs full OmniSafe), see `docs/questions-for-maintainer.md`.

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

### Q4 🔴 `[DISS]` Safe-RL cost signal definition
- **Known:** the 26-key `reward_params` (`deepracer-env .../agent_ctrl/constants.py:RewardParam`) offers
  candidate cost terms: off-track, crash, steering/jerk, near-edge time.
- **Next:** decide and document the cost + limit before implementing PPO/PID-Lagrangian (W-saferl).

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
