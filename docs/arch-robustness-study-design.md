# Architecture-robustness study — design dump (the "mastered-out" study)

> Exhaustive design dump from the 2026-06-29 session. The maintainer will refine.
> This is the MLP-vs-LSTM(-vs-prev-action) robustness study. The running version was
> STOPPED 2026-06-29 to make way for the deepracer-env car-uncapping upgrade (see
> `docs/deepracer-env-upgrade-handoff.md`). Rebuild this study AFTER the upgrade lands.

---

## 1. The scientific question

On a task with an **unobservable, per-episode-constant steering bias** (a hidden latent →
a POMDP for a memoryless policy), plus per-step feature noise, **how much memory does the
policy architecture need to stay robust?** Compare points on the memory spectrum:

| Arm | Memory of the hidden bias | Cost | Expectation |
|---|---|---|---|
| **MLP (blind)** | none — can't infer it (true POMDP) | cheapest | flat-lines under bias (the baseline) |
| **MLP + prev-action** | 1-step feedback (sees last commanded action + resulting state → instantaneous system-ID) | ~free (concat last action to obs) | partial, noisy compensation — the interesting middle |
| **LSTM** (sb3-contrib RecurrentPPO) | full history → integrates the estimate | heaviest | best compensation |

All arms: **asymmetric actor-critic** (actor sees the NOISED feature vector, critic sees
the CLEAN one), **no frame stacking** (the LSTM *is* the memory; the MLP is the control).
Decision history: maintainer first said "MLP and LSTM just", then on 2026-06-29 added the
**prev-action arm back** (cheapest + most scientifically interesting middle point).

### Why this is "Option A"
The per-episode bias is left **unobservable** on purpose — it models realistic actuator
miscalibration the policy must be robust to without being told. Keep it unobservable.

---

## 2. THE BIG DIRECTIVE (2026-06-29): max-parallelism big-rollout DR

The maintainer wants the OpenAI-ADR structure: **one policy, a big diverse rollout, many
DR'd envs per gradient step.** Diversity *within a single PPO batch* drives robustness —
NOT raw throughput. Explicit preferences:

- **Single Optuna parallelism** (`GYM_DR_HPO_PARALLEL=1`, trials sequential) — NOT many
  concurrent few-car trials.
- **Max `n_cars` per trial** — each car a different track + different DR draw → one big
  diverse rollout → one PPO update. Accepts slightly lower total throughput for this.

See memory `dr-max-parallelism-dr-rollout`. This **reverses** the current study, which is
the opposite (n_cars=1, 4 concurrent trials). The current single-car design exists only
because single-car can `set_world` for cheap in-loop held-out eval; the directive
supersedes that — go multi-car, and use the parallel multi-car eval (§4).

### The hard limit this runs into
One Gazebo world ≈ 8–16 cars before single-thread ODE RTF tanks (see upgrade handoff §4).
The launch caps at 8 today. **This study is BLOCKED on the deepracer-env upgrade** that
uncaps cars + measures the real RTF cliff. True OpenAI-scale (100s–1000s envs) needs a
multi-sim→one-policy VecEnv aggregation layer that dr-gym does not yet have.

---

## 3. Track universe — use ALL distinct layouts, proper 3-way split

**Current study uses only 26 of ~65 distinct layouts (40% coverage) and has NO val split.**
Audit (2026-06-29): 134 registry entries → **68 distinct base layouts** → **65 usable sim
layouts** (excluding 3 physical-reserved). The other 108 registry entries are mostly
direction/visual DUPLICATES (`_ccw`/`_cw`/`_eval`/`_building`/`_f1`/`_wide`/`_carpet`/…),
but ~39 are genuinely distinct layouts we simply weren't using (april, june, may, october,
september, arctic, penbay, red_star, Austin, Aragon, Belille, Singapore, Spain, Vegas,
China, …).

**Target:** all 65 distinct layouts, **70/15/15 train/val/test ≈ 45 / 10 / 10**, by base
layout (variants stay with their base — no leakage, same rule as the camera dataset split,
see memory `camera-cnn-dataset-run`). **Physical tracks reserved** for the final sim2real
number (never train/sim-eval): `reinvent_base`, `reInvent2019_track`, `Oval_track`.
**Val** drives Optuna + early-stop; **test** is touched once at the very end.

De-dup helper (run against `gym_dr.TRACKS`): strip suffixes `_ccw|_cw|_mirrored|_eval|_Eval`
and visual `_(building|f1|wide|carpet|concrete|wood|jeremiah)`, group, pick one canonical
per base. (`_open` vs `_pro` are DIFFERENT layouts — keep both.)

---

## 4. Eval — parallel multi-car held-out (maintainer's choice)

Current single-car eval (`callbacks.py:466-485`): every 40k steps the ONE sim hot-swaps
(`set_world`) to each held-out world sequentially, 5 deterministic episodes each
(8 worlds × 5 + 1 train × 5 = 45 episodes, serial). Objective = mean held-out
clean-completion.

**Wanted:** **parallel multi-car held-out eval** — N cars each pinned to a distinct
held-out (val) track, evaluated together (the "12 tracks at once, then the rest" idea).
Feasible once training is single-trial (only 1 training sim running leaves headroom for an
eval sim). Multi-car has no `set_world`, so it's a **dedicated eval sim** (or the upgrade
adds per-car world-swap and it collapses into one mechanism).

### Cadence conflict (must resolve at rebuild)
Per-chunk parallel multi-car eval alongside training was infeasible at 4 concurrent trials
(laptop hit load 84 on 22 cores with 4 single-car sims alone). With single Optuna
parallelism there's headroom, but still decide:
- **(a) Search-cheap + finalist-parallel** (was my recommendation): cheap proxy eval
  in-loop for pruning; parallel multi-car held-out eval on top-K finalists' checkpoints as
  the deciding number. Don't waste expensive eval on losers.
- **(b) In-loop parallel every chunk** as the live Optuna objective (now affordable at
  single Optuna parallelism). "Real" objective from step one.

Recurrent-aware eval already exists: `gym_dr/trainers/sb3/callbacks.py::_eval_policy`
threads LSTM state + episode_starts (SB3's `evaluate_policy` doesn't carry LSTM state).

---

## 5. HPO framing decision (open)

Single Optuna parallelism + 8–16-car multi-hour trials run sequentially = a 40-trial HPO
takes days. Two ways:
- **(a)** keep it an HPO but cut to ~12–15 trials, or
- **(b)** reframe: short shared-HP search → **3 definitive big runs** (one per arm) for the
  actual comparison. (My lean: (b) — the arms ARE the question.)

---

## 6. Concrete config state (where the code is)

Experiment file: `experiments/oracle_hpo.py` (NAME=`arch_robust_hpo`; was renamed from the
oracle's `oracle_hpo`). Current (pre-redesign) state:
- `n_cars=1` (← must become max), `GYM_DR_HPO_PARALLEL=4` (← must become 1).
- `TRAIN_WORLDS` = 18, `EVAL_WORLDS` = 8 (← must become the 45/10/10 split over 65).
- Task: `ADR(steering_bias=BIAS=10.0, speed_bias=0.5, feature_noise=Range(0,0.2),
  steering_noise=Range(0,3), …)` — fixed across trials so the arm comparison is fair.
- `search_space(trial)`: `arch ∈ {mlp, lstm}` (← add `mlp_prevaction`). mlp →
  `AsymmetricActorCriticPolicy`; lstm → `asymmetric_recurrent_policy()` +
  `lstm_hidden_size ∈ {64,128,256}` + `enable_critic_lstm`. `frame_stack=1` both.
- `eval_path_plots=True` (flipped on 2026-06-29 so TB IMAGES shows trajectory overlays).

Arch policies: `gym_dr/asymmetric.py` —
- `AsymmetricActorCriticPolicy` (MLP; actor reads `obs["actor"]` noised, critic
  `obs["critic"]` clean via `KeyExtractor`; vf extractor swapped + optimizer rebuilt in
  `_build`). Validated.
- `asymmetric_recurrent_policy()` — lazy factory returning a RecurrentActorCriticPolicy
  subclass. **WART:** returns a dynamic `<locals>` class (qualname has `<locals>`, not
  picklable). Consider making it a stable module-level class during the rebuild.

### prev-action arm — implementation note (not yet built)
Append the previous action (2D: steering, speed, normalized) to BOTH obs keys (actor +
critic — the action is the policy's own output, known exactly, not noised). Needs a small
per-car obs wrapper that tracks last action (zeros at reset) and concatenates it; grows the
obs space by 2 dims (the MLP input grows automatically). With multi-car it's per-car.
No new algorithm; arm name `mlp_prevaction` → `name="ppo"` + AsymmetricActorCriticPolicy +
the wrapper enabled via a config flag (e.g. `observation.include_prev_action`).

---

## 7. Fixes already landed this session (don't redo)

- **MLflow experiment-creation race** (`gym_dr/mlflow_utils.py::_set_experiment_racesafe`):
  concurrent HPO workers raced `set_experiment` on the file store; losers raised
  `MlflowException ... already exists` → trial FAILED (lstm-skewed by timing). Fixed:
  retry + bind-by-id. This was THE cause of "all lstm trials fail" — NOT an lstm bug
  (verified: lstm trains fine in isolation).
- **`eval_path_plots=False→True`** in `oracle_hpo.py` (TB IMAGES were empty by config, not
  a dead server).
- sb3-contrib RecurrentPPO wired: `algorithms.py::import_algos` registers `recurrent_ppo`
  (optional import); `docker_runner.py` bind-mounts host `sb3_contrib` into the container.

---

## 8. Rebuild checklist (post-upgrade)

1. [ ] Uncap cars (deepracer-env upgrade) + measure RTF-vs-n_cars cliff → pick `n_cars`.
2. [ ] `experiments/oracle_hpo.py`: `n_cars=<measured max>`, `GYM_DR_HPO_PARALLEL=1`.
3. [ ] Track split: 65 layouts → 45/10/10 by base layout; physical reserved.
4. [ ] Add `mlp_prevaction` arm + the prev-action obs wrapper.
5. [ ] Multi-car big-rollout config (each car: different track + DR; tune `n_steps` so
       `n_steps × n_cars` is a big batch).
6. [ ] Parallel multi-car held-out eval (decide §4 cadence a/b) — verify recurrent-aware.
7. [ ] Decide §5 HPO-vs-definitive-runs framing.
8. [ ] Make `asymmetric_recurrent_policy()` a stable module-level class (picklability).
9. [ ] Verify nonzero per-car `dr/ep_max_progress` (phantom-car / reward-clobber trap).
