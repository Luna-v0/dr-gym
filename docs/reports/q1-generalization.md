# Q1 — Why `time_trail_hard_track_trial_18` didn't generalize · `[DISS]` · 2026-06-21

## Question / goal
The maintainer reports this run "learned Oval but didn't generalize." Find the bottleneck with evidence,
and name the single highest-leverage fix. Separate "can't even master the training track" from "masters
training, fails held-out."

## What I did
- Read `artifacts/time_trail_hard_track_trial_18/{run_config,training_status}.json`.
- Summarized **every** MLflow metric for the run (`mlruns/983899726953030811/e4dde088c915426b835197e05146febe`):
  the SB3 PPO diagnostics, the `dr/ep_*` episode metrics, and `eval/mean_reward` (script:
  `/tmp/diagnose_trial18.py`).
- Read the reward code (`gym_dr/rewards.py`), the env factory + wrappers (`gym_dr/envs/time_trial.py`,
  `wrappers.py`), world scheduling (`gym_dr/worlds.py`), and the SB3 trainer/policy wiring.
- Ran the §4.5 killer list against the code.

## Evidence

**Run design** (`run_config.json`): trains on **`Oval_track` only** via the *legacy* `worlds` config
(`names:[Oval_track], rotations:1`) — **no `world_strategy`, no `eval_worlds`**. So eval ran on the *same*
Oval (the `SequentialRotation` fallback evaluates on the current training world). `learning_rate=2.05e-5`,
`total_timesteps=250k`, `n_epochs=5`, `frame_stack=4`, `reward=center_line`, `eval_reward=progress_safe`,
`speed_low=1.0` (crawling already ruled out).

**PPO diagnostics** (234 updates):

| metric | trajectory | reading |
|---|---|---|
| `explained_variance` | 0.003 → ~0.7 (peak 0.8) | **critic learns fine** |
| `clip_fraction` | ≈ **0** throughout (max 0.0096) | PPO's clip never engages → updates are tiny |
| `approx_kl` | ~1e-4 (max 2e-3) | policy steps are minuscule |
| `train/std` | 1.000 → 1.027 (rising) | actor distribution **not converging** — slightly *widening* |
| `entropy_loss` | -2.84 → -2.89 (flat) | no collapse, but no sharpening either |
| `learning_rate` | fixed **2.05e-5** | (HPO-sampled; very low) |

**Behavior** (`dr/ep_*`, 243 windows): `ep_max_progress` **plateaus ~60%** with no upward trend
(first 59 → last 50, mean 61); `ep_offtrack_rate ≈ 0.32` (flat); `ep_mean_speed` 1.1 → 2.3 (rises);
`ep_crash_count = 0`. `eval/mean_reward` ≈ 2700–3400, **flat across all 250k steps**; the only "eval"
numbers in the run are on Oval — **there is no held-out measurement anywhere**.

**Reward code:**
- `center_line` (training): `base = progress·speed/4 · band_multiplier`, off-track `-5`. No explicit
  lap-completion term; per-step value grows with cumulative `progress`.
- `progress_safe` (eval): on-track `(progress/steps)·100 + speed²`; off-track returns
  `OFFTRACK_PENALTY = -1.0` (`gym_dr/rewards.py:282`) — **the function docstring claims "-100" but the
  constant is -1.0 (stale docstring)**. So off-track is barely penalized and `speed²` dominates ⇒ the eval
  reward rises with speed even when progress stalls. This is why `eval_reward ≈ 3000` looked healthy while
  `max_progress ≈ 60%` and `offtrack_rate ≈ 0.32`.

**Action space:** `Box([-30,1.0],[30,4.0])` in **engineering units** (deg, m/s). No action-normalization
wrapper is applied (`gym_dr/envs/wrappers.py` = `ActionBounds` + `GrayscaleObs` only; the trainer adds only
`DummyVecEnv` + `VecFrameStack`). SB3 PPO uses a diagonal Gaussian with unit-init `log_std` (std≈1.0) over
the **raw** action space ⇒ steering explores ≈ ±1° (1.7% of its ±30° range) while speed explores ≈ ±1 m/s
(33% of its 3 m/s range). **Steering is grossly under-explored.** *Confirmed from the saved policy:* per-dim
`log_std` moved only 0.0→0.023 (steering) / 0.0→0.030 (speed) over 250k steps — final std ≈ (1.02°,
1.03 m/s), i.e. steering exploration stayed at ~±1° the whole run.

## Findings — ranked hypotheses

| # | Hypothesis | Strength | Evidence |
|---|---|---|---|
| 1 | **Generalization was never trained or measured.** Single track, no held-out worlds. | **Certain** | `worlds:{names:[Oval_track],rotations:1}`, no `world_strategy`/`eval_worlds`. |
| 2 | **Actor severely under-trained — LR far too low.** The policy is near its initialization. | **Strong** | `clip_fraction≈0`, `approx_kl≈1e-4`, `std` not converging (rising), `lr=2.05e-5`. |
| 3 | **Un-normalized action space ⇒ steering barely explored**, so it can't learn to corner. | **Confirmed** | measured per-dim `std`=(1.02°,1.03 m/s); steering exploration ~±1° over ±30°. |
| 4 | **Misleading eval metric.** `progress_safe` is dominated by `speed²` with a -1.0 off-track penalty (not -100). | **Strong** | `gym_dr/rewards.py:282,299`; `eval_reward≈3000` while progress flat. |
| 5 | **Never masters even the *training* Oval** (~60% progress, ~32% off-track). | **Observed** | consequence of #2/#3 (± reward shaping). |
| 6 | **Single-env correlated rollouts** (`DummyVecEnv([one])`) lower PPO sample quality. | **Background** | structural to the one-Gazebo-container design. |

**Headline:** trial_18 is *not* primarily a generalization failure — it is a **near-untrained policy
(LR/exploration too small) evaluated only on its own training track**, scored by a speed-inflated metric
that masked the lack of progress. Generalization was never attempted (single track) nor measured (no
held-out).

Killer-list items cleared: reward is non-constant and bounded; no crashes; critic explains variance (not a
broken value head); obs `normalize_images=False` is intentional. Open: episode-termination semantics with
mercy resets (Q3).

## Recommendation
Run the **W3 controlled experiment** as the decisive next step, but make it informative by first applying
three cheap, reversible, **sign-off-gated** changes (do not silently rewrite — these touch reward/config):
1. **Normalize the action space to [-1,1]** (gymnasium `RescaleAction`) so the unit Gaussian explores
   steering and speed comparably — *the single highest-leverage lever* (fixes #3).
2. **Use a sane learning rate** (~3e-4, the documented default) and/or a longer budget (fixes #2);
   re-check `clip_fraction` climbs off 0.
3. **Fix the eval metric** so it tracks lap completion/progress, not `speed²`; correct the
   `progress_safe` off-track penalty/docstring (fixes #4).
Then train with **`OrderedSplit`** (multi-track train, held-out eval) and report the **generalization gap**.
This isolates #1/#5 ("can't master training tracks") from a true generalization gap.

## Risks / open questions
- ✓ Confirmed: per-dim `log_std` (steering 0.023, speed 0.030) ≈ init; steering std ≈ ±1° over ±30° (#3).
- Q3: what actually terminates an episode (mercy resets make off-track survivable; `ep_length` 114–243).
- High `explained_variance` with a `[1024,1024,1024]` critic could be partly memorization of one track.

## Next steps
1. **W1** — scripted/pure-pursuit baseline to confirm the env+reward can be driven (separates env bugs from
   RL). 2. Get sign-off on the three changes above. 3. **W3** — Arm A (reproduce: single-track, held-out
   eval) vs Arm B (`OrderedSplit` multi-track), trace-on, multi-seed; compute the generalization gap with
   `deepracer-utils` (`GymTraceLog` + rliable, `phase='eval'`).
