# Scope review — "won't generalize" is measurement + sample-efficiency + optimization, not one bug · `[DISS]` · 2026-06-21

## Question / goal
The maintainer noted that curriculum / multi-track experiments were *already tried* and "were not good."
Re-scope Q1 against **all** the relevant runs (not just trial_18) and decide what to actually fix first.

## What I did
Pulled per-run held-out metrics across the multi-world / curriculum studies
(`/tmp/mlflow_summary.py` over `mlruns/`): `tt_multiworld` (21-trial HPO), `time_trial_trial18_10x` (16M),
`tt_curriculum_earlystop` (6M), `time_trial_multiworld`. Read the experiment scripts
(`experiments/time_trial_train.py`, `time_trial_earlystop.py`, `ordered_split_example.py`).

## Evidence

What was **already tried** (so these are *not* the missing ingredient):
- **Track variety + held-out split:** `tt_multiworld` trains 6 tracks, evaluates 3 held-out (`OrderedSplit`).
- **Learning-rate sweep:** `tt_multiworld` LR spans 1e-5 → 9.5e-4.
- **Reward sweep:** center_line / progress_and_speed / anti_zigzag / centerline_quadratic / progress_per_step.
- **Curriculum:** `tt_curriculum_earlystop` — 6 tracks easy→hard, mastery-gated advance.
- **Big budget:** `trial18_10x` — 16M steps, held-out split.

What happened (held-out `eval/<world>_mean_reward`, training `dr/ep_max_progress`, `train/{clip_fraction,std}`):

| Run | budget | train prog | clip_frac | std | held-out reward | failure mode |
|---|---|---|---|---|---|---|
| `tt_multiworld` (best, trial_12) | 600k | ~50% | 0.13 | 9.4 | ~2000–3400 | under-trained |
| `tt_multiworld` (high-LR trial_2/11) | 600k | ~0–14% | — | 1.6 / **63** | ~900–2200 | **LR instability / std blow-up** |
| `tt_multiworld` (most trials) | 600k | <30% | low | ~1 | ~2000–3000 | under-trained |
| `tt_curriculum_earlystop` | 6M | 28% | 0.76 | **0.03** | 2036 | **entropy/std collapse** |
| `trial18_10x` | 16M | **100%** | 0.67 | 1.9 | reInvent 22640 / Bowtie 5398 | learns train; **uneven transfer** |

Three findings stand out:

1. **The evaluation metric barely discriminates.** Across `tt_multiworld`, held-out `eval_reward` is
   ~2000–3000 whether training progress is 0.2% or 50%. `progress_safe` is dominated by `+speed²` and
   penalizes off-track by only −1.0 (`gym_dr/rewards.py:282`), so it rewards *fast* over *completes the lap
   cleanly* and can't rank policies. **You cannot tell "generalized" from "didn't" with this number.**
2. **Single-env PPO is grossly sample-inefficient here.** 600k steps over 6 tracks (=100k/track) never
   learns (prog <50%); only the 16M-step run reaches 100% on training tracks. So **the HPO study was
   structurally doomed** — every 600k trial was under-trained, and HPO optimized noise. This is a
   throughput/sample-efficiency problem, not a hyperparameter problem.
3. **Optimization is fragile.** High LR → `std` explodes and progress collapses; the curriculum run →
   `std` collapses to 0.03 (premature determinism). The action space is **un-normalized** in *every* run
   (raw deg/m·s), so the unit Gaussian both under-explores steering (±1° over ±30°) at low LR and is prone
   to scale blow-ups at high LR — a strong, still-untested common cause.

## Findings (re-scoped)
"Won't generalize" is **not one bug**. It is, in order of leverage:
1. **No trustworthy yardstick** — every "good/not good" verdict so far rests on a metric that can't
   distinguish a 0.2%-progress policy from a 50% one.
2. **Learning isn't affordable** — adequate training needs millions of steps in a single env; trials can't
   afford it, so they under-train.
3. **Optimization instability** — std collapse/explosion; un-normalized actions a likely common root.
4. Only *after* 1–3: the genuine generalization levers (curriculum that actually advances, domain
   randomization, and the perception-net / asymmetric-critic for visual transfer).

The earlier `q1-generalization.md` (trial_18 = under-trained + un-normalized + bad metric) is consistent
and now generalizes to the whole study.

## Recommendation — re-prioritize the plan
- **P1 `[DISS]` Trustworthy evaluation FIRST.** A completion/lap-time/progress metric + a fixed held-out
  protocol + actual rollout inspection (trace path-plots, VNC). Fix `progress_safe`. *Nothing else is
  measurable until this exists.* (Folds W3's gap metric + W-dash forward.)
- **P2 `[BOTH]` Make learning affordable.** Benchmark RTF/parallelism (Q3/Q4), set a realistic per-trial
  budget (≫600k), and/or cut sample needs. The single-env (`DummyVecEnv([one])`) ceiling is central.
- **P3 `[DISS]` Optimization robustness.** Normalize the action space to [-1,1] (cheap, untested,
  present in all failures); add entropy/std-collapse guards; revisit the oversized `[1024×3]` heads.
- **P4 `[DISS]`/`[REAL]` Generalization levers** — curriculum/DR/perception, once P1–P3 hold.
- **Quick sanity gate (cheap, parallel):** W1 scripted baseline — confirm the env+reward can be driven at
  all, so we're not chasing an RL fix for an env problem.

## Risks / open questions
- The eval metric being unreliable means even `trial18_10x`'s "uneven transfer" needs confirmation by
  watching/▶ tracing rollouts, not just the number.
- Is the instability primarily the un-normalized action space, the huge heads, or PPO settings? P3 tests it.
- What did "not good" look like to the maintainer (VNC behavior, specific tracks)? — would sharpen P1–P4.

## Next steps
Confirm the re-prioritization, then P1 (eval metric + protocol) as the new lead, with W1 as a cheap
parallel sanity check. Update the plan file's lead workstream accordingly.
