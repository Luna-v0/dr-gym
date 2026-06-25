# Reward search · `[DISS]` · 2026-06-23

The D3 baseline (pure PPO + curriculum, no DR) converged to a clear failure mode: **speed pinned at ~3.9/4.0,
~28% max progress, 0 completions, ~18–20% off-track** — "floor it, crash at the first hard corner." This
motivated a reward search.

## Offline shape filter (no sim) — `scripts/reward_ranking.py`
Before spending sim time, score every reward on synthetic trajectories with known `reward_params`
(`clean_lap`, `fast_crash`, `crawl`, `zigzag`) and check the *episode return* ordering. Two findings:

1. **Every existing reward already ranks `clean_lap > fast_crash`.** So the fast-crash is **not** primarily a
   reward-*shape* problem — it's an **optimization/exploration** problem: the policy can't yet learn to corner,
   so among reachable (all-crashing) policies, faster = more progress before crash = higher return. A reward
   with a denser corner signal helps, but curriculum/exploration/DR matter too.
2. **8 of 10 rewards have a *crawl trap*** (`crawl > clean_lap`): per-step rewards accumulate over more steps,
   so a slow 3×-steps lap out-scores a fast clean one (the classic AWS warning — the car learns to dawdle for
   points). Only **progress-normalized** rewards avoid it.

Result table (episode return; higher = preferred):

| reward | clean_lap | fast_crash | crawl | verdict |
|---|---|---|---|---|
| centerline_quadratic (D3) | 160 | 25 | 279 | crawl trap |
| center_line / progress_and_speed / anti_zigzag / … | — | — | — | crawl trap |
| **progress_per_step** | 10290 | 2170 | 10180 | **shape-OK** |
| **progress_complete** (new) | 188 | 15 | 164 | **shape-OK** (clean > crawl > crash) |

## New rewards (`gym_dr/rewards.py`)
- **`make_weighted_reward(...)`** — stateless weighted sum of load-bearing terms: centeredness, **speed gated
  by centered×aligned** (no reckless-speed reward), a **speed-into-corner penalty** (`speed×curvature`),
  alignment, lap-pace, steering penalty. Presets: `centered_speed`, `corner_aware`, `survive_first`.
- **`make_progress_reward(...)`** → preset **`progress_complete`**: progress-DELTA + per-step time penalty +
  completion bonus. Total ≈ (progress) − penalty·steps (+bonus) ⇒ bounded per lap (no crawl trap) and rewards
  finishing *fast*. Stateful (tracks previous progress, detects the per-episode reset). The two
  knobs/cancellation bugs (laps must reach 100% for the bonus; `center_bonus < step_penalty` or it cancels the
  time pressure) were caught by the offline filter before any training.

## Training search — `experiments/reward_search.py` (needs the sim; after D3)
Optuna study, **500k-step trials** on the held-out `StochasticCurriculum` split, **2 parallel workers**
(sw-render sweet spot), scored by the clean-completion eval. Sweeps the reward **family + hyperparameters**
(`progress_complete`'s `step_penalty`/`completion_bonus`/`center_bonus`, or `make_weighted_reward`'s weights)
plus a little PPO `lr`/`ent_coef` (since the fast-crash is partly optimization). Focused on the
progress-normalized families the offline filter cleared.

## Key takeaway
The reward search alone may not fully fix the fast-crash — the offline filter shows the *shape* is already
right, so it's also an optimization problem. Expect the fix to be a **combination**: a progress-normalized
reward with a corner-speed penalty (denser cornering signal) **+** the DR/curriculum (state coverage so the
policy actually learns to corner). The reward search narrows the reward half; Phase 2 (DR/ADR) addresses the
exploration half.

## Files
- `scripts/reward_ranking.py` — offline shape filter (no sim).
- `gym_dr/rewards.py` — `make_weighted_reward`, `make_progress_reward` + presets (in `REWARD_VARIANTS`).
- `experiments/reward_search.py` — the Optuna training search.
- `tests/test_rewards.py` — gating/corner-penalty/no-crawl-trap/completion-bonus tests (48 pass).
