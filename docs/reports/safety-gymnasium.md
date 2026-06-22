# Safety-Gymnasium integration — findings & plan · `[DISS]` · 2026-06-21

## What the `deepracer-env` `feat/safety-gymnasium` branch provides
(Inspected via `git show origin/feat/safety-gymnasium` — **not** checked out.)
- **`SafetyDeepRacerEnv`** (`deepracer_env/safety/safety_env.py`) — a `gym.Wrapper` that emits the
  Safety-Gymnasium **6-tuple** `(obs, reward, cost, terminated, truncated, info)`. Cost levels:
  - `safety-0` — `1.0` per off-track step;
  - `safety-1` — collision, combined with off-track via `max` / `add` / `crash-only`;
  - `safety-2` — weighted sum of `offtrack / crash / near_collision / steering_jerk`.
  - `terminate_on_cost` optional; cost derived from the `is_offtrack` / `is_crashed` / `objects_distance`
    info keys.
- **Adapters** `SafetyToGymnasium` / `GymnasiumToSafety` (cost carried via `info['cost']`).
- **`register_safety_envs()`** → `SafetyDeepRacer-OffTrack-v0`, `SafetyDeepRacer-Collision-v0`.
- **`examples/train_safe.py`** — SB3 PPO + a hand-rolled Lagrangian penalty (explicitly *illustrative, not
  benchmark-grade*).
- ⚠️ **The branch predates current `master`** (the diff shows master's world-swap/OA/world_swap work as
  deletions). It must be **rebased onto master** before use in the current single-container hot-swap stack.

## Done now (dr-gym side, autonomous + tested)
- **`gym_dr/costs.py`** — native cost callables (`cost_offtrack`, `cost_collision`, `make_composite_cost`)
  mirroring the branch's three levels, in the plain-`Callable[[dict], float]` shape (ADR-0002). Lets dr-gym
  compute the cost **in-process from the existing reward-params tap** (`gym_dr/metrics.py`) — no 6-tuple env
  required. Unit-tested (`tests/test_costs.py`, 7 passing).

## Integration options
- **(A) In-process cost + constrained trainer in dr-gym** — add a `cost` field to `ExperimentConfig`, tap it
  beside the reward, and a constrained trainer (OmniSafe PPO-Lag / PID-Lag, or a Lagrangian callback).
  Keeps the single-container hot-swap stack and the clean-completion eval.
- **(B) Branch's `SafetyDeepRacerEnv` + `safety_gymnasium` registry + an external constrained-RL lib** — run
  the algorithm on the **standard CMDP benchmarks first** to establish trust, then DeepRacer.
- **Recommend both:** validate the safe-RL algorithm on a Safety-Gymnasium task (B), then run DeepRacer
  constrained via (A) reusing `gym_dr/costs.py`.

## Validation plan
1. Rebase `feat/safety-gymnasium` onto `master`; resolve the deletions.
2. Add a dep group for `safety_gymnasium` (+ the chosen backend, OmniSafe).
3. Validate `SafetyDeepRacerEnv`: 6-tuple shape, cost fires on off-track/crash, `gym.make(<registered id>)`
   works (needs the branch image).
4. Reproduce a known safe-RL result on a Safety-Gymnasium task with the backend (algorithm sanity).
5. DeepRacer constrained run: `cost=cost_offtrack`, a cost limit; report reward vs cost-rate vs the
   clean-completion metric.

## Decisions / blockers (see `docs/questions-for-maintainer.md`)
- **D9** — safe-RL backend (OmniSafe recommended) + OK to install `safety_gymnasium` and rebase the branch.

## Status
`gym_dr/costs.py` done + tested. The rest is blocked on D9 + the branch rebase + deps + a constrained
trainer (W-saferl / architecture-review R3).
