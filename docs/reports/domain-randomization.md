# Domain randomization (W-dr) — status & ADR design · `[DISS]`→`[REAL]` · 2026-06-22

DR targets **environmental robustness** — a *separate* axis from track generalization (curriculum). Two
knobs: static randomization (done) and **automatic** (ADR, not yet built).

## Done — static DR
- `ActuatorNoise` (Gaussian on steering/speed, engineering units, applied before the action clip) and
  `ObservationNoise` (camera additive Gaussian + per-step brightness jitter) wrappers
  (`gym_dr/envs/wrappers.py`), opt-in via `DomainRandomizationConfig`, wired in the `time_trial` factory,
  unit-tested (`tests/test_domain_randomization.py`). On `main`.

## Pending — episode-reset DR (env-side; signed off, image-gated)
- **Random valid-start:** a `deepracer-env` reset change to sample `start_ndist` from `np_random` each reset
  (today only deterministic round-robin `CHANGE_START` / alternation `ALT_DIR` exist).
- **Random direction** per episode.
These ship with the deepracer-env edits batch (also `sim_time` exposure + episode-lifecycle config).

## ADR — Automatic Domain Randomization (NOT built; ready to)
ADR (OpenAI dactyl-style) **expands the randomization ranges as the agent succeeds and contracts them when it
fails**, so robustness auto-grows to the hardest level the policy can handle — no hand-tuned schedule.
**Prerequisite now satisfied:** a per-eval success signal — `eval/clean_completion_rate` from
`TrainingContext.evaluate` / the eval callbacks.

### Design
- An **`ADRController`** holding, per DR parameter (`actuator_steering_std`, `actuator_speed_std`,
  `obs_gaussian_std`, `obs_brightness_jitter`), a current value in `[0, max]` + a step size. After each eval:
  `clean_completion_rate ≥ promote` (e.g. 0.7) → widen each range one step toward `max`;
  `≤ demote` (e.g. 0.3) → narrow. Log `adr/<param>` so the growth is visible alongside the success curve.
- Make the noise wrappers read their std from a **shared mutable `ADRState`** (not a fixed value at
  construction), so the controller updates ranges live without rebuilding the env.
- **Hook:** call `controller.update(clean_completion_rate)` after each `ctx.evaluate`. For SB3, an
  `ADRCallback`; for custom trainers, one line in the loop (the trainer contract makes this uniform).
- Optional dactyl-style **boundary sampling** (sample at the current max a fraction of the time) for an
  ADR-progress / entropy metric.

### Effort & validation
Moderate, **GPU-free to build** (controller + mutable-range wrappers + hook). Validate with a
**robustness-vs-randomization curve** (success at increasing *fixed* DR vs ADR-grown), reported separately
from the curriculum's track-generalization gap.

## Status line
static DR ✅ · reset DR (env-side) ⏳ signed off · **ADR ⏳ designed, ready to build** (prerequisite met).
