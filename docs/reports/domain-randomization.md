# Domain randomization (W-dr) — status & ADR design · `[DISS]`→`[REAL]` · 2026-06-22

DR targets **environmental robustness** — a *separate* axis from track generalization (curriculum). Two
knobs: static randomization (done) and **automatic** (ADR, not yet built).

> ## API UPDATE — 2026-06-24 (Range/Choice + ADR + drag + friction)
> The flat `DomainRandomizationConfig(actuator_steering_std=..., adr=True, ...)` API was **replaced** (clean
> break) by typed value-specs and an `ADR` subclass. Authoring now goes through `EnvironmentConfig`
> (`gym_dr/environment.py`):
> ```python
> from gym_dr import EnvironmentConfig, FeatureObs, ACL, ADR, Range, Choice
> EnvironmentConfig(
>     observation=FeatureObs(),                     # or CameraObs()
>     curriculum=ACL(train_worlds=[...], eval_worlds=[...]),   # ACL = the former StochasticCurriculum
>     domain_randomization=ADR(                      # ADR(...) static-knob base = DomainRandomization
>         steering_noise=Range(0, 3), speed_noise=Range(0, 0.15),   # was actuator_*_std (Range high = old ceiling)
>         obs_gaussian=Range(0, 10), obs_brightness=Range(0, 0.2),
>         drag=Range(0.7, 1.0),         # per-episode throttle->speed factor (sim2real); 1.0 = off
>         friction=Range(0.8, 1.5),     # per-SPAWN wheel-mu multiplier (Gazebo has no runtime mu service)
>         random_start=True, random_direction=True,
>         step=0.1, promote=0.7, demote=0.3, seed=42))
> ```
> - **Value specs** (`gym_dr/randomization.py`): `Range(low, high)` (continuous, per-episode), `Choice([...])`
>   (discrete list), or a bare scalar (constant). `sample_spec` / `spec_bounds` helpers.
> - **`ADR`** widens each **noise** knob's `Range` cur_high from `low`→`high` as held-out clean-completion
>   clears `promote` (shrinks ≤ `demote`); logs `adr/<knob>_high`. `drag`/`friction` sample their full Range
>   each spawn/episode (their easy anchor is 1.0, not 0, so naive widening is a follow-up).
> - **`drag`** = `DragRandomization` action wrapper (per-episode speed scaling). **`friction`** = per-spawn
>   wheel μ via the `friction_mu` xacro/launch arg ← `GYM_DR_FRICTION_MU` (dr-gym `app.py` samples it);
>   per-EPISODE μ needs a Gazebo plugin (deepracer-env has no runtime surface-μ service — documented limit).
> - The old flat fields still exist *internally* on `ExperimentConfig` (the env factory reads them); the
>   authoring API is `EnvironmentConfig`. See `tests/test_environment.py` + `test_domain_randomization.py`.
> The historical design notes below use the pre-refactor names.

## Done — static DR
- `ActuatorNoise` (Gaussian on steering/speed, engineering units, applied before the action clip) and
  `ObservationNoise` (camera additive Gaussian + per-step brightness jitter) wrappers
  (`gym_dr/envs/wrappers.py`), opt-in via `DomainRandomizationConfig`, wired in the `time_trial` factory,
  unit-tested (`tests/test_domain_randomization.py`). On `main`.

## Episode-reset DR (env-side) — BUILT 2026-06-22
- **Random valid-start + random direction** are now implemented in `deepracer-env`: the `RANDOM_START` /
  `RANDOM_DIRECTION` controller-config modes (`agent_ctrl/constants.py`, `rollout_agent_ctrl.py:finish_episode`)
  sample a uniform normalized start distance along the centerline (any `ndist∈[0,1)` is a valid on-track pose)
  and/or a random direction each training episode, taking precedence over the deterministic `CHANGE_START`
  round-robin / `ALT_DIR` alternation. Uses a dedicated seeded RNG; `.get(...,False)` defaults keep old configs
  unchanged. `gym_dr/envs/time_trial.py` passes them through when `DomainRandomizationConfig.random_start/
  random_direction` are set; `experiments/end_to_end_ppo.py` enables both.
- **Deployment (no image rebuild needed):** bind-mount the local checkout over the container's package via
  `GYM_DR_DEEPRACER_ENV_SRC` (`gym_dr/docker_runner.py`) — validated: the patched `RANDOM_START`/`RANDOM_DIRECTION`
  load in the base image. (Proper long-term fix = rebuild the base sim image from deepracer-env source.)
- Still pending in the env-edits batch: `sim_time` exposure + episode-lifecycle config.

## ADR — Automatic Domain Randomization (BUILT)
**Implemented (2026-06-22):** `ADRController` + mutable `ADRState` (`gym_dr/domain_randomization.py`),
live-range `ActuatorNoise`/`ObservationNoise` (read the current std each step), factory wiring
(`DomainRandomizationConfig(adr=True, ...)`), and the post-eval hook in **both** the SB3
`MultiWorldEvalCallback` and the custom-trainer `TrainingContext.evaluate` — so it scales DR ranges from the
held-out `clean_completion_rate` and logs `adr/<dim>`. Tested (`tests/test_domain_randomization.py`). The
`actuator_*`/`obs_*` config values act as the ceilings; ranges start at 0 and grow. **Pending:** a
validation run + adding the env-side reset DR dims (start/direction) once they land. Design below.

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

## Multi-car DR warmup (the ADR substitute for `MultiCarVecEnv`) — 2026-06-28

Feedback-ADR needs an in-loop held-out clean-completion signal to widen knobs. The
multi-car VecEnv has none (it can't `set_world`, so the in-sim eval is disabled) and it
reads each DR knob as its **static high** (`spec_bounds(spec)[1]`) with **no ADR
controller**. Applying every magnitude at full strength from step 0 made the asym oracle
unlearnable: an unobservable ±15° per-episode steering bias + full feature noise gave
~2-step episodes / 100% off-track / flat 5% progress (see
[status-2026-06-28](status-2026-06-28.md)).

`MultiCarVecEnv` therefore has a **self-counted linear DR warmup** (`_dr_scale()`): a
factor ∈ [0,1] that grows linearly over the first `GYM_DR_DR_WARMUP_STEPS` timesteps and
multiplies every magnitude knob (per-episode steering/speed bias, feature noise, per-step
actuator noise). The policy learns to **drive first** (near-clean, survivable early
episodes), then to **counter** the perturbations as they reach full strength —
schedule-based, not feedback-based (no eval needed). `0` = off (full strength; the camera
run keeps it off). Forwarded into the container by `app.py`, read by the multi-car
factory. Pair with `frame_stack > 1` so the policy has temporal context to infer an
unobservable per-episode bias (its drift signature) — see
[asymmetric-architecture](asymmetric-architecture.md).

## Status line
static DR ✅ · **ADR ✅ built** (controller + live wrappers + SB3 & custom-trainer eval hooks; tested) ·
**multi-car DR warmup ✅** (linear, self-counted; multi-car's feedback-ADR substitute) ·
reset DR (env-side) ⏳ signed off · ADR/DR validation run ⏳.
