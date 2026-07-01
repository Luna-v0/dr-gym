# ADR-0004 — Orchestrator refactor: Study / Stage / Algorithm interface

- **Status:** Accepted (2026-07-01)
- **Deciders:** maintainer (decisions B1–B4, see `/BLOCKERS.md`) + Claude
- **Supersedes / touches:** the flat `train()`/`study()` app API, `ExperimentConfig`
  early-stop fields, `hpo.py` sampler seeding. Related: ADR-0002 (extend existing
  patterns), ADR-0003 (versioned contract surface).

## Context

TASKS.md Task 1 (refactor) + Task 8 (usability / no algorithm lock-in) ask for: an
explicit **Stage** pipeline (compose the MDP data-path with `>>`), tighter coupling of
**Study == training + hyperparameter search over one interface** with seeds from a single
root, **early stopping as interchangeable Strategy objects**, **config-driven curriculum**,
and an **algorithm-agnostic abstract class** giving users the whole ecosystem (TensorBoard,
MLflow, checkpointing, eval) without being locked to Stable-Baselines3.

The codebase already has most of the seams: `trainers/base.Trainer` (a `Protocol` with
`fit(env, ctx)`), `TrainingContext` (TB+MLflow+checkpoint+eval services), `WorldStrategy`
(curriculum ABC), and a complete `SeedManager` (master seed → eval/sampler/replicate streams
with an agent⊥domain split) whose consumers are still marked `#FUTURE`. The refactor is
therefore largely *formalising and wiring*, not green-field.

Maintainer decisions locking the shape:
- **B1 — clean break.** Replace the old `train()`/`study()`/flat-config API; migrate the
  canonical experiments + tests in the same unit. No deprecated shim.
- **B2 — hybrid Stage, algorithm-agnostic.** The performance win is *batched/vectorised
  rollout*, not SB3. So the pipeline is Stages; an `Algorithm` owns the rollout loop; SB3 is
  one (fast, vectorised) adapter; a bring-your-own-loop adapter can call `action = pipeline(obs)`
  literally. **Nothing in the core privileges SB3.**
- **B3 — refactor first**, with characterization tests as its opening step.

## Decision

Introduce four locked interfaces. They compose; each is independently testable.

### 1. `Stage[I, O]` — `gym_dr/pipeline.py` (DONE)
A tiny, dependency-free composable function `I -> O` with `>>`, name composition, and
flattening introspection (`len`, `iter`, `repr`). Two roles: **declarative assembly** of an
experiment's obs→encode→policy→action wiring (a batched adapter reads it and runs its own
loop), and a **literal single-obs data-path** for inference / ONNX export / on-car deploy /
the decoupled obs-encoder→policy evaluation. Neural stages wrap a `torch.nn.Module`; the
primitive stays torch-free.

### 2. `Trainer` (name kept) — formalise as an extendable ABC — `gym_dr/trainers/base.py`
**Decision (maintainer): keep the name `Trainer`.** Relift the `Trainer` `Protocol` to an
**abstract base class** users *extend* (Task 8's "abstract class they can extend and implement").
Contract unchanged: `fit(env, ctx: TrainingContext) -> TrainResult`. `Sb3Trainer` and
`FsrlTrainer` subclass it explicitly (`Sb3Trainer` currently only satisfies it structurally).
`TrainingContext` remains the single service surface (`log_metrics` TB+MLflow, `record_episode`,
`save_model`/`save_checkpoint`, `report_eval` +Optuna, `swap_world`, `evaluate`) — this is the
"whole ecosystem" a custom trainer gets for free. A documented `CustomTorchTrainer` example drives
its own loop via `pipeline(obs)` (no SB3).

### 3. `Study` — `gym_dr/study.py` (replaces `app.train`/`app.study`)
**One class, one interface.** A `Study` is *always* defined over a hyperparameter space
(`params`); a **single training run is a Study whose hyperparameters are all `Fixed`** — no
`None` sentinel, no separate `train()` function (maintainer decision).

```python
Study(
    experiment: ExperimentConfig,
    params: Mapping[str, Hyperparam | Any] = {},   # value or gym_dr.search.{Float,Int,Categorical}
    master_seed: int = 0,
    n_replicates: int = 1,
    n_trials: int = 1,          # forced to 1 when the space is all-Fixed (single run)
    n_parallel: int = 1,
).run() -> StudyResult
```

A bare constant in `params` is coerced to `Fixed`. When every entry is `Fixed` the space has no
search dimension → exactly one trial (optionally × `n_replicates`), and Optuna is skipped. Any
distribution present → HPO over `n_trials`. Seeds derive from `SeedManager(master_seed)`: TPE
sampler from `sampler_seed()` (replaces the weak `base.seed + worker_idx`), each replicate `k`
from `replicate(k).agent` (policy) with `replicate(k).domain` reserved for env randomization.
`run()` unifies today's host/worker dispatch, multi-world rotation, MLflow lifecycle, and the new
replicate loop. `StudyResult` exposes `best_params`, `best_value`, per-replicate run dirs.

### 5. `SearchSpace` / `Hyperparam` — `gym_dr/search.py` (DONE)
The declarative hyperparameter vocabulary the `Study` consumes: `Fixed`, `Float`, `Int`,
`Categorical` (frozen dataclasses; each knows how to `suggest(trial, name)`), plus `SearchSpace`
which maps dotted `ExperimentConfig` keys to them, reports `is_single_run`, and compiles to
per-trial overrides. Distinct from `gym_dr.randomization.{Range,Choice}` (those randomize the
*environment*, not the *hyperparameters*).

### 4. `EarlyStopStrategy` — `gym_dr/early_stopping.py` (DONE)
Frozen-dataclass hierarchy (`OfftrackRate`, `MetricThreshold`, `RewardThreshold`,
`CleanCompletion`, `AllOf`/`AnyOf`) + a stateful `EarlyStopController` (patience streak, per-chunk
reset). `TrainingConfig.early_stop: EarlyStopStrategy | None` replaces the three
`early_stop_*` fields. `OfftrackRate(0.0, patience=1)` reproduces the historical default exactly,
so behaviour is preserved. HPO can sweep `training.early_stop.threshold` etc.

### Config-driven curriculum
`WorldStrategy` stays the curriculum port. The mastery-gated variant (already sketched in
`ACL`'s docstring) is enabled by feeding the `EarlyStopController`'s per-chunk qualifying signal
back to the strategy — no new port, a runtime hook on the existing one.

### Config layout (clean break of the *authoring* surface)
`EnvironmentConfig` becomes the **canonical (only) authoring surface**. The dual-source
`__post_init__` unpack (root cause of the reward-clobber bug, see memory
`postinit-reward-clobber-bug`) is removed in favour of a single explicit
`ExperimentConfig.from_environment(env, *, name, trainer=..., training=..., tracking=...)`
builder that populates the flat internal fields **once**. The flat fields stay as the
orchestrator's internal representation (read by env factory, dispatch, docker_runner — not
rewritten), but are no longer part of the user-facing authoring API: experiments construct via
`from_environment`, not by passing `camera_obs=`/`domain_randomization=`/etc. directly. This is
the pragmatic clean break — one authoring path, the bug fixed — without a risky rewrite of every
internal reader.

## Alternatives considered
- **Additive (keep both APIs)** — rejected per B1 (two ways to do everything; doc burden).
- **Stages as the literal per-step datapath everywhere** — rejected per B2 (Python call-chain
  per step wrecks camera throughput at n_cars scale; batching is the real lever).
- **Runtime early-stop registry (by name)** — rejected: loses static typing / clean HPO sweep
  keys; frozen dataclasses serialise and hash for free.

## Consequences
- **Positive:** one canonical, explicit, algorithm-agnostic API; SB3 removable; reproducibility
  from a single master seed; early stopping and curriculum are declarative and sweepable; the
  decoupled obs-net↔policy experiment has a first-class Stage seam.
- **Cost / risk:** breaking change — every `experiments/*.py` and several tests migrate in the
  same unit (mitigated by characterization tests written first, per B3). The `Study` rewrite
  touches the host/container dispatch in `app.py` and `docker_runner.py`; done carefully behind
  the smoke pre-flight.
- **Foundations landed already:** `gym_dr/pipeline.py` + `gym_dr/early_stopping.py` + their
  tests (32 passing) + pytest markers. The invasive `Algorithm`/`Study`/config changes follow.
