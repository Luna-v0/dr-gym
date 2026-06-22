# Architecture & cross-repo design review · `[BOTH]` · 2026-06-21

> Standing design doc (maintainer asked: "re-write some parts and pass to another repo? design patterns?
> update something?"). **Proposals, not silent rewrites** — every refactor is small, reversible, and
> sign-off-gated. ADRs live in `docs/adr/`.

## 1. Current architecture (recap)
Three repos, coupling **schema-only** (see `docs/system-overview.md`): `dr-gym` (RL/training),
`deepracer-env` (ROS1/Gazebo sim), `deepracer-utils` (analysis, reads the trace format). dr-gym already
uses good patterns: **Strategy** (`gym_dr/worlds.py:WorldStrategy`), **ABC/plugin** (`Trainer` in
`gym_dr/trainers/`), **Factory** (`gym_dr/envs/`), **Adapter** (trace → deepracer-utils DataFrame),
**Wrapper** (`ActionBounds`/`GrayscaleObs`/`NormalizeActions`). The review extends these rather than
inventing new structure.

## 2. Repo-boundary audit — what belongs where, what should move

| Concern | Today | Recommendation |
|---|---|---|
| Reward functions | `gym_dr/rewards.py` | **Stay** — a training concern. But the *param contract* (the 26 keys) is owned by `deepracer-env`; document it as a versioned shared contract (§4). |
| Action/units (deg, m/s ↔ ServoCtrlMsg) | implicit; export in eng units, rescale formula only in a smoke test | **Extract** one source-of-truth rescale used by both export and the on-car node (R1). Removes a silent-failure surface. |
| Perception net (camera→features) | does not exist | **Train in dr-gym** (`gym_dr/perception/` — it has the sim ground truth); the trained net is a shipped artifact. |
| On-car inference node (ROS) | does not exist (documented only) | **New repo `deepracer-deploy`** (ADR-0001) — ROS2 + OpenVINO IR + perception, minimal deps, off the training graph. |

**Why a new deploy repo, not into an existing one:** the car runtime is ROS2 + OpenVINO on different
hardware with a different release cadence; folding it into `deepracer-env` (ROS1 sim) or `dr-gym` (py3.8
training) couples unrelated lifecycles and bloats both. The seam between training and deploy is already
just two artifacts — the **IR model** and **`model_metadata.json`** — so a clean repo split is natural.

## 3. Design patterns — extend the existing ones to the new axes
Keep the "plain callable + Strategy/Wrapper/ABC" style so new features are swappable, not bolted on
(ADR-0002):

| New axis (workstream) | Pattern to reuse | Shape |
|---|---|---|
| Curriculum (P4 / W-curriculum) | **Strategy** | a `WorldStrategy` subclass (success-gated / interleaved); already fits. |
| Domain randomization (W-dr) | **Wrapper** + Scheduler | a composable DR wrapper stack + an ADR controller that widens ranges on success. |
| Safe-RL cost (W-saferl) | **plain callable** (like `reward`) | `cost: Callable[[dict], float]` on `ExperimentConfig` + Lagrangian logic in a constrained `Trainer`. |
| Perception (W-perception) | **Strategy/Adapter** | a pluggable obs→features interface: raw-CNN ↔ perception-net ↔ privileged. |
| RL backends (W-extensibility) | **ABC contract** | the `Trainer` interface (§5). |

## 4. Contract surface — make the seams explicit and versioned (ADR-0003)
The cross-repo seams are load-bearing and currently implicit. Add a top-level `CONTRACTS.md` enumerating
them, each with a `schema_version`:
- **Trace schema** (`docs/trace-contract.md`) — add a `schema_version` column/field.
- **`reward_params`** — the 26 keys `deepracer-env` guarantees to the reward callback (owner: deepracer-env).
- **Action/units + `model_metadata.json`** — engineering units (or [-1,1] when `normalize_actions`), one
  rescale function as source of truth.
- **IR model I/O** — input (grayscale stack `(4,120,160)` uint8) and output (action-mean; eng units or
  [-1,1]). Bump deliberately; a change here is what breaks the car.

## 5. Backend extensibility contract (W-extensibility, pairs with this review)
A `Trainer` (see `gym_dr/trainers/base.py`) must: implement `fit(experiment, ctx) -> TrainResult`; honor
`TrainingContext` (`world_plan`, `eval_worlds`, `metrics_state`, `run_dir`, `report_eval`,
`save_checkpoint`, `action_space`); emit TensorBoard + MLflow scalars; checkpoint; run the **held-out eval
protocol** (`docs/eval-protocol.md`, clean-completion); support **curriculum** (`set_world` between chunks),
**DR** (env wrappers), and the **safe-RL cost** interface. Deliverable: `docs/trainer-contract.md` + a second
backend (adopt **CleanRL** for a lean PPO, or **OmniSafe** for constrained RL) validating the contract.

## 6. Concrete refactor candidates (build-vs-adopt · cost/benefit · all reversible, sign-off-gated)
- **R1 — one action-rescale source of truth** (low). Removes the on-car silent-unit-mismatch risk. Build.
- **R2 — `CONTRACTS.md` + trace `schema_version`** (low). Turns silent drift into a reviewed event. Build.
- **R3 — `cost` callable + constrained-PPO** (med). **Adopt** OmniSafe/safety-starter and validate on
  Safety-Gymnasium before DeepRacer, rather than hand-rolling Lagrangian PPO. (W-saferl)
- **R4 — perception as a pluggable feature-extractor** (med). Build the interface; train net on sim GT. (W-perception)
- **R5 — `deepracer-deploy` repo** (med). Clean separation for the on-car node + perception inference. (ADR-0001)
- **R6 — replace single-env `DummyVecEnv([one])` with a multi-car VecEnv** (med/high) once P2 picks the
  throughput design (N-cars-in-one-world). The single env is the sample-efficiency ceiling (scope review).
- **Do NOT rewrite:** the trace contract, the `WorldStrategy` pattern, and the ONNX→IR pipeline — sound.

## 7. Recommendation & sequencing
Land the cheap contract hygiene first (R1, R2) alongside the P1–P3 work, since they de-risk the on-car
path and the cross-repo seams at low cost. Defer R3–R6 behind their workstreams (and the P2 throughput
decision for R6). Stand up `deepracer-deploy` (R5/ADR-0001) when the first validated policy is ready to
leave the sim. The recurring discipline: **new features reuse the existing patterns; cross-repo seams are
versioned contracts, changed deliberately.**

## Open decisions
D6 (deploy repo location) — recommendation here is a new `deepracer-deploy` repo; confirm. See
`docs/questions-for-maintainer.md`.
