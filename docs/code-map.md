# Code Map — dr-gym ↔ deepracer-env (refactor-prep reference) · 2026-06-29

A function/module-level map of the whole camera-perception + oracle + multi-car training
stack, for planning a larger refactor. For each piece: **what it's responsible for**, the
**key functions/classes**, **coupling**, and **refactor-smells**. Built from a read-only
sweep of the actual code (5 parallel mappers).

> Companion: [system-overview](system-overview.md) (high-level), [status-2026-06-28](reports/status-2026-06-28.md)
> (recent changes), [domain-randomization](reports/domain-randomization.md), [asymmetric-architecture](reports/asymmetric-architecture.md).

---

## 0. System overview — layers & data flow

```
experiments/*.py  (author an ExperimentConfig; host/container split via GYM_DR_IN_CONTAINER)
      │  train()/study()
      ▼
gym_dr/app.py  ──host──►  docker_runner.py  ──►  Docker container (sim image, repo bind-mounted)
      │  container                                   │ re-imports the same experiment script
      ▼                                               ▼
gym_dr/trainer.py: run_training()
      │  install_metrics() wraps reward + env_wrapper
      ▼
env_factory = gym_dr/envs/dispatch.py: build_env()   ── 2×2 on (n_cars, camera_obs) ──►
      ├─ single-car: time_trial() / feature_time_trial()  (+ wrappers.py DR stack)
      └─ multi-car : multi_car()  → MultiCarVecEnv  (DR + per-car metrics + recorder inline)
      │                                   │ backend
      ▼                                   ▼
gym_dr/trainers/sb3: Sb3Trainer.fit()   deepracer_env.MultiAgentDeepRacerEnv / DeepRacerEnv
      │  SB3 callbacks (eval/metrics/plots/checkpoint/heartbeat)   │  Gazebo Classic + ROS1
      ▼                                                            ▼
TensorBoard / MLflow / checkpoints / perception_out shards    racecar_i + offset TrackData
```

**Key cross-cutting facts:**
- **Orchestration is by environment variable.** Host ↔ container IPC is `GYM_DR_*` env vars
  (the container re-imports the experiment script and must rebuild an identical config).
- **The env factory is a 2×2 dispatch** on `(n_cars ≤1?, camera_obs?)`.
- **DR logic is implemented twice** — single-car via composable gym wrappers (live ADR
  feedback), multi-car inline in `MultiCarVecEnv` (self-counted warmup ramp).
- **Multi-car has NO `set_world`** (N track instances in one world) → no in-loop held-out
  eval → held-out is a separate single-car / frozen-rollout pass.

---

## 1. Experiments (entry points) — `experiments/`

Each authors an `ExperimentConfig`/`EnvironmentConfig` and has a **host/container split** in
`__main__`: host runs a per-chunk spawn loop; inside the container (`GYM_DR_IN_CONTAINER`) it
trains ONE chunk on the env-provided tracks. All forward state via `GYM_DR_*`.

### experiments/camera_cnn_dataset.py
**Responsible for:** Train the camera (vision CNN, 4-frame stack) policy AND generate the
camera→features perception dataset (Phase-1 priority).
- `_split_tracks(seed=42)` — deterministic by-TRACK split: unique base tracks → 70/15/15
  train/val/test; `_cw/_ccw/_mirrored` VARIANTS + physical (reinvent/Oval) held out separately.
  No leakage (a base lives in exactly one split). Exposes `TRAIN/VAL/TEST/VARIANT_TRACKS`.
- `_train_pool()` → the train split; `_groups(tracks, n, passes)` — concatenate passes then
  chunk into N-car groups (pads only the final group → minimal dup).
- `build_experiment(group, resume)` — CameraObs, `n_cars=len(group)`, heavy ADR, reward
  `progress_per_step`, eval `clean_completion`, frame_stack=4, GPU; `eval_freq=CHUNK//4`.
- `N_CARS` default 8 (sets `GYM_DR_ALLOW_CAMERA_NCARS=1` at module level so the container gets it).
- `main()` host loop over groups (sets `GYM_DR_DEMO_WORLDS` per chunk, resumes `latest_model`).
- Coupling: `GYM_DR_PERCEPTION_OUT` (recorder), `GYM_DR_VISUAL_DR(_SEED)`, `GYM_DR_N_CARS`,
  `GYM_DR_DEMO_WORLDS`, `GYM_DR_ALLOW_CAMERA_NCARS`.

### experiments/perception_capture_heldout.py
**Responsible for:** Frozen-rollout (lr=0) capture of val/test/variants/physical frames the
training sweep never drives — no training on held-out tracks.
- `GYM_DR_CAPTURE_SPLIT` ∈ {val,test,variants,physical} → tracks from `cam.*_TRACKS`; output
  `artifacts/perception_capture_<split>/perception_out`. `build_capture()` reuses
  `cam.build_experiment` with `learning_rate=0`, eval disabled. `N_CARS` default **4**
  (lower than training's 8 → fewer track spawns → avoids Gazebo spawn-timeout flakiness).
- Coupling: `GYM_DR_CAPTURE_{SPLIT,RESUME,TRACKS,NCARS,STEPS,PASSES}`.

### experiments/oracle_asym_multicar.py
**Responsible for:** 12-car feature oracle, asymmetric critic, for robust state-based policy.
- N_CARS=12 (one per train track; feature obs scales ~linearly). EVAL_WORLDS empty (multi-car
  can't set_world). **Learnability fix:** `GYM_DR_DR_WARMUP_STEPS` (ramp all DR 0→full over
  ~20%) + `frame_stack=4` (observation memory to infer the unobservable per-episode bias).
- Coupling: `GYM_DR_FEATURE_SET=actor_extended`, `GYM_DR_DEMO_WORLDS`, `GYM_DR_DR_WARMUP_STEPS`.

### experiments/oracle_hpo.py
**Responsible for:** Optuna HPO for the asym oracle. **Single-car** (so it CAN run in-loop
held-out eval → a real objective; the searched HPs transfer to the multi-car run).
- Search: lr, ent_coef, n_steps, batch_size, gamma, gae_lambda, clip_range, n_epochs,
  target_kl, `frame_stack∈{1,2,4,8}`, net_width∈{64,128,256}, feature_noise_high. Objective =
  held-out clean_completion. Mild ±5° bias so frame_stack has something to infer. Short trials
  (240k) + pruning. `study()` spawns N parallel workers on one SQLite study.

### experiments/oracle_asym_robust.py
**Responsible for:** Single-car feature oracle (upper-bound + camera teacher); asym critic +
feature-noise ADR + ACL curriculum; 18 train / 8 held-out (incl. physical) measured in-loop.

---

## 2. Config & public API — `gym_dr/{config,environment,worlds,action_space,randomization,domain_randomization}.py`

### gym_dr/config.py
**Responsible for:** The (legacy-flat) `ExperimentConfig` users author + HPO mutation + serialization.
- Dataclasses: `TrainingConfig`, `TrackingConfig`, `WorldsConfig`, `TraceConfig`, `ExperimentConfig`.
- `ExperimentConfig.__post_init__` — **unpacks the composed `environment: EnvironmentConfig` into
  flat fields** (camera_obs, reward, n_cars, world_strategy…), with a `_fill`-when-at-default guard
  so it can't clobber the metrics-wrapped reward; **sets `GYM_DR_FEATURE_SET` + `GYM_DR_ASYM_CRITIC`
  env vars** so they survive container re-import.
- `effective_strategy()`, `to_dict()` (callables→dotted paths), `flat_params()` (MLflow),
  `with_overrides(**dotted)` (HPO, via `dataclasses.replace`). `load_config`/`load_search_space`.
- **Smell:** dual surface — flat fields AND `environment` coexist (R5 "rewire to EnvironmentConfig"
  pending); env-var side effects in `__post_init__`.

### gym_dr/environment.py
**Responsible for:** The new typed env-building API. `CameraObs` / `FeatureObs` (with
`asymmetric_critic`) union, `SafeRL`, and `EnvironmentConfig` (observation, action_space,
curriculum, domain_randomization, object_avoidance, safe_rl, n_cars, reward, eval_reward, gui;
derived `camera_obs`/`is_safe_rl`). `FeatureObs.__post_init__` defaults `features=ACTOR_FEATURES`.

### gym_dr/worlds.py
**Responsible for:** World-scheduling strategies (the curriculum). `WorldStrategy` ABC +
`FixedWorlds`, `OrderedSplit` (train vs held-out eval list), `ACL` (spaced-repetition expanding
window, deterministic from seed; **schedule-based unlock, not mastery-gated**). `WorldChunk`.

### gym_dr/action_space.py
**Responsible for:** `ContinuousActionSpaceConfig` / `DiscreteActionSpaceConfig` + DeepRacer
`model_metadata.json` serialization (`write_model_metadata`). `normalize_actions=True` default
→ policy sees [-1,1] → **on-car inference node must rescale to eng units**. `sensor` list passed
straight to the sim (a missing sensor blocks reset).

### gym_dr/randomization.py
**Responsible for:** The value-spec layer. `Range`, `Choice`, `ParamSpec = Range|Choice|float|int`;
`sample_spec` (per-episode draw), `spec_bounds` (low,high envelope for ADR/init), `is_randomized`,
`spec_to_dict`. Standalone, used everywhere.

### gym_dr/domain_randomization.py
**Responsible for:** DR config + automatic-DR controller. `DomainRandomization` (all knobs as
ParamSpec: steering/speed noise, steering/speed bias, obs gaussian/brightness/contrast/gamma,
feature_noise, drag, friction, random_start/direction) with `has_*`/`is_adr` props; `ADR` (adds
step/promote/demote); `ADRState` (live `cur_high` read by wrappers); `ADRController.update(success_rate)`
widens/narrows. **Smell:** `ADR_NOISE_DIMS` hardcoded (drag/friction NOT ADR-ramped); only single-car
uses ADR feedback (multi-car can't — no eval).

---

## 3. Observation / perception / policy — `gym_dr/{perception,feature_obs,asymmetric,rewards}.py`

### gym_dr/perception.py
**Responsible for:** Feature-set definitions (camera→features supervised targets) + the privileged
decomposition + the perception net.
- Tuples: `PERCEPTION_FEATURES`(6) deployable, `PRIVILEGED_EXTRA_FEATURES`(6) critic-only,
  `DYNAMIC_FEATURES`(3), `ALL_FEATURES`(9), `ACTOR_EXTRA_FEATURES`(2), **`ACTOR_FEATURES`(11)** =
  the actor vector (deliberately excludes `progress_frac` — not camera-recoverable).
- `perception_targets/privileged_state/critic_state/actor_targets/all_targets/dynamic_targets` —
  build label vectors from `reward_params` (+ prev for temporal). `enrich_reward_params`. `PerceptionNet`
  (lazy torch). Curvature lookahead k=3 (actor FOV) vs k=5 (privileged).

### gym_dr/envs/feature_obs.py
**Responsible for:** `FeatureObsWrapper` — replaces camera obs with a feature vector built from a
`params_source()` closure (the "reward tap"); applies `feature_noise`; `asymmetric=True` → Dict
`{actor:noised, critic:true}`. **Smell:** params_source closure coupling; asymmetric-dict logic
**duplicated** in `MultiCarVecEnv._obs_from`.

### gym_dr/asymmetric.py
**Responsible for:** `AsymmetricActorCriticPolicy` (+ `KeyExtractor`). Actor reads `obs["actor"]`,
critic `obs["critic"]`; `_build()` swaps `vf_features_extractor` to the critic key and **rebuilds
the optimizer** so the swapped params register. Validates Dict obs has both keys.

### gym_dr/rewards.py
**Responsible for:** Plain `Callable[[dict],float]` rewards + factories + `REWARD_VARIANTS` registry.
Training: `center_line`, `progress_per_step` ((progress/steps)*100 + speed²), `centerline_quadratic`,
`anti_zigzag`, `waypoint_anticipation`, `object_avoidance_aware`. Eval: **`clean_completion`** (default;
pace + linear speed + completion bonus, −10 off-track). Factories `make_weighted_reward`,
`make_progress_reward` (**stateful — one instance per worker, not vec-safe**).

---

## 4. Env factory & DR application — `gym_dr/envs/{dispatch,time_trial,multi_car,wrappers}.py`

### gym_dr/envs/dispatch.py
**Responsible for:** The 2×2 router. `build_env(experiment)` → `time_trial`/`feature_time_trial`
(n_cars≤1) or `multi_car` (n>1). `feature_time_trial` wires the reward-tap + `FeatureObsWrapper`,
reading `GYM_DR_FEATURE_SET`/`GYM_DR_ASYM_CRITIC`. **Smell:** env-var-driven feature-set branch in a
factory; reward-tap closure over a shared mutable dict.

### gym_dr/envs/time_trial.py
**Responsible for:** Single-car `DeepRacerEnv` + the composable wrapper stack (order is load-bearing):
`TimeLimit` → `ActionBounds` → `DragRandomization` → `ActuatorNoise` → `NormalizeActions` →
`GrayscaleObs` → `ObservationNoise`; optional ADR controller attached as `env.adr_controller`.
`random_start/random_direction` passed as `config=` to DeepRacerEnv (sim-side reset modes, not a wrapper).

### gym_dr/envs/multi_car.py
**Responsible for:** `MultiCarVecEnv` — adapts `MultiAgentDeepRacerEnv` to SB3 `VecEnv` with per-car
action/obs transforms, the **DR-warmup ramp**, per-car `_EpisodeMetrics`, and the perception recorder.
- `_dr_scale()` linear 0→1 over `dr_warmup_steps`; `_resample_drag(car)` per-episode drag+bias (scaled);
  `_to_engineering(action,car)` normalize→drag→bias→per-step noise→clip; `_obs_from()` camera (grayscale+
  jitter) or feature (noise + asym dict); `step_wait()` (records frame+metrics, per-car auto-reset,
  resample). `attach_metrics()` (two-phase, post-construction). `can_set_world = hasattr(backend,'set_world')`
  (False). `env_method` exposes `set_recorder_phase`/`set_metrics_eval_mode`.
- `multi_car(experiment)` factory: **23-arg constructor** wiring every DR magnitude via `spec_bounds`.
- **Smells:** huge param list; DR-warmup duplicates single-car ADR; feature-noise/asym duplicated vs
  FeatureObsWrapper; two-phase metrics fragile; camera n≤2 guard (`GYM_DR_ALLOW_CAMERA_NCARS` to override).

### gym_dr/envs/wrappers.py
**Responsible for:** Single-car action/obs wrappers — `ActionBounds`, `NormalizeActions`, `GrayscaleObs`
(BT.601 luma uint8), `ActuatorNoise` (per-step gaussian + per-episode bias; reads live ADRState),
`DragRandomization`, `ObservationNoise` (photometric jitter; live ADRState), `CostInfoWrapper`
(surfaces `info["cost"]` for safe-RL). `apply_image_jitter()` — shared with multi_car.

---

## 5. Orchestration — `gym_dr/{app,trainer,docker_runner,hpo}.py` + `trainers/sb3/__init__.py`

### gym_dr/app.py
**Responsible for:** Mode-dispatched entrypoints + host-side Docker orchestration + crash recovery.
- `train()`/`study()` dispatch on `GYM_DR_IN_CONTAINER`/`GYM_DR_WORKER`. `_train_host` spawns ONE
  container that hot-swaps tracks between chunks (`GYM_DR_ROTATE=1`), with a `SIM_RESTART_RC=75`
  recovery loop (reads `rotation_resume.json`). `_train_one_chunk` (container) → `run_training`.
  `_spawn_workers` for HPO. **The `GYM_DR_*` forwarding list lives here** (DEMO_*, PERCEPTION_OUT,
  VISUAL_DR*, DR_WARMUP_STEPS, FEATURE_SET, ASYM_CRITIC, ALLOW_CAMERA_NCARS, RTF_OVERRIDE, N_CARS, …).
- **Smell:** mode detection by env-var; forwarding list must be kept in sync or the container diverges.

### gym_dr/trainer.py
**Responsible for:** `run_training(experiment, trial=None)` — seed, paths, `model_metadata.json`,
`install_metrics`, resolve strategy, build `TrainingContext` (world plan if rotating), open MLflow,
`trainer.fit(env, ctx)`, return `final_eval_reward` (the HPO objective).

### gym_dr/trainers/sb3/__init__.py
**Responsible for:** `Sb3Trainer.fit` — GPU fallback, `VecFrameStack` if frame_stack>1, **computes
`_n_envs = env.num_envs`** and divides checkpoint+eval freq by it (so freq means TIMESTEPS at any
car count), builds the callback list, and runs the runtime world-rotation loop (`_swap_world` between
chunks, `reset_num_timesteps` only on chunk 0). `_boot_world_consumed` process-global tracks container
reuse across HPO trials. Returns `TrainResult`.

### gym_dr/docker_runner.py
**Responsible for:** Building the `docker run` argv + mounts + watchdog. `spawn_training_chunk`
(blocking, heartbeat watchdog → `SIM_RESTART_RC` on hang), `spawn_workers` (parallel HPO, restart on
crash). `_build_run_cmd` mounts repo + artifacts + mlruns + optuna.db; the **dev override**
(`GYM_DR_DEEPRACER_ENV_SRC`) bind-mounts the local `deepracer_env` package **and** the sim
`launch/`+`urdf/` over the image — this is why local launch/sensor edits take effect without a rebuild.

### gym_dr/hpo.py
**Responsible for:** `make_study` (SQLite Optuna, lenient `MedianPruner`, `TPESampler`), `build_objective`
(`with_overrides` per trial → `run_training`), `run_worker` (`study.optimize(..., catch=Exception)`,
seed offset per worker). `study_storage_default`.

---

## 6. Eval / metrics / plots / dataset — `gym_dr/{evaluate,metrics,perception_recorder}.py` + `trainers/sb3/{callbacks,plots}.py` + `scripts/`

### gym_dr/metrics.py
**Responsible for:** Per-episode metric accumulation + eval-reward switching + trajectory capture + trace.
- `_EpisodeMetrics` — per-step accumulator → `summary()` (the `dr/ep_*` dict stamped in
  `info["dr_episode"]`) + `path_payload()` (for plots). Mode flags (`use_eval_reward`, `capture_path`,
  `cost_fn`, `sink`) deliberately NOT reset per episode (toggled by callbacks).
- `_wrap_reward` (records train+eval reward, returns one per mode; keeps `__wrapped__`),
  `_MetricsEnvWrapper` (single-env), `install_metrics(experiment, run_dir)` — wires it all; for
  `MultiCarVecEnv` calls `attach_metrics`, else wraps single-env.

### gym_dr/trainers/sb3/callbacks.py
**Responsible for:** SB3 callbacks. `MultiWorldEvalCallback` (held-out eval) — `_can_set_world` detects
multi-car (`can_set_world=False`) and does ONE honest current-tracks eval instead of faking per-world
swaps; toggles recorder phase; **restores phase to "train" BEFORE the resume reset** (the
eval→train contamination fix); logs `generalization_gap`. `CtxEvalCallback` (single-world),
`CtxCheckpointCallback` (prunes old), `_EarlyStopMixin` (per-chunk streak), `Status/Mlflow/Reward/Heartbeat/WallClock`
callbacks. `_log_eval_paths` → plots. **freq//n_envs applied in `__init__`** (see §5).

### gym_dr/trainers/sb3/plots.py
**Responsible for:** Eval trajectory charts. `_load_route_borders` (route `.npy` inner/outer, memoised),
`_draw_skeleton` (track backdrop; **now frames axes to the full track extent** so a tiny/stationary path
still shows on the track — the recent "broken plot" fix), `render_overlay` (all eps), `render_episode`
(speed-coloured scatter + start/stop markers).

### gym_dr/perception_recorder.py
**Responsible for:** Per-episode (frame, target) capture → `.npz` shards. `PerceptionRecorder` (+ `_CarBuffer`):
`start_episode/record/flush_episode/flush_all`, train/eval `set_phase`, contiguous frames + `actor_targets`
+ diag, atomic `.tmp.npz`→`.npz`, chmod 0777 dirs / 0666 files, **drops shards if free disk < 3 GB**.
`recorder_from_env` (gated by `GYM_DR_PERCEPTION_OUT`).

### gym_dr/evaluate.py
**Responsible for:** Reconstruct an experiment from `run_config.json` (`experiment_for_model`,
`_reconstruct_experiment` **forces n_cars=1**) + `run_evaluation` (view-mode, deterministic, streams
metrics) + `evaluate_on_tracks` (set_world loop, aggregates clean_completion/completion/progress/offtrack).

### scripts/
- **perception_offload.py** — daemon draining `perception_out` → NVMe → Pi (rsync), keeps disk clear.
- **eval_physical_tracks.py** — out-of-loop eval on physical tracks via `evaluate_on_tracks`
  (forwards FEATURE_SET/ASYM_CRITIC). **evaluate.py** — host/container GUI (VNC) view-mode launcher.
- **multicar_throughput.py** — the n×rtf benchmark harness (override DRG/DENV/IMAGE for the laptop).

---

## 7. deepracer-env sim layer (the dependency)

### environments/multi_agent_env.py
**Responsible for:** N agents on separated track instances in one Gazebo world. `grid_offsets`,
per-car offset `TrackData` + `WorldSwapper.spawn_track_instance`, `step()` (free-running, one pass),
`reset_one(i)`. **NO `set_world`** (deleting one instance breaks all) — the root reason multi-car
can't hot-swap or run in-loop held-out eval.

### environments/deepracer_env.py
**Responsible for:** Single-car Gymnasium env. `DEFAULT_ACTION_SPACE` Box(2). `set_world(name)` —
between-episode track swap (pause, delete/spawn, rebuild TrackData, teleport, drain stale frames).

### environments/world_swap.py
**Responsible for:** Gazebo track delete/spawn over `gazebo_ros` services. `spawn_track_instance(world,
model, offset)` (offset embedded in SDF `<pose>`), wall-clock confirm polls, raises `WorldSwapError` on
the intermittent `delete_model` gzserver segfault (recoverable).

### track_geom/track_data.py
**Responsible for:** Track geometry + per-car offset. `TrackData.create(world, offset)` (non-singleton,
shifts waypoint columns), `TrackLine` (shapely), `get_racecar_start_pose(idx, racer_num, start)`.

### agent_ctrl/rollout_agent_ctrl.py
**Responsible for:** Per-agent control: reward eval, reset-rules (random_start/direction), DR magnitudes,
visual randomizer (gated `GYM_DR_VISUAL_DR=1`, primary car only), dedicated reset RNG.

### sensors/sensors_rollout.py
**Responsible for:** Sensors + `DoubleBuffer`. `Camera.get_state(block=True, timeout=120)` — **blocks if
no frame**; the single-OGRE-render-thread serialization is why >2 real cameras degrade RTF (the launch
limit, now lifted, exposes this true render ceiling).

### domain_randomizations/visual_randomizer.py
**Responsible for:** Per-episode track/background recolor (sim2real). Only track-surface + background
visuals are service-recolorable (not lane lines / skybox). Gated `GYM_DR_VISUAL_DR(_SEED)`.

### simulation/.../launch/racetrack_with_racecar.launch + scripts/car_node.py
**Responsible for:** ROS/Gazebo launch — spawns N car bodies. **Generalized:** `racecar_2..7` includes
are now conditional on `len(simapp_versions) > i` (was hardcoded 2); `car_node.py args=$(eval len(...))`.
`car_node.py` handles N cars (`get_racecar_names`, per-car start poses). Hard ceiling now 8 (racecar_7).

---

## 8. Cross-cutting concerns

### 8.1 The `GYM_DR_*` env-var protocol (host → container IPC)
| Var | Set by | Read by | Meaning |
|---|---|---|---|
| `GYM_DR_IN_CONTAINER` / `GYM_DR_WORKER` | docker_runner | app.py | mode detect (container / HPO worker) |
| `GYM_DR_ROTATE` | app.py | trainer/Sb3Trainer | enable in-container world rotation |
| `WORLD_NAME` / `RESUME_FROM` / `CHUNK_*` / `SEED` | app.py | _train_one_chunk | per-chunk overrides |
| `GYM_DR_N_CARS` / `GYM_DR_DEMO_WORLDS` / `GYM_DR_DEMO_SPACING` | experiment+app | multi_car / launch | multi-car spawn |
| `GYM_DR_ALLOW_CAMERA_NCARS` | experiment(module-level) | multi_car guard | allow camera n>2 |
| `GYM_DR_FEATURE_SET` / `GYM_DR_ASYM_CRITIC` | config.__post_init__ | dispatch/multi_car | feature set + asym Dict obs |
| `GYM_DR_DR_WARMUP_STEPS` | oracle experiment | multi_car `_dr_scale` | DR warmup ramp |
| `GYM_DR_PERCEPTION_OUT` | experiment | perception_recorder | enable recording + path |
| `GYM_DR_VISUAL_DR(_SEED)` | experiment | deepracer-env RolloutCtrl | sim-side visual recolor |
| `GYM_DR_CAPTURE_*` | capture experiment | perception_capture_heldout | held-out split capture |
| `GYM_DR_HEARTBEAT` / `GYM_DR_WATCHDOG*` | docker_runner | callbacks / host | hang detection |
| `GYM_DR_MAX_EPISODE_STEPS` / `RTF_OVERRIDE` / `GYM_DR_FRICTION_MU` | various | env / sim | episode cap, RTF, friction |

### 8.2 The 2×2 env dispatch
`(n_cars≤1, camera)` → `time_trial`; `(≤1, feature)` → `feature_time_trial`; `(>1, camera)` →
`multi_car`(camera, cap 2); `(>1, feature)` → `multi_car`(feature, scales to ~8/12).

### 8.3 DR applied twice
Single-car: composable wrappers reading **live `ADRState`** (per-step feedback). Multi-car: inline in
`MultiCarVecEnv` with a **self-counted linear warmup** (no eval signal available). Same intent, two
code paths → the #1 duplication to unify.

---

## 9. Refactor opportunities (consolidated, prioritized)

The mappers converged on these. Roughly highest-leverage first:

1. **Kill env-var-driven control flow → put it in the config object.** `GYM_DR_FEATURE_SET`,
   `GYM_DR_ASYM_CRITIC`, `GYM_DR_DR_WARMUP_STEPS`, `GYM_DR_N_CARS`, `GYM_DR_ALLOW_CAMERA_NCARS`,
   `GYM_DR_DEMO_WORLDS` are all really `EnvironmentConfig` fields tunneled through env vars because
   the container re-imports the script. Finish **R5 (rewire to EnvironmentConfig)** and pass the
   serialized config to the container instead of re-importing + re-deriving. Removes the silent-divergence
   class of bug (forwarding list out of sync).
2. **Unify the DR application.** One `DRScheduler` consumed by both single-car wrappers and
   `MultiCarVecEnv` (warmup ramp + ADR feedback as two strategies). Collapses the duplicated
   feature-noise/asym-obs/bias/jitter logic (currently in feature_obs.py, wrappers.py, multi_car.py).
3. **Shrink the `multi_car()` 23-arg constructor** — pass `domain_randomization` + `observation` objects
   and extract inside, like `time_trial` does. Make per-car metrics part of construction (not the
   fragile two-phase `attach_metrics`).
4. **Replace the host/container `GYM_DR_IN_CONTAINER` string-branch** with explicit `run_host` /
   `run_container` entry points; serialize the config (not re-import the script).
5. **Multi-car `set_world` (or a clean alternative).** Today it forces per-chunk fresh containers and
   blocks in-loop held-out eval. Either implement an all-N-instances swap in `MultiAgentDeepRacerEnv`,
   or formalize the "frozen single-car held-out pass" as the supported pattern.
6. **Extract the by-track split + the 18-track geometry map to versioned data** (`gym_dr/data/splits`
   + a `.json` artifact) instead of duplicating in each oracle experiment + `camera_cnn_dataset`.
7. **Make DR knobs ADR-ramp uniformly** (drag/friction currently bypass ADR) and add an explicit
   `frozen=True` on `Sb3Trainer` (vs `learning_rate=0` override for capture).
8. **Harden the asym contract** — `AsymmetricActorCriticPolicy` should assert its Dict obs has exactly
   `{actor,critic}` at build; `make_progress_reward` statefulness should be made vec-safe.
9. **Sim-layer:** parameterize the launch's 8-car ceiling; replace visual-DR string parsing with an
   enum; document the camera single-OGRE render ceiling as the real (not launch) limit.

---

---

## 10. Architecture-robustness study (MLP vs LSTM) + on-car deployment · 2026-06-29

A study comparing **how robust each policy architecture is** to the deployable handicaps — a
**noised** actor feature vector (perception noise) + an **unobservable per-episode steering
bias** (actuator miscalibration). All **asymmetric** (critic sees the clean vector), **no frame
stacking** on either arm (stacking is an implementation hack; the LSTM *is* the memory, the MLP
is the memoryless control). The task keeps the hidden bias (option A) so it mirrors real-car
miscalibration — which only **memory** can infer (a POMDP for a memoryless MLP).

**Code added for this:**
- `gym_dr/asymmetric.py::asymmetric_recurrent_policy()` — lazy factory for the LSTM analogue of
  `AsymmetricActorCriticPolicy`: actor LSTM on `obs["actor"]` (noised), a separate critic LSTM on
  `obs["critic"]` (clean), via `RecurrentActorCriticPolicy` + the `KeyExtractor` swap. Lazy so
  importing `asymmetric` never requires sb3-contrib.
- `gym_dr/trainers/sb3/algorithms.py` — `import_algos()` now registers `recurrent_ppo` →
  `sb3_contrib.RecurrentPPO` (optional import; absent → just not in the registry).
- `gym_dr/trainers/sb3/callbacks.py::_eval_policy()` — **recurrent-aware eval**: SB3's
  `evaluate_policy` calls `predict(obs)` *without* the LSTM state, so it would judge a recurrent
  net as if memoryless. `_eval_policy` threads `state`+`episode_start` (resets at episode end) for
  recurrent models; non-recurrent fall through to SB3's `evaluate_policy` unchanged. The
  `MultiWorldEvalCallback` eval calls route through it so the LSTM is scored fairly.
- `gym_dr/docker_runner.py` — bind-mounts the host venv's `sb3_contrib` into the container's
  `dist-packages` (it's not in the base image), like the `deepracer_env` dev-override.
- `experiments/oracle_hpo.py` (renamed study `arch_robust_hpo`) — search-space `arch ∈ {mlp, lstm}`:
  mlp → `ppo` + `AsymmetricActorCriticPolicy`; lstm → `recurrent_ppo` +
  `asymmetric_recurrent_policy()` (+ `lstm_hidden_size`). `frame_stack=1` both. Fixed
  `steering_bias=GYM_DR_HPO_BIAS` (default 10°) = the same task for every trial. Objective =
  held-out clean-completion (recurrent-aware). Optuna compares the two head-to-head.

**What the result decides — and the on-car deployment cost of each:**

| Arch | On-car state | ONNX / OpenVINO export | On-car inference node | Handles hidden miscalibration? |
|---|---|---|---|---|
| **MLP** (memoryless) | none — stateless | trivial (`obs → action`) | feed one obs/step | **No** (POMDP) — the control |
| **MLP + frame-stack** | none (model stateless; a **ring buffer of N obs** lives in the node) | straightforward (bigger input) | buffer last N obs, feed the stack (the DeepRacer node already buffers frames) | Yes (fixed window) |
| **MLP + prev-action** | none (node feeds back last command) | trivial (`[obs, a_{t-1}] → action`) | feed back the last commanded action | Yes, if 1-step suffices (bias ≈ achieved−commanded) |
| **LSTM** (recurrent) | **stateful** — carry `(h, c)` across steps | **harder**: ONNX must expose the LSTM state as extra **inputs+outputs** (stateful); OpenVINO IR via `gym_dr/optimize.py` then needs the state plumbed (Assign/ReadValue or external state mgmt) | maintain `(h,c)`, feed back each step, **reset at episode start** | Yes — arbitrary/learned memory |

So the study is also a **deployment decision**: if a memoryless MLP can't and an LSTM can, the
extra cost of stateful on-car inference (state carry + recurrent ONNX/OpenVINO export) is
justified; if a cheaper stateless option (stacking / prev-action) matched the LSTM, you'd ship
that instead and keep the car's inference node stateless. (`MLP+prev-action` and the simple-RNN
arms were validated but scoped out of this run; both remain one search-space entry away.)

**Hardware-relevant caveats for the LSTM path:**
- Export: a recurrent SB3 policy must be traced with the hidden state as I/O; the on-car node
  owns initialization + per-step feedback + episode-boundary reset (mishandling the reset =
  state leaks across laps).
- OpenVINO: LSTM ops are supported, but the project's `optimize.py` (ONNX→IR, [[onnx-openvino]])
  currently targets a stateless graph — stateful inference is the new work for an LSTM deploy.
- Latency/footprint: a 64–256-unit LSTM adds negligible compute on the car's Atom/Pi vs the CNN.

---

_Generated 2026-06-29 from a 5-way read-only code sweep, extended 2026-06-29 with the architecture
study + deployment section. Verify file:line against current code before acting — this is a
point-in-time map for a refactor that will move things._
