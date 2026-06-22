# Authoring your own trainer — the contract

The system is **not** tied to Stable-Baselines3. A trainer is *any* object with
`fit(env, ctx: TrainingContext) -> TrainResult`. Pass it as `ExperimentConfig(trainer=...)`
and the orchestrator runs the rest — Docker dispatch, MLflow lifecycle, status JSON, artifact archival,
Optuna — unchanged. Drop in a from-scratch PyTorch loop, CleanRL, JAX, RLlib, your own PID-Lagrangian —
whatever — and reuse all the plumbing via the **`TrainingContext` services** below.

## What you get (reuse these — don't reinvent them)
`gym_dr/trainers/base.py:TrainingContext`:

| Service | What it does |
|---|---|
| `ctx.save_model(fn, name=...)` | Write a top-level model (`initial_model`/`latest_model`/`final_model`) **+ DeepRacer `model_metadata.json` sidecar** (so any checkpoint is shippable to the car). |
| `ctx.save_checkpoint(fn, step=...)` | Periodic checkpoint under `checkpoints/` + sidecar. |
| `ctx.log_metrics(dict, step)` | Scalars to **TensorBoard *and* MLflow** in one call (loss curves, etc.). |
| `ctx.record_episode(info, step)` | Drains the `dr/ep_*` episode metrics from `info["dr_episode"]` (present when the env was built through the orchestrator) → TB+MLflow. |
| `ctx.evaluate(predict_fn, env, n_episodes, step)` | The project's **held-out clean-completion eval**: swaps to each `eval_worlds`, runs episodes, aggregates the success-criterion metrics, logs them, and calls `report_eval` (MLflow + Optuna pruning). Returns the aggregate. (Raw gymnasium env + `predict_fn(obs)->action`.) |
| `ctx.swap_world(env, world)` | Hot-swap the Gazebo track (curriculum / eval). Works for raw env or SB3 `VecEnv`. |
| `ctx.report_eval(mean, step)` | MLflow + Optuna pruning (also called inside `evaluate`). |
| `ctx.set_status(status, extra)` | Update `training_status.json` with progress. |

Fields: `run_dir`, `action_space`, `training` (`TrainingConfig`: `total_timesteps`, `eval_freq`,
`checkpoint_freq`, `n_eval_episodes`, …), `seed`, `metrics_state`, `world_plan`, `chunk_steps`,
`rotate_start_index`, `eval_worlds`, `trial`.

## Lifecycle to follow
1. `ctx.save_model(fn, name="initial_model")` before training.
2. Train. `ctx.log_metrics({...}, step)` for your losses; `ctx.record_episode(info, step)` on terminal steps.
3. **Curriculum:** if `ctx.world_plan` is set, train `ctx.chunk_steps` per world and `ctx.swap_world(env, next)`
   between chunks (start at `ctx.rotate_start_index`).
4. Every `training.eval_freq`: `ctx.evaluate(predict_fn, env, n_episodes=training.n_eval_episodes, step=...)`.
5. Every `training.checkpoint_freq`: `ctx.save_checkpoint(fn, step=...)`.
6. In a `finally`: `ctx.save_model(fn, name="latest_model")` (resume/crash target). On clean exit:
   `ctx.save_model(fn, name="final_model")`.
7. Return `TrainResult(final_eval_reward=..., final_model_path=..., extra={"timesteps_completed":...,
   "elapsed_seconds":...})`.

## Performance
Every service is **I/O at a boundary** (rollout/episode/eval), never per env-step — the only per-step hook is
the reward-params tap, which already runs and is cheap. So the interface adds **no measurable training
overhead**; your algorithm's compute dominates. (Keep `log_metrics` out of the inner step loop.)

## Example & reference
- `experiments/custom_trainer_example.py` — a copy-paste skeleton; drop your update in.
- `gym_dr/trainers/base.py` — the `Trainer` Protocol + `TrainingContext`.
- `gym_dr/trainers/sb3/` — the SB3 reference implementation.
- The planned **SB3 PID-Lagrangian** trainer (safe-RL, D9) will be a second backend built against this
  same contract (`docs/reports/safe-rl-backend.md`).
