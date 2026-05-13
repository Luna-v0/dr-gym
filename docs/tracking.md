# Experiment tracking: MLflow + TensorBoard

## MLflow

Local file store at `mlruns/` (bind-mounted into the container at `/workspace/mlruns`).

```bash
mlflow ui --backend-store-uri file://$PWD/mlruns
# open http://localhost:5000
```

Every training run logs:

- **Params** — every leaf of `ExperimentConfig` as a flat dotted key. Examples: `algorithm.kwargs.learning_rate`, `reward.params.reward_center`, `action_space.steering_high`, `training.total_timesteps`.
- **Metrics** — SB3 logger scalars mirrored on each rollout end via `gym_dr/callbacks/mlflow.py:MlflowSB3Callback`. Includes `rollout_ep_rew_mean`, `train_loss`, `train_entropy_loss`, `train_explained_variance`, `eval_mean_reward`, etc. (slashes become underscores for MLflow compat.)
- **Artifacts** — the full `artifacts/<run_name>/` tree at end-of-run: TB events, all model zips, every `*.model_metadata.json` sibling, rendered `reward_function.py`, `run_config.json`, `training_status.json`, `export_bundle/`.
- **Tags** — `algorithm`, `world_name`, `action_space_type`, plus user-supplied `cfg.tracking.tags`.

HPO runs are nested: parent run named `hpo:<study_name>` with one child per trial. See [hpo.md](hpo.md).

## TensorBoard

Per-run TB logs are still under `artifacts/<run_name>/tensorboard/`.

```bash
RUN_NAME=quick_test ./run_tensorboard.sh
# or for all runs:
./run_tensorboard.sh
```

More details in [tensorboard.md](tensorboard.md).
