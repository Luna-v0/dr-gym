# DeepRacer SB3 Training Environment

This repo trains a Stable-Baselines3 PPO policy against `seresheim/deepracer-env` inside Docker.

The default workflow is CPU training. It saves checkpoints, final models, run metadata, reward code, model metadata, and optional TensorBoard logs under `artifacts/`.

## Prerequisites

- Docker (with the daemon running and `buildx` available)
- `git`
- ~50 GB free in Docker's storage location for the first build

Nothing else is required — `bootstrap.sh` handles the upstream simulator image
and the project image.

## First-Time Setup

Run once on a fresh machine:

```bash
cd /mnt/hd/Repos/gym-dr
./bootstrap.sh
```

This will:

1. Run preflight checks (docker daemon, buildx, disk space)
2. Clone `github.com/seresheim/deepracer-env` into `.deepracer-env-upstream/` (if missing) and check out the pinned commit
3. Build the base simulator image `awsdeepracercommunity/deepracer-env:0.1-cpu` (if missing) — this step takes a while
4. Build the project training image `my-deepracer-project:cpu`
5. Run an import sanity check against the new image

The script is idempotent: images and source already present are reused, so
re-running it is fast.

### Options

```text
./bootstrap.sh [-a cpu|gpu] [-u UPSTREAM_DIR] [-r UPSTREAM_REF] [-h]
```

- `-a` — architecture (`cpu` default, `gpu` builds the CUDA base image)
- `-u` — where to clone/find the upstream source. Useful if you already have a checkout.
- `-r` — git ref of upstream to pin to. The default pin is a known-good commit.
- `-h` — show usage.

Equivalent env vars: `ARCH`, `UPSTREAM_DIR`, `UPSTREAM_REF`.

### Iterating Without Rebuilding

The project image contains only third-party dependencies. The project source
is bind-mounted into the container at run time, so editing these files does
**not** require rebuilding the image — just re-run `./run_cpu_training.sh`:

- PPO hyperparameters → edit a YAML in `configs/` (recommended) or pass env vars
- Reward shaping → edit `reward.py`
- Action space / sensors → edit `model_metadata.json`
- Training loop, callbacks, logging → edit `train.py`

Only rebuild (`./bootstrap.sh` or `docker build -t my-deepracer-project:cpu .`)
when `requirements.txt` changes.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Docker daemon not reachable` | Daemon stopped or you're not in `docker` group | `sudo systemctl start docker`; verify `docker info` works |
| `docker buildx not available` | Older Docker | Install buildx ([instructions](https://github.com/docker/buildx#installing)) |
| `Only N GB free` | Disk too full | Free up space; first build needs ~50 GB |
| Upstream `build.sh` fails on apt-get | Network / apt mirror issue | Retry; consider re-running with `-r <newer_ref>` |
| Sanity check fails | Bad pip layer | Delete `my-deepracer-project:cpu` and re-run `./bootstrap.sh` |

## Running a Training Run

Every run is described by a YAML in `configs/`. Pass the path as the first
argument:

```bash
cd /mnt/hd/Repos/gym-dr
./run_cpu_training.sh configs/quick.yaml
```

To create a new run, copy a config, edit it, and launch:

```bash
cp configs/quick.yaml configs/my_experiment.yaml
$EDITOR configs/my_experiment.yaml
./run_cpu_training.sh configs/my_experiment.yaml
```

The chosen YAML is copied into the run's artifacts dir as `config.yaml` for
reproducibility.

### Ad-hoc Overrides

Environment variables override anything in the YAML. Useful for quick tweaks
without editing the file:

```bash
TOTAL_TIMESTEPS=100 ./run_cpu_training.sh configs/quick.yaml
RUN_NAME=my_smoke_run ./run_cpu_training.sh configs/quick.yaml
```

### Included Configs

- `configs/quick.yaml` — ~5k-timestep smoke test
- `configs/default.yaml` — 4-hour wall-clock run at high requested RTF

### Output

Artifacts land under `artifacts/<run_name>/`:

```text
artifacts/<run_name>/config.yaml          # copy of the YAML used
artifacts/<run_name>/run_config.json      # fully resolved parameters
artifacts/<run_name>/training_status.json # live status
artifacts/<run_name>/latest_model.zip
artifacts/<run_name>/final_model.zip
artifacts/<run_name>/checkpoints/
artifacts/<run_name>/tensorboard/
artifacts/<run_name>/export_bundle/
```

## Detached Long Run With tmux

For unattended training, run inside `tmux`:

```bash
cd /mnt/hd/Repos/gym-dr
tmux new-session -s deepracer_train
```

Then start training:

```bash
./run_cpu_training.sh configs/default.yaml
```

Detach from tmux with:

```text
Ctrl-b d
```

Check the session later:

```bash
tmux attach -t deepracer_train
```

## Check Training Status

Inspect the status JSON:

```bash
cat artifacts/<run_name>/training_status.json
```

Follow checkpoint creation:

```bash
find artifacts/<run_name>/checkpoints -maxdepth 1 -type f | sort
```

Inspect the fully resolved run configuration:

```bash
cat artifacts/<run_name>/run_config.json
```

## Resume Training

Set `resume_from` in a YAML to the **container** path of a previous checkpoint
(the host `artifacts/` directory is mounted at `/workspace/artifacts` inside
the container):

```yaml
# configs/resume.yaml
run_name: resume_from_default_4h
resume_from: /workspace/artifacts/default_4h/latest_model.zip
total_timesteps: 1000000000
max_train_seconds: 14400
```

```bash
./run_cpu_training.sh configs/resume.yaml
```

## TensorBoard

TensorBoard logs are written to:

```text
artifacts/<RUN_NAME>/tensorboard/
```

Launch TensorBoard for one run:

```bash
cd /mnt/hd/Repos/gym-dr
RUN_NAME=long_4h_rtf100 ./run_tensorboard.sh
```

Open:

```text
http://localhost:6006
```

Launch TensorBoard for all runs:

```bash
cd /mnt/hd/Repos/gym-dr
./run_tensorboard.sh
```

More details are in [docs/tensorboard.md](docs/tensorboard.md).

## Configuration Reference

Every key below can appear in a YAML config (lowercase) or be passed as an
environment variable (uppercase). Env vars win when both are set.

| YAML key                | Env var                 | Default          | Notes                                              |
|-------------------------|-------------------------|------------------|----------------------------------------------------|
| `run_name`              | `RUN_NAME`              | timestamped      | Output dir under `artifacts/<run_name>/`.          |
| `world_name`            | `WORLD_NAME`            | `reinvent_base`  | DeepRacer track.                                   |
| `total_timesteps`       | `TOTAL_TIMESTEPS`       | `500000`         | SB3 timestep target.                               |
| `max_train_seconds`     | `MAX_TRAIN_SECONDS`     | _none_           | Wall-clock limit, e.g. `14400` for 4h.             |
| `rtf_override`          | `RTF_OVERRIDE`          | _none_           | Requested Gazebo real-time factor.                 |
| `checkpoint_freq`       | `CHECKPOINT_FREQ`       | `1000`           | Checkpoint save frequency in timesteps.            |
| `sb3_device`            | `SB3_DEVICE`            | `cpu`            | `cpu` or `cuda`.                                   |
| `resume_from`           | `RESUME_FROM`           | _none_           | Container path to a previous `.zip` checkpoint.    |
| `n_steps`               | `N_STEPS`               | `256`            | PPO rollout steps.                                 |
| `batch_size`            | `BATCH_SIZE`            | `64`             | PPO batch size.                                    |
| `learning_rate`         | `LEARNING_RATE`         | `3.0e-4`         | PPO learning rate.                                 |
| `ent_coef`              | `ENT_COEF`              | `0.01`           | PPO entropy coefficient.                           |
| `status_update_steps`   | `STATUS_UPDATE_STEPS`   | `1000`           | Min timesteps between status JSON updates.         |
| `status_update_seconds` | `STATUS_UPDATE_SECONDS` | `30`             | Min wall-clock seconds between status updates.     |

## Model Outputs

Each run directory contains:

```text
initial_model.zip
latest_model.zip
final_model.zip
checkpoints/
run_config.json
training_status.json
model_metadata.json
reward_function.py
export_bundle/
```

`latest_model.zip` is the safest resume target because it is saved in the `finally` block even when training stops early.

## Physical Car Caveat

The saved `.zip` files are Stable-Baselines3 checkpoints. They are useful for local resume and inference experiments.

They are not AWS DeepRacer-native model export bundles for the stock physical-car upload flow.

The current physical-car direction is documented in [docs/physical-car-integration-notes.md](docs/physical-car-integration-notes.md).
