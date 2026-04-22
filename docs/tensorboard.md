# TensorBoard Tutorial

Date: 2026-04-08

## What is already wired in this repo

The training script already writes TensorBoard logs to a per-run directory:

- `artifacts/<RUN_NAME>/tensorboard/`

The relevant code is in:

- [train.py](/mnt/hd/Repos/gym-dr/train.py)

The project now also includes:

- `tensorboard` in [requirements.txt](/mnt/hd/Repos/gym-dr/requirements.txt)
- a helper script: [run_tensorboard.sh](/mnt/hd/Repos/gym-dr/run_tensorboard.sh)

## Important caveat

TensorBoard event files are only created if the training container was started from an image that already has `tensorboard` installed.

That means:

- older runs started before the image rebuild will **not** have TensorBoard event files
- new runs started after rebuilding the image will have them

In particular, if a run started while the training script printed:

- `tensorboard is not installed; disabling tensorboard logging`

then that run will not contain TensorBoard event data.

## Step 1: rebuild the image

From the project root:

```bash
cd /mnt/hd/Repos/gym-dr
docker build -t my-deepracer-project:cpu .
```

After this rebuild, new training runs should write TensorBoard event files.

## Step 2: start a training run

Example:

```bash
cd /mnt/hd/Repos/gym-dr
RUN_NAME=tb_test \
TOTAL_TIMESTEPS=500000 \
MAX_TRAIN_SECONDS=3600 \
RTF_OVERRIDE=100 \
./run_cpu_training.sh
```

This creates a run directory like:

```text
artifacts/tb_test/
```

and the TensorBoard logs should appear under:

```text
artifacts/tb_test/tensorboard/
```

## Step 3: launch TensorBoard

### Option A: view one specific run

```bash
cd /mnt/hd/Repos/gym-dr
RUN_NAME=tb_test ./run_tensorboard.sh
```

Then open:

```text
http://localhost:6006
```

### Option B: view all runs together

```bash
cd /mnt/hd/Repos/gym-dr
./run_tensorboard.sh
```

This serves the entire `artifacts/` directory, so multiple runs can appear in the same TensorBoard instance.

## Step 4: change the port if needed

If `6006` is already in use:

```bash
cd /mnt/hd/Repos/gym-dr
PORT=6010 RUN_NAME=tb_test ./run_tensorboard.sh
```

Then open:

```text
http://localhost:6010
```

## How the helper works

The helper script runs TensorBoard inside Docker and mounts the local `artifacts/` folder into the container.

You do not need a local Python install for this workflow as long as the project image has been rebuilt.

## How to verify it is working

Check whether event files exist:

```bash
find artifacts/tb_test/tensorboard -type f
```

If TensorBoard logging is active, you should see files with names similar to:

```text
events.out.tfevents....
```

You can also inspect the run config:

```bash
cat artifacts/tb_test/run_config.json
```

## Troubleshooting

### TensorBoard page opens but no runs appear

Most common causes:

- the image was not rebuilt after adding `tensorboard`
- the run started from an older image
- the selected `RUN_NAME` is wrong
- the run has not progressed enough to flush event files yet

Check:

```bash
find artifacts/<RUN_NAME>/tensorboard -type f
```

### Training log says TensorBoard is disabled

If the log contains:

```text
tensorboard is not installed; disabling tensorboard logging
```

that run will not produce TensorBoard event files.

Rebuild the image and start a new run.

### I want TensorBoard for the currently running 4-hour job

If that job started from an image without TensorBoard installed, it will not begin emitting TensorBoard events mid-run.

You would need to:

1. rebuild the image
2. stop the old run or let it finish
3. start a new run or resume from `latest_model.zip`

Example resume flow:

```bash
cd /mnt/hd/Repos/gym-dr
RUN_NAME=tb_resume \
RESUME_FROM=/workspace/artifacts/long_4h_rtf100_20260408_145000/latest_model.zip \
TOTAL_TIMESTEPS=1000000000 \
MAX_TRAIN_SECONDS=14400 \
RTF_OVERRIDE=100 \
./run_cpu_training.sh
```

Then launch:

```bash
cd /mnt/hd/Repos/gym-dr
RUN_NAME=tb_resume ./run_tensorboard.sh
```

## Related files

- [requirements.txt](/mnt/hd/Repos/gym-dr/requirements.txt)
- [train.py](/mnt/hd/Repos/gym-dr/train.py)
- [run_cpu_training.sh](/mnt/hd/Repos/gym-dr/run_cpu_training.sh)
- [run_tensorboard.sh](/mnt/hd/Repos/gym-dr/run_tensorboard.sh)
