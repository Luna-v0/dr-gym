# TensorBoard

`Sb3Trainer` writes TensorBoard event files automatically on every training chunk, under:

```text
artifacts/<chunk_name>/tensorboard/
```

`<chunk_name>` is `<experiment.name>_rot<r>_<world>` — e.g. `quick_test_rot0_reinvent_base`. The toggle is `TrackingConfig.tensorboard` (default `True`).

## View runs

The helper script runs TensorBoard **on the host** (the simapp container ships an old TB 2.14 that's incompatible with modern protobuf — a recent uv-installed TB sidesteps the issue):

```bash
# All chunks under ./artifacts/ — useful for comparing runs side by side
./run_tensorboard.sh

# One specific chunk
./run_tensorboard.sh quick_test_rot0_reinvent_base

# Different port (if 6006 is taken)
PORT=6007 ./run_tensorboard.sh
```

Then open <http://localhost:6006>.

## What you'll see

SB3 writes the standard scalars:

- `rollout/ep_rew_mean` — mean episode reward (the primary signal you want to climb).
- `rollout/ep_len_mean` — episode length in steps.
- `train/loss`, `train/policy_loss`, `train/value_loss`, `train/entropy_loss`, `train/explained_variance`, `train/clip_fraction` — PPO training diagnostics.
- `time/fps`, `time/iterations` — throughput.
- `eval/mean_reward`, `eval/mean_ep_length` — periodic eval rollouts.

If you serve the whole `artifacts/` dir, every chunk shows up as a separate run in the left sidebar. Use the regex filter at the top of the sidebar to narrow them down (`^quick_test_` to see only your latest experiment, etc.).

## Old runs from before the refactor

Pre-refactor run dirs (e.g. `long_4h_rtf100_*`, `test_cpu_persist*`, `test_long_limit_rtf100`) are still on disk. Hide them with the sidebar regex filter, or delete:

```bash
rm -rf artifacts/long_4h_rtf100_* artifacts/test_*
```

## Verify event files exist

```bash
find artifacts/<chunk_name>/tensorboard -type f
```

You should see `events.out.tfevents.*` files. If the dir is empty, the chunk hasn't flushed events yet (give it ~30 s of training) or `cfg.tracking.tensorboard` was set to `False`.

## Troubleshooting

### `MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`

You're running the *container's* old TensorBoard. Use `./run_tensorboard.sh` (which now invokes the host's TB via `uv run tensorboard`) — not `docker run ... tensorboard`.

### Port already in use

```bash
PORT=6007 ./run_tensorboard.sh
```

### Multiple chunks, want one continuous reward curve

The host orchestrator tags every chunk with `run_group=<experiment.name>`. In TensorBoard the chunks appear as separate runs (they technically are), but the MLflow UI groups them via the `run_group` tag — easier comparison there. See [tracking.md](tracking.md).
