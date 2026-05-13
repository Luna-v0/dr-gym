# Artifact layout

Every training run writes to `artifacts/<run_name>/`. The layout is:

```text
artifacts/<run_name>/
├── run_config.json                       # fully resolved cfg.to_dict()
├── training_status.json                  # live status (initialized → running → completed|...|failed)
├── model_metadata.json                   # DeepRacer-compatible sidecar (run-level)
├── reward_function.py                    # rendered reward source for this run
│
├── initial_model.zip
├── initial_model.model_metadata.json     # ← every .zip gets a sibling
├── latest_model.zip
├── latest_model.model_metadata.json
├── final_model.zip
├── final_model.model_metadata.json
│
├── best_model/
│   ├── best_model.zip
│   └── best_model.model_metadata.json
│
├── checkpoints/
│   ├── ppo_checkpoint_<steps>_steps.zip
│   ├── ppo_checkpoint_<steps>_steps.model_metadata.json
│   └── ...
│
├── tensorboard/                          # SB3 TB events
├── eval/                                  # SB3 EvalCallback log
└── export_bundle/
    ├── model_metadata.json               # ready-to-ship metadata copy
    └── reward_function.py
```

## Failure / interruption variants

| Status | Extra zip + metadata |
|---|---|
| `completed` | `final_model.zip` written |
| `time_limit_reached` | `final_model.zip` written; wall-clock hit |
| `interrupted` (SIGINT/SIGTERM) | `interrupted_model.zip` written |
| `failed` (Python exception) | `crash_recovery_model.zip` written |

In every case, `latest_model.zip` is updated in the `finally` block too. It's the safest resume target.

## Why every checkpoint has a sidecar

DeepRacer requires `model_metadata.json` next to the model to know the action space. If you cherry-pick a single `.zip` from `checkpoints/` to ship to the physical car, the sibling `.model_metadata.json` travels with it. Losing the metadata = unshippable model.

This is enforced by:

- `gym_dr/trainer.py:_save_with_metadata` — wraps every explicit `model.save(...)` call.
- `gym_dr/callbacks/checkpoint.py:MetadataAwareCheckpointCallback` — subclass of `CheckpointCallback` that writes the sibling on every periodic checkpoint.
- `gym_dr/callbacks/eval.py:MetadataAwareEvalCallback` — writes the sibling for `best_model.zip` whenever it's saved.

## Verifying integrity

After a run, no `*.zip` under the run dir should be orphaned:

```bash
find artifacts/<run_name> -name '*.zip' | while read z; do
  test -f "${z%.zip}.model_metadata.json" || echo "MISSING: $z"
done
```

Empty output = all good.
