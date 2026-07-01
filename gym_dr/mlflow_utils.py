"""MLflow lifecycle helpers used by ``gym_dr.trainer``.

Each training chunk opens its own MLflow run via ``start_run``. The host
orchestrator does NOT pre-open a parent run — that pattern is fragile across
MLflow versions (host and container often have different MLflow builds, and
the older one can't always parse run metadata written by the newer one).

Chunks of one multi-world rotation, and trials of one HPO study, are
grouped via the ``run_group`` tag. The host sets the ``MLFLOW_RUN_GROUP``
env var when spawning each container; ``start_run`` reads it and tags the
run. In MLflow UI, filter by ``tags.run_group = "<your_experiment_name>"``
to see them all together.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from gym_dr.config import ExperimentConfig


def _mlflow():
    import mlflow

    return mlflow


def _set_experiment_racesafe(mlflow, name: str) -> None:
    """``mlflow.set_experiment`` is NOT concurrency-safe on the file store.

    When N HPO workers boot together (or a ``--rm`` worker restarts), each one
    sees the experiment "doesn't exist" and races to ``create_experiment``; the
    losers raise ``MlflowException: Experiment '<name>' already exists`` and the
    whole trial is marked FAILED. (Empirically this skews toward whichever arm
    sets up a hair slower — e.g. the LSTM trials lost the create race to the MLP
    trial every time.) Retry by binding to the now-existing experiment by id.
    """
    from mlflow.exceptions import MlflowException

    for attempt in range(6):
        try:
            mlflow.set_experiment(name)
            return
        except MlflowException:
            exp = mlflow.get_experiment_by_name(name)
            if exp is not None:                      # someone else won the race
                mlflow.set_experiment(experiment_id=exp.experiment_id)
                return
            if attempt == 5:
                raise                                # genuinely cannot create it
            time.sleep(0.25 * (attempt + 1))         # transient FS contention


def _tags_for(cfg: ExperimentConfig) -> dict[str, str]:
    trainer_name = getattr(cfg.trainer, "name", type(cfg.trainer).__name__)
    tags: dict[str, str] = {
        "trainer": str(trainer_name),
        "worlds": ",".join(cfg.worlds.names),
        "action_space_type": cfg.action_space.action_space_type,
    }
    if (group := os.getenv("MLFLOW_RUN_GROUP")):
        tags["run_group"] = group
    tags.update(cfg.tracking.tags)
    return tags


@contextmanager
def start_run(cfg: ExperimentConfig) -> Iterator[Any]:
    """Open an MLflow run for one training chunk.

    Tags it with ``trainer``, ``worlds``, ``action_space_type``, the
    ``run_group`` env tag (set by the host orchestrator), and any tags from
    ``cfg.tracking.tags``. Logs the flat config as params.
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(cfg.tracking.mlflow_tracking_uri)
    _set_experiment_racesafe(mlflow, cfg.tracking.mlflow_experiment)
    with mlflow.start_run(run_name=cfg.name) as run:
        mlflow.set_tags(_tags_for(cfg))
        mlflow.log_params(_stringify_params(cfg.flat_params()))
        yield run


def _stringify_params(params: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in params.items():
        s = "" if value is None else str(value)
        if len(s) > 500:
            s = s[:497] + "..."
        out[key] = s
    return out


def log_run_artifacts(
    run_dir: Path, *, exclude: tuple[str, ...] = ("checkpoints",)
) -> None:
    """Upload the run dir tree to MLflow if a run is active.

    By default the ``checkpoints/`` subdir is **excluded**: with a local MLflow
    file store ``log_artifacts`` *copies* every file into ``mlruns/``, so logging
    the periodic checkpoints would duplicate them (often tens of GB per run) on
    the same disk for no benefit — they already persist under
    ``artifacts/<run>/checkpoints/``. The shippable models (``best_model/``,
    ``final_model.zip``, ``latest_model.zip``), configs, reward source and
    TensorBoard events are still logged. Pass ``exclude=()`` to log everything.
    """
    mlflow = _mlflow()
    if mlflow.active_run() is None:
        return
    run_dir = Path(run_dir)
    if not exclude:
        mlflow.log_artifacts(str(run_dir))
        return
    for entry in sorted(run_dir.iterdir()):
        if entry.name in exclude:
            continue
        if entry.is_dir():
            mlflow.log_artifacts(str(entry), artifact_path=entry.name)
        else:
            mlflow.log_artifact(str(entry))
