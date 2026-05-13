"""MLflow lifecycle helpers used by ``gym_dr.trainer`` and ``gym_dr.app``.

``start_run`` opens a (possibly nested) MLflow run, tags it, and logs the
flat config as params. ``start_parent_run`` is the host-side helper that
opens a parent run for HPO studies and multi-world rotations; its run_id
gets passed to children via the ``MLFLOW_PARENT_RUN_ID`` env var.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from gym_dr.config import ExperimentConfig


def _mlflow():
    import mlflow

    return mlflow


def _tags_for(cfg: ExperimentConfig) -> dict[str, str]:
    trainer_name = getattr(cfg.trainer, "name", type(cfg.trainer).__name__)
    return {
        "trainer": str(trainer_name),
        "worlds": ",".join(cfg.worlds.names),
        "action_space_type": cfg.action_space.action_space_type,
        **cfg.tracking.tags,
    }


@contextmanager
def start_run(cfg: ExperimentConfig, parent_run_id: str | None = None) -> Iterator[Any]:
    """Open an MLflow run for one training chunk; nested if a parent id is given."""
    mlflow = _mlflow()
    mlflow.set_tracking_uri(cfg.tracking.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.tracking.mlflow_experiment)
    nested = parent_run_id is not None
    parent_arg: dict[str, Any] = {}
    if parent_run_id is not None:
        parent_arg["parent_run_id"] = parent_run_id
    with mlflow.start_run(run_name=cfg.name, nested=nested, **parent_arg) as run:
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


def log_run_artifacts(run_dir: Path) -> None:
    """Upload the entire run dir tree to MLflow if a run is active."""
    mlflow = _mlflow()
    if mlflow.active_run() is None:
        return
    mlflow.log_artifacts(str(run_dir))


def start_parent_run(cfg: ExperimentConfig, study_name: str) -> str:
    """Open and immediately close a parent MLflow run; return its run_id.

    Children nest under this parent by passing ``MLFLOW_PARENT_RUN_ID``.
    Used both by multi-world rotations (one parent per ``train(experiment)``
    invocation) and by HPO (one parent per study).
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(cfg.tracking.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.tracking.mlflow_experiment)
    with mlflow.start_run(run_name=f"parent:{study_name}") as run:
        mlflow.set_tags({"parent": "true", "study_name": study_name, **_tags_for(cfg)})
        return run.info.run_id


def parent_run_id_from_env() -> str | None:
    rid = os.getenv("MLFLOW_PARENT_RUN_ID")
    return rid if rid else None
