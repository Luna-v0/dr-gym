from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from gym_dr.config import ExperimentConfig


def _mlflow():
    import mlflow

    return mlflow


@contextmanager
def start_run(cfg: ExperimentConfig, parent_run_id: str | None = None) -> Iterator[Any]:
    mlflow = _mlflow()
    mlflow.set_tracking_uri(cfg.tracking.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.tracking.mlflow_experiment)
    nested = parent_run_id is not None
    parent_arg: dict[str, Any] = {}
    if parent_run_id is not None:
        parent_arg["parent_run_id"] = parent_run_id
    with mlflow.start_run(run_name=cfg.name, nested=nested, **parent_arg) as run:
        tags = {
            "algorithm": cfg.algorithm.name,
            "world_name": cfg.world_name,
            "action_space_type": cfg.action_space.action_space_type,
            **cfg.tracking.tags,
        }
        mlflow.set_tags(tags)
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
    mlflow = _mlflow()
    if mlflow.active_run() is None:
        return
    mlflow.log_artifacts(str(run_dir))


def start_parent_run(cfg: ExperimentConfig, study_name: str) -> str:
    mlflow = _mlflow()
    mlflow.set_tracking_uri(cfg.tracking.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.tracking.mlflow_experiment)
    with mlflow.start_run(run_name=f"hpo:{study_name}") as run:
        mlflow.set_tags(
            {
                "hpo": "true",
                "study_name": study_name,
                "algorithm": cfg.algorithm.name,
                "world_name": cfg.world_name,
            }
        )
        return run.info.run_id


def parent_run_id_from_env() -> str | None:
    rid = os.getenv("MLFLOW_PARENT_RUN_ID")
    return rid if rid else None
