from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

from gym_dr.config import ExperimentConfig


def train(experiment: ExperimentConfig) -> float:
    from gym_dr.trainer import run_training

    return run_training(experiment)


def study(
    base: ExperimentConfig,
    search_space: Callable[[Any], dict[str, Any]],
    *,
    study_name: str,
    n_trials: int,
    n_parallel: int = 1,
    storage: str | None = None,
    image_tag: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    if os.getenv("GYM_DR_WORKER"):
        _run_worker(base, search_space, study_name, storage)
        return 0

    return _spawn_workers(
        base=base,
        study_name=study_name,
        n_trials=n_trials,
        n_parallel=n_parallel,
        storage=storage,
        image_tag=image_tag,
        extra_env=extra_env or {},
    )


def inspect(experiment: ExperimentConfig) -> None:
    import json

    print(json.dumps(experiment.to_dict(), indent=2, default=str))
    print("\n# Flat MLflow params:")
    for k, v in experiment.flat_params().items():
        print(f"  {k} = {v}")


def _run_worker(
    base: ExperimentConfig,
    search_space: Callable[[Any], dict[str, Any]],
    study_name: str,
    storage: str | None,
) -> None:
    from gym_dr.hpo import run_worker

    storage_url = storage or os.getenv("STUDY_STORAGE", "sqlite:////workspace/optuna.db")
    n_trials_per_worker = int(os.getenv("N_TRIALS_PER_WORKER", "1"))
    run_worker(base, search_space, study_name, storage_url, n_trials_per_worker)


def _spawn_workers(
    *,
    base: ExperimentConfig,
    study_name: str,
    n_trials: int,
    n_parallel: int,
    storage: str | None,
    image_tag: str | None,
    extra_env: dict[str, str],
) -> int:
    from gym_dr.action_space import write_model_metadata
    from gym_dr.docker_runner import spawn_workers
    from gym_dr.mlflow_utils import start_parent_run

    experiment_path = _resolve_experiment_path()
    project_dir = Path(os.getenv("PROJECT_DIR", Path.cwd())).resolve()
    write_model_metadata(project_dir / "model_metadata.json", base.action_space)

    storage_url = storage or os.getenv("STUDY_STORAGE", "sqlite:////workspace/optuna.db")
    image = image_tag or os.getenv("IMAGE_TAG", "my-deepracer-project:cpu")

    parent_run_id = start_parent_run(base, study_name)

    env = {
        "GYM_DR_WORKER": "1",
        "STUDY_STORAGE": storage_url,
        "MLFLOW_PARENT_RUN_ID": parent_run_id,
        "WORLD_NAME": base.world_name,
        "EXPERIMENT_PATH": _to_container_path(experiment_path, project_dir),
        **extra_env,
    }

    return spawn_workers(
        image_tag=image,
        study_name=study_name,
        n_trials=n_trials,
        n_parallel=n_parallel,
        base_env=env,
    )


def _resolve_experiment_path() -> Path:
    env = os.getenv("GYM_DR_EXPERIMENT_FILE")
    if env:
        return Path(env).resolve()
    main_mod = sys.modules.get("__main__")
    if main_mod and getattr(main_mod, "__file__", None):
        return Path(main_mod.__file__).resolve()
    raise RuntimeError(
        "Could not locate the experiment script. "
        "Set GYM_DR_EXPERIMENT_FILE to its absolute path."
    )


def _to_container_path(host_path: Path, project_dir: Path) -> str:
    host_path = host_path.resolve()
    try:
        rel = host_path.relative_to(project_dir)
    except ValueError as exc:
        raise RuntimeError(
            f"Experiment file {host_path} must live inside PROJECT_DIR {project_dir}"
        ) from exc
    return f"/workspace/{rel.as_posix()}"
