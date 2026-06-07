"""User-facing entrypoints: ``train(experiment)`` and ``study(...)``.

The user's ``app.py`` ends with ``if __name__ == "__main__": train(experiment)``.
Running ``python app.py`` from the host kicks off the *host orchestrator*:
the orchestrator pre-generates ``model_metadata.json`` and then ``docker run``s
a *single* container that trains the entire multi-world rotation in-process.
Gazebo loads ``worlds.names[0]`` at startup; the in-container trainer then
swaps tracks between chunks at runtime via ``DeepRacerEnv.set_world`` — no
container restart per ``(rotation, world)``. Inside the container the same
``app.py`` runs, but ``train()`` detects worker mode via the
``GYM_DR_IN_CONTAINER`` env var and runs the rotation.

Environment-variable protocol between host and container
--------------------------------------------------------
The host orchestrator sets these on the container's env; the in-container
``_train_one_chunk`` reads them:

- ``GYM_DR_IN_CONTAINER=1`` — tells the script "you are the worker".
- ``GYM_DR_ROTATE=1`` — switch the trainer into runtime-rotation mode: walk
  ``experiment.worlds`` (names × rotations) in one container, swapping tracks
  with ``set_world`` between ``chunk_steps``-sized chunks.
- ``WORLD_NAME`` — the *first* world; consumed by the upstream simapp at
  container startup (not by Python). Subsequent worlds are loaded via
  ``set_world``.
- ``CHUNK_NAME`` — becomes ``experiment.name`` (the ``artifacts/<name>/`` dir).
- ``RESUME_FROM`` — container path to a starting ``latest_model.zip``;
  overrides ``experiment.training.resume_from`` (applies to the first chunk).
- ``CHUNK_STEPS`` — overrides ``experiment.training.total_timesteps`` (legacy
  single-chunk path; the rotation path reads ``worlds.chunk_steps`` instead).
- ``MLFLOW_PARENT_RUN_ID`` — children open nested runs under this parent.

For HPO the orchestrator additionally sets ``GYM_DR_WORKER=1``,
``STUDY_NAME``, ``STUDY_STORAGE``, ``N_TRIALS_PER_WORKER``. See
``gym_dr/hpo.py`` and ``gym_dr/docker_runner.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

from gym_dr.config import ExperimentConfig


def train(experiment: ExperimentConfig) -> Any:
    """Run a training. Mode-dispatched.

    - On the *host* (no ``GYM_DR_IN_CONTAINER`` env): launch a *single*
      Docker container that trains the whole multi-world rotation in-process,
      swapping tracks at runtime via ``DeepRacerEnv.set_world`` (no per-chunk
      container restart). Returns the host path of the run's
      ``latest_model.zip``.
    - *Inside a container* (``GYM_DR_IN_CONTAINER=1``): apply per-chunk
      env-var overrides, then run ``gym_dr.trainer.run_training``. Returns
      the final eval reward (float).
    """
    if os.getenv("GYM_DR_IN_CONTAINER"):
        return _train_one_chunk(experiment)
    return _train_host(experiment)


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
    """Run an Optuna study. Mode-dispatched like ``train``.

    On the host: pre-generate ``model_metadata.json``, open a parent MLflow
    run, spawn ``n_parallel`` worker containers that each pull trials from
    one shared SQLite-backed Optuna study. World is fixed per study (uses
    ``base.worlds.names[0]``; multi-world HPO is out of scope).

    Inside a worker container (``GYM_DR_WORKER=1``): loop
    ``study.optimize(objective, n_trials=N_TRIALS_PER_WORKER)``.
    """
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
    """Pretty-print the resolved experiment + the flat MLflow param keys.

    Useful for dry-running: ``python -c "from app import experiment; \\
    from gym_dr import inspect; inspect(experiment)"``. No Docker, no sim.
    """
    import json

    print(json.dumps(experiment.to_dict(), indent=2, default=str))
    print("\n# Flat MLflow params:")
    for k, v in experiment.flat_params().items():
        print(f"  {k} = {v}")


# ----------------------------- single training ----------------------------- #

def _train_one_chunk(experiment: ExperimentConfig) -> float:
    """Container side: apply per-chunk env-var overrides, then train one chunk."""
    overrides: dict[str, Any] = {}
    if (name := os.getenv("CHUNK_NAME")):
        overrides["name"] = name
    if (resume := os.getenv("RESUME_FROM")):
        overrides["training.resume_from"] = resume
    if (chunk_steps := os.getenv("CHUNK_STEPS")):
        overrides["training.total_timesteps"] = int(chunk_steps)
    if overrides:
        experiment = experiment.with_overrides(**overrides)

    from gym_dr.trainer import run_training

    return run_training(experiment)


def _train_host(experiment: ExperimentConfig) -> str | None:
    """Host side: pre-gen metadata, then docker-run each (rotation, world) chunk in turn.

    Chunks are NOT opened under a host-side MLflow parent run — the host
    and container often run incompatible MLflow versions and the older one
    can't read run metadata written by the newer one. Each chunk opens its
    own MLflow run and tags it with ``run_group=<experiment.name>`` so the
    MLflow UI can group them via a tag filter
    (``tags.run_group = "quick_test"``).
    """
    from gym_dr.action_space import write_model_metadata
    from gym_dr.docker_runner import spawn_training_chunk

    experiment_path = _resolve_experiment_path()
    project_dir = Path(os.getenv("PROJECT_DIR", Path.cwd())).resolve()
    write_model_metadata(project_dir / "model_metadata.json", experiment.action_space)

    image = os.getenv("IMAGE_TAG") or _default_image(experiment.use_gpu)
    container_experiment_path = _to_container_path(experiment_path, project_dir)

    worlds = experiment.worlds
    n_chunks = worlds.rotations * len(worlds.names)

    # Single-container runtime rotation. One container loads worlds.names[0] at
    # startup (the simapp reads WORLD_NAME once), then the in-process trainer
    # walks the remaining chunks by calling DeepRacerEnv.set_world() between
    # them — swapping the Gazebo track without restarting gzserver, and keeping
    # the policy weights + PPO optimizer state in memory the whole time. The
    # container reads the rotation plan (names, chunk_steps, rotations) straight
    # off experiment.worlds; GYM_DR_ROTATE just switches the trainer into the
    # rotation code path.
    container_name = f"gym-dr-{experiment.name}"
    env = {
        "GYM_DR_IN_CONTAINER": "1",
        "GYM_DR_ROTATE": "1",
        "WORLD_NAME": worlds.names[0],
        "CHUNK_NAME": experiment.name,
        "MLFLOW_RUN_GROUP": experiment.name,
        "EXPERIMENT_PATH": container_experiment_path,
    }
    if experiment.training.resume_from:
        env["RESUME_FROM"] = experiment.training.resume_from
    if experiment.training.rtf_override is not None:
        env["RTF_OVERRIDE"] = str(experiment.training.rtf_override)
    ports: list[tuple[int, int]] | None = None
    if experiment.enable_gui:
        env["ENABLE_GUI"] = "True"
        ports = [(5900, 5900)]

    print(
        f"[train] single-container rotation: {n_chunks} chunk(s) over "
        f"worlds={worlds.names} x{worlds.rotations}, "
        f"chunk_steps={worlds.chunk_steps}, first_world={worlds.names[0]!r}"
        + ("  (GUI on vnc://localhost:5900)" if experiment.enable_gui else ""),
        flush=True,
    )
    rc = spawn_training_chunk(
        image_tag=image,
        container_name=container_name,
        base_env=env,
        published_ports=ports,
        use_gpu=experiment.use_gpu,
    )
    if rc != 0:
        print(f"[train] container {container_name} exited rc={rc}; aborting", flush=True)
        return None

    return f"/workspace/artifacts/{experiment.name}/latest_model.zip"


# ----------------------------------- HPO ----------------------------------- #

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

    experiment_path = _resolve_experiment_path()
    project_dir = Path(os.getenv("PROJECT_DIR", Path.cwd())).resolve()
    write_model_metadata(project_dir / "model_metadata.json", base.action_space)

    storage_url = storage or os.getenv("STUDY_STORAGE", "sqlite:////workspace/optuna.db")
    image = image_tag or os.getenv("IMAGE_TAG") or _default_image(base.use_gpu)

    world = base.worlds.names[0]
    env = {
        "GYM_DR_WORKER": "1",
        "STUDY_STORAGE": storage_url,
        "MLFLOW_RUN_GROUP": f"study:{study_name}",
        "WORLD_NAME": world,
        "EXPERIMENT_PATH": _to_container_path(experiment_path, project_dir),
        **extra_env,
    }
    vnc_base = None
    if base.enable_gui:
        env["ENABLE_GUI"] = "True"
        vnc_base = 5900
        print(
            f"[hpo] GUI enabled — VNC on vnc://localhost:{vnc_base}"
            + (f"..{vnc_base + n_parallel - 1}" if n_parallel > 1 else ""),
            flush=True,
        )

    return spawn_workers(
        image_tag=image,
        study_name=study_name,
        n_trials=n_trials,
        n_parallel=n_parallel,
        base_env=env,
        vnc_base_port=vnc_base,
        use_gpu=base.use_gpu,
    )


# ---------------------------- path helpers --------------------------------- #

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


def _default_image(use_gpu: bool) -> str:
    """Pick the right project image tag based on the experiment's GPU flag.

    Keeps ``--gpus all`` (added when ``use_gpu=True``) and the image arch in
    sync — otherwise we'd pass GPU access to a CPU-only image, which fails
    silently if the user only rebuilt one arch."""
    return "my-deepracer-project:gpu" if use_gpu else "my-deepracer-project:cpu"


def _to_container_path(host_path: Path, project_dir: Path) -> str:
    host_path = host_path.resolve()
    try:
        rel = host_path.relative_to(project_dir)
    except ValueError as exc:
        raise RuntimeError(
            f"Experiment file {host_path} must live inside PROJECT_DIR {project_dir}"
        ) from exc
    return f"/workspace/{rel.as_posix()}"
