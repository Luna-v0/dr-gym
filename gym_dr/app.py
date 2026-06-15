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
  the strategy's training chunks (names × rotations) in one container, swapping
  tracks with ``set_world`` between ``chunk_steps``-sized chunks. Set by the
  single-container ``train`` host path, and by the HPO host for multi-world
  studies so every trial rotates through the training worlds.
- ``WORLD_NAME`` — the *first* world; consumed by the upstream simapp at
  container startup (not by Python). Subsequent worlds are loaded via
  ``set_world``.
- ``CHUNK_NAME`` — becomes ``experiment.name`` (the ``artifacts/<name>/`` dir).
- ``RESUME_FROM`` — container path to a starting ``latest_model.zip``;
  overrides ``experiment.training.resume_from`` (applies to the first chunk).
- ``CHUNK_STEPS`` — overrides ``experiment.training.total_timesteps`` (legacy
  single-chunk path; the rotation path reads ``worlds.chunk_steps`` instead).
- ``SEED`` — overrides ``experiment.seed``. Lets the host run the *same*
  experiment script across several seeds (one container per seed) without the
  script hard-coding a seed — see ``experiments/multiseed_ordered_split.py``.
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

# Container exit code meaning "gzserver crashed mid-rotation; relaunch me to
# resume from the checkpoint". Chosen as EX_TEMPFAIL (75) so it can't collide
# with a normal 0/1 exit. Shared by _train_one_chunk (sets it) and _train_host
# (acts on it).
_SIM_RESTART_RC = 75


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
    one shared SQLite-backed Optuna study. The world schedule comes from
    ``base.effective_strategy()``: a single-world study trains every trial on
    ``first_world()``, while a multi-world strategy (multi-world
    ``SequentialRotation`` or ``OrderedSplit``) makes each trial rotate through
    the training worlds — the worker hot-swaps the Gazebo track between chunks
    via ``DeepRacerEnv.set_world`` (one container, no per-world restart).
    Held-out eval worlds (``OrderedSplit.eval_worlds``) are measured each
    evaluation regardless, giving a track-generalisation HPO objective.

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
    if (seed := os.getenv("SEED")):
        overrides["seed"] = int(seed)
    if overrides:
        experiment = experiment.with_overrides(**overrides)

    from gym_dr.trainer import run_training

    try:
        return run_training(experiment)
    except BaseException as ex:  # noqa: BLE001
        # A mid-rotation gzserver segfault (WorldSwapError) is recoverable: the
        # trainer has persisted rotation_resume.json. Exit with the agreed
        # restart code so the host relaunches the container on the crashed
        # world from the checkpoint. (Matched by class name to avoid importing
        # deepracer_env on the host / in stub tests.)
        if type(ex).__name__ == "WorldSwapError":
            print("[train] gzserver died mid-rotation; exiting rc="
                  f"{_SIM_RESTART_RC} for host-side container restart", flush=True)
            sys.exit(_SIM_RESTART_RC)
        raise


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

    # The world schedule comes from the strategy (custom one, or a
    # SequentialRotation derived from experiment.worlds). It decides the
    # training order, the initial WORLD_NAME, and any held-out eval worlds.
    strategy = experiment.effective_strategy()
    chunks = strategy.training_chunks()
    n_chunks = len(chunks)
    first_world = strategy.first_world()
    eval_worlds = strategy.evaluation_worlds()

    # Single-container runtime rotation. One container loads first_world at
    # startup (the simapp reads WORLD_NAME once), then the in-process trainer
    # walks the remaining chunks by calling DeepRacerEnv.set_world() between
    # them — swapping the Gazebo track without restarting gzserver, and keeping
    # the policy weights + PPO optimizer state in memory the whole time. The
    # container rebuilds the same strategy from the experiment script;
    # GYM_DR_ROTATE just switches the trainer into the rotation code path.
    base_env = {
        "GYM_DR_IN_CONTAINER": "1",
        "GYM_DR_ROTATE": "1",
        "CHUNK_NAME": experiment.name,
        "MLFLOW_RUN_GROUP": experiment.name,
        "EXPERIMENT_PATH": container_experiment_path,
    }
    if experiment.training.rtf_override is not None:
        base_env["RTF_OVERRIDE"] = str(experiment.training.rtf_override)
    if experiment.seed is not None:
        # Propagate the seed so a multi-seed host loop yields distinct,
        # reproducible per-seed runs (the container re-imports the script, so
        # the seed can't ride along on the experiment object — only via env).
        base_env["SEED"] = str(experiment.seed)
    ports: list[tuple[int, int]] | None = None
    if experiment.enable_gui:
        base_env["ENABLE_GUI"] = "True"
        ports = [(5900, 5900)]

    train_order = [c.world for c in chunks]
    print(
        f"[train] {strategy.name}: {n_chunks} chunk(s); "
        f"train_order={train_order}, "
        f"eval_worlds={eval_worlds or '(current training world)'}, "
        f"first_world={first_world!r}"
        + ("  (GUI on vnc://localhost:5900)" if experiment.enable_gui else ""),
        flush=True,
    )

    # Single-container rotation with crash recovery. Normally the one container
    # runs the whole rotation. If gzserver segfaults mid-swap (an intermittent
    # Gazebo bug on mesh delete_model), the container exits with _SIM_RESTART_RC
    # after writing rotation_resume.json; we relaunch it on the crashed world,
    # resuming from the checkpoint, and continue where it left off.
    import json

    artifacts_dir = Path(
        os.getenv("ARTIFACTS_DIR", str(project_dir / "artifacts"))).resolve()
    resume_file = artifacts_dir / experiment.name / "rotation_resume.json"
    if resume_file.exists():  # clear stale state from a previous run
        resume_file.unlink()

    max_restarts = int(os.getenv("GYM_DR_MAX_SIM_RESTARTS", "20"))
    start_index = 0
    resume_from: str | None = experiment.training.resume_from
    restarts = 0

    while True:
        env = dict(base_env)
        env["WORLD_NAME"] = train_order[start_index]
        env["ROTATE_START_INDEX"] = str(start_index)
        if resume_from:
            env["RESUME_FROM"] = resume_from
        container_name = f"gym-dr-{experiment.name}" + (
            f"-r{restarts}" if restarts else "")
        rc = spawn_training_chunk(
            image_tag=image,
            container_name=container_name,
            base_env=env,
            published_ports=ports,
            use_gpu=experiment.use_gpu,
        )
        if rc == 0:
            return f"/workspace/artifacts/{experiment.name}/latest_model.zip"

        if rc == _SIM_RESTART_RC and restarts < max_restarts and resume_file.exists():
            state = json.loads(resume_file.read_text(encoding="utf-8"))
            start_index = int(state["start_index"])
            resume_from = state.get("resume_from")
            resume_file.unlink()
            restarts += 1
            print(
                f"[train] gzserver crashed mid-rotation; relaunching on chunk "
                f"{start_index + 1}/{n_chunks} (world={train_order[start_index]!r}) "
                f"from {resume_from!r} [restart {restarts}/{max_restarts}]",
                flush=True,
            )
            continue

        print(f"[train] container {container_name} exited rc={rc}; aborting", flush=True)
        return None


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

    strategy = base.effective_strategy()
    chunks = strategy.training_chunks()
    env = {
        "GYM_DR_WORKER": "1",
        "STUDY_STORAGE": storage_url,
        "MLFLOW_RUN_GROUP": f"study:{study_name}",
        "WORLD_NAME": strategy.first_world(),
        "EXPERIMENT_PATH": _to_container_path(experiment_path, project_dir),
        **extra_env,
    }
    # Multi-world HPO. When the strategy schedules more than one training chunk
    # (a multi-world SequentialRotation, or an OrderedSplit with several
    # train_worlds), put the worker into runtime track-rotation mode so EVERY
    # trial trains across the whole world schedule, hot-swapping the Gazebo
    # track between chunks via DeepRacerEnv.set_world (no container restart).
    # Single-world studies leave this unset and keep the legacy
    # one-world-per-trial path. Held-out eval worlds need no flag — run_training
    # always reads strategy.evaluation_worlds(), so OrderedSplit's track
    # generalisation metric works in HPO with or without training rotation.
    if len(chunks) > 1:
        env["GYM_DR_ROTATE"] = "1"
        print(
            f"[hpo] multi-world HPO enabled: each trial rotates "
            f"{[c.world for c in chunks]}; "
            f"eval_worlds={strategy.evaluation_worlds() or '(current training world)'}",
            flush=True,
        )
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
