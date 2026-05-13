"""Host-side Docker spawners.

Two flavours:

- ``spawn_training_chunk`` — blocking, single container, returns rc. Used by
  ``gym_dr.app._train_host`` to run one ``(rotation, world)`` chunk at a time.
- ``spawn_workers`` — non-blocking spawn of N parallel containers that all
  share an Optuna study + MLflow tree. Used by ``gym_dr.app._spawn_workers``
  for HPO.

Both share ``_build_run_cmd`` for mount + env layout.
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _resolve_project_dir() -> Path:
    env = os.getenv("PROJECT_DIR")
    if env:
        return Path(env).resolve()
    return Path.cwd().resolve()


def _resolve_artifacts_dir(project_dir: Path) -> Path:
    env = os.getenv("ARTIFACTS_DIR")
    return Path(env).resolve() if env else project_dir / "artifacts"


def _build_run_cmd(
    image_tag: str,
    project_dir: Path,
    artifacts_dir: Path,
    container_name: str,
    env_vars: dict[str, str],
    published_ports: list[tuple[int, int]] | None = None,
    use_gpu: bool = False,
) -> list[str]:
    optuna_db = project_dir / "optuna.db"
    optuna_db.touch(exist_ok=True)
    mlruns_dir = project_dir / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "-v",
        f"{project_dir}:/workspace:rw",
        "-v",
        f"{artifacts_dir}:/workspace/artifacts",
        "-v",
        f"{mlruns_dir}:/workspace/mlruns",
        "-v",
        f"{optuna_db}:/workspace/optuna.db",
    ]
    if use_gpu:
        argv.extend(["--gpus", "all"])
    for host_port, container_port in published_ports or []:
        argv.extend(["-p", f"{host_port}:{container_port}"])
    for key, value in env_vars.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.append(image_tag)
    return argv


def spawn_training_chunk(
    image_tag: str,
    container_name: str,
    base_env: dict[str, str],
    published_ports: list[tuple[int, int]] | None = None,
    use_gpu: bool = False,
) -> int:
    """Run one training chunk in a Docker container; block until exit.

    Returns the container's exit code. SIGINT/SIGTERM in the host process
    docker-kills the container. ``published_ports`` is a list of
    ``(host_port, container_port)`` pairs forwarded via ``-p`` — used for
    the VNC GUI when ``enable_gui=True``. ``use_gpu`` adds ``--gpus all``
    to expose host GPUs to the container (requires NVIDIA Container Toolkit).
    """
    project_dir = _resolve_project_dir()
    artifacts_dir = _resolve_artifacts_dir(project_dir)
    argv = _build_run_cmd(
        image_tag, project_dir, artifacts_dir, container_name, base_env,
        published_ports, use_gpu=use_gpu,
    )
    print(f"[train] spawning {container_name}: {' '.join(argv)}", flush=True)

    proc = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)

    def kill(_signum=None, _frame=None) -> None:
        subprocess.run(["docker", "kill", container_name], check=False, capture_output=True)

    prev_int = signal.signal(signal.SIGINT, kill)
    prev_term = signal.signal(signal.SIGTERM, kill)
    try:
        return proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


def spawn_workers(
    image_tag: str,
    study_name: str,
    n_trials: int,
    n_parallel: int,
    base_env: dict[str, str],
    vnc_base_port: int | None = None,
    use_gpu: bool = False,
) -> int:
    """Spawn N parallel HPO workers; wait on all; return worst exit code.

    If ``vnc_base_port`` is set, each worker publishes its container port
    5900 to host port ``vnc_base_port + worker_idx`` for Gazebo VNC.
    """
    project_dir = _resolve_project_dir()
    artifacts_dir = _resolve_artifacts_dir(project_dir)
    per_worker = max(1, math.ceil(n_trials / max(1, n_parallel)))

    processes: list[tuple[str, subprocess.Popen]] = []

    def kill_outstanding(_signum=None, _frame=None) -> None:
        for name, _ in processes:
            subprocess.run(["docker", "kill", name], check=False, capture_output=True)

    signal.signal(signal.SIGINT, kill_outstanding)
    signal.signal(signal.SIGTERM, kill_outstanding)

    for idx in range(n_parallel):
        env_vars = dict(base_env)
        env_vars.update(
            {
                "STUDY_NAME": study_name,
                "N_TRIALS_PER_WORKER": str(per_worker),
                "WORKER_INDEX": str(idx),
            }
        )
        name = f"gym-dr-hpo-{study_name}-{idx}"
        ports = [(vnc_base_port + idx, 5900)] if vnc_base_port is not None else None
        argv = _build_run_cmd(
            image_tag, project_dir, artifacts_dir, name, env_vars, ports, use_gpu=use_gpu,
        )
        print(f"[hpo] spawning {name}: {' '.join(argv)}", flush=True)
        proc = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)
        processes.append((name, proc))

    overall_rc = 0
    for name, proc in processes:
        rc = proc.wait()
        print(f"[hpo] worker {name} exited rc={rc}", flush=True)
        if rc != 0 and overall_rc == 0:
            overall_rc = rc
    return overall_rc


def env_from_pairs(pairs: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"invalid --env entry {pair!r}; expected KEY=VAL")
        k, v = pair.split("=", 1)
        out[k] = v
    return out
