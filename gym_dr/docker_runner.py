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
    for key, value in env_vars.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.append(image_tag)
    return argv


def spawn_workers(
    image_tag: str,
    study_name: str,
    n_trials: int,
    n_parallel: int,
    base_env: dict[str, str],
) -> int:
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
        argv = _build_run_cmd(image_tag, project_dir, artifacts_dir, name, env_vars)
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
