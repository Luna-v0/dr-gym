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
import time
from pathlib import Path

# Returned when the host watchdog kills a wedged-but-alive container (a sim
# *hang*, not a crash) so the caller relaunches it through the same resume path
# as a gzserver crash. Mirrors gym_dr.app._SIM_RESTART_RC (imported there).
SIM_RESTART_RC = 75

# Watchdog: a container that hasn't touched its heartbeat for TIMEOUT seconds is
# treated as hung and killed. BOOT_GRACE allows the (slow) Gazebo boot before the
# first heartbeat. Disable with GYM_DR_WATCHDOG=0. See docs/reports/d3-hang-postmortem.md.
_WATCHDOG_ON = os.getenv("GYM_DR_WATCHDOG", "1") != "0"
_WATCHDOG_TIMEOUT = int(os.getenv("GYM_DR_WATCHDOG_TIMEOUT", "600"))
_WATCHDOG_BOOT_GRACE = int(os.getenv("GYM_DR_WATCHDOG_BOOT_GRACE", "360"))


def _docker_kill(name: str) -> None:
    subprocess.run(["docker", "kill", name], check=False, capture_output=True)


def _docker_rm_f(name: str) -> None:
    """Force-remove any existing container with this name so a (re)spawn can reuse
    it. Guards against orphans from a killed launcher and against the watchdog's
    own kill+relaunch racing container teardown (both reuse fixed names)."""
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)


def _heartbeat_paths(artifacts_dir: Path, container_name: str) -> tuple[Path, str]:
    """(host path, container path) for this container's heartbeat file. The
    artifacts dir is bind-mounted at /workspace/artifacts, so a file the
    container touches there is visible to the host here."""
    host = artifacts_dir / f".heartbeat-{container_name}"
    container = f"/workspace/artifacts/.heartbeat-{container_name}"
    return host, container


def _is_hung(proc: "subprocess.Popen", host_heartbeat: Path, started: float) -> bool:
    """True if the container has produced no heartbeat progress within the
    configured window (boot grace until the first heartbeat, then TIMEOUT)."""
    now = time.monotonic()
    if host_heartbeat.exists():
        return (time.time() - host_heartbeat.stat().st_mtime) > _WATCHDOG_TIMEOUT
    return (now - started) > _WATCHDOG_BOOT_GRACE  # never even started rolling
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
    # Opt-in dev override: bind-mount a local deepracer-env checkout over the
    # base image's installed package, so edits to the sim (e.g. the W-dr
    # random_start / random_direction reset modes) take effect WITHOUT rebuilding
    # the image. Point GYM_DR_DEEPRACER_ENV_SRC at the repo's `deepracer_env/`
    # package dir. No-op when unset (normal runs use the baked-in package).
    dr_env_src = os.getenv("GYM_DR_DEEPRACER_ENV_SRC")
    if dr_env_src:
        src = Path(dr_env_src).resolve()
        if not (src / "agent_ctrl" / "constants.py").exists():
            raise RuntimeError(
                f"GYM_DR_DEEPRACER_ENV_SRC={src} is not a deepracer_env package dir "
                "(expected agent_ctrl/constants.py inside it)."
            )
        argv += [
            "-v",
            # ROS 2 Lyrical base ships Python 3.14 (was 3.8 on the Noetic base).
            f"{src}:/usr/local/lib/python3.14/dist-packages/deepracer_env:ro",
        ]
        # Also overlay the catkin `simulation` package's launch + urdf (which live
        # OUTSIDE the python package, in the image's catkin share dir) so sim-asset
        # edits — the camera-off `include_camera` toggle, the multicar launch —
        # take effect too. Derived from the repo root (parent of the python pkg).
        repo_root = src.parent
        sim_launch = repo_root / "simulation" / "src" / "deepracer_simulation_environment" / "launch"
        sim_urdf = repo_root / "simulation" / "urdf"
        # ROS 2 colcon --merge-install layout: share lives at <install>/share/<pkg>
        # (the Noetic catkin path was deepracer_simulation_environment/share/...).
        share = "/opt/simapp/share/deepracer_simulation_environment"
        if sim_launch.is_dir():
            argv += ["-v", f"{sim_launch}:{share}/launch:ro"]
        if sim_urdf.is_dir():
            argv += ["-v", f"{sim_urdf}:{share}/urdf:ro"]
    # sb3-contrib (RecurrentPPO / LSTM policies) is NOT in the base image; mount the
    # host venv's package into the container's dist-packages so the LSTM architecture
    # arm can import it without rebuilding the image. No-op if not installed on the host.
    try:
        import sb3_contrib as _sb3c

        _contrib_dir = Path(_sb3c.__file__).resolve().parent
        argv += ["-v", f"{_contrib_dir}:/usr/local/lib/python3.14/dist-packages/sb3_contrib:ro"]
    except Exception:  # noqa: BLE001
        pass
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
    host_hb, container_hb = _heartbeat_paths(artifacts_dir, container_name)
    host_hb.unlink(missing_ok=True)  # clear a stale heartbeat from a prior run
    env = dict(base_env, GYM_DR_HEARTBEAT=container_hb)
    argv = _build_run_cmd(
        image_tag, project_dir, artifacts_dir, container_name, env,
        published_ports, use_gpu=use_gpu,
    )
    print(f"[train] spawning {container_name}: {' '.join(argv)}", flush=True)

    _docker_rm_f(container_name)  # clear any orphan with this name first
    proc = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)

    def kill(_signum=None, _frame=None) -> None:
        _docker_kill(container_name)

    prev_int = signal.signal(signal.SIGINT, kill)
    prev_term = signal.signal(signal.SIGTERM, kill)
    started = time.monotonic()
    try:
        if not _WATCHDOG_ON:
            return proc.wait()
        while True:
            try:
                return proc.wait(timeout=15)        # exited on its own
            except subprocess.TimeoutExpired:
                pass
            if _is_hung(proc, host_hb, started):
                print(f"[watchdog] {container_name} hung (no heartbeat); killing + "
                      f"requesting restart", flush=True)
                _docker_kill(container_name)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                return SIM_RESTART_RC
    finally:
        host_hb.unlink(missing_ok=True)
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
    max_restarts = int(os.getenv("GYM_DR_MAX_WORKER_RESTARTS", "10"))

    workers: dict[int, dict] = {}

    def kill_outstanding(_signum=None, _frame=None) -> None:
        for w in workers.values():
            _docker_kill(w["name"])

    signal.signal(signal.SIGINT, kill_outstanding)
    signal.signal(signal.SIGTERM, kill_outstanding)

    def launch(idx: int) -> dict:
        name = f"gym-dr-hpo-{study_name}-{idx}"
        host_hb, container_hb = _heartbeat_paths(artifacts_dir, name)
        host_hb.unlink(missing_ok=True)
        env_vars = dict(base_env, STUDY_NAME=study_name,
                        N_TRIALS_PER_WORKER=str(per_worker), WORKER_INDEX=str(idx),
                        GYM_DR_HEARTBEAT=container_hb)
        ports = [(vnc_base_port + idx, 5900)] if vnc_base_port is not None else None
        argv = _build_run_cmd(image_tag, project_dir, artifacts_dir, name, env_vars,
                              ports, use_gpu=use_gpu)
        print(f"[hpo] spawning {name}: {' '.join(argv)}", flush=True)
        _docker_rm_f(name)  # clear any orphan/leftover with this name first
        proc = subprocess.Popen(argv, stdout=sys.stdout, stderr=sys.stderr)
        return {"idx": idx, "name": name, "proc": proc, "hb": host_hb,
                "started": time.monotonic(), "restarts": 0}

    for idx in range(n_parallel):
        workers[idx] = launch(idx)

    overall_rc = 0
    done: set[int] = set()
    try:
        while len(done) < n_parallel:
            time.sleep(15)
            for idx, w in list(workers.items()):
                if idx in done:
                    continue
                rc = w["proc"].poll()
                if rc is not None:                       # exited
                    print(f"[hpo] worker {w['name']} exited rc={rc}", flush=True)
                    if rc != 0 and w["restarts"] < max_restarts:
                        # nonzero exit (incl. a watchdog-killed sibling): relaunch;
                        # it rejoins the shared Optuna study and pulls new trials.
                        w2 = launch(idx); w2["restarts"] = w["restarts"] + 1
                        workers[idx] = w2
                    else:
                        if rc != 0 and overall_rc == 0:
                            overall_rc = rc
                        done.add(idx)
                    continue
                if _WATCHDOG_ON and _is_hung(w["proc"], w["hb"], w["started"]):
                    print(f"[watchdog] hpo worker {w['name']} hung; killing", flush=True)
                    _docker_kill(w["name"])
                    try:
                        w["proc"].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    if w["restarts"] < max_restarts:
                        w2 = launch(idx); w2["restarts"] = w["restarts"] + 1
                        workers[idx] = w2
                    else:
                        overall_rc = overall_rc or SIM_RESTART_RC
                        done.add(idx)
    finally:
        for w in workers.values():
            w["hb"].unlink(missing_ok=True)
    return overall_rc


def env_from_pairs(pairs: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"invalid --env entry {pair!r}; expected KEY=VAL")
        k, v = pair.split("=", 1)
        out[k] = v
    return out
