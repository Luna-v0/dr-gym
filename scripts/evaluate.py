#!/usr/bin/env python3
"""Run a trained model in the simulator in "view mode" — watch + inspect.

Host usage:

    uv run python scripts/evaluate.py \\
        --model artifacts/hpo_trial_15/final_model.zip \\
        --app app.py \\
        --episodes 5

Then attach a VNC client to localhost:5900 to watch the car; per-step and
per-episode detail streams to this terminal. ``--loop`` runs forever until
you Ctrl-C.

This is a host/container dispatcher (same pattern as ``app.py``):
- On the host: pre-generates ``model_metadata.json``, then ``docker run``s
  the sim container with the GUI on, ``EXPERIMENT_PATH`` pointed back at
  this script, and the model/app/episode count passed via env vars.
- Inside the container (``GYM_DR_IN_CONTAINER=1``): loads the experiment +
  model and calls ``gym_dr.evaluate.run_evaluation``.

Frame stacking is auto-detected from the model's ``run_config.json``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the project root importable when run as a bare script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _container_mode() -> int:
    """Inside the sim container: load experiment + model, run evaluation."""
    from gym_dr.config import load_config
    from gym_dr.evaluate import run_evaluation

    model = os.environ["GYM_DR_EVAL_MODEL"]
    app = os.environ["GYM_DR_EVAL_APP"]
    episodes = int(os.environ.get("GYM_DR_EVAL_EPISODES", "5"))
    loop = os.environ.get("GYM_DR_EVAL_LOOP", "0") == "1"

    experiment = load_config(app)
    run_evaluation(experiment, Path(model), n_episodes=episodes, loop=loop)
    return 0


def _host_mode(args: argparse.Namespace) -> int:
    """On the host: pre-gen metadata, spawn the GUI sim container."""
    from gym_dr.action_space import write_model_metadata
    from gym_dr.config import load_config
    from gym_dr.docker_runner import spawn_training_chunk

    project_dir = Path(os.getenv("PROJECT_DIR", _PROJECT_ROOT)).resolve()
    model_path = args.model.resolve()
    app_path = args.app.resolve()

    for label, p in (("model", model_path), ("app", app_path)):
        if not p.exists():
            print(f"{label} not found: {p}", file=sys.stderr)
            return 1
        try:
            p.relative_to(project_dir)
        except ValueError:
            print(f"{label} must live inside the project dir ({project_dir}): {p}",
                  file=sys.stderr)
            return 1

    experiment = load_config(app_path)
    write_model_metadata(project_dir / "model_metadata.json", experiment.action_space)

    world = args.world or experiment.worlds.names[0]
    image = os.getenv("IMAGE_TAG", "my-deepracer-project:cpu")

    def to_container(p: Path) -> str:
        return f"/workspace/{p.relative_to(project_dir).as_posix()}"

    env = {
        "GYM_DR_IN_CONTAINER": "1",
        "WORLD_NAME": world,
        "ENABLE_GUI": "True",
        "EXPERIMENT_PATH": to_container(Path(__file__).resolve()),
        "GYM_DR_EVAL_MODEL": to_container(model_path),
        "GYM_DR_EVAL_APP": to_container(app_path),
        "GYM_DR_EVAL_EPISODES": str(args.episodes),
        "GYM_DR_EVAL_LOOP": "1" if args.loop else "0",
    }
    if experiment.training.rtf_override is not None:
        env["RTF_OVERRIDE"] = str(experiment.training.rtf_override)

    print(f"[evaluate] world={world!r} model={model_path.name}  "
          f"GUI on vnc://localhost:5900", flush=True)
    return spawn_training_chunk(
        image_tag=image,
        container_name=f"gym-dr-eval-{model_path.parent.name}",
        base_env=env,
        published_ports=[(5900, 5900)],
        use_gpu=experiment.use_gpu,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if os.getenv("GYM_DR_IN_CONTAINER"):
        return _container_mode()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to a trained SB3 .zip (inside the project dir)")
    parser.add_argument("--app", type=Path, default=Path("app.py"),
                        help="Experiment script for env/reward/action-space (default app.py)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes to run (default 5; ignored with --loop)")
    parser.add_argument("--loop", action="store_true",
                        help="Run forever until Ctrl-C — just watch")
    parser.add_argument("--world", default=None,
                        help="Override WORLD_NAME (default: experiment.worlds.names[0])")
    args = parser.parse_args(argv)
    return _host_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
