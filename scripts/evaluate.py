#!/usr/bin/env python3
"""Run a trained model in the simulator in "view mode" — watch + inspect.

Host usage (the common case — no --app needed):

    uv run python scripts/evaluate.py \\
        --model artifacts/hpo_trial_15/final_model.zip

Then attach a VNC client to localhost:5900 to watch the car; per-step and
per-episode detail streams to this terminal.

Flags:
  --episodes N   how many episodes to run (default 5; ignored with --loop)
  --loop         run forever until Ctrl-C — just watch
  --world W      override the track (default: the model's training world)
  --rtf R        simulator real-time factor. Default 1.0 = human-watchable
                 real time. The training config's rtf_override (often 10+
                 for fast HPO) is *not* inherited — eval is for watching.
  --app PATH     optional. By default the experiment is reconstructed from
                 the model's run_config.json. Pass --app only if the run used
                 callables defined inline in the script (which can't be
                 resolved by import path).
  --run-config PATH
                 optional. Explicit run_config.json to reconstruct from.
                 By default it's auto-discovered next to the model, then in
                 the nearest parent directory — so a model in best_model/
                 still finds the trial's run_config.json one level up.

Host/container dispatch (same pattern as app.py):
- On the host: reconstructs the experiment, pre-generates model_metadata.json,
  ``docker run``s the sim container with the GUI on and EXPERIMENT_PATH
  pointed back at this script.
- Inside the container (GYM_DR_IN_CONTAINER=1): loads the experiment + model
  and calls gym_dr.evaluate.run_evaluation.

Frame stacking is auto-detected from the model's run_config.json.
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
    """Inside the sim container: resolve experiment + model, run evaluation."""
    from gym_dr.evaluate import experiment_for_model, run_evaluation

    model = Path(os.environ["GYM_DR_EVAL_MODEL"])
    app = os.environ.get("GYM_DR_EVAL_APP") or None
    run_config = os.environ.get("GYM_DR_EVAL_RUN_CONFIG") or None
    episodes = int(os.environ.get("GYM_DR_EVAL_EPISODES", "5"))
    loop = os.environ.get("GYM_DR_EVAL_LOOP", "0") == "1"

    run_config_path = Path(run_config) if run_config else None
    experiment = experiment_for_model(
        model, Path(app) if app else None, run_config_path
    )
    run_evaluation(
        experiment, model, n_episodes=episodes, loop=loop,
        run_config_path=run_config_path,
    )
    return 0


def _host_mode(args: argparse.Namespace) -> int:
    """On the host: reconstruct experiment, pre-gen metadata, spawn the GUI sim."""
    from gym_dr.action_space import write_model_metadata
    from gym_dr.app import _default_image
    from gym_dr.docker_runner import spawn_training_chunk
    from gym_dr.evaluate import experiment_for_model

    project_dir = Path(os.getenv("PROJECT_DIR", _PROJECT_ROOT)).resolve()
    model_path = args.model.resolve()
    app_path = args.app.resolve() if args.app else None
    run_config_path = args.run_config.resolve() if args.run_config else None

    to_check = [("model", model_path)]
    if app_path is not None:
        to_check.append(("app", app_path))
    if run_config_path is not None:
        to_check.append(("run-config", run_config_path))
    for label, p in to_check:
        if not p.exists():
            print(f"{label} not found: {p}", file=sys.stderr)
            return 1
        try:
            p.relative_to(project_dir)
        except ValueError:
            print(f"{label} must live inside the project dir ({project_dir}): {p}",
                  file=sys.stderr)
            return 1

    experiment = experiment_for_model(model_path, app_path, run_config_path)
    write_model_metadata(project_dir / "model_metadata.json", experiment.action_space)

    world = args.world or experiment.worlds.names[0]
    # Select the image the same way training/app.py does: match the
    # experiment's GPU flag so the image arch and --gpus all stay in sync.
    image = os.getenv("IMAGE_TAG") or _default_image(experiment.use_gpu)

    def to_container(p: Path) -> str:
        return f"/workspace/{p.relative_to(project_dir).as_posix()}"

    env = {
        "GYM_DR_IN_CONTAINER": "1",
        "WORLD_NAME": world,
        "ENABLE_GUI": "True",
        "EXPERIMENT_PATH": to_container(Path(__file__).resolve()),
        "GYM_DR_EVAL_MODEL": to_container(model_path),
        "GYM_DR_EVAL_EPISODES": str(args.episodes),
        "GYM_DR_EVAL_LOOP": "1" if args.loop else "0",
        # Real-time factor: default 1.0 (human-watchable). The training
        # config's rtf_override is deliberately NOT inherited.
        "RTF_OVERRIDE": str(args.rtf),
    }
    if app_path is not None:
        env["GYM_DR_EVAL_APP"] = to_container(app_path)
    if run_config_path is not None:
        env["GYM_DR_EVAL_RUN_CONFIG"] = to_container(run_config_path)
    # Feature-obs models select their vector (9 vs the 11-feature actor set) via
    # GYM_DR_FEATURE_SET, which the experiment SCRIPT sets at import but
    # run_config.json doesn't capture — so reconstruction here would default to the
    # 9-feature vector and mismatch an 11-input policy. Forward it from the host
    # env when set (no-op for camera models / the default 9-feature path).
    feature_set = os.environ.get("GYM_DR_FEATURE_SET")
    if feature_set:
        env["GYM_DR_FEATURE_SET"] = feature_set

    print(f"[evaluate] world={world!r} model={model_path.name}  rtf={args.rtf}  "
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
    parser.add_argument("--app", type=Path, default=None,
                        help="Optional experiment script override. Default: reconstruct "
                             "from the model's run_config.json")
    parser.add_argument("--run-config", type=Path, default=None,
                        help="Explicit run_config.json to reconstruct the experiment from. "
                             "Default: auto-discover next to the model, then in the nearest "
                             "parent directory (e.g. the trial dir when the model is in "
                             "best_model/)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes to run (default 5; ignored with --loop)")
    parser.add_argument("--loop", action="store_true",
                        help="Run forever until Ctrl-C — just watch")
    parser.add_argument("--world", default=None,
                        help="Override WORLD_NAME (default: the model's training world)")
    parser.add_argument("--rtf", type=float, default=1.0,
                        help="Simulator real-time factor (default 1.0 = human-watchable)")
    args = parser.parse_args(argv)
    return _host_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
