#!/usr/bin/env python3
"""Out-of-loop evaluation on the PHYSICAL tracks (D7).

The maintainer physically owns only ``reInvent2019_track`` and ``Oval_track``.
Those are reserved OUT of sim training/eval (see docs/eval-protocol.md) so they
stay a true sim-to-real held-out signal. This script scores a trained model on
them *outside the training loop* and reports the success-criterion metrics
(clean-completion rate first).

Host usage (reconstructs the experiment from the model's run_config.json):

    uv run python scripts/eval_physical_tracks.py \\
        --model artifacts/p1p3_validation/best_model/best_model.zip --episodes 5

Host/container dispatch mirrors scripts/evaluate.py. Results print to stdout and
are written to ``<model_dir>/physical_eval.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# The maintainer's physical tracks — the whole point of this script.
PHYSICAL_TRACKS = ["reInvent2019_track", "Oval_track"]


def _container_mode() -> int:
    from gym_dr.evaluate import evaluate_on_tracks, experiment_for_model

    model = Path(os.environ["GYM_DR_EVAL_MODEL"])
    run_config = os.environ.get("GYM_DR_EVAL_RUN_CONFIG") or None
    episodes = int(os.environ.get("GYM_DR_EVAL_EPISODES", "5"))
    tracks = [t for t in os.environ.get("GYM_DR_EVAL_TRACKS", "").split(",") if t]
    tracks = tracks or PHYSICAL_TRACKS
    rc = Path(run_config) if run_config else None

    experiment = experiment_for_model(model, None, rc)
    results = evaluate_on_tracks(experiment, model, tracks, n_episodes=episodes, run_config_path=rc)

    print("\n=== Physical-track evaluation (out-of-loop) ===")
    for track, r in results.items():
        print(f"  {track:22s} clean_completion={r['clean_completion_rate']:.2f}  "
              f"completion={r['completion_rate']:.2f}  progress={r['mean_max_progress']:.1f}%  "
              f"speed={r['mean_speed']:.2f}  offtrack_rate={r['mean_offtrack_rate']:.2f}")
    out = model.parent / "physical_eval.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def _host_mode(args: argparse.Namespace) -> int:
    from gym_dr.action_space import write_model_metadata
    from gym_dr.app import _default_image
    from gym_dr.docker_runner import spawn_training_chunk
    from gym_dr.evaluate import experiment_for_model

    project_dir = Path(os.getenv("PROJECT_DIR", _PROJECT_ROOT)).resolve()
    model_path = args.model.resolve()
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1
    try:
        model_path.relative_to(project_dir)
    except ValueError:
        print(f"model must live inside the project dir ({project_dir})", file=sys.stderr)
        return 1

    experiment = experiment_for_model(model_path, None, None)
    write_model_metadata(project_dir / "model_metadata.json", experiment.action_space)
    image = os.getenv("IMAGE_TAG") or _default_image(experiment.use_gpu)
    tracks = args.tracks.split(",") if args.tracks else PHYSICAL_TRACKS

    def to_container(p: Path) -> str:
        return f"/workspace/{p.relative_to(project_dir).as_posix()}"

    env = {
        "GYM_DR_IN_CONTAINER": "1",
        "WORLD_NAME": tracks[0],
        "EXPERIMENT_PATH": to_container(Path(__file__).resolve()),
        "GYM_DR_EVAL_MODEL": to_container(model_path),
        "GYM_DR_EVAL_EPISODES": str(args.episodes),
        "GYM_DR_EVAL_TRACKS": ",".join(tracks),
        "RTF_OVERRIDE": str(args.rtf),
    }
    # Feature/asym models pick their obs shape from env vars the run_config.json
    # doesn't capture: GYM_DR_FEATURE_SET (9- vs 11-feature actor vector) and
    # GYM_DR_ASYM_CRITIC (=1 -> Dict{actor,critic} obs for the asymmetric value
    # net). Reconstruction defaults to the 9-feature Box, which mismatches an
    # 11-input / Dict-obs policy — so forward them from the host when set (no-op
    # for camera / default-feature models). dispatch.feature_time_trial reads both.
    for _var in ("GYM_DR_FEATURE_SET", "GYM_DR_ASYM_CRITIC"):
        if os.environ.get(_var):
            env[_var] = os.environ[_var]
    print(f"[eval-physical] model={model_path.name} tracks={tracks} rtf={args.rtf}", flush=True)
    return spawn_training_chunk(
        image_tag=image,
        container_name=f"gym-dr-physeval-{model_path.parent.name}",
        base_env=env,
        use_gpu=experiment.use_gpu,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if os.getenv("GYM_DR_IN_CONTAINER"):
        return _container_mode()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--tracks", default="", help="comma-separated; default the physical tracks")
    p.add_argument("--rtf", type=float, default=10.0)
    return _host_mode(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
