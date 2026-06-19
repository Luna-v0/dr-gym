"""Collect REAL camera observations from the DeepRacer simulator.

Random pixels make a trained policy collapse to a near-constant action, so they are a
weak parity stimulus. This drives a trained model through the Gazebo sim for a few steps
and saves the *policy-ready* observations (channels-first uint8, frame-stacked exactly as
the network consumes them) to an ``.npz`` — which ``smoke_test_2_parity.py --obs-npz``
then replays through SB3 / onnxruntime / OpenVINO.

Host/container dispatch mirrors ``scripts/evaluate.py``::

    # host: spawns the sim container, writes tmp/sim_obs.npz back to the project dir
    uv run python scripts/collect_sim_obs.py \\
        --model artifacts/time_trial_trial18_10x/best_model/best_model.zip \\
        --steps 30 --out tmp/sim_obs.npz

The npz holds one array per sensor key, shape ``(steps, C, H, W)`` uint8.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _container_mode() -> int:
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from gym_dr.evaluate import _frame_stack_from_run_config, experiment_for_model
    from gym_dr.export import load_sb3_zip
    from gym_dr.metrics import install_metrics

    model_path = Path(os.environ["GYM_DR_COLLECT_MODEL"])
    out_path = Path(os.environ["GYM_DR_COLLECT_OUT"])
    steps = int(os.environ.get("GYM_DR_COLLECT_STEPS", "30"))

    experiment = experiment_for_model(model_path)
    frame_stack = _frame_stack_from_run_config(model_path)
    wrapped, env_wrapper, _ = install_metrics(experiment)
    base_env = env_wrapper(wrapped.env_factory(wrapped))
    venv = DummyVecEnv([lambda: base_env])
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)

    model = load_sb3_zip(model_path)
    frames: list[dict] = []
    obs = venv.reset()
    for i in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        # The exact tensor the policy consumes: channels-first, frame-stacked,
        # uint8 (normalize_images=False -> no /255). This is the ONNX input.
        tensor_obs, _ = model.policy.obs_to_tensor(obs)
        frames.append({k: v.cpu().numpy() for k, v in tensor_obs.items()})
        obs, _, dones, _ = venv.step(action)
        print(f"[collect] step {i} action={np.asarray(action).ravel()}", flush=True)

    keys = list(frames[0].keys())
    arrs = {k: np.concatenate([f[k] for f in frames], axis=0).astype(np.uint8) for k in keys}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrs)
    print(f"[collect] wrote {out_path}: {[(k, arrs[k].shape) for k in keys]}", flush=True)
    venv.close()
    return 0


def _host_mode(args: argparse.Namespace) -> int:
    from gym_dr.action_space import write_model_metadata
    from gym_dr.app import _default_image
    from gym_dr.docker_runner import spawn_training_chunk
    from gym_dr.evaluate import experiment_for_model

    project_dir = Path(os.getenv("PROJECT_DIR", _PROJECT_ROOT)).resolve()
    model_path = args.model.resolve()
    out_path = (project_dir / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    experiment = experiment_for_model(model_path)
    write_model_metadata(project_dir / "model_metadata.json", experiment.action_space)
    world = args.world or experiment.worlds.names[0]
    image = os.getenv("IMAGE_TAG") or _default_image(experiment.use_gpu)

    def to_container(p: Path) -> str:
        return f"/workspace/{p.relative_to(project_dir).as_posix()}"

    env = {
        "GYM_DR_IN_CONTAINER": "1",
        "WORLD_NAME": world,
        "ENABLE_GUI": "False",
        "RTF_OVERRIDE": str(args.rtf),
        "EXPERIMENT_PATH": to_container(Path(__file__).resolve()),
        "GYM_DR_COLLECT_MODEL": to_container(model_path),
        "GYM_DR_COLLECT_OUT": to_container(out_path),
        "GYM_DR_COLLECT_STEPS": str(args.steps),
    }
    print(f"[collect] world={world!r} model={model_path.name} steps={args.steps} -> {args.out}",
          flush=True)
    rc = spawn_training_chunk(
        image_tag=image,
        container_name=f"gym-dr-collect-{model_path.parent.name}",
        base_env=env,
        use_gpu=experiment.use_gpu,
    )
    if rc == 0 and out_path.exists():
        print(f"[collect] OK: {out_path}")
    return rc


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if os.getenv("GYM_DR_IN_CONTAINER"):
        return _container_mode()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("tmp/sim_obs.npz"))
    ap.add_argument("--world", default=None)
    ap.add_argument("--rtf", type=float, default=10.0)
    return _host_mode(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
