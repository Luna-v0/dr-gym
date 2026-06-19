"""Closed-loop precision experiment: does FP16 inference compound in the loop?

Per-step parity (the smoke tests) is OPEN-loop and cannot answer whether a small
deployment-precision error compounds once the policy drives itself (action -> new
observation -> new action). This experiment answers it directly: it drives a trained
oval-track policy through the Gazebo sim under three inference precisions and compares
TASK metrics (lap progress, off-track rate), not per-step action diffs.

Variants (all drive the same env, same seed, N episodes each):
  - ``torch``    : the policy as trained (PyTorch FP32) — the reference.
  - ``ort_fp32`` : onnxruntime on the FP32 ONNX — harness sanity (must match torch).
  - ``ort_fp16`` : onnxruntime on an IEEE-FP16 ONNX (internal compute in half, FP32 IO).
                   This is the faithful numerical proxy for an OpenVINO FP16 IR — Gate 2
                   already proved onnxruntime == OpenVINO == PyTorch at FP32, so the only
                   new variable here is the FP16 rounding.

Why onnxruntime and not OpenVINO in the loop: OpenVINO isn't in the sim container and its
NumPy pin conflicts there; onnxruntime is a single wheel and numerically equivalent for
this question. The car's FP32 path is the Atom CPU (no bf16); FP16 is the Gen9 iGPU path.

Host/container dispatch mirrors scripts/evaluate.py::

    uv run python scripts/precision_experiment.py \\
        --model artifacts/time_trail_hard_track_trial_20/best_model/best_model.zip \\
        --episodes 8

The host exports both ONNX files, spawns the sim, then prints the comparison table.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
LOG = logging.getLogger("precision_exp")

METRIC_KEYS = [
    "dr/ep_max_progress", "dr/ep_ended_offtrack", "dr/ep_offtrack_count",
    "dr/ep_reward", "dr/ep_length", "dr/ep_mean_speed",
]


# --------------------------------------------------------------------------- #
# Container side: drive the sim under each precision
# --------------------------------------------------------------------------- #

def _container_mode() -> int:
    import numpy as np

    # onnxruntime isn't baked into the sim image; install at startup (internet is up).
    try:
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError:
        print("[pe] pip install onnxruntime ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)
    import onnxruntime as ort

    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from gym_dr.evaluate import _frame_stack_from_run_config, experiment_for_model
    from gym_dr.export import load_sb3_zip
    from gym_dr.metrics import install_metrics

    model_path = Path(os.environ["GYM_DR_PE_MODEL"])
    onnx_fp32 = Path(os.environ["GYM_DR_PE_ONNX_FP32"])
    onnx_fp16 = Path(os.environ["GYM_DR_PE_ONNX_FP16"])
    out_path = Path(os.environ["GYM_DR_PE_OUT"])
    episodes = int(os.environ.get("GYM_DR_PE_EPISODES", "8"))
    max_steps = int(os.environ.get("GYM_DR_PE_MAX_STEPS", "3000"))
    seed = int(os.environ.get("GYM_DR_PE_SEED", "123"))

    experiment = experiment_for_model(model_path)
    frame_stack = _frame_stack_from_run_config(model_path)
    wrapped, env_wrapper, _ = install_metrics(experiment)
    base_env = env_wrapper(wrapped.env_factory(wrapped))
    venv = DummyVecEnv([lambda: base_env])
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)

    model = load_sb3_zip(model_path)
    policy = model.policy
    (key,) = list(policy.observation_space.spaces.keys())
    low, high = policy.action_space.low, policy.action_space.high

    def torch_predict(obs):
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action)

    def make_ort_predict(onnx_path):
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        want_u8 = sess.get_inputs()[0].type == "tensor(uint8)"

        def predict(obs):
            tensor_obs, _ = policy.obs_to_tensor(obs)  # channels-first, matches ONNX
            arr = tensor_obs[key].cpu().numpy()
            arr = arr.astype(np.uint8) if want_u8 else arr.astype(np.float32)
            mean = sess.run(None, {in_name: arr})[0].astype(np.float32)
            # Replicate SB3 predict(deterministic): clip raw mean to the action Box.
            return np.clip(mean, low, high)

        return predict

    predictors = {
        "torch": torch_predict,
        "ort_fp32": make_ort_predict(onnx_fp32),
        "ort_fp16": make_ort_predict(onnx_fp16),
    }

    results: dict[str, list[dict]] = {}
    for variant, predict in predictors.items():
        print(f"\n[pe] === variant {variant}: {episodes} episodes ===", flush=True)
        try:
            venv.seed(seed)
        except Exception:  # noqa: BLE001
            pass
        summaries: list[dict] = []
        obs = venv.reset()
        ep = step = 0
        while ep < episodes:
            obs, _, dones, infos = venv.step(predict(obs))
            step += 1
            if dones[0] or step >= max_steps:
                summ = infos[0].get("dr_episode", {})
                summ = {k: float(summ.get(k, float("nan"))) for k in METRIC_KEYS}
                summ["_truncated"] = float(step >= max_steps and not dones[0])
                summaries.append(summ)
                print(f"[pe] {variant} ep {ep}: progress={summ['dr/ep_max_progress']:.1f} "
                      f"offtrack={summ['dr/ep_offtrack_count']:.0f} "
                      f"reward={summ['dr/ep_reward']:.0f} steps={step}", flush=True)
                ep += 1
                step = 0
                if step >= max_steps:
                    obs = venv.reset()
        results[variant] = summaries

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[pe] wrote {out_path}", flush=True)
    venv.close()
    return 0


# --------------------------------------------------------------------------- #
# Host side: export ONNX (fp32 + fp16), spawn sim, summarize
# --------------------------------------------------------------------------- #

def _export_onnx(model_path: Path, out_dir: Path):
    """Export the SB3 policy to FP32 ONNX and an IEEE-FP16 ONNX (host, dr-gym venv)."""
    from gym_dr.export import sb3_to_onnx

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "agent_fp32.onnx"
    fp16 = out_dir / "agent_fp16.onnx"
    sb3_to_onnx(model_path, fp32, opset_version=11)

    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(fp32))
    # keep_io_types: FP32/uint8 IO, FP16 internal compute (mirrors an OpenVINO FP16 IR).
    model16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model16, str(fp16))
    LOG.info("exported %s and %s", fp32.name, fp16.name)
    return fp32, fp16


def _host_mode(args) -> int:
    from gym_dr.action_space import write_model_metadata
    from gym_dr.app import _default_image
    from gym_dr.docker_runner import spawn_training_chunk
    from gym_dr.evaluate import experiment_for_model

    project_dir = Path(os.getenv("PROJECT_DIR", _PROJECT_ROOT)).resolve()
    model_path = args.model.resolve()
    out_dir = (project_dir / "tmp/precision_exp").resolve()
    fp32, fp16 = _export_onnx(model_path, out_dir)
    results_path = out_dir / "results.json"

    experiment = experiment_for_model(model_path)
    write_model_metadata(project_dir / "model_metadata.json", experiment.action_space)
    world = args.world or experiment.worlds.names[0]
    image = os.getenv("IMAGE_TAG") or _default_image(experiment.use_gpu)

    def to_c(p: Path) -> str:
        return f"/workspace/{p.relative_to(project_dir).as_posix()}"

    env = {
        "GYM_DR_IN_CONTAINER": "1",
        "WORLD_NAME": world,
        "ENABLE_GUI": "False",
        "RTF_OVERRIDE": str(args.rtf),
        "EXPERIMENT_PATH": to_c(Path(__file__).resolve()),
        "GYM_DR_PE_MODEL": to_c(model_path),
        "GYM_DR_PE_ONNX_FP32": to_c(fp32),
        "GYM_DR_PE_ONNX_FP16": to_c(fp16),
        "GYM_DR_PE_OUT": to_c(results_path),
        "GYM_DR_PE_EPISODES": str(args.episodes),
        "GYM_DR_PE_SEED": str(args.seed),
    }
    print(f"[pe] world={world!r} model={model_path.name} episodes={args.episodes}", flush=True)
    rc = spawn_training_chunk(
        image_tag=image,
        container_name=f"gym-dr-precision-{model_path.parent.name}",
        base_env=env,
        use_gpu=experiment.use_gpu,
    )
    if rc != 0 or not results_path.exists():
        print(f"[pe] FAILED (rc={rc}); results missing", file=sys.stderr)
        return rc or 1
    _summarize(json.loads(results_path.read_text()))
    return 0


def _summarize(results: dict) -> None:
    import statistics as st

    def agg(vals):
        vals = [v for v in vals if v == v]  # drop nan
        return (st.mean(vals) if vals else float("nan"),
                st.pstdev(vals) if len(vals) > 1 else 0.0)

    variants = list(results)
    n = len(next(iter(results.values()))) if results else 0
    print(f"\n{'='*78}\nCLOSED-LOOP PRECISION COMPARISON  ({n} episodes/variant)\n{'='*78}")
    header = f"{'metric':<26}" + "".join(f"{v:>17}" for v in variants)
    print(header)
    print("-" * len(header))
    pretty = {
        "dr/ep_max_progress": "max_progress (%)",
        "dr/ep_ended_offtrack": "ended_offtrack (frac)",
        "dr/ep_offtrack_count": "offtrack_count",
        "dr/ep_reward": "reward",
        "dr/ep_length": "length (steps)",
        "dr/ep_mean_speed": "mean_speed",
    }
    for k in METRIC_KEYS:
        row = f"{pretty.get(k, k):<26}"
        for v in variants:
            m, s = agg([e[k] for e in results[v]])
            row += f"{m:>10.2f}±{s:<6.2f}"
        print(row)
    print("-" * len(header))
    # Headline: does fp16 degrade lap completion vs torch?
    base = agg([e["dr/ep_max_progress"] for e in results["torch"]])[0]
    print("\nverdict (progress vs torch baseline):")
    for v in variants:
        m = agg([e["dr/ep_max_progress"] for e in results[v]])[0]
        print(f"  {v:<10} Δprogress = {m - base:+.2f} pp")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if os.getenv("GYM_DR_IN_CONTAINER"):
        return _container_mode()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--world", default=None)
    ap.add_argument("--rtf", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=123)
    return _host_mode(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
