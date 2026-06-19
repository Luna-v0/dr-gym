"""Cross-venv IR worker: convert an ONNX model to IR and run it on saved feeds.

Both smoke tests run their TensorFlow / onnxruntime / SB3 parts in one venv but need
the OpenVINO IR converted + executed in a *different* venv (legacy 2021.x ``mo`` vs the
modern toolchain, each with incompatible NumPy pins). This worker is invoked as a
subprocess with the target venv's python; it exchanges arrays as ``.npz`` files so the
"same fixed input through every runtime" contract holds by construction.

It imports ``gym_dr/optimize.py`` *by file path* on purpose: the OpenVINO venvs don't
have stable-baselines3, so importing the ``gym_dr`` package (whose ``__init__`` pulls in
SB3) would fail. ``optimize.py`` itself is stdlib-only.

Usage::

    python _ir_runner.py --onnx model.onnx --feeds in.npz --out out.npz \
        --backend legacy --input-shape 1,120,160,1 --ir-dir ir/
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

_OPT_PATH = Path(__file__).resolve().parents[1] / "gym_dr" / "optimize.py"


def _load_optimize():
    spec = importlib.util.spec_from_file_location("dr_optimize", _OPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--feeds", type=Path, required=True, help="npz of name->array inputs")
    ap.add_argument("--out", type=Path, required=True, help="npz to write outputs into")
    ap.add_argument("--backend", default="auto", choices=["auto", "legacy", "modern"])
    ap.add_argument("--input-shape", default=None, help='e.g. "1,120,160,1"')
    ap.add_argument("--reverse-input-channels", action="store_true")
    ap.add_argument("--ir-dir", type=Path, default=None)
    args = ap.parse_args()

    opt = _load_optimize()
    backend = opt.detect_backend() if args.backend == "auto" else args.backend
    ir_dir = args.ir_dir or args.onnx.parent / f"ir_{backend}"
    shape = [int(x) for x in args.input_shape.split(",")] if args.input_shape else None

    xml, _bin = opt.onnx_to_ir(
        args.onnx,
        ir_dir,
        data_type="FP32",
        input_shape=shape,
        reverse_input_channels=args.reverse_input_channels,
        backend=backend,
    )

    feeds = {k: np.asarray(v) for k, v in np.load(args.feeds).items()}
    outputs = opt.run_ir(xml, feeds, backend=backend)
    outputs = {k: np.asarray(v) for k, v in outputs.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **outputs)
    print(f"[_ir_runner] backend={backend} xml={xml}")
    print(f"[_ir_runner] outputs: {[(k, v.shape) for k, v in outputs.items()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
