"""Cross-venv onnxruntime worker: run an ONNX model on saved feeds.

Companion to ``_ir_runner.py``. The dr-gym export venv has torch/SB3/onnx but not
onnxruntime; onnxruntime lives in the modern OpenVINO venv. This worker runs there,
exchanging arrays via ``.npz`` so every runtime sees the identical fixed input.

Usage::

    python _ort_runner.py --onnx agent.onnx --feeds in.npz --out out.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--feeds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    want = {i.name: i.type for i in sess.get_inputs()}
    feeds = {}
    for k, v in np.load(args.feeds).items():
        v = np.asarray(v)
        # Match the graph's declared input element type (e.g. uint8 vs float).
        if want.get(k) == "tensor(uint8)":
            v = v.astype(np.uint8)
        elif want.get(k) == "tensor(float)":
            v = v.astype(np.float32)
        feeds[k] = v

    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    result = {n: np.asarray(o) for n, o in zip(names, outs)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **result)
    print(f"[_ort_runner] outputs: {[(n, result[n].shape) for n in names]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
