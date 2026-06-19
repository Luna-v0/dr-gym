"""Smoke Test 1 — conversion-pipeline correctness (TF -> ONNX -> IR), CPU, FP32.

Gate 1 of the ONNX-support plan. Algorithm-agnostic: proves the ``.onnx -> IR`` branch
(``gym_dr.optimize.onnx_to_ir``) yields IR numerically faithful to the source graph,
*independently of any RL model*. No SB3, no dr-gym env.

Chain: a stock DeepRacer TF frozen graph (``model_15.pb``) -> tf2onnx (opset 11) ->
OpenVINO IR via BOTH the legacy 2021.x ``mo`` and the modern toolchain -> a four-way
inference comparison (TensorFlow / onnxruntime / OpenVINO-legacy / OpenVINO-modern) on a
single fixed input.

Run with the MODERN venv (has tensorflow + tf2onnx + onnxruntime + modern openvino); the
legacy IR step is shelled out to the legacy venv's python::

    .venv-ov-modern/bin/python scripts/smoke_test_1_pipeline.py

Gate (two parts, both required):

1. **Numeric**: TF vs onnxruntime vs each OpenVINO IR within rtol=1e-4/atol=1e-5, on a
   normalized [0,1] input. With OpenVINO forced to FP32 inference, all four runtimes agree
   to ~1e-8.
2. **Decision equivalence**: argmax(output) identical across every runtime, on BOTH the
   normalized input and a raw [0,255] stress input.

Key finding: OpenVINO auto-selects **bf16** inference for an FP32 IR on AVX512_BF16 CPUs.
Left on, that injects ~3e-4 (normalized) / ~1e-3 (raw) error — and BOTH the 2021.4 and
2024 toolchains do it, agreeing with each other, which can masquerade as a "toolchain
precision floor". ``gym_dr.optimize.run_ir(force_fp32=True)`` disables it; we also print
the raw-[0,255] matrix as a diagnostic. Max-abs-diff is always printed for every pair.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PB = (
    REPO.parent
    / "deepracer-utils/tests/deepracer/model/sample-model/model/model_15.pb"
)
DEFAULT_LEGACY_PY = REPO / ".venv-ov-legacy/bin/python"
IR_RUNNER = REPO / "scripts/_ir_runner.py"

# Tolerances vs the TF reference. With OpenVINO forced to FP32 inference (run_ir's
# force_fp32=True, which disables auto-bf16 on AVX512_BF16 CPUs), every runtime — TF,
# onnxruntime, and both OpenVINO toolchains — agrees to ~1e-8 on this model, on both
# normalized and raw input. So a single tight tolerance suffices.
#
# NB: WITHOUT force_fp32, OpenVINO silently runs the FP32 IR in bf16 and the diff jumps
# to ~3e-4 (normalized) / ~1e-3 (raw); both legacy and modern do this and agree with each
# other, which is what made it look like a "toolchain precision floor". It is not — it is
# bf16. That is the real, reportable finding of this gate.
TOL = {
    "onnxruntime":     dict(rtol=1e-4, atol=1e-5),
    "openvino-modern": dict(rtol=1e-4, atol=1e-5),
    "openvino-legacy": dict(rtol=1e-4, atol=1e-5),
}


# --------------------------------------------------------------------------- #
# TensorFlow: discover graph IO + run reference
# --------------------------------------------------------------------------- #

def load_graphdef(pb_path: Path):
    import tensorflow as tf

    gdef = tf.compat.v1.GraphDef()
    gdef.ParseFromString(Path(pb_path).read_bytes())
    g = tf.Graph()
    with g.as_default():
        tf.compat.v1.import_graph_def(gdef, name="")
    return g


def discover_io(g):
    """Return (input_op_name, output_tensor_name) for a DeepRacer policy graph."""
    ops = g.get_operations()
    inputs = [op for op in ops if op.type == "Placeholder"]
    if len(inputs) != 1:
        raise RuntimeError(f"expected 1 Placeholder, found {[o.name for o in inputs]}")
    consumed = {inp.name for op in ops for inp in op.inputs}
    leaves = [
        o.name
        for op in ops
        if op.type not in ("Const", "Assign", "NoOp", "Placeholder")
        for o in op.outputs
        if o.name not in consumed
    ]
    # Prefer the policy head if present.
    policy = [n for n in leaves if "policy" in n.lower()]
    out_name = (policy or leaves)[0]
    return inputs[0].name, out_name


def tf_infer(g, in_op_name, out_tensor_name, x):
    import tensorflow as tf

    with tf.compat.v1.Session(graph=g) as sess:
        return sess.run(out_tensor_name, {in_op_name + ":0": x})


# --------------------------------------------------------------------------- #
# tf2onnx + onnxruntime
# --------------------------------------------------------------------------- #

def tf_to_onnx(pb_path: Path, in_tensor: str, out_tensor: str, onnx_path: Path, opset: int = 11):
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--graphdef", str(pb_path),
        "--inputs", in_tensor,
        "--outputs", out_tensor,
        "--opset", str(opset),
        "--output", str(onnx_path),
    ]
    print("[tf2onnx]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not onnx_path.exists():
        raise RuntimeError(f"tf2onnx failed:\n{proc.stdout}\n{proc.stderr}")
    return onnx_path


def ort_infer(onnx_path: Path, x):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: x})[0]
    return in_name, out


# --------------------------------------------------------------------------- #
# IR worker (subprocess into the right venv)
# --------------------------------------------------------------------------- #

def run_ir_worker(python: Path, onnx_path: Path, feeds: dict, out_npz: Path,
                  backend: str, input_shape=None):
    feeds_npz = out_npz.with_name(out_npz.stem + "_feeds.npz")
    np.savez(feeds_npz, **feeds)
    cmd = [
        str(python), str(IR_RUNNER),
        "--onnx", str(onnx_path),
        "--feeds", str(feeds_npz),
        "--out", str(out_npz),
        "--backend", backend,
    ]
    if input_shape is not None:
        cmd += ["--input-shape", ",".join(str(int(d)) for d in input_shape)]
    print(f"[ir:{backend}]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.returncode != 0 or not out_npz.exists():
        raise RuntimeError(f"IR worker ({backend}) failed:\n{proc.stdout}\n{proc.stderr}")
    data = np.load(out_npz)
    return data[data.files[0]]  # single-output graph


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def diff_matrix(results: dict, title: str):
    names = list(results)
    print(f"\n=== max-abs-diff matrix [{title}] ===")
    print("           " + "".join(f"{n:>16}" for n in names))
    for a in names:
        row = f"{a:>11}"
        for b in names:
            d = float(np.max(np.abs(results[a].astype(np.float64) - results[b].astype(np.float64))))
            row += f"{d:>16.2e}"
        print(row)


def numeric_gate(results: dict):
    """TF vs ONNX (tight) and TF vs each IR (FP32 envelope)."""
    ref = "tensorflow"
    print("\n=== numeric gate vs tensorflow ===")
    ok = True
    for n in results:
        if n == ref:
            continue
        rtol, atol = TOL[n]["rtol"], TOL[n]["atol"]
        passed = np.allclose(results[ref], results[n], rtol=rtol, atol=atol)
        ok = ok and passed
        print(f"  {ref} vs {n:<18}: {'PASS' if passed else 'FAIL'} "
              f"(max abs diff {np.max(np.abs(results[ref]-results[n])):.2e}, "
              f"rtol={rtol:.0e} atol={atol:.0e})")
    return ok


def decision_gate(results: dict):
    """argmax (the discrete action / decision) must match the TF reference everywhere."""
    ref_arg = int(np.argmax(results["tensorflow"]))
    print(f"\n=== decision gate (argmax, tf={ref_arg}) ===")
    ok = True
    for n, v in results.items():
        a = int(np.argmax(v))
        match = a == ref_arg
        ok = ok and match
        print(f"  {n:<18}: argmax={a} {'OK' if match else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pb", type=Path, default=DEFAULT_PB)
    ap.add_argument("--legacy-python", type=Path, default=DEFAULT_LEGACY_PY)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.pb.exists():
        print(f"ERROR: frozen graph not found: {args.pb}", file=sys.stderr)
        return 2

    tmp = args.workdir or Path(tempfile.mkdtemp(prefix="smoke1_"))
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {tmp}")

    # 1. Discover IO.
    g = load_graphdef(args.pb)
    in_op, out_tensor = discover_io(g)
    in_tensor = in_op + ":0"
    print(f"input : {in_tensor}")
    print(f"output: {out_tensor}")
    shape = [1, 120, 160, 1]  # static (batch pinned for legacy mo)

    # 2. Convert once (tf2onnx); discover the ONNX input name.
    onnx_path = tmp / "model.onnx"
    tf_to_onnx(args.pb, in_tensor, out_tensor, onnx_path, opset=11)
    onnx_in_name, _ = ort_infer(onnx_path, np.zeros(shape, np.float32))
    print(f"[onnx] input name '{onnx_in_name}'")
    have_legacy = args.legacy_python.exists()
    if not have_legacy:
        print(f"WARN: legacy python not found at {args.legacy_python}; skipping legacy IR")

    def run_all(x, tag):
        res = {"tensorflow": np.asarray(tf_infer(g, in_op, out_tensor, x))}
        _, ort_out = ort_infer(onnx_path, x)
        res["onnxruntime"] = np.asarray(ort_out)
        res["openvino-modern"] = run_ir_worker(
            Path(sys.executable), onnx_path, {onnx_in_name: x},
            tmp / f"out_modern_{tag}.npz", backend="modern", input_shape=shape)
        if have_legacy:
            res["openvino-legacy"] = run_ir_worker(
                args.legacy_python, onnx_path, {onnx_in_name: x},
                tmp / f"out_legacy_{tag}.npz", backend="legacy", input_shape=shape)
        return res

    rng = np.random.default_rng(args.seed)
    raw = rng.uniform(0, 255, size=shape).astype(np.float32)
    norm = raw / 255.0  # normalized [0,1]: the realistic vision-model regime

    # --- Primary gate: normalized input ---
    print("\n########## NORMALIZED [0,1] (numeric + decision gate) ##########")
    res_norm = run_all(norm, "norm")
    diff_matrix(res_norm, "normalized [0,1]")
    num_ok = numeric_gate(res_norm)
    dec_ok_norm = decision_gate(res_norm)

    # --- Diagnostic: raw input (decision gate only; numeric reported) ---
    print("\n########## RAW [0,255] (diagnostic; decision gate) ##########")
    res_raw = run_all(raw, "raw")
    diff_matrix(res_raw, "raw [0,255] — FP32 amplification, NOT gated numerically")
    dec_ok_raw = decision_gate(res_raw)

    # --- legacy vs modern OpenVINO, the requested comparison ---
    if have_legacy:
        print("\n=== legacy-vs-modern OpenVINO (max abs diff vs onnxruntime ref) ===")
        for tag, res in [("normalized", res_norm), ("raw", res_raw)]:
            dm = np.max(np.abs(res["openvino-modern"] - res["onnxruntime"]))
            dl = np.max(np.abs(res["openvino-legacy"] - res["onnxruntime"]))
            print(f"  {tag:>10}: modern={dm:.2e}  legacy={dl:.2e}  "
                  f"(legacy/modern = {dl/max(dm,1e-12):.0f}x)")

    ok = num_ok and dec_ok_norm and dec_ok_raw
    print("\n=== GATE 1: %s ===" % ("PASS" if ok else "FAIL"))
    print(f"    numeric(normalized)={num_ok}  decision(norm)={dec_ok_norm}  decision(raw)={dec_ok_raw}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
