#!/usr/bin/env python3
"""On-car inference-engine comparison (run ON the Pi, in the install venv).

Benchmarks the SAME policy network across whichever engines are installed and
have a model file present, reporting **latency** (mean/p50/p95 ms), process
**RSS** (MB), and **thermals** (sysfs zone, °C) before/after — with a
**cool-down interval + thermal guard between engines** (the maintainer flagged
the Pi's cooling needs an upgrade).

Engines (auto-detected):
  * onnxruntime  — runs ``<model-dir>/agent.onnx`` directly.
  * openvino     — also reads ``agent.onnx`` (ARM CPU plugin).
  * tflite       — runs ``<model-dir>/agent.tflite`` if present (conversion TBD).
  * executorch   — runs ``<model-dir>/agent.pte``   if present (conversion TBD).

Inputs are introspected from the ONNX graph (name/shape/dtype), random data of
the right dtype is generated once, and reused across ONNX-based engines for a
fair comparison.

    python bench_engines.py --model-dir ~/oncar_bench --iters 300 --cooldown 60 --temp-limit 70
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

_THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")


def temp_c():
    try:
        return round(int(_THERMAL.read_text().strip()) / 1000.0, 1)
    except Exception:
        return None


def rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


_ONNX_DTYPE = {
    "tensor(float)": np.float32, "tensor(uint8)": np.uint8,
    "tensor(double)": np.float64, "tensor(int64)": np.int64,
}


def _onnx_inputs(onnx_path):
    """Return [(name, shape, np_dtype)] from the ONNX graph via onnxruntime."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    specs = []
    for i in sess.get_inputs():
        shape = [(1 if (not isinstance(d, int) or d <= 0) else d) for d in i.shape]
        specs.append((i.name, shape, _ONNX_DTYPE.get(i.type, np.float32)))
    return specs


def _make_feed(specs, rng):
    feed = {}
    for name, shape, dt in specs:
        if np.issubdtype(dt, np.integer):
            feed[name] = rng.integers(0, 256, size=shape).astype(dt)
        else:
            feed[name] = rng.standard_normal(size=shape).astype(dt)
    return feed


def _time_loop(run_once, iters, warmup):
    for _ in range(warmup):
        run_once()
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat.sort()
    return {
        "mean_ms": round(statistics.mean(lat), 3),
        "p50_ms": round(lat[len(lat) // 2], 3),
        "p95_ms": round(lat[int(len(lat) * 0.95)], 3),
        "max_ms": round(lat[-1], 3),
    }


def bench_onnxruntime(onnx_path, feed, iters, warmup):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    return _time_loop(lambda: sess.run(out_names, feed), iters, warmup)


def bench_openvino(onnx_path, feed, iters, warmup):
    import openvino as ov
    core = ov.Core()
    compiled = core.compile_model(str(onnx_path), "CPU")
    # Map our feed (by name) to OpenVINO inputs by order.
    arrays = list(feed.values())
    return _time_loop(lambda: compiled(arrays), iters, warmup)


def bench_tflite(model_path, iters, warmup):
    try:
        from ai_edge_litert.interpreter import Interpreter  # LiteRT (successor to tflite-runtime)
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception:
            from tensorflow.lite import Interpreter  # type: ignore
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    rng = np.random.default_rng(0)
    for d in inp:
        interp.set_tensor(d["index"], rng.standard_normal(d["shape"]).astype(d["dtype"]))
    return _time_loop(interp.invoke, iters, warmup)


def _cooldown(label, cooldown, temp_limit, max_wait=300):
    if cooldown > 0:
        print(f"[{label}] cooldown {cooldown}s ...", flush=True)
        time.sleep(cooldown)
    waited = 0
    while temp_limit and (temp_c() or 0) > temp_limit and waited < max_wait:
        print(f"[{label}] temp {temp_c()}°C > {temp_limit}°C — waiting ...", flush=True)
        time.sleep(15)
        waited += 15


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=Path.home() / "oncar_bench")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--cooldown", type=int, default=60, help="seconds between engines (thermals)")
    ap.add_argument("--temp-limit", type=float, default=70.0, help="°C; wait below this before each engine")
    args = ap.parse_args()

    onnx_path = args.model_dir / "agent.onnx"
    rng = np.random.default_rng(0)
    feed = None
    if onnx_path.exists():
        specs = _onnx_inputs(onnx_path)
        feed = _make_feed(specs, rng)
        print(f"model inputs: {[(n, s, str(d)) for n, s, d in specs]}", flush=True)

    plan = []
    if onnx_path.exists():
        plan += [("onnxruntime", lambda: bench_onnxruntime(onnx_path, feed, args.iters, args.warmup)),
                 ("openvino", lambda: bench_openvino(onnx_path, feed, args.iters, args.warmup))]
    if (args.model_dir / "agent.tflite").exists():
        plan.append(("tflite", lambda: bench_tflite(args.model_dir / "agent.tflite", args.iters, args.warmup)))
    # executorch hook left for when a .pte is provided.

    results = []
    for i, (name, fn) in enumerate(plan):
        _cooldown(name, args.cooldown if i else 0, args.temp_limit)
        t_before = temp_c()
        try:
            lat = fn()
            row = {"engine": name, **lat, "rss_mb": rss_mb(),
                   "temp_before_c": t_before, "temp_after_c": temp_c()}
        except Exception as exc:  # noqa: BLE001
            row = {"engine": name, "error": str(exc)[:200]}
        print(f"[{name}] {row}", flush=True)
        results.append(row)

    out = args.model_dir / "engine_benchmark.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("\n=== engine comparison ===")
    print(f"{'engine':>12} {'mean_ms':>8} {'p50_ms':>8} {'p95_ms':>8} {'rss_mb':>8} {'temp_after':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['engine']:>12}  ERROR: {r['error']}")
        else:
            print(f"{r['engine']:>12} {r['mean_ms']:>8} {r['p50_ms']:>8} {r['p95_ms']:>8} "
                  f"{str(r['rss_mb']):>8} {str(r['temp_after_c']):>10}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
