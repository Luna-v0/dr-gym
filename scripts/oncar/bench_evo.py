#!/usr/bin/env python3
"""On-**EVO** inference benchmark — stock AWS DeepRacer (Intel Atom + OpenVINO **2021.1**).

The EVO analog of ``bench_engines.py`` / ``bench_model_sizes.py``, but using the
**legacy Inference Engine API** (``openvino.inference_engine.IECore``) the stock car
actually ships — the modern ``openvino.Core`` / onnxruntime used on the Pi are absent
here. Mirrors how the DeepRacer ``inference_node`` runs: ``read_network`` →
``load_network`` → ``infer`` loop.

Key facts that shaped this:
  * OpenVINO 2021.1 ``IECore.read_network`` reads ``.onnx`` **directly** (native C++
    importer) — no ``mo`` and no python-``onnx`` needed. The compiled network (hence
    latency) is identical whether the source is ``.onnx`` or a converted ``.xml`` IR,
    so timing the ONNX is a faithful proxy for the deployed IR's latency.
  * The deploy net has a dynamic batch dim, so we ``reshape`` to a static
    ``[1,4,120,160]`` before loading.

Per model × device it reports model **load/compile time**, inference **latency**
(mean/p50/p95/max ms) and **FPS**, process **RSS** (MB), and **thermals** (°C)
before→after, with a cool-down + thermal guard between runs.

    # CPU only (no root):
    python3 bench_evo.py --model-dir ~/evo_bench --devices CPU --iters 300
    # CPU + iGPU (needs the Intel NEO OpenCL driver + render/video group):
    python3 bench_evo.py --model-dir ~/evo_bench --devices CPU,GPU --iters 300
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import time
from pathlib import Path

import numpy as np
from openvino.inference_engine import IECore

_THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
_PREC_TO_NP = {"U8": np.uint8, "FP32": np.float32, "FP16": np.float16,
               "I32": np.int32, "I64": np.int64, "BOOL": np.bool_}


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


def _cooldown(label, cooldown, temp_limit, max_wait=300):
    if cooldown > 0:
        print(f"[{label}] cooldown {cooldown}s ...", flush=True)
        time.sleep(cooldown)
    waited = 0
    while temp_limit and (temp_c() or 0) > temp_limit and waited < max_wait:
        print(f"[{label}] temp {temp_c()}°C > {temp_limit}°C — waiting ...", flush=True)
        time.sleep(15)
        waited += 15


def _build_feed(net, rng):
    feed = {}
    for name, info in net.input_info.items():
        dt = _PREC_TO_NP.get(info.precision, np.float32)
        shape = info.input_data.shape
        if np.issubdtype(dt, np.integer):
            feed[name] = rng.integers(0, 256, size=shape).astype(dt)
        else:
            feed[name] = rng.standard_normal(size=shape).astype(dt)
    return feed


def _reshape_static(net, input_shape):
    """Pin every input to a static shape (batch=1) so load_network/infer work."""
    new = {}
    for name, info in net.input_info.items():
        cur = list(info.input_data.shape)
        if len(cur) == len(input_shape):
            new[name] = list(input_shape)
        else:  # fall back: replace any dynamic/<=0 dim with 1
            new[name] = [d if isinstance(d, int) and d > 0 else 1 for d in cur] or list(input_shape)
    net.reshape(new)
    return new


def bench_one(ie, model_path, device, input_shape, iters, warmup, rng):
    t0 = time.perf_counter()
    net = ie.read_network(model=str(model_path))
    t_read = (time.perf_counter() - t0) * 1e3
    shapes = _reshape_static(net, input_shape)
    feed = _build_feed(net, rng)

    t0 = time.perf_counter()
    exe = ie.load_network(network=net, device_name=device)
    t_load = (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        exe.infer(inputs=feed)
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        exe.infer(inputs=feed)
        lat.append((time.perf_counter() - t0) * 1e3)
    lat.sort()
    mean = statistics.mean(lat)
    return {
        "input_shapes": shapes,
        "read_ms": round(t_read, 1),
        "load_ms": round(t_load, 1),
        "mean_ms": round(mean, 3),
        "p50_ms": round(lat[len(lat) // 2], 3),
        "p95_ms": round(lat[min(int(len(lat) * 0.95), len(lat) - 1)], 3),
        "max_ms": round(lat[-1], 3),
        "fps": round(1000.0 / mean, 1),
        "rss_mb": rss_mb(),
    }


def _params_of(model_path, manifest):
    stem = Path(model_path).stem
    if stem in manifest and "params" in manifest[stem]:
        return manifest[stem]["params"]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=Path.home() / "evo_bench")
    ap.add_argument("--glob", default="*.onnx", help="model files to bench (also *.xml)")
    ap.add_argument("--devices", default="CPU", help="comma list, e.g. CPU or CPU,GPU")
    ap.add_argument("--input-shape", default="1,4,120,160")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--cooldown", type=int, default=30, help="seconds between runs (thermals)")
    ap.add_argument("--temp-limit", type=float, default=75.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    input_shape = [int(x) for x in args.input_shape.split(",")]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    rng = np.random.default_rng(0)

    ie = IECore()
    print(f"IECore devices available: {ie.available_devices}", flush=True)

    manifest = {}
    mpath = args.model_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    models = sorted(glob.glob(str(args.model_dir / args.glob)) +
                    glob.glob(str(args.model_dir / "*.xml")))
    models = [m for m in models if Path(m).stem != "manifest"]
    # sort by param count when known (size sweep), else by name
    models.sort(key=lambda m: (_params_of(m, manifest) or 0, m))
    print(f"models: {[Path(m).name for m in models]}", flush=True)

    results = []
    first = True
    for model_path in models:
        for device in devices:
            label = f"{Path(model_path).stem}/{device}"
            _cooldown(label, 0 if first else args.cooldown, args.temp_limit)
            first = False
            t_before = temp_c()
            row = {"model": Path(model_path).name, "device": device,
                   "params": _params_of(model_path, manifest),
                   "temp_before_c": t_before}
            try:
                row.update(bench_one(ie, model_path, device, input_shape,
                                     args.iters, args.warmup, rng))
            except Exception as exc:  # noqa: BLE001
                row["error"] = str(exc)[:300]
            row["temp_after_c"] = temp_c()
            print(f"[{label}] {row}", flush=True)
            results.append(row)

    out = args.out or (args.model_dir / "evo_benchmark.json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("\n=== EVO benchmark (OpenVINO 2021.1 IECore, single-stream) ===")
    hdr = (f"{'model':>10} {'dev':>4} {'params':>10} {'load_ms':>8} "
           f"{'mean_ms':>8} {'p50':>7} {'p95':>7} {'fps':>6} {'rss_mb':>7} {'temp':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['model']:>10} {r['device']:>4}  ERROR: {r['error'][:60]}")
        else:
            p = f"{r['params']/1e6:.1f}M" if r.get("params") else "-"
            print(f"{r['model']:>10} {r['device']:>4} {p:>10} {r['load_ms']:>8.0f} "
                  f"{r['mean_ms']:>8.2f} {r['p50_ms']:>7.2f} {r['p95_ms']:>7.2f} "
                  f"{r['fps']:>6.1f} {str(r['rss_mb']):>7} {str(r['temp_after_c']):>6}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
