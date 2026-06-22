#!/usr/bin/env python3
"""On-Pi model-size / memory-budget sweep — how big a model can the edge run?

For each ``*.onnx`` in --model-dir, load it under onnxruntime and measure inference
latency (mean/p50/p95 ms), the model's loaded RSS cost (MB), and temperature, with
a thermal-paced cooldown between models. With ``manifest.json`` present it sorts by
param count, so the output is a params -> latency -> memory curve: the *edge* is the
largest model still under the control budget within the Pi's RAM.

    python bench_model_sizes.py --model-dir ~/oncar_models --iters 200 --cooldown 45
"""
from __future__ import annotations

import argparse
import glob
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


def bench_one(path, iters, warmup):
    import onnxruntime as ort

    base = rss_mb() or 0.0
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    loaded = rss_mb() or 0.0
    inp = sess.get_inputs()[0]
    shape = [1 if (not isinstance(d, int) or d <= 0) else d for d in inp.shape]
    dtype = np.uint8 if "uint8" in inp.type else np.float32
    x = (np.random.randint(0, 256, shape).astype(dtype) if dtype == np.uint8
         else np.random.randn(*shape).astype(np.float32))
    feed = {inp.name: x}
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        sess.run(outs, feed)
    lat = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(outs, feed)
        lat.append((time.perf_counter() - t) * 1000.0)
    lat.sort()
    return {
        "mean_ms": round(statistics.mean(lat), 3),
        "p50_ms": round(lat[len(lat) // 2], 3),
        "p95_ms": round(lat[int(len(lat) * 0.95)], 3),
        "model_rss_mb": round(loaded - base, 1),
        "proc_rss_mb": rss_mb(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=Path.home() / "oncar_models")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cooldown", type=int, default=45)
    ap.add_argument("--temp-limit", type=float, default=70.0)
    a = ap.parse_args()

    man = {}
    mf = a.model_dir / "manifest.json"
    if mf.exists():
        man = json.loads(mf.read_text())
    models = sorted(glob.glob(str(a.model_dir / "*.onnx")),
                    key=lambda p: man.get(Path(p).stem, {}).get("params", 0))
    rows = []
    for i, m in enumerate(models):
        name = Path(m).stem
        if i and a.cooldown:
            print(f"[{name}] cooldown {a.cooldown}s ...", flush=True)
            time.sleep(a.cooldown)
            waited = 0
            while a.temp_limit and (temp_c() or 0) > a.temp_limit and waited < 240:
                time.sleep(10); waited += 10
        try:
            r = bench_one(m, a.iters, a.warmup)
            r.update({"model": name, "params": man.get(name, {}).get("params"),
                      "onnx_mb": round(man.get(name, {}).get("onnx_bytes", 0) / 1e6, 1),
                      "temp_after_c": temp_c()})
        except Exception as exc:  # noqa: BLE001
            r = {"model": name, "error": str(exc)[:200]}
        print(f"[{name}] {r}", flush=True)
        rows.append(r)

    out = a.model_dir / "model_size_benchmark.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print("\n=== model-size sweep (onnxruntime, Pi) ===")
    print(f"{'model':>8} {'params':>13} {'mean_ms':>8} {'p95_ms':>8} {'model_MB':>9} {'proc_MB':>8} {'temp':>6}")
    for r in rows:
        if "error" in r:
            print(f"{r['model']:>8}  ERROR: {r['error']}")
            continue
        print(f"{r['model']:>8} {str(r.get('params')):>13} {r['mean_ms']:>8} {r['p95_ms']:>8} "
              f"{str(r['model_rss_mb']):>9} {str(r['proc_rss_mb']):>8} {str(r['temp_after_c']):>6}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
