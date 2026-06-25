"""On-car OpenVINO inference benchmark (OpenVINO 2021.1, Inference Engine API).

Mirrors how the AWS DeepRacer ``inference_node`` runs a model: read the IR network,
``load_network`` onto a device, then run single-frame ``infer`` calls in a loop. We
report the metrics that matter for the car:

  * model load/compile time (``load_network``) — the "GPU is slower to load" cost,
  * per-frame inference latency (mean / median / p95 / min / max),
  * sustained single-stream throughput (FPS = 1000 / mean_ms).

Single-stream/synchronous on purpose: the car infers one camera frame at a time, so
latency-per-frame (not batched throughput) is the deployment-relevant number.

Usage (after sourcing setupvars.sh):
    python3 car_ov_bench.py --xml ir_fp32/agent.xml --device CPU --iters 300 --warmup 30
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
from openvino.inference_engine import IECore

_PREC_TO_NP = {"U8": np.uint8, "FP32": np.float32, "FP16": np.float16,
               "I32": np.int32, "I64": np.int64}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--force-fp32", action="store_true",
                    help="disable auto-bf16 on capable CPUs (Atom has none; no-op here)")
    args = ap.parse_args()

    ie = IECore()
    bin_path = args.xml[:-4] + ".bin"

    t0 = time.perf_counter()
    net = ie.read_network(model=args.xml, weights=bin_path)
    t_read = time.perf_counter() - t0

    # Build a fixed random input matching each declared input's shape + precision.
    feeds = {}
    for name, info in net.input_info.items():
        dt = _PREC_TO_NP.get(info.precision, np.float32)
        shape = info.input_data.shape
        if np.issubdtype(dt, np.integer):
            feeds[name] = np.random.randint(0, 256, size=shape).astype(dt)
        else:
            feeds[name] = np.random.rand(*shape).astype(dt)
        print(f"[input] {name} shape={shape} prec={info.precision}")

    config = {}
    if args.force_fp32:
        config["ENFORCE_BF16"] = "NO"

    t0 = time.perf_counter()
    exec_net = ie.load_network(network=net, device_name=args.device, config=config)
    t_load = time.perf_counter() - t0
    print(f"[load] read_network={t_read*1e3:.1f} ms  "
          f"load_network({args.device})={t_load*1e3:.1f} ms")

    for _ in range(args.warmup):
        exec_net.infer(inputs=feeds)

    lat = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        exec_net.infer(inputs=feeds)
        lat.append((time.perf_counter() - t0) * 1e3)

    lat.sort()
    mean = statistics.mean(lat)
    p95 = lat[int(0.95 * len(lat)) - 1]
    print(f"\n=== {args.device}  ({args.iters} iters, single-stream/sync) ===")
    print(f"  mean   {mean:7.2f} ms   ({1000.0/mean:6.1f} FPS)")
    print(f"  median {statistics.median(lat):7.2f} ms")
    print(f"  p95    {p95:7.2f} ms")
    print(f"  min    {lat[0]:7.2f} ms")
    print(f"  max    {lat[-1]:7.2f} ms")
    print(f"  stdev  {statistics.pstdev(lat):7.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
