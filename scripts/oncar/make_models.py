#!/usr/bin/env python3
"""Generate DeepRacer-shaped policy nets at a range of sizes, export to ONNX, for
the on-car **model-size / memory budget** study (run on the workstation; needs torch).

The point is to find the *edge*: how big a model the Pi can run within the control
budget, and how much RAM each size costs. Weights are random — only architecture
(latency + memory) matters for this sweep. Input matches deployment:
``(1, 4, 120, 160)`` uint8 (4-frame grayscale stack), cast to float inside (the
``normalize_images=False`` path). Output is a 2-D action (steering, speed).

    uv run --no-sync python scripts/oncar/make_models.py --out /tmp/oncar_models

Writes ``<size>.onnx`` per size + a ``manifest.json`` (params + file bytes), ready
to scp to the Pi for ``bench_model_sizes.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

# name -> (conv stack [(out_ch, kernel, stride)], features_dim, mlp_head widths)
SIZES = {
    "tiny":   ([(16, 8, 4), (32, 4, 2)], 128, [64]),
    "small":  ([(32, 8, 4), (64, 4, 2), (64, 3, 1)], 256, [128, 128]),         # ~p1p3 deploy net
    "medium": ([(32, 8, 4), (64, 4, 2), (64, 3, 1)], 512, [256, 256]),
    "large":  ([(32, 8, 4), (64, 4, 2), (128, 3, 1)], 512, [512, 512]),
    "xl":     ([(32, 8, 4), (64, 4, 2), (128, 3, 2), (128, 3, 1)], 512, [1024, 1024, 1024]),  # ~trial_18
    "xxl":    ([(64, 8, 4), (128, 4, 2), (256, 3, 2), (256, 3, 1)], 1024, [2048, 2048, 2048]),
}


class Net(nn.Module):
    def __init__(self, convs, features_dim, head):
        super().__init__()
        layers, c = [], 4
        for (o, k, s) in convs:
            pad = 0 if s > 1 else k // 2
            layers += [nn.Conv2d(c, o, k, s, padding=pad), nn.ReLU()]
            c = o
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 4, 120, 160)).flatten(1).shape[1]
        mlp, prev = [nn.Linear(flat, features_dim), nn.ReLU()], features_dim
        for h in head:
            mlp += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        mlp += [nn.Linear(prev, 2)]
        self.head = nn.Sequential(*mlp)

    def forward(self, x):
        return self.head(self.conv(x.float()).flatten(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("/tmp/oncar_models"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 4, 120, 160, dtype=torch.uint8)
    manifest = {}
    for name, (convs, fd, head) in SIZES.items():
        net = Net(convs, fd, head).eval()
        params = sum(p.numel() for p in net.parameters())
        path = args.out / f"{name}.onnx"
        torch.onnx.export(
            net, dummy, str(path),
            input_names=["FRONT_FACING_CAMERA"], output_names=["action"],
            opset_version=11,
        )
        manifest[name] = {"params": params, "onnx_bytes": path.stat().st_size}
        print(f"{name:8s} params={params:>12,}  onnx={path.stat().st_size/1e6:6.1f} MB")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(manifest)} models + manifest to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
