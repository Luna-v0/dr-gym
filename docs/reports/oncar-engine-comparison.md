# On-car inference-engine comparison · `[REAL]` · 2026-06-22

## Result — Raspberry Pi 4 (aarch64), trial_18 net, input `(1,4,120,160)` uint8
| engine | mean ms | p50 | p95 | RSS MB | temp before→after |
|---|--:|--:|--:|--:|--:|
| **onnxruntime 1.27** | **11.7** | 11.7 | 12.0 | 120.5 | 54.0 → 59.4 °C |
| OpenVINO 2026.2 (ARM CPU) | 13.6 | 13.6 | 13.8 | 110.9 | 54.0 → 58.9 °C |

## Findings
- **onnxruntime is ~14% faster** (11.7 vs 13.6 ms); OpenVINO-ARM is marginally lower RSS + cooler.
- **~12 ms ⇒ ~80 Hz max** inference, vs a ~15 Hz control loop ⇒ **comfortable real-time margin even on this
  oversized net** (trial_18's `[1024]×3` MLP heads). The deployment net (`p1p3`'s `[256,256]`) will be faster.
- Thermals rose only ~5 °C over the short benchmark — fine pre-cooling-upgrade for short runs; **re-measure
  under sustained driving** once the cooling upgrade is in.

## D8 — resolved
**onnxruntime is the engine for the aarch64 custom car** (fastest here, simplest — runs our exported ONNX
directly, no conversion). OpenVINO-ARM is a working fallback. TFLite/ExecuTorch remain pending converted
models (completeness only; unlikely to beat onnxruntime given the above).

## Two deploy targets — track the differences (maintainer #4)
| target | board | arch | engine | model format | status |
|---|---|---|---|---|---|
| **custom car** | Pi 4 | aarch64 | onnxruntime | ONNX | benchmarked ✓ |
| **stock car** | DeepRacer | x86 (Atom) | OpenVINO **IR** | `.xml/.bin` | x86 IR pipeline built (2 gates); car not on-network yet |

Keep a side-by-side (latency / memory / thermals / action-accuracy) as both come online.

## Quantization evaluation (maintainer #4)
- **Pi (aarch64 CPU): int8 dynamic quantization** (`onnxruntime.quantization.quantize_dynamic`) is the real
  CPU lever (speed + memory). **fp16 usually gives no CPU speedup** (upcast to fp32). Measure latency +
  action-mean accuracy vs fp32.
- **x86 / GPU: fp16** — but mind the OpenVINO bf16 auto-cast gotcha (`docs/onnx-support-status.md`;
  `run_ir(force_fp32=...)`).
- **Accuracy gate:** action-mean within tolerance of fp32 (reuse the smoke-test parity harness,
  `scripts/smoke_test_2_parity.py`).
- Context: at ~12 ms we're already well under the 15 Hz budget, so quantization is **headroom / thermals /
  battery**, not a necessity — but worth quantifying, especially post-cooling for sustained runs.

## Next
1. int8-quantize `agent.onnx`, push, re-bench on the Pi (latency + accuracy vs fp32).
2. TFLite/ExecuTorch model conversions for a complete 4-way table.
3. `deepracer-deploy` repo (ADR-0001): wrap onnxruntime + the `[-1,1]→ServoCtrlMsg` rescale + a watchdog.
