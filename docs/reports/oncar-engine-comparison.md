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

## Model-size / memory "edge" (Pi 4, onnxruntime, random-weight nets)
How big a model the Pi can run vs the control budget (15 Hz = 66.7 ms/step) and its RAM cost:

| model | params | mean ms | p95 ms | model RSS | proc RSS | temp |
|---|--:|--:|--:|--:|--:|--:|
| tiny | 1.0 M | 3.0 | 3.1 | 15 MB | 63 MB | 56 °C |
| small (~p1p3 deploy) | 4.0 M | 8.1 | 9.0 | 11 MB | 74 MB | 57 °C |
| xl (~trial_18 conv) | 6.0 M | 10.2 | 10.3 | 21 MB | 96 MB | 58 °C |
| medium | 7.9 M | 12.1 | 12.2 | 16 MB | 112 MB | 59 °C |
| large | 16.0 M | 21.4 | 21.8 | 59 MB | 171 MB | 58 °C |
| xxl | 24.1 M | 34.6 | 34.9 | 66 MB | 178 MB | 61 °C |

**The Pi is not the constraint.** At 15 Hz you have ~67 ms/inference; even the **24 M-param** net runs in
**~35 ms (half the budget)** using **<200 MB of ~3.4 GB** RAM. So:
- **Memory is a non-issue** — GBs of headroom; the model itself is ≤66 MB resident.
- **Latency is the real limit.** Comfortable to ~24 M params at 15 Hz; extrapolating (~1.4 ms/M for these
  MLP-heavy nets) the 67 ms ceiling sits near ~45 M params (conv-heavy nets cost more/param, so lower).
  At **30 Hz** (33 ms) the cap is ~xl/medium (6–8 M); the current deploy net (`small`, 4 M) is **8 ms** —
  huge headroom either way.
- Latency tracks **compute, not param count** (xl's 6 M @10.2 ms vs medium's 7.9 M @12 ms — xl has deeper
  conv). Budget by FLOPs/conv depth, not just params.
- Thermals stayed 56–61 °C over the short bench; **re-check under sustained driving + post-cooling-upgrade**.

**Takeaway:** you can use a substantially bigger network than trial_18 on the custom car without missing the
control deadline — the edge is ~tens of ms of latency, not memory.

## Next
1. int8-quantize `agent.onnx`, push, re-bench on the Pi (latency + accuracy vs fp32).
2. TFLite/ExecuTorch model conversions for a complete 4-way table.
3. `deepracer-deploy` repo (ADR-0001): wrap onnxruntime + the `[-1,1]→ServoCtrlMsg` rescale + a watchdog.
