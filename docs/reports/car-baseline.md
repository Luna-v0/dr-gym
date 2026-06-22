# On-car baseline — custom car @ 192.168.15.5 · `[REAL]` · 2026-06-21

## What I did
SSH'd to `deepracer@192.168.15.5` (key auth, motors disconnected) and inventoried the system + checked for
an inference runtime. Read-only; did **not** open `/opt/aws/deepracer/password.txt`.

## Findings
| | |
|---|---|
| Board | **Raspberry Pi 4 Model B Rev 1.2** (aarch64) — this is the **custom** car, not the stock x86 Atom car |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8 |
| CPU / RAM | 4 cores · 3784 MB (≈3.4 GB available) |
| Python | 3.12.3 |
| Idle temp | **~53.6 °C** (`/sys/class/thermal/thermal_zone0/temp`). `vcgencmd` unavailable (`/dev/vcio` missing) — use the sysfs thermal zone. Cooling upgrade pending per maintainer. |
| Inference runtime | **none installed** — `import openvino` and `import onnxruntime` both fail |
| DeepRacer stack | `/opt/aws/deepracer` present (`camera`, `lib`, `nginx`, `start_ros.sh`, `util`, `artifacts`) |

## Key implication (deployment architecture)
The ONNX→OpenVINO **IR pipeline is x86-validated** (`gym_dr/optimize.py`, the 2021.4 device toolchain, the
bf16 gotcha — all Intel). This custom car is **aarch64**, where the practical engines are **onnxruntime
(strong aarch64 support)** or the **OpenVINO ARM CPU plugin** (different from the x86 IR runtime), or
**TFLite**. So there are effectively **two deploy targets**:
- **Stock car (x86 + OpenVINO)** — the existing IR pipeline fits.
- **Custom car (Pi4 aarch64)** — needs an aarch64 runtime; OpenVINO IR may not be the right artifact here.

This reinforces ADR-0001 (a dedicated `deepracer-deploy` repo) and adds a **runtime-choice decision** for the
aarch64 target.

## Blocked on (to actually benchmark memory/latency/thermals)
1. **Pick the aarch64 engine** (onnxruntime vs OpenVINO-ARM vs TFLite) — architectural, surfaced as a new
   decision (D8).
2. Install it on the Pi (a system change — will confirm before modifying the car).
3. Push a model (the exported ONNX) + a tiny harness that runs the perception/policy and logs latency,
   RSS, and the sysfs temperature under load.

## Next steps
Add D8 (engine choice). Once chosen + confirmed: install runtime, push the `p1p3_validation` ONNX export,
micro-benchmark (latency/mem/thermals) — thermals provisional until the cooling upgrade.
