# DeepRacer **EVO** on-car baseline · `[REAL]` · 2026-06-25

Phase 0 of `evo-car-test-plan.md` — the x86/OpenVINO-IR analog of `car-baseline.md` (Pi).
Stock AWS DeepRacer at `192.168.0.103` (user `deepracer`, key auth). Read-only inventory.

## Hardware
| field | value |
|---|---|
| Board | AWS DeepRacer (orig.), hostname `amss-aou6`, deeplens image |
| CPU | **Intel Atom E3930** @ 1.30 GHz (turbo 1.8), **2 cores / 2 threads**, Apollo Lake |
| ISA | SSE4.2, AES, SHA — **no AVX/AVX2** ⇒ no bf16 (the OpenVINO auto-bf16 gotcha is moot here) |
| iGPU | **Intel HD Graphics 500** (Gen9 LP, 12 EU) — `/dev/dri/card0`, `renderD128` present |
| RAM | 3769 MB (≈2.5 GB available at idle) |

## Software
| field | value |
|---|---|
| OS | Ubuntu 20.04.1 LTS, kernel `4.15.0-1005-deeplens` |
| OpenVINO | **2021.1.0** (`2021.1.0-1237-bece22ac675-releases/2021/1`) — legacy `IECore` API only |
| Python | 3.8.5 (system) |
| onnxruntime | **absent** (data point — OpenVINO IR is the deployment engine on x86) |
| Model Optimizer | `mo.py` present; needs python-`onnx` (we avoid it — see below) |
| DeepRacer stack | `deepracer-core.service` (runs as **root**); stock models under `/opt/aws/deepracer/artifacts` |
| Idle thermals | `thermal_zone0` ≈ 27–33 °C |
| Idle CPU load | high — `sensor_fusion_node` ~90 %, `camera_node` ~30 % (load-avg ~4 on 2 cores) |

## Inference-engine devices (OpenVINO `IECore.available_devices`)
- **Out of the box: `['CPU', 'GNA']`** — no `GPU`. The Gen9 iGPU was present but the **Intel
  NEO OpenCL runtime was not installed** (`/etc/OpenCL/vendors/` missing, no `libigdrcl.so`);
  only the ICD *loader* `ocl-icd-libopencl1` was present. `GNA` (audio) is irrelevant here.
- **After enabling (2026-06-25): `['CPU', 'GNA', 'GPU']`** → `GPU = Intel(R) Gen9 HD Graphics
  (iGPU)`. Steps (root): side-loaded the 5 Intel NEO **19.41.14441** `.deb`s (Gen9-compatible,
  the version OpenVINO's `install_NEO_OCL_driver.sh` pins) to `~/evo_bench/neo_debs/` and
  `sudo dpkg -i *.deb` (the car's clock reads 2021 → on-car `wget`/`pip` fail TLS, so the
  debs were fetched on the workstation); then `sudo usermod -aG render,video deepracer` for
  `/dev/dri` access. Reversible: `sudo dpkg -r intel-opencl intel-ocloc intel-igc-opencl
  intel-igc-core intel-gmmlib` + `gpasswd -d`.

## How we run inference (no `mo`, no python-`onnx`)
OpenVINO 2021.1 `IECore.read_network()` reads `.onnx` **directly** via its native C++
importer; the compiled network — hence latency — is identical to a converted `.xml` IR.
So `scripts/oncar/bench_evo.py` times the ONNX directly (after `reshape` to a static
`[1,4,120,160]`). python-`onnx` is currently broken on the car (protobuf/C-ext mismatch),
and python-side IR `serialize` isn't implemented in 2021.1 — neither blocks the benchmark.

## Sensors / ROS introspection (Phase 4)
Pending (read-only; optional). The deploy net is single-camera grayscale `(1,4,120,160)`;
EVO stereo + LiDAR are extra inputs not fed to the current policy — to be recorded later
for object-avoidance / SafeRL work.
