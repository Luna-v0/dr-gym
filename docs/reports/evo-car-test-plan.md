# DeepRacer **EVO** on-car test plan · stock x86 (Intel Atom) + OpenVINO IR

Date: 2026-06-25 · target: **AWS DeepRacer EVO** (the official car; NOT the Pi4 custom car at
`192.168.15.5`). This fills the missing x86/IR side of the latency·memory·thermal·parity table —
the EVO was the *"car not on-network yet"* row in `oncar-engine-comparison.md`.

These mirror the **Raspberry-Pi benchmarks** (`bench_engines.py`, `bench_model_sizes.py`,
`smoke_test_2_parity.py`) but on the EVO's **Intel Atom + OpenVINO IR** runtime, plus EVO-only
sensor introspection (stereo cameras + LiDAR).

## Ground rules (every test)
- **Motors disconnected / car on a stand. No driving.** These are inference + introspection
  benchmarks only — exactly like the Pi tests.
- **Read-only on the device** except installing a Python runtime into a *user* venv (confirm first;
  never touch `/opt/aws/deepracer/...` or the stock model path).
- Each engine/run: **cooldown + thermal guard** between runs (`bench_engines.py --cooldown 60
  --temp-limit 70`). Atom thermals differ from the Pi — record them fresh.
- Capture everything to JSON on the car, scp back, drop the numbers into
  `oncar-engine-comparison.md` next to the Pi rows.

---

## Phase 0 — Inventory / baseline  (read-only, ~10 min)
The EVO analog of `car-baseline.md`. SSH in (key auth), inventory, **do not** open
`password.txt`. Record:

| field | how |
|---|---|
| Board / arch | `uname -m` (expect `x86_64`), `lscpu` (Intel **Atom** model + cores) |
| OS / kernel | `lsb_release -a`, `uname -r` (stock DeepRacer device image) |
| RAM | `free -m` |
| Idle temp | `sensors` if present, else `cat /sys/class/thermal/thermal_zone*/temp` |
| OpenVINO | `python3 -c "import openvino; print(openvino.__version__)"` — **expect 2021.x** (the `mo`/IR the car actually runs) |
| onnxruntime | `python3 -c "import onnxruntime"` (likely absent — that's a data point) |
| DeepRacer stack | `ls /opt/aws/deepracer`; `systemctl --type=service | grep -i deepracer` |
| **EVO sensors** | `ros2 topic list` — confirm **stereo** camera topics + **LiDAR** scan topic (see Phase 4) |

**Output:** `docs/reports/evo-car-baseline.md`.

---

## Phase 1 — Engine / IR latency benchmark  (the core Pi-test analog, ~20 min)
Latency (mean/p50/p95 ms), process RSS, thermals before→after — on the **same net** the Pi ran
(`trial_18` conv, input `(1,4,120,160)` uint8), so the rows are directly comparable.

On x86 the engines of interest are:
1. **OpenVINO IR (`.xml`/`.bin`)** — the runtime the stock car uses. **The headline number.**
2. OpenVINO reading `agent.onnx` directly (sanity).
3. onnxruntime-x86 (if installable) — cross-check.

Artifacts (built on the workstation, scp'd to the car):
- `agent.onnx` — the deploy net (from `make_models.py`'s `small`/`xl`, or a real export).
- `agent.xml` / `agent.bin` — IR from **both** backends via `gym_dr/optimize.py`:
  - **legacy** (`.venv-ov-legacy`, OpenVINO 2021.x `mo`) — *what the EVO actually runs*.
  - **modern** (`.venv-ov-modern`) — for the IR-vs-IR diff.
  - ⚠️ **bf16 auto-cast gotcha**: on some OpenVINO builds the IR silently casts to bf16 →
    accuracy drift. Convert/run with `force_fp32` (see `onnx-support-status.md`) and verify in
    Phase 3.

Run on the car:
```
python3 scripts/oncar/bench_engines.py --model-dir ~/evo_bench --iters 300 \
    --cooldown 60 --temp-limit 70
```
(`bench_engines.py` already auto-detects openvino + onnxruntime; **extend it to also time the
compiled `agent.xml` IR** — small add via `scripts/oncar/_ir_runner.py`, prepped below.)

**Pass:** mean inference **≪ control budget** (15 Hz = 66.7 ms; 30 Hz = 33 ms). Compare Atom-IR
vs Pi-onnxruntime (11.7 ms). Flag if the Atom is slower than the Pi (older single-thread).

---

## Phase 2 — Model-size / latency edge sweep  (~25 min)
The EVO analog of the Pi size sweep — how big a net the Atom runs within budget, and its RAM cost.
```
# workstation: build the size ladder once
uv run --no-sync python scripts/oncar/make_models.py --out /tmp/evo_models
# convert each size to IR (legacy + modern) — prepped helper below
.venv-ov-legacy/bin/python scripts/oncar/make_ir.py --in /tmp/evo_models --backend legacy
# car: bench latency/RSS/thermals across sizes
python3 scripts/oncar/bench_model_sizes.py --model-dir ~/evo_models --iters 200 \
    --cooldown 45 --temp-limit 70
```
**Output:** an Atom column beside the Pi's tiny→xxl table. Report the param/latency edge at 15 Hz
and 30 Hz (the Pi's edge was ~24 M @ 35 ms; the Atom will differ).

---

## Phase 3 — Numerical parity gate  (correctness, ~10 min)
Proves the **deployed IR computes the same action** as the trained policy — the silent-failure
surface. Uses the existing harness:
```
.venv/bin/python scripts/smoke_test_2_parity.py
```
Checks SB3 `forward(deterministic)` action-mean vs onnxruntime vs **OpenVINO IR (legacy + modern)**,
atol ~1e-4, **fp32 and fp16** (mind the bf16 gotcha → `force_fp32`). Also confirms the
engineering-units → `ServoCtrlMsg [-1,1]` rescale contract (deploy-time silent failure if missed).

**Pass:** IR action-mean within tolerance of the SB3 reference, fp32 **and** fp16. A fail here = do
**not** deploy that IR.

---

## Phase 4 — EVO sensor + ROS introspection  (read-only, optional, ~15 min)
EVO-specific and a precondition for the `customrl` deploy path
(`physical-car-integration-notes.md` steps 1–2). Non-destructive:
- `ros2 topic list` / `ros2 topic info` — confirm **stereo** camera topics (EVO has two
  `FRONT_FACING_CAMERA`s → `STEREO_CAMERAS`) + the **LiDAR** `/scan`, their msg types + rates
  (`ros2 topic hz`).
- Confirm the **actuation** topic/service (`ServoCtrlMsg`) and the `ctrl_pkg` mode services
  (`vehicle_state` / `enable_state`).
- **Record a short camera rosbag** (`ros2 bag record <camera topic> -d 30`) on the stand — feeds
  the rosbag→features perception pipeline (Task #17) and lets us check sim-vs-real image stats
  (brightness/scale) for the DR `obs_*` ranges.

**Note:** our deploy net is **single front camera** grayscale `(1,4,120,160)`. The EVO's stereo +
LiDAR are extra inputs we are *not* feeding the current policy — record them now for later
object-avoidance / SafeRL work, but the Phase 1–3 benchmark stays single-camera for Pi
comparability.

---

## What's ready vs. what I'll prep
| item | status |
|---|---|
| `bench_engines.py`, `bench_model_sizes.py`, `make_models.py`, `smoke_test_2_parity.py` | ✅ exist |
| `optimize.py` ONNX→IR (legacy 2021.x + modern), `.venv-ov-legacy/-modern` | ✅ exist |
| `make_ir.py` (batch-convert the size ladder to IR) | ⏳ I'll add (thin loop over `optimize.onnx_to_ir`) |
| IR row in `bench_engines.py` (time compiled `agent.xml`) | ⏳ small add via `_ir_runner` |
| `evo_bench/` bundle (agent.onnx + agent.xml/.bin) ready to scp | ⏳ I'll build once you confirm |
| real trained camera ONNX (vs random-weight) | ⚠️ none exported yet — random nets are fine for latency/mem/thermal + IR-vs-ONNX parity; a real export needed only for *trained-action* parity |

## Sequencing on the day
0 → 1 → 3 first (baseline, headline latency, parity). 2 if time. 4 last (read-only, no model
needed). All safe with motors disconnected.
