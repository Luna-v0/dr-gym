# On-Car OpenVINO Benchmark — Progress & Resume Notes

> ✅ **COMPLETED 2026-06-25.** Results live in **`docs/reports/evo-car-baseline.md`** (Phase 0
> inventory + iGPU enablement) and **`docs/reports/oncar-engine-comparison.md`** (the EVO
> CPU-vs-iGPU latency table + findings). Benchmark script: `scripts/oncar/bench_evo.py`.
> Headline: deploy net (4.4 M) = **6.3 ms / 158 FPS CPU**, **4.7 ms / 211 FPS iGPU** (clean),
> both ~10–14× under the control budget. The notes below are the original 2026-06-19 plan.

_Paused 2026-06-19. Goal: measure whether a dr-gym PyTorch (SB3 PPO) policy, exported
PyTorch → ONNX → OpenVINO IR, runs well on the physical DeepRacer's **CPU** and **iGPU**
— latency, FPS, and model-load time. This is the unfinished "Stage 3 on-device CPU-vs-GPU
benchmark" deferred in `docs/onnx-support-status.md`._

## The car (192.168.0.103, user `deepracer`, no password)

- Original AWS DeepRacer, hostname `amss-aou6`.
- **CPU:** Intel Atom E3930 @ 1.3 GHz (turbo 1.8), 2 cores / 2 threads, **no AVX**
  → no bf16, so the "auto-bf16 precision" caveat from `onnx-support-status.md` is moot here.
- **iGPU:** Intel HD Graphics 500 (Gen9), present at `/dev/dri/card0` + `renderD128`.
- RAM 3.7 GiB. **OpenVINO 2021.1.110** at `/opt/intel/openvino_2021`. Python 3.8.5.
- `mo.py`: `/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py`.
- **OpenVINO IE `available_devices` = `['CPU', 'GNA']` — GPU NOT exposed out of the box.**
  (GNA = audio accelerator, useless here.) Causes: `/etc/OpenCL/vendors/` is **empty**
  (no OpenCL ICD registered) and `deepracer` is **not in the `video`/`render` groups**.
  The GPU plugin lib `libclDNNPlugin.so` *does* exist.
- `deepracer-core.service` runs as **root**.
- **Car clock is wrong (~2021-03-05)** → on-car `pip` TLS handshakes fail, no PyPI.
  Worked around by side-loading wheels from the host.
- **Idle load is high:** `sensor_fusion_node` ~86%, `camera_node` ~58%, `web_video_server`
  ~38%, load-avg ~4 on 2 cores. CPU benchmark numbers will be depressed unless we stop
  `deepracer-core` first.

## The model under test

- `tmp/precision_exp/agent_fp32.onnx` (committed in the `onnx` commit). Architecture
  `DEEP_CONVOLUTIONAL_NETWORK_SHALLOW`; input `FRONT_FACING_CAMERA [N,4,120,160]` uint8
  (4-frame grayscale stack), output `action [N,2]` (steering, speed).
- Trained SB3 `.zip` sources live in `dr-gym/artifacts/` (e.g.
  `tt_multiworld_trial_*/best_model/best_model.zip`, `time_trial_demo/final_model.zip`).
  Re-export with `gym_dr.export.sb3_to_onnx` if a different model is wanted; perf depends
  on architecture, not weights, so the committed ONNX is representative.

## ✅ Done

1. Reviewed the `onnx` commit (`gym_dr/optimize.py`, smoke tests, `precision_experiment.py`,
   `docs/onnx-support-status.md`). The car deploys OpenVINO **IR**, so the on-device test
   is: ONNX → IR (via car's `mo`) → `IECore.load_network` → timed `infer` loop.
2. Confirmed community stack `aws-deepracer-community/deepracer-custom-car` runs OpenVINO
   **GPU/iGPU inference on the original DeepRacer** ("reduces CPU load, increases model load
   time") — so the iGPU path is legitimately viable on this car.
3. SSH OK. Copied `agent_fp32.onnx` → car `~/ov_bench/`.
4. Wrote `scripts/car_ov_bench.py` (uses the **same IECore API the real `inference_node`
   uses**: `read_network` → `load_network` → timed single-frame `infer`; reports load time,
   mean/median/p95/min/max latency, FPS). Copied to car `~/ov_bench/`.
5. Installed `onnx==1.10.2` + `typing_extensions` on the car (`pip --user --no-index
   --no-deps`, from host-cross-downloaded cp38 wheels in `~/ov_bench/wheels/`). Needed
   because the car's `mo.py` requires `onnx` to read ONNX. **System numpy 1.19.5 /
   protobuf 3.15.5 left untouched; root ROS services unaffected.**

## ⏭️ Next (resume here)

1. **Convert ONNX → IR on the car** (safe — writes only to `~/ov_bench`). This was the
   next step when we paused:
   ```bash
   ssh deepracer@192.168.0.103
   source /opt/intel/openvino_2021/bin/setupvars.sh
   cd ~/ov_bench
   MO=/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py
   python3 $MO --input_model agent_fp32.onnx --output_dir ir_fp32 --model_name agent \
       --data_type FP32 --input_shape "[1,4,120,160]"
   python3 $MO --input_model agent_fp32.onnx --output_dir ir_fp16 --model_name agent \
       --data_type FP16 --input_shape "[1,4,120,160]"
   ```
2. **CPU benchmark (FP32):**
   ```bash
   python3 ~/ov_bench/car_ov_bench.py --xml ~/ov_bench/ir_fp32/agent.xml --device CPU \
       --iters 300 --warmup 30
   ```
   Run **twice** for a fair picture: once as-is (realistic, services running) and once with
   `deepracer-core` stopped (clean) — `sudo systemctl stop deepracer-core` then `start`.
3. **Enable + benchmark iGPU (FP16)** — needs **root** (user offered):
   - Grant device access: `sudo usermod -aG render,video deepracer` (then re-login), OR run
     the GPU benchmark under `sudo`.
   - `python3 ~/ov_bench/car_ov_bench.py --xml ~/ov_bench/ir_fp16/agent.xml --device GPU`
   - If it errors `[CLDNN ERROR] No GPU device was found` / clGetPlatformIDs -1001, the
     Intel OpenCL runtime isn't active: try OpenVINO's
     `install_NEO_OCL_driver.sh` (under `/opt/intel/openvino_2021/.../install_dependencies/`)
     or register an ICD in `/etc/OpenCL/vendors/`. Expect a **slow first `load_network`**
     on GPU (clDNN kernel JIT).
4. Record metrics → answer "optimal on CPU vs iGPU?" Target: the car's camera is ~15–30 fps,
   so per-frame latency well under ~33–66 ms = real-time-capable.

## Root commands the user can run (they offered)

- `sudo systemctl stop deepracer-core` / `start` — free CPU for a clean CPU benchmark.
- `sudo usermod -aG render,video deepracer` — grant iGPU (`/dev/dri`) access. Re-login after.
- (Optional) fix clock for on-car pip: not required — wheels are side-loaded.
- (If GPU plugin can't find the device) enable Intel OpenCL/NEO runtime.

## Cleanup when fully done (optional)

- `rm -rf ~/ov_bench` on the car; `pip uninstall onnx typing_extensions` (user) if desired.
- Reverse any group/clock changes if made.

## Local artifacts created this session

- `scripts/car_ov_bench.py` — the on-car IE benchmark (committed-worthy).
- This file.
- Host wheels cached at `/tmp/onnx_wheels/` (onnx 1.10.2 cp38 + typing_extensions).
- On car: `~/ov_bench/{agent_fp32.onnx, car_ov_bench.py, wheels/}`.
