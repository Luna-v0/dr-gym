#!/usr/bin/env python3
"""Throughput benchmark — parallel instances (threads) vs RTF (W-throughput / D4).

Answers two things a single training run cannot:

  * **fps is NOT the simulation clock.** SB3's ``time/fps`` is *agent env-steps
    per wall-second* (training throughput). The simulation real-time factor
    (RTF = sim-seconds / wall-second) is separate; they relate only through the
    control period (``fps ≈ control_rate_Hz × RTF``). This script measures
    BOTH: aggregate agent-steps/s across workers, and the **effective RTF**
    sampled directly from each container's ROS ``/clock``.
  * **threads vs RTF.** Sweeps ``(n_workers × rtf_override)`` and reports
    aggregate throughput + per-instance effective RTF, so we can pick the
    operating point (the maintainer saw 1@160×→~43fps vs 7@10×→~3fps; the sweet
    spot is between).

Host mode (default) launches detached sim containers, times them by wall clock,
and samples RTF. Container mode (``GYM_DR_IN_CONTAINER=1``) just trains the
time-capped benchmark config (the standard container entrypoint calls this).

    uv run --no-sync python scripts/throughput_benchmark.py             # default sweep
    uv run --no-sync python scripts/throughput_benchmark.py --workers 4 --rtf 40 --seconds 120
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

SECONDS_DEFAULT = 120
TRACK = "Spain_track"  # a non-physical track present in the sim image
IMAGE = os.getenv("IMAGE_TAG", "my-deepracer-project:gpu")


def _bench_experiment():
    """A tiny, time-capped single-track training used only to measure throughput."""
    from gym_dr import (
        ContinuousActionSpaceConfig, ExperimentConfig, Sb3Trainer,
        SequentialRotation, TrackingConfig, TrainingConfig,
        progress_and_speed, time_trial,
    )
    from gym_dr.networks import DeepRacerCNN

    seconds = int(os.getenv("BENCH_SECONDS", SECONDS_DEFAULT))
    return ExperimentConfig(
        name="throughput_bench",
        env_factory=time_trial,
        reward=progress_and_speed,
        trainer=Sb3Trainer(
            name="ppo", policy="MultiInputPolicy",
            kwargs={
                "n_steps": 256, "batch_size": 64, "learning_rate": 3e-4,
                "policy_kwargs": {
                    "share_features_extractor": False, "normalize_images": False,
                    "features_extractor_class": DeepRacerCNN,
                    "features_extractor_kwargs": {
                        "conv_layers": [[32, 8, 4], [64, 4, 2], [64, 3, 1]],
                        "features_dim": 256,
                    },
                    "net_arch": {"pi": [128], "vf": [128]},
                },
            },
            frame_stack=4, device=os.getenv("BENCH_DEVICE", "cuda"),
        ),
        action_space=ContinuousActionSpaceConfig(speed_low=1.0),
        world_strategy=SequentialRotation(names=[TRACK], chunk_steps=10_000_000, rotations=1),
        training=TrainingConfig(
            total_timesteps=10_000_000, max_train_seconds=seconds,
            eval_freq=10_000_000, checkpoint_freq=10_000_000,
        ),
        tracking=TrackingConfig(mlflow_experiment="throughput_bench"),
        use_gpu=(os.getenv("BENCH_DEVICE", "cuda") == "cuda"), seed=42,
    )


# ----------------------------- host orchestration ----------------------------- #

def _docker_run_detached(name, rtf, seconds, proj, *, image=IMAGE, use_gpu=True, device="cuda", sw_render=False):
    cmd = [
        "docker", "run", "-d", "--rm", "--name", name,
        "-v", f"{proj}:/workspace:rw",
        "-v", f"{proj}/artifacts:/workspace/artifacts",
        "-v", f"{proj}/mlruns:/workspace/mlruns",
    ]
    if use_gpu:
        cmd += ["--gpus", "all"]
    cmd += [
        "-e", "GYM_DR_IN_CONTAINER=1",
        "-e", f"CHUNK_NAME={name}",
        "-e", "EXPERIMENT_PATH=/workspace/scripts/throughput_benchmark.py",
        "-e", f"RTF_OVERRIDE={rtf}",
        "-e", f"WORLD_NAME={TRACK}",
        "-e", f"BENCH_SECONDS={seconds}",
        "-e", f"BENCH_DEVICE={device}",
    ]
    if sw_render:
        # Force Mesa software GL so Gazebo renders the camera on the CPU even
        # when the container can see the GPU (for CUDA inference).
        cmd += ["-e", "LIBGL_ALWAYS_SOFTWARE=1", "-e", "GALLIUM_DRIVER=llvmpipe"]
    cmd += [image]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def _measure_rtf(container: str, wall: float = 8.0):
    """Effective RTF = (sim_time2 - sim_time1) / wall, sampled from ROS /clock."""
    def _clock():
        try:
            out = subprocess.run(
                ["docker", "exec", container, "bash", "-lc",
                 "source /opt/ros/noetic/setup.bash 2>/dev/null; rostopic echo -n1 /clock 2>/dev/null"],
                capture_output=True, text=True, timeout=20).stdout
        except Exception:
            return None
        secs = nsecs = None
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("secs:"):
                secs = int(s.split(":")[1])
            elif s.startswith("nsecs:"):
                nsecs = int(s.split(":")[1])
        return (secs + nsecs / 1e9) if secs is not None else None

    t1 = _clock()
    if t1 is None:
        return None
    time.sleep(wall)
    t2 = _clock()
    if t2 is None:
        return None
    return round((t2 - t1) / wall, 2)


def _steps_per_sec(name: str, proj: str):
    p = Path(proj) / "artifacts" / name / "training_status.json"
    try:
        d = json.loads(p.read_text())
        ts = int(d.get("timesteps_completed", 0))
        el = int(d.get("elapsed_seconds", 0)) or 1
        return ts / el
    except Exception:
        return None


def _run_point(workers, rtf, seconds, proj, *, image=IMAGE, use_gpu=True, device="cuda", tag="", sw_render=False):
    pre = f"bench{('_' + tag) if tag else ''}_w{workers}_r{rtf}"
    pre = re.sub(r"[^a-zA-Z0-9_.-]", "-", pre)  # docker --name allows only [a-zA-Z0-9_.-]
    names = [f"{pre}_{i}" for i in range(workers)]
    for n in names:
        _docker_run_detached(n, rtf, seconds, proj, image=image, use_gpu=use_gpu, device=device, sw_render=sw_render)
    time.sleep(min(60, max(20, seconds // 2)))   # let Gazebo settle
    rtf_eff = _measure_rtf(names[0])
    deadline = time.time() + seconds + 300
    while time.time() < deadline:
        running = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name={pre}_"],
            capture_output=True, text=True).stdout.strip()
        if not running:
            break
        time.sleep(5)
    rates = [r for r in (_steps_per_sec(n, proj) for n in names) if r]
    per = round(sum(rates) / len(rates), 1) if rates else 0.0
    return {
        "workers": workers, "rtf_set": rtf, "rtf_effective": rtf_eff,
        "device": device, "use_gpu": use_gpu, "sw_render": sw_render,
        "per_worker_steps_s": per, "aggregate_steps_s": round(sum(rates), 1),
        "n_reported": len(rates),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--rtf", type=int)
    ap.add_argument("--seconds", type=int, default=SECONDS_DEFAULT)
    ap.add_argument("--device-sweep", action="store_true",
                    help="compare GPU-render+CUDA-NN vs GPU-render+CPU-NN vs pure-CPU (1 worker)")
    ap.add_argument("--gpu-image", default=IMAGE)
    ap.add_argument("--cpu-image", default="my-deepracer-project:cpu")
    args = ap.parse_args()
    proj = str(Path(os.getenv("PROJECT_DIR", _PROJECT_ROOT)).resolve())

    if args.device_sweep:
        rtf = args.rtf or 30
        # 2x2 of render-device x NN-device. (tag, image, use_gpu, device, sw_render)
        configs = [
            ("gpu-render_cuda-nn", args.gpu_image, True, "cuda", False),
            ("gpu-render_cpu-nn", args.gpu_image, True, "cpu", False),
            ("cpu-render_cuda-nn", args.gpu_image, True, "cuda", True),   # CPU sim + GPU inference
            ("cpu-render_cpu-nn", args.cpu_image, False, "cpu", False),
        ]
        results = []
        for tag, img, gpu, dev, sw in configs:
            print(f"[bench] device-sweep {tag} (image={img} gpu={gpu} device={dev} sw_render={sw}) ...", flush=True)
            res = _run_point(1, rtf, args.seconds, proj, image=img, use_gpu=gpu, device=dev, tag=tag, sw_render=sw)
            res["config"] = tag
            print(f"[bench] {res}", flush=True)
            results.append(res)
        out = Path(proj) / "artifacts" / "device_benchmark.json"
        out.write_text(json.dumps(results, indent=2) + "\n")
        print("\n=== device comparison (1 worker; steps/s = agent env-steps/sec; rtf_eff = sim/wall) ===")
        print(f"{'config':>20} {'rtf_eff':>8} {'steps_s':>8}")
        for r in results:
            print(f"{r['config']:>20} {str(r['rtf_effective']):>8} {r['per_worker_steps_s']:>8}")
        print(f"\nwrote {out}")
        return 0

    points = [(args.workers, args.rtf)] if (args.workers and args.rtf) else \
             [(1, 160), (1, 10), (4, 40), (7, 10)]
    results = []
    for w, r in points:
        print(f"[bench] workers={w} rtf={r} seconds={args.seconds} ...", flush=True)
        res = _run_point(w, r, args.seconds, proj)
        print(f"[bench] {res}", flush=True)
        results.append(res)

    out = Path(proj) / "artifacts" / "throughput_benchmark.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("\n=== throughput summary (steps/s = agent env-steps/sec; rtf_eff = sim/wall) ===")
    print(f"{'workers':>7} {'rtf_set':>7} {'rtf_eff':>8} {'per_wkr_sps':>12} {'agg_sps':>9}")
    for r in results:
        print(f"{r['workers']:>7} {r['rtf_set']:>7} {str(r['rtf_effective']):>8} "
              f"{r['per_worker_steps_s']:>12} {r['aggregate_steps_s']:>9}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    if os.getenv("GYM_DR_IN_CONTAINER"):
        from gym_dr import train
        train(_bench_experiment())
        sys.exit(0)
    raise SystemExit(main())
