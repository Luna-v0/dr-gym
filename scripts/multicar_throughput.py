#!/usr/bin/env python3
"""Multi-car throughput benchmark: cars-in-ONE-sim x {sim-render gpu/cpu} x
{policy gpu/cpu} x rtf_override -> aggregate agent-steps/s + effective RTF.

Unlike scripts/throughput_benchmark.py (which scales SEPARATE containers), this
sweeps N cars inside a SINGLE Gazebo via the multi-agent VecEnv (the new path),
to find the optimal cars-per-sim and device/rtf operating point on THIS machine.

steps/s = SB3 ``time/fps`` (aggregate env-steps/s across the N cars) parsed from
the container log; RTF = sim-sec/wall-sec sampled from the container's /clock.

Runs each config sequentially (one container at a time), headless. Results ->
artifacts/multicar_throughput_camerafree.json + printed table.
"""
from __future__ import annotations
import json, os, re, subprocess, time
from pathlib import Path

ROOT = Path("/home/lunav0/Projects")
DRG = ROOT / "dr-gym"
DENV = ROOT / "deepracer-env"
IMAGE = "my-deepracer-project:gpu"
NAME = "gym-dr-mcbench"
STEPS = 2_000_000          # high cap; we tear down by time
WARMUP_S = 150             # max wait for first fps line
MEASURE_S = 45             # collect fps/RTF over this window after first reading

# config tuples: (tag, n_cars, camera, rtf, device, sw_render)
# Comprehensive grid (maintainer request): camera env 2x2 {render gpu/cpu} x
# {inference gpu/cpu} across n=2..8, plus an rtf_override sweep to find the stable
# optimum (too-high rtf at high n outruns the CPU -> service timeouts -> crash).
OUT_JSON = "multicar_grid.json"
CONFIGS = []
# --- camera env: 2x2 (render x inference) across car count ---
for _n in range(2, 9):                       # n = 2..8
    for _sw in (False, True):                # GPU render vs CPU (software) render
        for _dev in ("cuda", "cpu"):         # inference GPU vs CPU
            _r = "swR" if _sw else "gpuR"
            _d = "gpuNN" if _dev == "cuda" else "cpuNN"
            CONFIGS.append((f"cam_n{_n}_{_r}_{_d}", _n, 1, 20, _dev, _sw))
# --- rtf_override sweep (feature n4 cpu-NN, and camera n2 gpuR/gpuNN) ---
for _rtf in (5, 10, 40, 80):
    CONFIGS.append((f"feat_n4_rtf{_rtf}", 4, 0, _rtf, "cpu", False))
    CONFIGS.append((f"cam_n2_rtf{_rtf}", 2, 1, _rtf, "cuda", False))
# --- rtf stability probe at high n (does a lower rtf keep n=8 alive?) ---
for _rtf in (2, 5, 10):
    CONFIGS.append((f"feat_n8_rtf{_rtf}", 8, 0, _rtf, "cpu", False))


def _rm():
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)


def _launch(n, cam, rtf, device, sw):
    _rm()
    env = {
        "GYM_DR_IN_CONTAINER": "1", "GYM_DR_ROTATE": "1",
        "EXPERIMENT_PATH": "/workspace/experiments/multicar_demo.py",
        "GYM_DR_DEMO_NCARS": str(n), "GYM_DR_DEMO_CAMERA": str(cam),
        "GYM_DR_DEMO_SPACING": "50", "GYM_DR_DEMO_STEPS": str(STEPS),
        "GYM_DR_DEMO_WORLD": "Spain_track", "GYM_DR_DEMO_RTF": str(rtf),
        "GYM_DR_DEMO_DEVICE": device, "GYM_DR_DEMO_GUI": "0",
        "RTF_OVERRIDE": str(rtf), "SEED": "42", "GYM_DR_N_CARS": str(n),
        "ENABLE_GUI": "False", "WORLD_NAME": "Spain_track", "ROTATE_START_INDEX": "0",
        "CHUNK_NAME": "mcbench", "MLFLOW_RUN_GROUP": "mcbench",
        # Allow camera n>2 for the >2-camera-car render benchmark (the dr-gym guard
        # is for the un-generalised launch; with racecar_2..N launch blocks the cars
        # have real cameras). Forwarded from the host; default off keeps old runs safe.
        "GYM_DR_ALLOW_CAMERA_NCARS": os.getenv("GYM_DR_ALLOW_CAMERA_NCARS", "0"),
    }
    if sw:
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"; env["GALLIUM_DRIVER"] = "llvmpipe"
    cmd = ["docker", "run", "--rm", "-d", "--name", NAME,
           "-v", f"{DRG}:/workspace:rw",
           "-v", f"{DRG}/artifacts:/workspace/artifacts",
           "-v", f"{DENV}/deepracer_env:/usr/local/lib/python3.8/dist-packages/deepracer_env:ro",
           "-v", f"{DENV}/simulation/src/deepracer_simulation_environment/launch:/opt/simapp/deepracer_simulation_environment/share/deepracer_simulation_environment/launch:ro",
           "-v", f"{DENV}/simulation/urdf:/opt/simapp/deepracer_simulation_environment/share/deepracer_simulation_environment/urdf:ro"]
    cmd += ["--gpus", "all"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(IMAGE)
    subprocess.run(cmd, capture_output=True, check=True)


def _log():
    return subprocess.run(["docker", "logs", NAME], capture_output=True, text=True).stderr + \
           subprocess.run(["docker", "logs", NAME], capture_output=True, text=True).stdout


def _alive():
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    return NAME in r.stdout


def _fps(log):
    # SB3 logs the rate as "|    fps                  | 45 |"
    return [int(x) for x in re.findall(r"\bfps\s*\|\s*(\d+)", log)]


def _rtf():
    code = ('import rospy;from rosgraph_msgs.msg import Clock;'
            'rospy.init_node("b",anonymous=True,disable_signals=True);'
            'print(rospy.wait_for_message("/clock",Clock).clock.to_sec())')
    def clk():
        r = subprocess.run(["docker", "exec", NAME, "bash", "-lc",
                            f"source /opt/simapp/setup.bash 2>/dev/null; python3 -c '{code}'"],
                           capture_output=True, text=True, timeout=20)
        m = re.findall(r"[\d.]+", r.stdout)
        return float(m[-1]) if m else None
    try:
        t0 = clk(); w0 = time.time(); time.sleep(8); t1 = clk(); w1 = time.time()
        if t0 and t1: return round((t1 - t0) / (w1 - w0), 2)
    except Exception:
        return None
    return None


def run_one(tag, n, cam, rtf, device, sw):
    print(f"\n=== {tag}: n_cars={n} cam={cam} rtf={rtf} dev={device} sw={sw} ===", flush=True)
    try:
        _launch(n, cam, rtf, device, sw)
    except subprocess.CalledProcessError as e:
        return {"tag": tag, "error": "launch_failed", "stderr": e.stderr.decode()[-300:]}
    t0 = time.time(); first = None
    while time.time() - t0 < WARMUP_S:
        if not _alive():
            return {"tag": tag, "n_cars": n, "error": "died_during_warmup"}
        if _fps(_log()):
            first = time.time(); break
        time.sleep(10)
    if first is None:
        _rm(); return {"tag": tag, "n_cars": n, "error": "no_fps_in_warmup"}
    rtf_eff = _rtf()
    while time.time() - first < MEASURE_S and _alive():
        time.sleep(10)
    fps = _fps(_log())
    _rm()
    steps_s = int(sum(fps[-3:]) / len(fps[-3:])) if fps else None
    return {"tag": tag, "n_cars": n, "camera": cam, "rtf_set": rtf, "device": device,
            "sw_render": sw, "steps_per_s": steps_s, "rtf_effective": rtf_eff,
            "per_car_steps_s": round(steps_s / n, 1) if steps_s else None,
            "n_fps_samples": len(fps)}


def main():
    out = DRG / "artifacts" / OUT_JSON
    results = []
    for cfg in CONFIGS:
        r = run_one(*cfg)
        results.append(r)
        print("  ->", {k: r.get(k) for k in ("steps_per_s", "per_car_steps_s", "rtf_effective", "error")}, flush=True)
        out.write_text(json.dumps(results, indent=2))
    _rm()
    print("\n==== SUMMARY ====")
    print(f"{'tag':22} {'n':>2} {'steps/s':>8} {'/car':>7} {'RTF':>6}")
    for r in results:
        print(f"{r['tag']:22} {r.get('n_cars','?'):>2} {str(r.get('steps_per_s') or r.get('error','')):>8} "
              f"{str(r.get('per_car_steps_s') or ''):>7} {str(r.get('rtf_effective') or ''):>6}")
    print("results ->", out)


if __name__ == "__main__":
    main()
