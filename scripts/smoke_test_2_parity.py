"""Smoke Test 2 — framework parity (SB3 PyTorch -> ONNX -> IR), CPU, FP32.

Gate 2 of the ONNX-support plan. Proves the *real* dr-gym continuous-PPO policy survives
the full pipeline with its deterministic action (the diagonal-Gaussian mean) preserved.

Run with the dr-gym EXPORT venv (has stable-baselines3 + torch + onnx)::

    .venv/bin/python scripts/smoke_test_2_parity.py

It exports the SB3 ``.zip`` to ONNX via the existing ``gym_dr.export.sb3_to_onnx`` (the
``_DictExporter`` returns ``policy(obs, deterministic=True)[0]`` — exactly the action-mean
we treat as the parity target). onnxruntime and OpenVINO IR run in the OV venvs, fed the
identical fixed observation via ``.npz``.

Two pass levels:

  (i)  raw action-mean closeness — SB3 forward(deterministic) vs onnxruntime vs OpenVINO
       IR (modern + legacy), atol ~1e-4. The meaningful numeric gate.
  (ii) post-processed action — SB3 ``predict(deterministic=True)`` (clips to the action
       Box) vs the onnxruntime mean run through the same clip. Confirms the engineering-
       units contract.

Also reports the action-units reconciliation: dr-gym's action Box is in ENGINEERING units
(degrees, m/s), NOT [-1,1] — so on-car deployment must rescale to ServoCtrlMsg's [-1,1]
(throttle [0,1]). This is the plan's flagged silent-failure surface.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = REPO / "artifacts/hpo_trial_0/final_model.zip"
DEFAULT_MODERN_PY = REPO / ".venv-ov-modern/bin/python"
DEFAULT_LEGACY_PY = REPO / ".venv-ov-legacy/bin/python"
IR_RUNNER = REPO / "scripts/_ir_runner.py"
ORT_RUNNER = REPO / "scripts/_ort_runner.py"

# Actions are in ENGINEERING units (steering up to +-30 deg), so parity must be
# magnitude-aware: a pure absolute threshold is unrealistically tight on a 30-deg output.
# We use np.allclose(rtol, atol): rtol dominates for large actions, atol covers near-zero
# ones. With OpenVINO forced to FP32 (run_ir force_fp32=True; otherwise it runs bf16 and
# the action is off by ~1e-2 — see Smoke Test 1), all runtimes agree to ~1e-4 over real
# sim frames, so a single tight envelope works for every backend.
TOL = {
    "onnxruntime":     dict(rtol=1e-4, atol=1e-4),
    "openvino-modern": dict(rtol=1e-4, atol=1e-4),
    "openvino-legacy": dict(rtol=1e-4, atol=1e-4),
}


def _run(cmd):
    print("[run]", " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess failed:\n{proc.stdout}\n{proc.stderr}")
    return proc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--obs-npz", type=Path, default=None,
                    help="real observations from collect_sim_obs.py (key -> (N,C,H,W) uint8); "
                         "default: a single random frame")
    ap.add_argument("--modern-python", type=Path, default=DEFAULT_MODERN_PY)
    ap.add_argument("--legacy-python", type=Path, default=DEFAULT_LEGACY_PY)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/smoke2"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args(argv)

    if not args.zip.exists():
        print(f"ERROR: model zip not found: {args.zip}", file=sys.stderr)
        return 2
    tmp = args.workdir
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {tmp}")

    import torch
    from gym_dr.export import load_sb3_zip, sb3_to_onnx

    # 1. Export the policy to ONNX (reuses the existing exporter).
    onnx_path = tmp / "agent.onnx"
    sb3_to_onnx(args.zip, onnx_path, opset_version=args.opset)
    print(f"[export] {onnx_path}")

    # 2. Load model; confirm the parity-relevant config.
    model = load_sb3_zip(args.zip)
    policy = model.policy
    policy.eval()
    obs_space = policy.observation_space
    act_space = policy.action_space
    (key,) = list(obs_space.spaces.keys())
    obs_shape = obs_space.spaces[key].shape  # NCHW, e.g. (3,120,160)
    print(f"[cfg] obs '{key}' {obs_shape} uint8 | act {act_space} "
          f"| squash={policy.squash_output} use_sde={getattr(policy,'use_sde',False)} "
          f"| dist={type(policy.action_dist).__name__}")
    assert not policy.squash_output and not getattr(policy, "use_sde", False), \
        "parity target assumes no tanh-squash and no SDE"

    # 3. Fixed observation batch: raw 0-255 uint8, NCHW. No /255 (normalize_images=False).
    if args.obs_npz:
        loaded = np.load(args.obs_npz)
        obs = np.asarray(loaded[key] if key in loaded.files else loaded[loaded.files[0]])
        if obs.ndim == len(obs_shape):  # single frame -> add batch dim
            obs = obs[None]
        obs = obs.astype(np.uint8)
        print(f"[obs] real frames from {args.obs_npz}: {obs.shape}")
    else:
        rng = np.random.default_rng(args.seed)
        obs = rng.integers(0, 256, size=(1, *obs_shape), dtype=np.uint8)
        print(f"[obs] random frame: {obs.shape} (degenerate stimulus; pass --obs-npz for real)")
    B = obs.shape[0]
    feeds = tmp / "feeds.npz"
    np.savez(feeds, **{key: obs})
    static_shape = [B, *obs_shape]

    # 4. SB3 references (same interpreter), over the whole batch.
    obs_t = {key: torch.as_tensor(obs)}
    with torch.no_grad():
        a2_mean, _, _ = policy(obs_t, deterministic=True)  # the action-mean (parity target)
    a2_mean = a2_mean.cpu().numpy()
    a1_predict, _ = model.predict({key: obs}, deterministic=True)  # clipped/post-processed
    print(f"\n[sb3] raw mean   a2[0] = {a2_mean[0]}")
    print(f"[sb3] predict    a1[0] = {np.asarray(a1_predict)[0]}")
    if B > 1:  # show the stimulus is non-degenerate
        print(f"[sb3] action variety over {B} frames: "
              f"steering[min={a2_mean[:,0].min():.2f} max={a2_mean[:,0].max():.2f} "
              f"std={a2_mean[:,0].std():.3f}]  "
              f"speed[min={a2_mean[:,1].min():.2f} max={a2_mean[:,1].max():.2f} "
              f"std={a2_mean[:,1].std():.3f}]")

    results = {"sb3_mean": a2_mean}

    # 5. onnxruntime (modern venv).
    ort_out = tmp / "ort.npz"
    _run([args.modern_python, ORT_RUNNER, "--onnx", onnx_path, "--feeds", feeds, "--out", ort_out])
    results["onnxruntime"] = np.load(ort_out)["action"]

    # 6. OpenVINO IR — modern.
    irm = tmp / "ir_modern.npz"
    _run([args.modern_python, IR_RUNNER, "--onnx", onnx_path, "--feeds", feeds,
          "--out", irm, "--backend", "modern", "--input-shape", ",".join(map(str, static_shape))])
    d = np.load(irm)
    results["openvino-modern"] = d[d.files[0]]

    # 7. OpenVINO IR — legacy.
    have_legacy = args.legacy_python.exists()
    if have_legacy:
        irl = tmp / "ir_legacy.npz"
        _run([args.legacy_python, IR_RUNNER, "--onnx", onnx_path, "--feeds", feeds,
              "--out", irl, "--backend", "legacy", "--input-shape", ",".join(map(str, static_shape))])
        d = np.load(irl)
        results["openvino-legacy"] = d[d.files[0]]
    else:
        print(f"WARN: legacy python not found at {args.legacy_python}; skipping legacy IR")

    # ---------------- Level (i): raw action-mean parity ---------------- #
    print(f"\n=== level (i): raw action-mean parity (vs sb3_mean), worst over {B} frame(s) ===")
    ref = results["sb3_mean"]
    lvl1_ok = True
    for n, v in results.items():
        if n == "sb3_mean":
            print(f"  {n:<18} action[0]={v[0]}  (reference)")
            continue
        v = np.asarray(v).reshape(ref.shape)
        d = float(np.max(np.abs(ref - v)))
        rel = float(np.max(np.abs(ref - v) / (np.abs(ref) + 1e-6)))
        rtol, atol = TOL[n]["rtol"], TOL[n]["atol"]
        passed = bool(np.allclose(ref, v, rtol=rtol, atol=atol))
        lvl1_ok = lvl1_ok and passed
        print(f"  {n:<18} action[0]={v[0]}  max|Δ|={d:.2e}  max-rel={rel:.2e}  "
              f"{'PASS' if passed else 'FAIL'} (rtol={rtol:.0e} atol={atol:.0e})")

    # ---------------- Level (ii): post-processed action ---------------- #
    print("\n=== level (ii): post-processed action (clip to action Box) ===")
    low, high = act_space.low, act_space.high
    ort_clipped = np.clip(results["onnxruntime"].reshape(ref.shape), low, high)
    a1 = np.asarray(a1_predict).reshape(ref.shape)
    lvl2_diff = float(np.max(np.abs(ort_clipped - a1)))
    lvl2_ok = bool(np.allclose(ort_clipped, a1, rtol=1e-4, atol=1e-3))
    print(f"  sb3.predict      action[0] = {a1[0]}")
    print(f"  onnxruntime+clip action[0] = {ort_clipped[0]}")
    print(f"  max|Δ| = {lvl2_diff:.2e}  {'PASS' if lvl2_ok else 'FAIL'} (rtol=1e-04 atol=1e-03)")

    # ---------------- Action-units reconciliation ---------------- #
    print("\n=== action-units reconciliation (silent-failure surface) ===")
    print(f"  action Box low/high = {low} / {high}  -> ENGINEERING units (deg, m/s), NOT [-1,1]")
    print("  on-car ServoCtrlMsg expects angle in [-1,1], throttle in [0,1]; deployment")
    print("  must map: angle = steering_deg / 30.0 ; throttle = (speed - low) / (high - low).")
    steer_deg, speed_ms = float(a1[0, 0]), float(a1[0, 1])
    norm_angle = steer_deg / max(abs(low[0]), abs(high[0]))
    norm_throttle = (speed_ms - low[1]) / (high[1] - low[1])
    print(f"  example: predict[0]={a1[0]} -> servo angle={norm_angle:+.3f}, throttle={norm_throttle:.3f}")

    ok = lvl1_ok and lvl2_ok
    print("\n=== GATE 2: %s ===" % ("PASS" if ok else "FAIL"))
    print(f"    level(i) raw-mean={lvl1_ok}  level(ii) post-processed={lvl2_ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
