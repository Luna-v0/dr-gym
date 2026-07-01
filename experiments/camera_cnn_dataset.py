"""Camera-CNN PPO over (nearly) all tracks with heavy DR — and the perception
dataset generator (W-perception / Phase-1).

TWO purposes (maintainer):
  1. Train a "classic" vision policy: MultiInputPolicy + DeepRacerCNN on the
     grayscale 4-frame stack, TWO cars in one Gazebo each on its own track
     (n_cars=2 = the Gazebo-Classic camera ceiling), heavy domain randomization so
     it generalizes (sim-side track/background recolor + image brightness/contrast/
     gamma/Gaussian + actuator/drag/friction + random start/direction).
  2. Generate the camera->features dataset (THE priority): the recorder
     (gym_dr.perception_recorder, GYM_DR_PERCEPTION_OUT) captures contiguous
     per-episode frames + ground-truth ACTOR_FEATURES targets from BOTH the
     training rollouts AND the held-out evaluation — temporally consistent for the
     4-frame stack — for the CNN->feature distillation.

Track usage = 100% (maintainer): every track is train OR held-out. The real
physical car track ``reinvent_base`` (+ the reInvent family + Oval) is RESERVED for
held-out sim2real eval and never trained. The 2 cars rotate through all training
tracks in PAIRS across chunks (camera multi-car can't hot-swap tracks in-process,
so each chunk is a fresh container on a new pair, resuming the model — robust for an
unattended multi-hour run). Per-chunk **mastery early-stop** ends a pair early once
it's driven cleanly, so easy pairs don't waste time (the maintainer's "better early
stopping"); the dataset accumulates across every chunk regardless.

    GYM_DR_DEEPRACER_ENV_SRC=/home/lunav0/Projects/deepracer-env/deepracer_env \
      uv run --no-sync python experiments/camera_cnn_dataset.py
    # smoke (1 short chunk, validate the pipeline): GYM_DR_CAM_SMOKE=1
"""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

from gym_dr import (                                              # noqa: E402
    ADR, ContinuousActionSpaceConfig, CameraObs, EnvironmentConfig,
    ExperimentConfig, OrderedSplit, Range, Sb3Trainer, TraceConfig,
    TrackingConfig, TrainingConfig, clean_completion, progress_per_step,
    existing_tracks, Study, OfftrackRate,
)
from gym_dr.app import train as _train                           # noqa: E402  (host loop calls this)
from gym_dr.envs.dispatch import build_env                       # noqa: E402
from gym_dr.networks import DeepRacerCNN                         # noqa: E402

NAME = "camera_cnn_dataset"
SMOKE = os.getenv("GYM_DR_CAM_SMOKE") == "1"
# GYM_DR_CAM_CHUNK_STEPS overrides the per-chunk step budget — lets a smoke run
# force a *tiny* chunk so the container reaches teardown fast (e.g. exercising the
# multi-car camera exit path) without waiting on a full 2k/60k rollout.
CHUNK_STEPS = int(os.getenv("GYM_DR_CAM_CHUNK_STEPS", "2000" if SMOKE else "60000"))
PASSES = 1 if SMOKE else 2                          # times to cycle all train tracks
# Cars per chunk = DISTINCT tracks aggregated per rollout. More cars = stronger DR for
# generalization (maintainer: prefer instances over raw throughput). Camera renders
# serialize on one OGRE thread so RTF drops with N (benchmarked: n4=0.99x, ~0.5x at n8)
# — acceptable here. Bounded by the launch's racecar_0..7 (=8) and the ~12-13 Gazebo
# separate-track-instance spawn cap. Override with GYM_DR_CAM_NCARS.
N_CARS = int(os.getenv("GYM_DR_CAM_NCARS", "2" if SMOKE else "8"))
# Set at MODULE level (not in main()) so the spawned container — which re-imports this
# file and runs ONLY the GYM_DR_IN_CONTAINER branch, never main() — also clears the
# dr-gym camera n>2 guard. (>2 needs the generalised racecar_2..N launch, mounted via
# GYM_DR_DEEPRACER_ENV_SRC.) app.py also forwards it, belt-and-suspenders.
if N_CARS > 2:
    os.environ["GYM_DR_ALLOW_CAMERA_NCARS"] = "1"
# Recorder output: a CONTAINER path under the mounted /workspace/artifacts (lands on
# host artifacts/NAME/perception_out); scripts/perception_offload.py moves shards to
# /mnt/models then the Pi. Set here, forwarded into the container by app.py.
PERCEPTION_OUT = f"/workspace/artifacts/{NAME}/perception_out"

# ---- proper by-TRACK train/val/test split (no leakage) ------------------------
_RESERVED = re.compile(r"(reinvent|reInvent|Oval)", re.IGNORECASE)  # physical / sim2real
_VARIANT = re.compile(r"_(cw|ccw|mirrored)$")


def _split_tracks(seed: int = 42):
    """Deterministic by-TRACK split. The UNIQUE base tracks (no variant suffix) are
    split train/val/test 70/15/15 — a base lives in exactly ONE split, so val/test
    measure true track-generalization with NO leakage and NO duplication. The
    ``_cw/_ccw/_mirrored`` VARIANTS are NOT trained on; they form a separate held-out
    TRANSFORMATION-robustness set (same/known geometry, reversed/mirrored). The
    reserved physical family (reinvent/Oval) is excluded entirely (sim2real test,
    captured separately)."""
    seen, bases, variants = set(), [], []
    for t in existing_tracks():
        base = _VARIANT.sub("", t)
        if _RESERVED.search(base):
            continue
        if _VARIANT.search(t):
            variants.append(t)
        elif base not in seen:
            seen.add(base)
            bases.append(t)
    rng = random.Random(seed)
    rng.shuffle(bases)
    n = len(bases)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    return (sorted(bases[:n_tr]), sorted(bases[n_tr:n_tr + n_va]),
            sorted(bases[n_tr + n_va:]), sorted(variants))


TRAIN_TRACKS, VAL_TRACKS, TEST_TRACKS, VARIANT_TRACKS = _split_tracks()
# Nominal held-out for the curriculum config (multi-car can't set_world, so the in-loop
# eval runs on the CURRENT train group — honest in-distribution; the real val/test/
# variants frames come from the held-out capture pass, perception_capture_heldout.py).
EVAL_WORLDS = VAL_TRACKS[:4]


def _train_pool() -> list[str]:
    """Tracks the policy TRAINS on = the train split only (unique bases, no variants)."""
    return TRAIN_TRACKS


def _groups(tracks: list[str], n: int, passes: int, seed: int = 42) -> list[tuple[str, ...]]:
    """Chunk the (passes x shuffled) tracks into groups of ``n`` — one car per track
    per chunk, so every rollout aggregates ``n`` DISTINCT track+DR variations. Passes
    are concatenated THEN chunked, so only the single final group is padded (<= n-1
    repeats total) instead of once per pass — minimal duplication."""
    rng = random.Random(seed)
    seq: list[str] = []
    for _ in range(passes):
        shuf = tracks[:]
        rng.shuffle(shuf)
        seq += shuf
    while len(seq) % n:                          # pad ONLY the final partial group
        seq.append(rng.choice(tracks))
    out = [tuple(seq[i:i + n]) for i in range(0, len(seq), n)]
    return out


def build_experiment(group: tuple[str, ...], resume: str | None) -> ExperimentConfig:
    # Heavy DR (ADR): noise ceilings grow with held-out clean-completion. Visual DR
    # is image-space here (brightness/contrast/gamma/gaussian) + sim-side track/bg
    # recolor via GYM_DR_VISUAL_DR (deepracer_env VisualRandomizer).
    # STRONG action noise so the EXECUTED steering/speed get dragged up & down — this
    # is what spreads the dataset across the speed range (the speed_mps targets span
    # [1,4]) and adds steering variety, instead of the data bunching at one operating
    # point. (multi_car applies the Range's high as the actuator-noise std.)
    #   speed_noise 1.0 m/s on a [1,4] range; steering_noise 6 deg on +-30.
    # Plus the camera input noise (gaussian/brightness/contrast/gamma).
    dr = ADR(
        steering_noise=Range(0.0, 6.0), speed_noise=Range(0.0, 1.0),  # per-step jitter
        # per-EPISODE constant lean (miscalibrated actuator): "0 steering -> up to ±20
        # deg", motor offset up to ±1 m/s — the policy must detect the drift + correct.
        steering_bias=20.0, speed_bias=1.0,
        obs_gaussian=Range(0.0, 18.0), obs_brightness=Range(0.0, 0.4),
        obs_contrast=Range(0.0, 0.5), obs_gamma=Range(0.0, 0.5),
        # per-EPISODE speed regime: drag 0.2..1.0 makes whole episodes drive slow→fast,
        # so the dataset's executed-speed distribution spans the full range (with
        # speed_low=0.2 the slow episodes reach ~0.2 m/s, fast ones ~peak).
        drag=Range(0.2, 1.0), friction=Range(0.7, 1.6),
        random_start=True, random_direction=True,
        step=0.1, promote=0.7, demote=0.3, seed=42,
    )
    env = EnvironmentConfig(
        observation=CameraObs(),                          # vision (grayscale 4-stack)
        action_space=ContinuousActionSpaceConfig(
            # speed_low=0.2 (was 1.0) so drag/bias-slowed episodes actually execute
            # really slow — the dataset's speed_mps targets then span ~0.2..4.0 m/s.
            steering_low=-30.0, steering_high=30.0, speed_low=0.2, speed_high=4.0,
            normalize_actions=True),
        # first_world = group[0] => the container's WORLD_NAME; the N cars' actual
        # tracks come from GYM_DR_DEMO_WORLDS=group (set in the loop). eval_worlds is
        # the held-out set (multi-car eval runs on the current group — honest
        # in-distribution; true held-out is perception_capture_heldout.py).
        curriculum=OrderedSplit(train_worlds=[group[0]], eval_worlds=EVAL_WORLDS,
                                chunk_steps=CHUNK_STEPS, rotations=1),
        domain_randomization=dr,
        # progress_per_step = (progress/steps)*100 + speed^2 — the speed^2 term makes
        # crawling unrewarding, so the policy must commit to pace (fixes the prior
        # slow-driving exploit under centerline_quadratic). eval stays clean_completion.
        n_cars=len(group), reward=progress_per_step, eval_reward=clean_completion,
    )
    return ExperimentConfig.from_environment(env,
        name=NAME,
        env_factory=build_env,
        trainer=Sb3Trainer(
            name="ppo", policy="MultiInputPolicy",
            kwargs={
                "n_steps": 1024, "batch_size": 256, "learning_rate": 3.0e-4,
                "ent_coef": 0.01, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "n_epochs": 10, "target_kl": 0.08,
                "policy_kwargs": {
                    "share_features_extractor": False, "normalize_images": False,
                    "features_extractor_class": DeepRacerCNN,
                    "features_extractor_kwargs": {
                        "conv_layers": [[32, 8, 4], [64, 4, 2], [64, 3, 1]],
                        "features_dim": 512},
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]}},
            },
            frame_stack=4, device=os.getenv("GYM_DR_DEVICE", "cuda")),
        training=TrainingConfig(
            total_timesteps=CHUNK_STEPS, checkpoint_freq=CHUNK_STEPS,
            checkpoint_keep_last=3,
            # eval_freq must give >= early_stop_patience eval rounds PER chunk or the
            # mastery early-stop can never trigger (each chunk is a fresh container that
            # resets the streak). Trainer divides this by n_cars internally, so ~4
            # rounds/chunk here. (multi-car eval runs on the CURRENT pair — honest
            # in-distribution; true held-out is the separate single-car capture pass.)
            eval_freq=max(1, CHUNK_STEPS // 4), n_eval_episodes=3,
            resume_from=resume, rtf_override=60, eval_path_plots=True,
            # Per-chunk mastery early-stop: stop a pair once its off-track rate stays
            # <=10% for 3 consecutive eval rounds (don't over-train solved pairs).
            early_stop=(OfftrackRate(max_offtrack_rate=0.10, patience=3) if not SMOKE else None)),
        tracking=TrackingConfig(mlflow_experiment=NAME),
        trace=TraceConfig(enabled=False),         # the perception recorder is the dataset
        seed=42, use_gpu=True,
    )


def main() -> int:
    pool = _train_pool()
    groups = _groups(pool, N_CARS, PASSES)
    if SMOKE:
        groups = groups[:1]
    print(f"[camera_cnn] TRAIN split={len(pool)} tracks (no variants), val={len(VAL_TRACKS)} "
          f"test={len(TEST_TRACKS)} variants={len(VARIANT_TRACKS)} held-out, "
          f"{N_CARS} cars/chunk, {len(groups)} chunk(s) x {CHUNK_STEPS} steps (smoke={SMOKE})",
          flush=True)

    os.environ["GYM_DR_N_CARS"] = str(N_CARS)
    os.environ["GYM_DR_PERCEPTION_OUT"] = PERCEPTION_OUT
    os.environ["GYM_DR_VISUAL_DR"] = "1"   # GYM_DR_ALLOW_CAMERA_NCARS set at module level

    resume = None
    container_latest = f"/workspace/artifacts/{NAME}/latest_model.zip"
    for i, group in enumerate(groups):
        os.environ["GYM_DR_DEMO_WORLDS"] = ",".join(group)
        os.environ["GYM_DR_VISUAL_DR_SEED"] = str(1000 + i)
        print(f"\n[camera_cnn] === chunk {i + 1}/{len(groups)}: {len(group)} cars on {group} ===",
              flush=True)
        try:
            _train(build_experiment(group, resume))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — one bad pair shouldn't kill the sweep
            print(f"[camera_cnn] chunk {i + 1} FAILED ({exc}); continuing", flush=True)
            continue
        # After the first successful chunk, resume the accumulated model each time.
        if Path(f"artifacts/{NAME}/latest_model.zip").exists():
            resume = container_latest
    print("[camera_cnn] sweep complete.", flush=True)
    return 0


if __name__ == "__main__":
    # Host/container split (mirrors every other experiment's `train(experiment)`):
    # on the HOST, main() runs the per-pair spawn loop; INSIDE a spawned container
    # (GYM_DR_IN_CONTAINER), this same file is re-executed — it must train ONE chunk
    # on the env-provided pair, NOT re-run the host loop (which would loop in-container
    # and re-spawn the 2nd car's track -> "entity already exists").
    if os.getenv("GYM_DR_IN_CONTAINER"):
        _group = tuple(w for w in (os.getenv("GYM_DR_DEMO_WORLDS") or "").split(",") if w)
        if not _group:
            _w = os.getenv("WORLD_NAME") or _train_pool()[0]
            _group = (_w,) * N_CARS
        _train(build_experiment(_group, resume=os.getenv("RESUME_FROM") or None))
    else:
        sys.exit(main())
