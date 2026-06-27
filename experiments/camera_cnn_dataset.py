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
    existing_tracks, train,
)
from gym_dr.app import train as _train                           # noqa: E402  (host loop calls this)
from gym_dr.envs.dispatch import build_env                       # noqa: E402
from gym_dr.networks import DeepRacerCNN                         # noqa: E402

NAME = "camera_cnn_dataset"
SMOKE = os.getenv("GYM_DR_CAM_SMOKE") == "1"
CHUNK_STEPS = 2_000 if SMOKE else 60_000
PASSES = 1 if SMOKE else 2                          # times to cycle all train pairs
# Recorder output: a CONTAINER path under the mounted /workspace/artifacts (lands on
# host artifacts/NAME/perception_out); scripts/perception_offload.py moves shards to
# /mnt/models then the Pi. Set here, forwarded into the container by app.py.
PERCEPTION_OUT = f"/workspace/artifacts/{NAME}/perception_out"

# ---- track split: 100% usage, physical car track held out ---------------------
_RESERVED = re.compile(r"(reinvent|reInvent|Oval)", re.IGNORECASE)  # physical / sim2real
_VARIANT = re.compile(r"_(cw|ccw|mirrored)$")
# Held-out eval set (drives the in-container eval each chunk): the REAL physical
# track first, plus a few diverse held-outs. Small => eval stays fast.
EVAL_WORLDS = ["reinvent_base", "Oval_track", "jyllandsringen_pro", "penbay_pro"]


def _train_pool() -> list[str]:
    seen, pool = set(), []
    for t in existing_tracks():
        base = _VARIANT.sub("", t)
        if base in seen or _RESERVED.search(base) or base in EVAL_WORLDS:
            continue
        seen.add(base)
        pool.append(base)
    return pool


def _pairs(tracks: list[str], passes: int, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for p in range(passes):
        shuf = tracks[:]
        rng.shuffle(shuf)
        if len(shuf) % 2:
            shuf.append(rng.choice(tracks))     # odd -> pad with a repeat
        out += [(shuf[i], shuf[i + 1]) for i in range(0, len(shuf), 2)]
    return out


def build_experiment(pair: tuple[str, str], resume: str | None) -> ExperimentConfig:
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
        # first_world = pair[0] => the container's WORLD_NAME; the 2 cars' actual
        # tracks come from GYM_DR_DEMO_WORLDS=pair (set in the loop). eval_worlds is
        # the held-out set the in-container eval scores each chunk.
        curriculum=OrderedSplit(train_worlds=[pair[0]], eval_worlds=EVAL_WORLDS,
                                chunk_steps=CHUNK_STEPS, rotations=1),
        domain_randomization=dr,
        # progress_per_step = (progress/steps)*100 + speed^2 — the speed^2 term makes
        # crawling unrewarding, so the policy must commit to pace (fixes the prior
        # slow-driving exploit under centerline_quadratic). eval stays clean_completion.
        n_cars=2, reward=progress_per_step, eval_reward=clean_completion,
    )
    return ExperimentConfig(
        name=NAME,
        environment=env,
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
            frame_stack=4, device="cuda"),
        training=TrainingConfig(
            total_timesteps=CHUNK_STEPS, checkpoint_freq=CHUNK_STEPS,
            checkpoint_keep_last=3, eval_freq=CHUNK_STEPS, n_eval_episodes=3,
            resume_from=resume, rtf_override=60, eval_path_plots=True,
            # Per-chunk mastery early-stop: stop a pair once held-out off-track rate
            # stays <=10% for 3 consecutive eval rounds (don't over-train solved pairs).
            early_stop_enabled=(not SMOKE), early_stop_max_offtrack_rate=0.10,
            early_stop_patience=3),
        tracking=TrackingConfig(mlflow_experiment=NAME),
        trace=TraceConfig(enabled=False),         # the perception recorder is the dataset
        seed=42, use_gpu=True,
    )


def main() -> int:
    pool = _train_pool()
    pairs = _pairs(pool, PASSES)
    if SMOKE:
        pairs = pairs[:1]
    print(f"[camera_cnn] train pool={len(pool)} tracks, held-out={EVAL_WORLDS}, "
          f"{len(pairs)} chunk(s) x {CHUNK_STEPS} steps (smoke={SMOKE})", flush=True)

    os.environ["GYM_DR_N_CARS"] = "2"
    os.environ["GYM_DR_PERCEPTION_OUT"] = PERCEPTION_OUT
    os.environ["GYM_DR_VISUAL_DR"] = "1"

    resume = None
    container_latest = f"/workspace/artifacts/{NAME}/latest_model.zip"
    for i, pair in enumerate(pairs):
        os.environ["GYM_DR_DEMO_WORLDS"] = ",".join(pair)
        os.environ["GYM_DR_VISUAL_DR_SEED"] = str(1000 + i)
        print(f"\n[camera_cnn] === chunk {i + 1}/{len(pairs)}: cars on {pair} ===", flush=True)
        try:
            _train(build_experiment(pair, resume))
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
        _pair = tuple((os.getenv("GYM_DR_DEMO_WORLDS") or "").split(",")[:2])
        if len(_pair) < 2:
            _w = os.getenv("WORLD_NAME") or _train_pool()[0]
            _pair = (_w, _w)
        _train(build_experiment(_pair, resume=os.getenv("RESUME_FROM") or None))
    else:
        sys.exit(main())
