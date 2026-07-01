"""Held-out perception capture — render frames for the val / test / variants / physical
splits that the training sweep never drives on.

The camera training sweep (``camera_cnn_dataset.py``) trains the policy ONLY on the
TRAIN split and records its frames. The held-out splits (val, test, the
transformation-robustness VARIANTS, and the physical sim2real tracks) must NOT be
trained on — so this script loads the finished policy and rolls it out on them with
**``learning_rate=0``** (frozen: no gradient, no training on held-out tracks) while the
perception recorder captures camera frames + ground-truth feature targets. Reuses the
proven N-car camera recorder path (``camera_cnn_dataset.build_experiment``), so frame/
target alignment, visual DR, and shard layout are identical. Each split lands in its own
``artifacts/perception_capture_<split>/perception_out`` for consolidation to
``/mnt/models/dr_perception/<split>/``.

    GYM_DR_DEEPRACER_ENV_SRC=.../deepracer_env \
      GYM_DR_CAPTURE_RESUME=/workspace/artifacts/camera_cnn_dataset/final_model.zip \
      GYM_DR_CAPTURE_SPLIT=val \
      uv run --no-sync python experiments/perception_capture_heldout.py

Splits (GYM_DR_CAPTURE_SPLIT): ``val`` / ``test`` / ``variants`` (from the by-track
split in camera_cnn_dataset) / ``physical`` (reinvent_base + Oval, sim2real). Override
the tracks with GYM_DR_CAPTURE_TRACKS (comma list). Frozen lr=0; N cars = cam.N_CARS.
"""
from __future__ import annotations

import dataclasses
import os
import sys

from gym_dr.app import train as _train                           # noqa: E402
try:                                                             # noqa: E402
    import experiments.camera_cnn_dataset as cam                 # run as a module
except ModuleNotFoundError:                                      # run as a script
    import camera_cnn_dataset as cam

SPLIT = os.getenv("GYM_DR_CAPTURE_SPLIT", "physical").strip()
NAME = f"perception_capture_{SPLIT}"
_SPLIT_TRACKS = {
    "val": cam.VAL_TRACKS,
    "test": cam.TEST_TRACKS,
    "variants": cam.VARIANT_TRACKS,
    "physical": ["reinvent_base", "Oval_track"],
}
_env_tracks = [t for t in os.getenv("GYM_DR_CAPTURE_TRACKS", "").split(",") if t]
TRACKS = _env_tracks or _SPLIT_TRACKS.get(SPLIT, [])
# Captures only collect frames (no DR-aggregation benefit from high n), and 8 distinct
# track-instance spawns occasionally hit a Gazebo spawn-service timeout for some track
# mixes. A lower capture car count = fewer spawns/chunk = reliable. Default 4; override
# with GYM_DR_CAPTURE_NCARS.
N_CARS = int(os.getenv("GYM_DR_CAPTURE_NCARS", "4"))
CAPTURE_STEPS = int(os.getenv("GYM_DR_CAPTURE_STEPS", "40000"))
PASSES = int(os.getenv("GYM_DR_CAPTURE_PASSES", "1"))
PERCEPTION_OUT = f"/workspace/artifacts/{NAME}/perception_out"
RESUME = os.getenv("GYM_DR_CAPTURE_RESUME") or None


def build_capture(group: tuple[str, ...], resume: str | None):
    """camera_cnn_dataset's experiment, FROZEN (lr=0) and capture-sized: no learning on
    the held-out tracks, eval disabled — just rollouts + recorded frames."""
    exp = cam.build_experiment(group, resume)
    trainer = dataclasses.replace(
        exp.trainer, kwargs={**exp.trainer.kwargs, "learning_rate": 0.0})
    training = dataclasses.replace(
        exp.training, total_timesteps=CAPTURE_STEPS, resume_from=resume,
        eval_freq=10 ** 12, early_stop_enabled=False)
    env = dataclasses.replace(
        exp.environment,
        curriculum=dataclasses.replace(exp.environment.curriculum, eval_worlds=[]))
    return dataclasses.replace(
        exp, name=NAME, environment=env, trainer=trainer, training=training)


def main() -> int:
    if not TRACKS:
        print(f"[capture] no tracks for split={SPLIT!r}; set GYM_DR_CAPTURE_SPLIT or "
              f"GYM_DR_CAPTURE_TRACKS. nothing to do.", flush=True)
        return 0
    if RESUME is None:
        print("[capture] WARNING: GYM_DR_CAPTURE_RESUME unset — rolling out a RANDOM "
              "policy. Set it to the trained camera model for a useful dataset.", flush=True)
    groups = cam._groups(TRACKS, N_CARS, PASSES)
    print(f"[capture] split={SPLIT}: {len(TRACKS)} tracks, {N_CARS} cars/chunk, "
          f"{len(groups)} chunk(s) x {CAPTURE_STEPS} steps, frozen lr=0, resume={RESUME}",
          flush=True)

    os.environ["GYM_DR_N_CARS"] = str(N_CARS)
    os.environ["GYM_DR_PERCEPTION_OUT"] = PERCEPTION_OUT
    os.environ["GYM_DR_VISUAL_DR"] = "1"
    if N_CARS > 2:
        os.environ["GYM_DR_ALLOW_CAMERA_NCARS"] = "1"

    for i, group in enumerate(groups):
        os.environ["GYM_DR_DEMO_WORLDS"] = ",".join(group)
        os.environ["GYM_DR_VISUAL_DR_SEED"] = str(7000 + i)
        print(f"\n[capture] === {SPLIT} chunk {i + 1}/{len(groups)}: {group} ===", flush=True)
        try:
            _train(build_capture(group, RESUME))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[capture] {SPLIT} chunk {i + 1} FAILED ({exc}); continuing", flush=True)
            continue
    print(f"[capture] split={SPLIT} complete.", flush=True)
    return 0


if __name__ == "__main__":
    if os.getenv("GYM_DR_IN_CONTAINER"):
        _group = tuple(w for w in (os.getenv("GYM_DR_DEMO_WORLDS") or "").split(",") if w)
        if not _group:
            _w = os.getenv("WORLD_NAME") or (TRACKS[0] if TRACKS else "reinvent_base")
            _group = (_w,) * N_CARS
        _train(build_capture(_group, resume=os.getenv("RESUME_FROM") or RESUME))
    else:
        sys.exit(main())
