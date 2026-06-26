"""Perception dataset recorder — camera frames paired with ground-truth feature
labels, captured live from the camera-CNN PPO rollout **and** its evaluation.

This is the data engine for the camera->features distillation (W-perception /
Phase-1): the CNN must learn ``g(camera) -> ACTOR_FEATURES``, so we need many
(grayscale frame stack, feature target) pairs across many tracks and heavy visual
domain randomization. Rather than a separate scripted collection, we tap the real
training run so the data distribution is exactly what the policy visits — and we
record the **eval** rollouts too (held-out tracks), which the maintainer wants in
the training set.

TEMPORAL CONSISTENCY: the CNN consumes a 4-frame stack, so frames must stay in
contiguous per-episode order (NO per-frame subsampling). We store each episode as
one shard holding the ordered single-frame sequence ``(T, H, W)`` uint8; any
4-stack the policy used is just four consecutive frames, reconstructable at train
time. One shard per (episode, car).

Shard layout (``np.savez_compressed``)::

    frames    uint8  (T, H, W)        ordered grayscale frames (the newest frame
                                      per step; stack = frames[t-3:t+1])
    targets   f32    (T, F)           actor feature labels per frame (ACTOR_FEATURES)
    diag      f32    (T, D)           diagnostics (progress, speed, offtrack, x, y, heading)
    features  str    (F,)             target column names
    diag_cols str    (D,)             diagnostic column names
    meta      str    ()              JSON: track, car, phase(train/eval), episode id,
                                      dr (drag/friction/noise) + visual_dr flag

Enable by setting ``GYM_DR_PERCEPTION_OUT`` to an output dir (bind-mounted to fast
storage, e.g. /mnt/models). Disabled (no-op) when unset, so other runs are
unaffected. A host-side monitor offloads finished shards to the archive (gdrive)
and keeps the capture disk from filling — this class only writes + drops shards
gracefully when space is critically low (never crashes training).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from gym_dr.perception import ACTOR_FEATURES, actor_targets

LOG = logging.getLogger(__name__)

_DIAG_COLS = ("progress", "speed_mps", "is_offtrack", "x", "y", "heading")
_MIN_FREE_GB = 3.0  # below this on the capture disk, DROP shards (host monitor frees space)


def _free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


class _CarBuffer:
    __slots__ = ("frames", "targets", "diag", "prev", "track", "phase", "ep_id", "meta")

    def __init__(self) -> None:
        self.reset(track="", phase="train", ep_id=-1, meta={})

    def reset(self, *, track: str, phase: str, ep_id: int, meta: dict) -> None:
        self.frames: list = []
        self.targets: list = []
        self.diag: list = []
        self.prev: Optional[dict] = None
        self.track = track
        self.phase = phase
        self.ep_id = ep_id
        self.meta = meta


class PerceptionRecorder:
    """Capture per-episode (frame, feature-target) sequences for N cars.

    Designed to live inside ``MultiCarVecEnv`` (and reachable in single-car eval):
    call :meth:`start_episode` on each reset, :meth:`record` every step with the
    grayscale frame + that car's ``reward_params``, and :meth:`flush_episode` when
    a car's episode ends. Thread-free, single process; all writes are guarded so a
    full/slow disk never kills the run.
    """

    def __init__(self, out_dir: str | Path, n_cars: int,
                 features: Sequence[str] = ACTOR_FEATURES,
                 phase: str = "train") -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.n_cars = int(n_cars)
        self.features = tuple(features)
        self.phase = phase                 # toggled to "eval" by the eval callback
        self._buf = [_CarBuffer() for _ in range(self.n_cars)]
        self._ep_counter = 0
        self.written = 0
        self.dropped = 0
        self.frames_written = 0

    # -- lifecycle ----------------------------------------------------------- #
    def set_phase(self, phase: str) -> None:
        self.phase = "eval" if str(phase).startswith("eval") else "train"

    def start_episode(self, car: int, *, track: str, dr_meta: Optional[dict] = None) -> None:
        if not (0 <= car < self.n_cars):
            return
        self._ep_counter += 1
        meta = dict(dr_meta or {})
        meta.update({"track": track, "car": car, "phase": self.phase,
                     "episode": self._ep_counter, "ts": time.time()})
        self._buf[car].reset(track=track, phase=self.phase,
                             ep_id=self._ep_counter, meta=meta)

    def record(self, car: int, frame_gray: np.ndarray, params: dict) -> None:
        """Append one step's frame + derived feature target for ``car``.

        ``frame_gray`` is the newest grayscale frame ``(H, W)`` uint8 (the policy's
        stack is the last 4 of these). ``params`` is the car's ``reward_params``."""
        if not (0 <= car < self.n_cars):
            return
        b = self._buf[car]
        try:
            frame = np.asarray(frame_gray, dtype=np.uint8)
            if frame.ndim == 3:          # (H, W, 1) -> (H, W)
                frame = frame.squeeze(-1)
            tgt = actor_targets(params, b.prev).astype(np.float32)
            diag = np.array([
                float(params.get("progress", 0.0)),
                float(params.get("speed", 0.0)),
                1.0 if params.get("is_offtrack", False) else 0.0,
                float(params.get("x", 0.0)),
                float(params.get("y", 0.0)),
                float(params.get("heading", 0.0)),
            ], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 — a bad frame must not kill training
            LOG.debug("recorder.record skipped: %s", exc)
            return
        b.frames.append(frame)
        b.targets.append(tgt)
        b.diag.append(diag)
        if params:
            b.prev = dict(params)

    def flush_episode(self, car: int) -> None:
        """Write ``car``'s buffered episode as one compressed shard (if non-empty)."""
        if not (0 <= car < self.n_cars):
            return
        b = self._buf[car]
        n = len(b.frames)
        if n == 0:
            return
        if _free_gb(self.out) < _MIN_FREE_GB:   # host monitor hasn't freed space yet
            self.dropped += 1
            b.reset(track=b.track, phase=self.phase, ep_id=b.ep_id, meta=b.meta)
            if self.dropped % 20 == 1:
                LOG.warning("perception recorder DROPPING shards: capture disk < %.0f GB free",
                            _MIN_FREE_GB)
            return
        # subdir per phase/track keeps the archive browsable; .tmp -> rename so the
        # host monitor never offloads a half-written file.
        d = self.out / b.phase / _safe(b.track)
        d.mkdir(parents=True, exist_ok=True)
        # The container runs as root, so make the shard dirs world-writable — the
        # host-side offload monitor (a non-root user) must be able to MOVE/DELETE
        # finished shards out of here (delete needs write on the parent DIR).
        for _d in (self.out, self.out / b.phase, d):
            try:
                os.chmod(_d, 0o777)
            except OSError:
                pass
        stem = f"ep{b.ep_id:07d}_car{car}"
        # NB: np.savez_compressed APPENDS ".npz" unless the name already ends in it,
        # so the temp name must end in ".npz" (".tmp.npz") — else it writes
        # "<stem>.npz.tmp.npz" and the rename below misses. The offload monitor skips
        # ".tmp.npz" so it never grabs a half-written shard.
        tmp, final = d / (stem + ".tmp.npz"), d / (stem + ".npz")
        try:
            np.savez_compressed(
                tmp,
                frames=np.stack(b.frames, axis=0),
                targets=np.stack(b.targets, axis=0),
                diag=np.stack(b.diag, axis=0),
                features=np.array(self.features),
                diag_cols=np.array(_DIAG_COLS),
                meta=np.array(json.dumps(b.meta)),
            )
            os.replace(tmp, final)
            try:
                os.chmod(final, 0o666)   # world-readable/deletable for the offloader
            except OSError:
                pass
            self.written += 1
            self.frames_written += n
        except Exception as exc:  # noqa: BLE001 — never crash training over a write
            self.dropped += 1
            LOG.warning("perception recorder write failed (%s); dropped %d frames", exc, n)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        b.reset(track=b.track, phase=self.phase, ep_id=b.ep_id, meta=b.meta)

    def flush_all(self) -> None:
        for c in range(self.n_cars):
            self.flush_episode(c)

    def stats(self) -> dict:
        return {"shards": self.written, "frames": self.frames_written,
                "dropped": self.dropped, "free_gb": round(_free_gb(self.out), 1)}


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or "unknown"))


def recorder_from_env(n_cars: int, features: Sequence[str] = ACTOR_FEATURES
                      ) -> Optional[PerceptionRecorder]:
    """Build a recorder if ``GYM_DR_PERCEPTION_OUT`` is set, else ``None`` (no-op)."""
    out = os.getenv("GYM_DR_PERCEPTION_OUT")
    if not out:
        return None
    try:
        rec = PerceptionRecorder(out, n_cars, features=features)
        LOG.info("perception recorder ON -> %s (%d cars)", out, n_cars)
        return rec
    except Exception as exc:  # noqa: BLE001
        LOG.warning("perception recorder disabled (init failed): %s", exc)
        return None
