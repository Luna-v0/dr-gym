"""Canonical Tier-1 per-step trace — the simtrace-equivalent.

This is the in-process producer of the trace contract (``docs/trace-contract.md``,
§2 / §6). It taps the one place that sees the *full* DeepRacer reward-param dict
— the reward callback (``gym_dr/metrics.py``) — and writes one row per env step
to per-episode Parquet shards under ``<run_dir>/trace/steps/``.

Why per-episode shards instead of a single ``step.parquet``
----------------------------------------------------------
Each finished episode is flushed to its own file (``ep_000123.parquet``), so a
crash or wall-clock kill never corrupts a half-written global file — every
completed episode is already durable. ``load_steps(run_dir)`` concatenates the
shards back into the single Tier-1 DataFrame the contract describes. The column
names match the ``deepracer-utils`` *internal* DataFrame
(``deepracer-utils/docs/output-format.md``) so vendored analysis runs on them
unmodified — but nothing here imports ``deepracer-utils``.

What this producer can and cannot fill
--------------------------------------
The reward callback only receives ``params`` (the reward-param dict). It does
**not** see ``obs`` (camera / LiDAR — those are Tier 2, from the ROS bag) and it
carries **no simulator clock**. Per the contract's fallback clause, this sink
stamps ``wall_time`` (``time.time()``) and leaves ``sim_time`` null; the
bag→trace path (or a future ``DeepRacerEnv`` change that surfaces ``/clock``)
fills ``sim_time``. ``episode_status`` is *derived* best-effort from the terminal
step's flags, since the reward params don't carry the upstream status enum.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd

LOG = logging.getLogger(__name__)


# Canonical Tier-1 column order. Grouped to mirror docs/trace-contract.md §2.
# CSV-alias note: the deepracer-utils DataFrame uses these exact names
# (steer→steering_angle, throttle→speed, all_wheels_on_track→on_track), so
# its analysis utilities consume this frame as-is.
STEP_COLUMNS: List[str] = [
    # identity & sync keys
    "run_id", "episode", "steps", "sim_time", "wall_time",
    # track / world (hot-swap aware)
    "world_name", "chunk_index",
    # pose / kinematics
    "x", "y", "yaw", "steering_angle", "speed", "action",
    # progress / track geometry
    "progress", "closest_waypoint", "track_len", "on_track",
    "distance_from_center", "track_width",
    # outcome
    "reward", "eval_reward", "done", "episode_status", "phase",
    # object-avoidance extension
    "oa_enabled", "is_crashed", "is_offtrack",
    "closest_object_prev", "closest_object_next", "n_objects", "object_in_camera",
]


def build_step_row(
    params: Dict[str, Any],
    *,
    step: int,
    reward: float,
    eval_reward: float,
    phase: str = "train",
    wall_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Map one reward-param dict to the step-level (episode-independent) fields.

    Episode-level fields (``run_id``, ``episode``, ``world_name``,
    ``chunk_index``, terminal ``episode_status``, ``done``) are stamped later by
    :meth:`TraceSink.flush_episode` — they aren't known per-step. ``sim_time`` is
    intentionally left ``None`` (see module docstring).

    ``phase`` is ``"eval"`` for steps recorded during an SB3 evaluation rollout
    (the metrics state's ``use_eval_reward`` is set), else ``"train"`` — this is
    what lets downstream analysis isolate the *last eval* episode's path.

    Uses ``.get`` defensively so a partial ``params`` (test stubs, older envs)
    never raises mid-rollout.
    """
    closest = params.get("closest_waypoints", [None, None]) or [None, None]
    prev_wp = closest[0] if len(closest) > 0 else None
    next_wp = closest[1] if len(closest) > 1 else None

    objs = params.get("objects_location", []) or []
    closest_obj = params.get("closest_objects", [-1, -1]) or [-1, -1]

    on_track = bool(params.get("all_wheels_on_track", True))
    return {
        "steps": int(step),
        "sim_time": None,  # filled by the bag→trace path; in-process has no /clock
        "wall_time": float(wall_time if wall_time is not None else time.time()),
        "x": _f(params.get("x")),
        "y": _f(params.get("y")),
        "yaw": _f(params.get("heading")),
        "steering_angle": _f(params.get("steering_angle")),
        "speed": _f(params.get("speed")),
        # Continuous action spaces have no discrete index; -1 matches the
        # simtrace convention (output-format.md: action is -1 when unavailable).
        "action": int(params.get("action", -1)),
        "progress": _f(params.get("progress")),
        "closest_waypoint": _i(next_wp),
        "track_len": _f(params.get("track_length")),
        "on_track": on_track,
        "distance_from_center": _f(params.get("distance_from_center")),
        "track_width": _f(params.get("track_width")),
        "reward": float(reward),
        "eval_reward": float(eval_reward),
        "phase": phase,
        "oa_enabled": bool(params.get("object_avoidance_enabled", len(objs) > 0)),
        "is_crashed": bool(params.get("is_crashed", False)),
        "is_offtrack": bool(params.get("is_offtrack", not on_track)),
        "closest_object_prev": _i(closest_obj[0] if len(closest_obj) > 0 else -1),
        "closest_object_next": _i(closest_obj[1] if len(closest_obj) > 1 else -1),
        "n_objects": int(len(objs)),
        "object_in_camera": bool(params.get("object_in_camera", False)),
    }


def terminal_status(row: Dict[str, Any]) -> str:
    """Best-effort episode_status for the terminal step, derived from flags.

    The reward-param dict carries no upstream status enum, so we infer one in
    the same vocabulary as ``deepracer_env/metrics/constants.py`` (so vendored
    stability analysis that groups by ``episode_status`` keeps working).
    Priority: crash > off-track > lap-complete > time/step-up.
    """
    if row.get("is_crashed"):
        return "crashed"
    if row.get("is_offtrack") or not row.get("on_track", True):
        return "off_track"
    progress = row.get("progress")
    if progress is not None and float(progress) >= 99.999:
        return "lap_complete"
    return "time_up"


class TraceSink:
    """Buffers per-step rows for the current episode, flushes one Parquet shard.

    One instance per training run. ``add`` accumulates rows; ``flush_episode``
    stamps the episode-level columns, derives the terminal status, writes
    ``trace/steps/ep_<NNNNNN>.parquet`` and clears the buffer. Pandas/pyarrow are
    imported lazily so importing this module is cheap and an environment without
    them degrades to a logged warning instead of crashing the run.
    """

    def __init__(self, run_dir: Path, *, compression: str = "snappy") -> None:
        self.steps_dir = Path(run_dir) / "trace" / "steps"
        self.compression = compression
        self._buffer: List[Dict[str, Any]] = []
        self._episode = 0
        self._enabled = self._probe_backend()
        if self._enabled:
            self.steps_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _probe_backend() -> bool:
        try:
            import pandas  # noqa: F401
            import pyarrow  # noqa: F401

            return True
        except ImportError as exc:  # pragma: no cover - environment dependent
            LOG.warning("trace sink disabled: %s (need pandas + pyarrow)", exc)
            return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def add(self, row: Dict[str, Any]) -> None:
        if self._enabled:
            self._buffer.append(row)

    def flush_episode(
        self,
        *,
        world_name: Optional[str],
        chunk_index: int,
        run_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Write the buffered episode to a Parquet shard and reset the buffer.

        Stamps every row with the episode-level keys and marks the last row's
        ``done`` + derived ``episode_status``. Returns the shard path (or None if
        the sink is disabled or the episode was empty).
        """
        if not self._enabled or not self._buffer:
            self._buffer = []
            return None

        import pandas as pd

        rows = self._buffer
        for r in rows:
            r["run_id"] = run_id
            r["episode"] = self._episode
            r["world_name"] = world_name
            r["chunk_index"] = int(chunk_index)
            r["done"] = False
            r["episode_status"] = "in_progress"
        rows[-1]["done"] = True
        rows[-1]["episode_status"] = terminal_status(rows[-1])

        # Reindex to the canonical column order so every shard has an identical
        # schema and load_steps() can concat without column drift.
        df = pd.DataFrame(rows).reindex(columns=STEP_COLUMNS)
        path = self.steps_dir / f"ep_{self._episode:06d}.parquet"
        try:
            df.to_parquet(path, compression=self.compression, index=False)
        except Exception as exc:  # pragma: no cover - disk/serialisation issues
            LOG.warning("trace sink: failed to write %s: %s", path, exc)
            path = None  # type: ignore[assignment]

        self._episode += 1
        self._buffer = []
        return path

    def abandon_episode(self) -> None:
        """Drop the current buffer without writing (e.g. on env reset mid-episode)."""
        self._buffer = []


def load_steps(run_dir: Path) -> "pd.DataFrame":
    """Concatenate all per-episode shards into the single Tier-1 DataFrame.

    This is the read side of the contract: the analysis layer calls this instead
    of any ``deepracer-utils`` loader. Returns an empty DataFrame (with the
    canonical columns) when no shards exist.
    """
    import pandas as pd

    steps_dir = Path(run_dir) / "trace" / "steps"
    shards = sorted(steps_dir.glob("ep_*.parquet"))
    if not shards:
        return pd.DataFrame(columns=STEP_COLUMNS)
    frames = [pd.read_parquet(p) for p in shards]
    return pd.concat(frames, ignore_index=True)


def _f(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _i(value: Any) -> Optional[int]:
    return None if value is None else int(value)
