"""Structured provenance store (Task 5) — ports & adapters.

Training produces data in several places (dr-gym episode metrics, the applied
domain randomization, per-episode outcomes) that today land in different formats
with no shared key. This module is the small, queryable **system** that ties them
together: one row per *episode* with a stable ``episode_uuid``, and child rows for
the applied DR knobs and the episode metrics — so later analysis can answer "which
episodes ran under which randomization regime, and how did they score?".

Design (ports & adapters, per the repo convention):

- :class:`DataStore` — the port. The metrics layer depends only on this.
- :class:`SQLiteDataStore` — the default adapter: a single ACID ``.sqlite`` file
  that travels with the run artifact, no external service. Raw blobs (camera
  frames, per-step Parquet) stay in their own files; this store holds the
  per-episode index + DR + metrics that make them joinable.
- :class:`NullDataStore` — a no-op adapter (e.g. HPO, where thousands of episodes
  would bloat the DB) so callers can always hold a store unconditionally.

It is deliberately independent of the simulator, so it is unit-tested without one.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Mapping, Optional


class DataStore(ABC):
    """Port: register runs/episodes and their DR + metrics; read them back.

    Usable as a context manager (closes on exit).
    """

    @abstractmethod
    def register_run(self, run_id: str, *, experiment: str = "",
                     config: "Optional[Mapping[str, Any]]" = None) -> None: ...

    @abstractmethod
    def start_episode(self, run_id: str, *, episode: int, car: int = 0,
                      world: str = "", phase: str = "train") -> str:
        """Register an episode; return its ``episode_uuid`` (the shared join key)."""

    @abstractmethod
    def record_dr(self, episode_uuid: str, dr: "Mapping[str, Any]") -> None:
        """Persist the applied domain-randomization values for an episode."""

    @abstractmethod
    def record_metrics(self, episode_uuid: str, metrics: "Mapping[str, Any]") -> None:
        """Persist per-episode metrics (e.g. a ``dr_episode`` summary)."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class NullDataStore(DataStore):
    """No-op store — recording is disabled but callers need not branch."""

    def register_run(self, run_id, *, experiment="", config=None) -> None:
        pass

    def start_episode(self, run_id, *, episode, car=0, world="", phase="train") -> str:
        return ""

    def record_dr(self, episode_uuid, dr) -> None:
        pass

    def record_metrics(self, episode_uuid, metrics) -> None:
        pass

    def close(self) -> None:
        pass


def _split_value(value: Any) -> "tuple[Optional[float], Optional[str]]":
    """Return ``(numeric, text)`` — a value goes in exactly one column so numeric
    DR/metrics are range-queryable while categorical ones (track name, reverse) are
    still recorded."""
    if isinstance(value, bool):
        return (float(value), None)
    if isinstance(value, (int, float)):
        return (float(value), None)
    return (None, json.dumps(value, default=str) if not isinstance(value, str) else value)


class SQLiteDataStore(DataStore):
    """SQLite adapter — one file, foreign keys from dr/metrics rows to episodes.

    Single-process/single-thread (a multi-car VecEnv steps serially); the DB file
    lives under the run dir so it ships with the artifact.
    """

    def __init__(self, path: "str | Path") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id     TEXT PRIMARY KEY,
                experiment TEXT,
                config     TEXT,
                created    REAL
            );
            CREATE TABLE IF NOT EXISTS episodes (
                episode_uuid TEXT PRIMARY KEY,
                run_id       TEXT,
                episode      INTEGER,
                car          INTEGER,
                world        TEXT,
                phase        TEXT,
                created      REAL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            );
            CREATE TABLE IF NOT EXISTS dr_params (
                episode_uuid TEXT,
                key          TEXT,
                value        REAL,
                value_str    TEXT,
                FOREIGN KEY (episode_uuid) REFERENCES episodes (episode_uuid)
            );
            CREATE TABLE IF NOT EXISTS metrics (
                episode_uuid TEXT,
                key          TEXT,
                value        REAL,
                FOREIGN KEY (episode_uuid) REFERENCES episodes (episode_uuid)
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes (run_id);
            CREATE INDEX IF NOT EXISTS idx_dr_ep ON dr_params (episode_uuid);
            CREATE INDEX IF NOT EXISTS idx_metrics_ep ON metrics (episode_uuid);
            """
        )
        self._conn.commit()

    def register_run(self, run_id, *, experiment="", config=None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, experiment, config, created) VALUES (?, ?, ?, ?)",
            (run_id, experiment, json.dumps(dict(config or {}), default=str), time.time()),
        )
        self._conn.commit()

    def start_episode(self, run_id, *, episode, car=0, world="", phase="train") -> str:
        episode_uuid = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO episodes (episode_uuid, run_id, episode, car, world, phase, created)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (episode_uuid, run_id, int(episode), int(car), world, phase, time.time()),
        )
        self._conn.commit()
        return episode_uuid

    def record_dr(self, episode_uuid, dr) -> None:
        rows = []
        for key, value in dict(dr).items():
            num, text = _split_value(value)
            rows.append((episode_uuid, str(key), num, text))
        self._conn.executemany(
            "INSERT INTO dr_params (episode_uuid, key, value, value_str) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def record_metrics(self, episode_uuid, metrics) -> None:
        rows = []
        for key, value in dict(metrics).items():
            num, _ = _split_value(value)
            if num is not None:
                rows.append((episode_uuid, str(key), num))
        self._conn.executemany(
            "INSERT INTO metrics (episode_uuid, key, value) VALUES (?, ?, ?)", rows
        )
        self._conn.commit()

    # ------------------------------ read side ------------------------------ #
    def episode_count(self, run_id: "Optional[str]" = None) -> int:
        if run_id is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE run_id = ?", (run_id,)).fetchone()[0])

    def read_dr(self, episode_uuid: str) -> "dict[str, Any]":
        out: "dict[str, Any]" = {}
        for key, num, text in self._conn.execute(
            "SELECT key, value, value_str FROM dr_params WHERE episode_uuid = ?", (episode_uuid,)
        ):
            out[key] = text if num is None else num
        return out

    def read_metrics(self, episode_uuid: str) -> "dict[str, float]":
        return {
            key: value
            for key, value in self._conn.execute(
                "SELECT key, value FROM metrics WHERE episode_uuid = ?", (episode_uuid,)
            )
        }

    def read_episodes(self, run_id: "Optional[str]" = None) -> "List[dict]":
        """All episodes (optionally for one run), each merged with its DR + metrics."""
        q = "SELECT episode_uuid, run_id, episode, car, world, phase FROM episodes"
        args: tuple = ()
        if run_id is not None:
            q += " WHERE run_id = ?"
            args = (run_id,)
        q += " ORDER BY created"
        out: "List[dict]" = []
        for uid, rid, ep, car, world, phase in self._conn.execute(q, args).fetchall():
            row = {"episode_uuid": uid, "run_id": rid, "episode": ep, "car": car,
                   "world": world, "phase": phase}
            row["dr"] = self.read_dr(uid)
            row["metrics"] = self.read_metrics(uid)
            out.append(row)
        return out

    def close(self) -> None:
        self._conn.close()


def make_data_store(path: "Optional[str | Path]") -> DataStore:
    """Factory: an :class:`SQLiteDataStore` at ``path``, or a :class:`NullDataStore`
    when ``path`` is falsy (recording disabled)."""
    return SQLiteDataStore(path) if path else NullDataStore()
