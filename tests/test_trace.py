"""Tests for the Tier-1 per-step trace (gym_dr/trace.py) and its wiring into
the metrics recorder (gym_dr/metrics.py).

These exercise the in-process producer of the trace contract without SB3,
Docker, or the upstream DeepRacer env — the sink records off the reward-param
dict, so a hand-built params stream is enough.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gym_dr.config import ExperimentConfig, TraceConfig
from gym_dr.metrics import _EpisodeMetrics, install_metrics
from gym_dr.trace import (
    STEP_COLUMNS,
    TraceSink,
    build_step_row,
    load_steps,
    terminal_status,
)


def _params(**over):
    """A representative DeepRacer reward-param dict (keys per RewardParam)."""
    base = {
        "x": 1.5,
        "y": -2.0,
        "heading": 42.0,
        "steering_angle": -7.5,
        "speed": 2.25,
        "progress": 13.0,
        "closest_waypoints": [11, 12],
        "track_length": 17.6,
        "track_width": 0.76,
        "distance_from_center": 0.05,
        "all_wheels_on_track": True,
        "is_crashed": False,
        "is_offtrack": False,
        "closest_objects": [-1, -1],
        "objects_location": [],
        "object_in_camera": False,
    }
    base.update(over)
    return base


# --- build_step_row ---------------------------------------------------------


def test_build_step_row_maps_param_names():
    row = build_step_row(_params(), step=3, reward=1.0, eval_reward=0.5)
    assert row["steps"] == 3
    assert row["yaw"] == 42.0  # heading -> yaw
    assert row["speed"] == 2.25  # throttle alias
    assert row["closest_waypoint"] == 12  # next waypoint
    assert row["track_len"] == 17.6  # track_length -> track_len
    assert row["on_track"] is True
    assert row["reward"] == 1.0
    assert row["eval_reward"] == 0.5
    assert row["action"] == -1  # continuous -> -1
    assert row["phase"] == "train"  # default phase
    # in-process producer cannot stamp sim time
    assert row["sim_time"] is None
    assert row["wall_time"] is not None


def test_build_step_row_phase_eval():
    row = build_step_row({}, step=1, reward=0.0, eval_reward=0.0, phase="eval")
    assert row["phase"] == "eval"


def test_build_step_row_object_avoidance_fields():
    row = build_step_row(
        _params(
            objects_location=[(1.0, 2.0), (3.0, 4.0)],
            closest_objects=[5, 6],
            is_crashed=True,
            object_in_camera=True,
        ),
        step=1,
        reward=0.0,
        eval_reward=0.0,
    )
    assert row["n_objects"] == 2
    assert row["oa_enabled"] is True
    assert row["closest_object_prev"] == 5
    assert row["closest_object_next"] == 6
    assert row["is_crashed"] is True
    assert row["object_in_camera"] is True


def test_build_step_row_tolerates_empty_params():
    # Test stubs / older envs may pass a partial dict — must not raise.
    row = build_step_row({}, step=1, reward=0.0, eval_reward=0.0)
    assert row["on_track"] is True
    assert row["n_objects"] == 0
    assert row["closest_waypoint"] is None


# --- terminal_status --------------------------------------------------------


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"is_crashed": True, "is_offtrack": False, "on_track": True, "progress": 50}, "crashed"),
        ({"is_crashed": False, "is_offtrack": True, "on_track": False, "progress": 50}, "off_track"),
        ({"is_crashed": False, "is_offtrack": False, "on_track": True, "progress": 100.0}, "lap_complete"),
        ({"is_crashed": False, "is_offtrack": False, "on_track": True, "progress": 60.0}, "time_up"),
    ],
)
def test_terminal_status(row, expected):
    assert terminal_status(row) == expected


# --- TraceSink + load_steps -------------------------------------------------


def test_sink_writes_shards_and_load_steps_concats(tmp_path):
    sink = TraceSink(tmp_path)
    assert sink.enabled  # pandas + pyarrow present in this env

    # Episode 0: 3 steps on reinvent_base / chunk 0.
    for s in range(1, 4):
        sink.add(build_step_row(_params(progress=s * 10), step=s, reward=1.0, eval_reward=0.0))
    sink.flush_episode(world_name="reinvent_base", chunk_index=0, run_id="run-abc")

    # Episode 1: 2 steps after a hot swap to Bowtie_track / chunk 1, ends crashed.
    sink.add(build_step_row(_params(progress=5.0), step=1, reward=1.0, eval_reward=0.0))
    sink.add(build_step_row(_params(progress=8.0, is_crashed=True), step=2, reward=-5.0, eval_reward=0.0))
    sink.flush_episode(world_name="Bowtie_track", chunk_index=1, run_id="run-abc")

    df = load_steps(tmp_path)

    # Schema is exactly the canonical column set, in order.
    assert list(df.columns) == STEP_COLUMNS
    assert len(df) == 5

    # Episode-level stamping.
    assert set(df["episode"].unique()) == {0, 1}
    assert df.loc[df["episode"] == 0, "world_name"].eq("reinvent_base").all()
    assert df.loc[df["episode"] == 1, "world_name"].eq("Bowtie_track").all()
    assert df.loc[df["episode"] == 1, "chunk_index"].eq(1).all()
    assert df["run_id"].eq("run-abc").all()

    # `done` is True only on the terminal row of each episode.
    assert df.groupby("episode")["done"].sum().eq(1).all()
    # Derived terminal status: episode 1 ended on a crash.
    ep1_last = df[df["episode"] == 1].iloc[-1]
    assert ep1_last["episode_status"] == "crashed"
    assert ep1_last["done"]


def test_sink_abandon_drops_partial(tmp_path):
    sink = TraceSink(tmp_path)
    sink.add(build_step_row(_params(), step=1, reward=0.0, eval_reward=0.0))
    sink.abandon_episode()
    sink.flush_episode(world_name="w", chunk_index=0)
    assert load_steps(tmp_path).empty


def test_load_steps_empty_when_no_shards(tmp_path):
    df = load_steps(tmp_path)
    assert df.empty
    assert list(df.columns) == STEP_COLUMNS


# --- metrics wiring ---------------------------------------------------------


def test_episode_metrics_records_to_sink(tmp_path):
    """record_step -> sink.add, flush_episode -> shard, carrying world context."""
    sink = TraceSink(tmp_path)
    state = _EpisodeMetrics(sink=sink, world_name="reinvent_base", chunk_index=2)

    for s in range(1, 6):
        state.record_step(_params(progress=s * 5.0), reward=1.0)
    state.flush_episode()

    df = load_steps(tmp_path)
    assert len(df) == 5
    assert df["world_name"].eq("reinvent_base").all()
    assert df["chunk_index"].eq(2).all()
    # steps are the recorder's 1-based per-episode counter
    assert list(df["steps"]) == [1, 2, 3, 4, 5]


def test_install_metrics_sink_off_by_default(tmp_path):
    exp = ExperimentConfig(name="t")
    _, _, state = install_metrics(exp, run_dir=tmp_path)
    assert state.sink is None  # trace disabled by default


def test_install_metrics_sink_on_when_enabled(tmp_path):
    exp = ExperimentConfig(name="t", trace=TraceConfig(enabled=True))
    _, _, state = install_metrics(exp, run_dir=tmp_path)
    assert state.sink is not None and state.sink.enabled


def test_install_metrics_no_sink_without_run_dir():
    exp = ExperimentConfig(name="t", trace=TraceConfig(enabled=True))
    _, _, state = install_metrics(exp)  # no run_dir -> nowhere to write
    assert state.sink is None


# --- end-to-end through the env wrapper -------------------------------------


class _StubEnv:
    """Minimal env that calls the (wrapped) reward each step and terminates
    after ``ep_len`` steps — enough to exercise _MetricsEnvWrapper flush/reset."""

    def __init__(self, reward_fn, ep_len=4):
        self._reward_fn = reward_fn
        self._ep_len = ep_len
        self._step = 0

    def reset(self, **kwargs):
        self._step = 0
        return {}, {}

    def step(self, action):
        self._step += 1
        progress = self._step * (100.0 / self._ep_len)
        params = _params(progress=progress)
        reward = self._reward_fn(params)  # drives the metrics/trace recorder
        terminated = self._step >= self._ep_len
        return {}, reward, terminated, False, {}

    def close(self):
        pass


def test_env_wrapper_flushes_episode_on_terminal_step(tmp_path):
    exp = ExperimentConfig(name="t", trace=TraceConfig(enabled=True))
    exp, env_wrapper, state = install_metrics(exp, run_dir=tmp_path)
    state.world_name = "reinvent_base"
    env = env_wrapper(_StubEnv(exp.reward, ep_len=4))

    # Two full episodes.
    for _ in range(2):
        env.reset()
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(None)
            done = term or trunc

    df = load_steps(tmp_path)
    assert len(df) == 8  # 2 episodes x 4 steps
    assert set(df["episode"].unique()) == {0, 1}
    assert df["world_name"].eq("reinvent_base").all()
    # Each episode's last step is the lap completion (progress hit 100).
    assert df.groupby("episode")["episode_status"].last().eq("lap_complete").all()
