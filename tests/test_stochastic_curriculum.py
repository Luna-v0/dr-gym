"""Tests for the D5 StochasticCurriculum world strategy (anti-forgetting)."""
from __future__ import annotations

from gym_dr.worlds import StochasticCurriculum, WorldChunk


def _strat(**kw):
    kw.setdefault("train_worlds", ["A", "B", "C"])
    kw.setdefault("eval_worlds", ["X", "Y"])
    kw.setdefault("chunk_steps", 1000)
    kw.setdefault("n_chunks", 30)
    kw.setdefault("unlock_every", 3)
    kw.setdefault("seed", 0)
    return StochasticCurriculum(**kw)


def test_deterministic_plan():
    a = _strat().training_chunks()
    b = _strat().training_chunks()
    assert a == b  # same seed ⇒ identical plan (host & container must agree)
    assert all(isinstance(c, WorldChunk) and c.steps == 1000 for c in a)
    assert len(a) == 30


def test_first_world_and_first_chunk_are_track0():
    s = _strat()
    assert s.first_world() == "A"
    assert s.training_chunks()[0].world == "A"


def test_window_expands_over_time():
    s = _strat(n_chunks=30, unlock_every=3)
    chunks = [c.world for c in s.training_chunks()]
    # Before track 1 unlocks (chunks 0..2) only "A" is reachable.
    assert set(chunks[:3]) == {"A"}
    # By the end, all three tracks have been unlocked and sampled.
    assert set(chunks) == {"A", "B", "C"}


def test_only_train_worlds_sampled():
    s = _strat()
    assert set(c.world for c in s.training_chunks()) <= {"A", "B", "C"}


def test_older_tracks_keep_being_revisited():
    # With many chunks, the oldest track must still appear after newer ones
    # unlock — the anti-forgetting property.
    s = _strat(n_chunks=60)
    worlds = [c.world for c in s.training_chunks()]
    late = worlds[30:]  # well after all tracks unlocked
    assert "A" in late, "oldest track should still be revisited late in training"


def test_evaluation_worlds_held_out():
    assert _strat().evaluation_worlds() == ["X", "Y"]


def test_recency_favours_newer_tracks():
    # Over a long run, the most-recently-unlocked track is sampled more than the
    # oldest (recency_weight > 1).
    s = _strat(n_chunks=300, recency_weight=2.0)
    worlds = [c.world for c in s.training_chunks()]
    assert worlds.count("C") > worlds.count("A")
