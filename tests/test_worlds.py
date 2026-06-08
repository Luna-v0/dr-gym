"""Unit tests for the world-scheduling strategies (gym_dr/worlds.py) and their
wiring into ExperimentConfig."""
from __future__ import annotations

from gym_dr import (
    ExperimentConfig,
    OrderedSplit,
    SequentialRotation,
    WorldsConfig,
    WorldStrategy,
)
from gym_dr.worlds import WorldChunk


# --- SequentialRotation -----------------------------------------------------


def test_sequential_rotation_chunk_order_and_steps():
    s = SequentialRotation(names=["A", "B"], chunk_steps=1000, rotations=2)
    chunks = s.training_chunks()
    assert [c.world for c in chunks] == ["A", "B", "A", "B"]
    assert all(c.steps == 1000 for c in chunks)
    assert s.first_world() == "A"
    assert s.evaluation_worlds() == []  # eval on current training world


def test_sequential_rotation_coerces_bare_string():
    assert SequentialRotation(names="Oval_track").names == ["Oval_track"]


# --- OrderedSplit -----------------------------------------------------------


def test_ordered_split_train_order_and_held_out_eval():
    s = OrderedSplit(
        train_worlds=["A", "B", "C"],
        eval_worlds=["D", "E"],
        chunk_steps=2000,
        rotations=2,
    )
    chunks = s.training_chunks()
    assert [c.world for c in chunks] == ["A", "B", "C", "A", "B", "C"]  # rotations repeat
    assert all(c.steps == 2000 for c in chunks)
    assert s.evaluation_worlds() == ["D", "E"]  # ordered, independent of rotations
    assert s.first_world() == "A"


def test_ordered_split_is_a_world_strategy():
    assert isinstance(OrderedSplit(train_worlds=["A"], eval_worlds=["B"]), WorldStrategy)


def test_ordered_split_requires_a_train_world():
    import pytest

    with pytest.raises(ValueError):
        OrderedSplit(train_worlds=[], eval_worlds=["B"]).training_chunks  # noqa: B018
    # __post_init__ raises at construction
    with pytest.raises(ValueError):
        OrderedSplit(train_worlds=[], eval_worlds=["B"])


# --- ExperimentConfig.effective_strategy ------------------------------------


def test_effective_strategy_defaults_to_sequential_from_worlds():
    exp = ExperimentConfig(
        name="t", worlds=WorldsConfig(names=["X", "Y"], chunk_steps=500, rotations=3)
    )
    s = exp.effective_strategy()
    assert isinstance(s, SequentialRotation)
    assert [c.world for c in s.training_chunks()] == ["X", "Y"] * 3
    assert s.chunk_steps == 500
    assert s.evaluation_worlds() == []


def test_effective_strategy_uses_explicit_strategy():
    strat = OrderedSplit(train_worlds=["A"], eval_worlds=["B", "C"], chunk_steps=10)
    exp = ExperimentConfig(name="t", world_strategy=strat)
    assert exp.effective_strategy() is strat
    assert exp.effective_strategy().evaluation_worlds() == ["B", "C"]


def test_world_strategy_serialized_in_to_dict():
    strat = OrderedSplit(train_worlds=["A", "B"], eval_worlds=["C"], chunk_steps=10)
    exp = ExperimentConfig(name="t", world_strategy=strat)
    d = exp.to_dict()
    assert d["world_strategy"]["__class__"].endswith("OrderedSplit")
    assert d["world_strategy"]["train_worlds"] == ["A", "B"]
    assert d["world_strategy"]["eval_worlds"] == ["C"]
    # default (no strategy) serializes to None
    assert ExperimentConfig(name="t").to_dict()["world_strategy"] is None


def test_worldchunk_fields():
    c = WorldChunk("track", 123)
    assert c.world == "track" and c.steps == 123
