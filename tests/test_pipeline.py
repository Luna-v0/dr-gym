"""Tests for the composable Stage pipeline primitive (gym_dr.pipeline)."""
from __future__ import annotations

import pytest

from gym_dr.pipeline import Stage, as_stage, compose, identity, stage


def test_single_stage_calls_wrapped_fn():
    s = Stage(lambda x: x + 1)
    assert s(3) == 4
    assert len(s) == 1
    assert list(s) == [s]


def test_composition_runs_left_to_right():
    pipe = Stage(lambda x: x + 1) >> Stage(lambda x: x * 2)
    # (3 + 1) * 2 == 8, NOT (3 * 2) + 1 == 7
    assert pipe(3) == 8


def test_composition_with_bare_callable_right():
    pipe = Stage(lambda x: x + 1, name="inc") >> (lambda x: x * 2)
    assert pipe(3) == 8


def test_composition_with_bare_callable_left():
    # __rrshift__: bare callable on the left of a Stage
    pipe = (lambda x: x + 1) >> Stage(lambda x: x * 2, name="dbl")
    assert pipe(3) == 8


def test_names_compose():
    inc = Stage(lambda x: x + 1, name="inc")
    dbl = Stage(lambda x: x * 2, name="dbl")
    assert (inc >> dbl).name == "inc→dbl"


def test_pipeline_is_introspectable_and_flattens():
    a = Stage(lambda x: x + 1, name="a")
    b = Stage(lambda x: x * 2, name="b")
    c = Stage(lambda x: x - 3, name="c")
    pipe = a >> b >> c
    # Flattened: three leaf stages, in order.
    assert len(pipe) == 3
    assert [s.name for s in pipe] == ["a", "b", "c"]
    assert pipe(3) == ((3 + 1) * 2) - 3  # == 5
    assert "a >> b >> c" in repr(pipe)


def test_associativity_of_composition():
    a = Stage(lambda x: x + 1)
    b = Stage(lambda x: x * 2)
    c = Stage(lambda x: x - 3)
    left = (a >> b) >> c
    right = a >> (b >> c)
    for x in (-2, 0, 5, 10):
        assert left(x) == right(x)


def test_identity_is_neutral():
    inc = Stage(lambda x: x + 1, name="inc")
    assert (identity() >> inc)(4) == 5
    assert (inc >> identity())(4) == 5


def test_compose_helper_matches_operator():
    a = Stage(lambda x: x + 1)
    b = Stage(lambda x: x * 2)
    c = Stage(lambda x: x - 3)
    built = compose(a, b, c)
    manual = a >> b >> c
    assert built(7) == manual(7)
    assert len(built) == 3


def test_compose_requires_at_least_one_stage():
    with pytest.raises(ValueError):
        compose()


def test_stage_decorator_plain():
    @stage
    def grayscale(obs):
        return obs * 0.5

    assert isinstance(grayscale, Stage)
    assert grayscale.name == "grayscale"
    assert grayscale(2.0) == 1.0


def test_stage_decorator_with_name():
    @stage(name="adr-input")
    def add_noise(obs):
        return obs + 1

    assert isinstance(add_noise, Stage)
    assert add_noise.name == "adr-input"


def test_as_stage_is_idempotent():
    s = Stage(lambda x: x, name="keep")
    assert as_stage(s) is s
    wrapped = as_stage(lambda x: x + 1)
    assert isinstance(wrapped, Stage)
    assert wrapped(1) == 2


def test_non_callable_raises():
    with pytest.raises(TypeError):
        Stage(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        as_stage(42)  # type: ignore[arg-type]


def test_rename_preserves_substages_and_behaviour():
    pipe = Stage(lambda x: x + 1, name="a") >> Stage(lambda x: x * 2, name="b")
    renamed = pipe.rename("obs_to_action")
    assert renamed.name == "obs_to_action"
    assert renamed(3) == pipe(3)
    assert len(renamed) == len(pipe)


def test_fn_property_exposes_wrapped_callable():
    def encode(obs):
        return obs

    s = Stage(encode)
    assert s.fn is encode
