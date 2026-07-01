"""Tests for the unified hyperparameter space (gym_dr.search)."""
from __future__ import annotations

import pytest

from gym_dr.search import (
    Categorical,
    Fixed,
    Float,
    Hyperparam,
    Int,
    SearchSpace,
    as_hyperparam,
)


class FakeTrial:
    """Minimal Optuna-Trial stand-in recording suggest_* calls."""

    def __init__(self, floats=None, ints=None, cats=None):
        self.floats = floats or {}
        self.ints = ints or {}
        self.cats = cats or {}
        self.calls = []

    def suggest_float(self, name, low, high, log=False, step=None):
        self.calls.append(("float", name, low, high, log, step))
        return self.floats.get(name, (low + high) / 2)

    def suggest_int(self, name, low, high, log=False, step=1):
        self.calls.append(("int", name, low, high, log, step))
        return self.ints.get(name, low)

    def suggest_categorical(self, name, choices):
        self.calls.append(("cat", name, tuple(choices)))
        return self.cats.get(name, choices[0])


# ------------------------------------------------------------- hyperparams

def test_fixed_ignores_trial_and_returns_constant():
    f = Fixed(3e-4)
    assert f.is_fixed is True
    assert f.suggest(FakeTrial(), "lr") == 3e-4


def test_float_calls_suggest_float_with_scale():
    hp = Float(1e-5, 1e-3, log=True)
    trial = FakeTrial(floats={"lr": 2e-4})
    assert hp.suggest(trial, "lr") == 2e-4
    assert trial.calls == [("float", "lr", 1e-5, 1e-3, True, None)]
    assert hp.is_fixed is False


def test_float_rejects_low_above_high():
    with pytest.raises(ValueError):
        Float(1.0, 0.0)


def test_float_rejects_log_and_step_together():
    with pytest.raises(ValueError):
        Float(1e-5, 1e-3, log=True, step=1e-5)


def test_int_calls_suggest_int():
    hp = Int(2, 8, step=2)
    trial = FakeTrial(ints={"n_steps": 4})
    assert hp.suggest(trial, "n_steps") == 4
    assert trial.calls == [("int", "n_steps", 2, 8, False, 2)]


def test_int_rejects_bad_bounds():
    with pytest.raises(ValueError):
        Int(9, 1)


def test_categorical_coerces_list_to_tuple_and_suggests():
    hp = Categorical(["ppo", "sac", "td3"])
    assert hp.choices == ("ppo", "sac", "td3")
    trial = FakeTrial(cats={"algo": "sac"})
    assert hp.suggest(trial, "algo") == "sac"
    assert trial.calls == [("cat", "algo", ("ppo", "sac", "td3"))]


def test_categorical_rejects_empty_and_non_sequence():
    with pytest.raises(ValueError):
        Categorical([])
    with pytest.raises(TypeError):
        Categorical("notalist")  # type: ignore[arg-type]


def test_hyperparams_are_frozen_hashable():
    s = {Fixed(1), Fixed(1), Float(0.0, 1.0)}
    assert len(s) == 2


def test_as_hyperparam_coerces_bare_value():
    assert as_hyperparam(0.99) == Fixed(0.99)
    hp = Float(0.0, 1.0)
    assert as_hyperparam(hp) is hp
    assert isinstance(as_hyperparam(5), Hyperparam)


# -------------------------------------------------------------- SearchSpace

def test_all_fixed_space_is_single_run():
    space = SearchSpace({"trainer.kwargs.learning_rate": 3e-4, "trainer.kwargs.gamma": 0.99})
    assert space.is_single_run is True
    assert space.search_dims == []
    assert space.fixed_overrides() == {
        "trainer.kwargs.learning_rate": 3e-4,
        "trainer.kwargs.gamma": 0.99,
    }


def test_mixed_space_is_search():
    space = SearchSpace(
        {
            "trainer.kwargs.learning_rate": Float(1e-5, 1e-3, log=True),
            "trainer.kwargs.gamma": 0.99,
        }
    )
    assert space.is_single_run is False
    assert space.search_dims == ["trainer.kwargs.learning_rate"]


def test_overrides_uses_config_paths_as_keys():
    space = SearchSpace(
        {
            "trainer.kwargs.learning_rate": Float(1e-5, 1e-3, log=True),
            "trainer.kwargs.gamma": 0.99,
        }
    )
    trial = FakeTrial(floats={"trainer.kwargs.learning_rate": 5e-4})
    ov = space.overrides(trial)
    # Fixed contributes its constant; searched draws from the trial; keys are the
    # dotted config paths so this plugs straight into with_overrides(**ov).
    assert ov == {
        "trainer.kwargs.learning_rate": 5e-4,
        "trainer.kwargs.gamma": 0.99,
    }
    # The suggest was keyed by the full dotted path.
    assert trial.calls == [
        ("float", "trainer.kwargs.learning_rate", 1e-5, 1e-3, True, None)
    ]


def test_empty_space_is_single_run():
    space = SearchSpace()
    assert space.is_single_run is True
    assert len(space) == 0
    assert space.fixed_overrides() == {}


def test_searchspace_mapping_dunders():
    space = SearchSpace({"a": 1, "b": Float(0.0, 1.0)})
    assert len(space) == 2
    assert "a" in space
    assert set(iter(space)) == {"a", "b"}
    assert isinstance(space["b"], Float)


def test_describe_is_readable():
    space = SearchSpace({"a": 1, "lr": Float(1e-5, 1e-3, log=True)})
    d = space.describe()
    assert d["a"] == "Fixed(1)"
    assert "log" in d["lr"]
