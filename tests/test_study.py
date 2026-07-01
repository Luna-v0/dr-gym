"""Tests for the Study facade (gym_dr.study) — dispatch, seeds, delegation."""
from __future__ import annotations

import pytest

from gym_dr import ExperimentConfig, Float, SearchSpace, Study
from gym_dr.seeding import SeedManager
from gym_dr.study import StudyResult


@pytest.fixture
def exp():
    return ExperimentConfig(name="study_t")


@pytest.fixture
def host_mode(monkeypatch):
    """Neither worker nor in-container: the host dispatch paths."""
    monkeypatch.delenv("GYM_DR_WORKER", raising=False)
    monkeypatch.delenv("GYM_DR_IN_CONTAINER", raising=False)


@pytest.fixture
def spy(monkeypatch):
    """Record calls to the delegated app.train / app.study without running them."""
    calls = {"train": [], "study": []}

    def fake_train(experiment):
        calls["train"].append(experiment)
        return f"/path/{experiment.name}.zip"

    def fake_study(base, search_space, **kwargs):
        calls["study"].append({"base": base, "search_space": search_space, **kwargs})
        return 0

    monkeypatch.setattr("gym_dr.app.train", fake_train)
    monkeypatch.setattr("gym_dr.app.study", fake_study)
    return calls


# ----------------------------------------------------------- construction

def test_rejects_non_experiment():
    with pytest.raises(TypeError):
        Study(object())  # type: ignore[arg-type]


def test_params_dict_coerced_to_searchspace(exp):
    s = Study(exp, params={"trainer.kwargs.learning_rate": 3e-4})
    assert isinstance(s.space, SearchSpace)
    assert s.is_single_run is True


def test_search_space_passed_through(exp):
    space = SearchSpace({"trainer.kwargs.learning_rate": Float(1e-5, 1e-3, log=True)})
    s = Study(exp, params=space)
    assert s.space is space
    assert s.is_single_run is False


def test_omitted_params_is_single_run(exp):
    assert Study(exp).is_single_run is True


# --------------------------------------------------------- single-run host

def test_single_run_delegates_to_train_once(exp, host_mode, spy):
    res = Study(exp, master_seed=7).run()
    assert isinstance(res, StudyResult)
    assert len(spy["train"]) == 1
    assert not spy["study"]
    passed = spy["train"][0]
    assert passed.name == "study_t"
    # Seed derives from the master seed via SeedManager (reproducible), not None.
    assert passed.seed == SeedManager(7, n_replicates=1).replicate(0).agent
    assert res.run_paths == ["/path/study_t.zip"]


def test_replicates_loop_distinct_seeds_and_names(exp, host_mode, spy):
    res = Study(exp, master_seed=7, n_replicates=3).run()
    assert len(spy["train"]) == 3
    sm = SeedManager(7, n_replicates=3)
    names = [e.name for e in spy["train"]]
    seeds = [e.seed for e in spy["train"]]
    assert names == ["study_t_rep0", "study_t_rep1", "study_t_rep2"]
    assert seeds == [sm.replicate(k).agent for k in range(3)]
    assert len(set(seeds)) == 3  # independent
    assert res.n_replicates == 3
    assert len(res.run_paths) == 3


def test_fixed_params_applied_to_experiment(exp, host_mode, spy):
    Study(exp, params={"trainer.kwargs.gamma": 0.97}).run()
    passed = spy["train"][0]
    assert passed.trainer.kwargs["gamma"] == 0.97


def test_master_seed_is_reproducible(exp, host_mode, spy):
    Study(exp, master_seed=123).run()
    Study(exp, master_seed=123).run()
    assert spy["train"][0].seed == spy["train"][1].seed


# ---------------------------------------------------------------- HPO host

def test_hpo_delegates_to_study(exp, host_mode, spy):
    space = {"trainer.kwargs.learning_rate": Float(1e-5, 1e-3, log=True)}
    res = Study(exp, params=space, n_trials=15, n_parallel=2).run()
    assert not spy["train"]
    assert len(spy["study"]) == 1
    call = spy["study"][0]
    assert call["n_trials"] == 15
    assert call["n_parallel"] == 2
    assert res.n_trials == 15


def test_hpo_adapter_produces_overrides(exp, host_mode, spy):
    space = {"trainer.kwargs.learning_rate": Float(1e-5, 1e-3, log=True), "trainer.kwargs.gamma": 0.99}
    Study(exp, params=space, n_trials=3).run()
    adapter = spy["study"][0]["search_space"]

    class FakeTrial:
        def suggest_float(self, name, low, high, log=False, step=None):
            return 5e-4

    ov = adapter(FakeTrial())
    # dotted config keys preserved; fixed dim carried through; searched dim drawn.
    assert ov == {"trainer.kwargs.learning_rate": 5e-4, "trainer.kwargs.gamma": 0.99}


def test_hpo_seeds_sampler_from_master(exp, host_mode, spy):
    Study(exp, params={"trainer.kwargs.gamma": Float(0.9, 0.99)}, master_seed=55).run()
    # experiment.seed was None -> Study seeds it from master_seed for a
    # reproducible search.
    assert spy["study"][0]["base"].seed == 55


# ----------------------------------------------------- container / worker

def test_container_mode_runs_single_replicate(exp, monkeypatch, spy):
    monkeypatch.delenv("GYM_DR_WORKER", raising=False)
    monkeypatch.setenv("GYM_DR_IN_CONTAINER", "1")
    # Even with n_replicates=3, the container runs exactly one (host loops replicates).
    res = Study(exp, master_seed=7, n_replicates=3).run()
    assert len(spy["train"]) == 1
    assert spy["train"][0] is exp  # the base experiment, unmodified (env applies overrides)
    assert res.n_replicates == 1


def test_worker_mode_delegates_to_study(exp, monkeypatch, spy):
    monkeypatch.setenv("GYM_DR_WORKER", "1")
    Study(exp, params={"trainer.kwargs.gamma": Float(0.9, 0.99)}, n_trials=4).run()
    assert len(spy["study"]) == 1
    assert not spy["train"]


def test_callable_params_routes_to_hpo(exp, host_mode, spy):
    """A legacy imperative search_space(trial) -> dict callable is accepted and
    routes to the HPO path (eases migration of existing HPO experiments)."""
    def search_space(trial):
        return {"trainer.kwargs.gamma": 0.95}

    s = Study(exp, params=search_space, n_trials=2)
    assert s.is_single_run is False
    s.run()
    assert len(spy["study"]) == 1
    adapter = spy["study"][0]["search_space"]
    assert adapter(object()) == {"trainer.kwargs.gamma": 0.95}


def test_repr_shows_mode(exp):
    assert "single-run" in repr(Study(exp))
    assert "hpo" in repr(Study(exp, params={"trainer.kwargs.gamma": Float(0.9, 0.99)}, n_trials=5))
