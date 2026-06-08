"""Tests for ``gym_dr.seeding.SeedManager``.

The invariants under test mirror the design contract:

  - **Pure & deterministic** — same master seed reproduces every derived seed.
  - **Independence** — agent ⊥ domain within a replicate; replicates differ;
    eval/sampler/replicate roles don't collide.
  - **Fixed eval suite** — built once, cached, identical across calls.
  - **Stable role order** — adding a role at the end leaves earlier streams
    unchanged (so past runs stay reproducible).
"""
from __future__ import annotations

import numpy as np
import pytest

from gym_dr.seeding import SeedManager


def test_deterministic_same_master_reproduces_everything():
    a = SeedManager(1234)
    b = SeedManager(1234)
    assert a.eval_seeds() == b.eval_seeds()
    assert a.sampler_seed() == b.sampler_seed()
    assert [r.agent for r in a.replicates()] == [r.agent for r in b.replicates()]
    assert [r.domain for r in a.replicates()] == [r.domain for r in b.replicates()]


def test_different_master_differs():
    a = SeedManager(1)
    b = SeedManager(2)
    assert a.eval_seeds() != b.eval_seeds()
    assert a.sampler_seed() != b.sampler_seed()
    assert a.agent_seed(0) != b.agent_seed(0)


def test_agent_and_domain_are_independent_within_replicate():
    mgr = SeedManager(7, n_replicates=5)
    for rep in mgr.replicates():
        assert rep.agent != rep.domain


def test_replicates_are_distinct():
    mgr = SeedManager(7, n_replicates=8)
    agents = [r.agent for r in mgr.replicates()]
    domains = [r.domain for r in mgr.replicates()]
    assert len(set(agents)) == len(agents)
    assert len(set(domains)) == len(domains)
    # agent and domain pools shouldn't overlap either
    assert set(agents).isdisjoint(domains)


def test_roles_do_not_collide():
    mgr = SeedManager(99)
    pool = set(mgr.eval_seeds())
    pool.add(mgr.sampler_seed())
    pool.update(r.agent for r in mgr.replicates())
    pool.update(r.domain for r in mgr.replicates())
    expected = mgr.n_eval_seeds + 1 + 2 * mgr.n_replicates
    assert len(pool) == expected


def test_named_access_matches_indexed():
    mgr = SeedManager(55)
    assert mgr.agent_seed(2) == mgr.replicate(2).agent
    assert mgr.domain_seed(2) == mgr.replicate(2).domain


def test_replicate_index_bounds():
    mgr = SeedManager(0, n_replicates=3)
    with pytest.raises(IndexError):
        mgr.replicate(3)
    with pytest.raises(IndexError):
        mgr.replicate(-1)


def test_invalid_counts_rejected():
    with pytest.raises(ValueError):
        SeedManager(0, n_eval_seeds=0)
    with pytest.raises(ValueError):
        SeedManager(0, n_replicates=0)
    with pytest.raises(ValueError):
        SeedManager(0, n_eval_scenarios=0)


def test_eval_suite_built_once_and_cached():
    mgr = SeedManager(3, n_eval_seeds=2, n_eval_scenarios=3)
    calls = []

    def place(eval_seed, scenario_index):
        calls.append((eval_seed, scenario_index))
        return (eval_seed, scenario_index)

    suite1 = mgr.build_eval_suite(place)
    suite2 = mgr.build_eval_suite(place)
    # Cached: identical object, place() invoked only on the first build.
    assert suite1 is suite2
    assert len(calls) == 2 * 3
    assert len(suite1) == 2 * 3
    # Pooled across (eval_seed, scenario_index) for the fixed eval seeds.
    expected = [
        (s, i) for s in mgr.eval_seeds() for i in range(mgr.n_eval_scenarios)
    ]
    assert list(suite1) == expected


def test_eval_seeds_fixed_regardless_of_replicate_count():
    # The eval suite must not move when you change how many replicates you run.
    a = SeedManager(42, n_eval_seeds=5, n_replicates=3)
    b = SeedManager(42, n_eval_seeds=5, n_replicates=20)
    assert a.eval_seeds() == b.eval_seeds()
    assert a.sampler_seed() == b.sampler_seed()


def test_adding_a_role_does_not_shift_earlier_roles():
    # Roles are spawned positionally from the master in a fixed order
    # (eval=0, sampler=1, replicate=2). Re-deriving the first three children
    # directly must match what SeedManager exposes — proving that appending a
    # 4th role later would leave eval/sampler/replicate untouched.
    master = 2024
    mgr = SeedManager(master)
    root = np.random.SeedSequence(master)
    eval_root, sampler_root, replicate_root = root.spawn(3)
    expected_eval = [
        int(s.generate_state(1, dtype=np.uint32)[0])
        for s in eval_root.spawn(mgr.n_eval_seeds)
    ]
    assert mgr.eval_seeds() == expected_eval
    assert mgr.sampler_seed() == int(sampler_root.generate_state(1, dtype=np.uint32)[0])


def test_domain_rng_is_a_generator():
    mgr = SeedManager(11)
    rng = mgr.replicate(0).domain_rng()
    assert isinstance(rng, np.random.Generator)
    # Deterministic from the domain stream.
    again = SeedManager(11).replicate(0).domain_rng()
    assert rng.integers(0, 10_000) == again.integers(0, 10_000)
