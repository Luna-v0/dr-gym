"""Deterministic seed orchestration: one master seed → all RNG streams.

``SeedManager`` is the single source of truth that turns *one* recorded
master seed into every independent seed stream the pipeline needs. You record
one number; the whole experiment becomes reproducible; and no two roles ever
share RNG state.

Why ``SeedSequence.spawn`` (and not hand-picked distinct integers)
------------------------------------------------------------------
Statistical independence between streams comes from *spawning*, not from the
streams happening to be different integers. ``numpy.random.SeedSequence``
hashes ``(entropy, spawn_key)`` through a high-quality mixing function;
children produced by ``.spawn(n)`` get spawn keys ``(..., 0), (..., 1), …``
and are guaranteed independent. Two seeds that merely *look* different (``41``
vs ``42``) give no such guarantee — adjacent seeds can produce correlated
low-order bits in many PRNGs. So the manager only ever derives streams by
spawning from the master ``SeedSequence``.

The derivation tree
-------------------
On construction with ``master_seed`` the manager spawns three top-level role
streams (in this fixed order — append new roles at the *end* so existing
streams never shift)::

    master
      ├── eval       ──> spawn(n_eval_seeds)     # the fixed eval suite seeds
      ├── sampler    ──> one seed for Optuna's TPESampler
      └── replicate  ──> spawn(n_replicates)
                           └── each: spawn(2) -> (agent ⊥ domain)

- **Eval stream.** Spawns ``n_eval_seeds`` eval seeds. These build the fixed
  evaluation suite *once*: for each eval seed, ``n_eval_scenarios`` scenarios
  are placed and pooled into one heterogeneous suite that is never
  regenerated (so every trial and every replicate is scored on the *same*
  scenarios). See :meth:`SeedManager.build_eval_suite`.
- **Sampler stream.** One seed for Optuna's ``TPESampler`` so the search
  trajectory is reproducible. See :meth:`SeedManager.sampler_seed`.
- **Replicate stream.** Spawns ``n_replicates`` training seeds — the runs you
  loop over to capture pipeline variance.

The one invariant that genuinely matters: agent ⊥ domain
--------------------------------------------------------
For each training replicate the manager hands out **two independent
sub-seeds**:

- an **agent seed** — policy initialization, action sampling, minibatch order;
- a **domain seed** — drives scenario randomization (``place``) during
  *training* episodes.

These must be independent even though they belong to the same replicate:
scenario randomization should never perturb policy stochasticity, and vice
versa. Everything else here is bookkeeping; this decoupling is the part that
matters statistically.

Two contexts for domain randomization
-------------------------------------
The same ``place`` function is seeded from two different sources:

- **eval** placements are *fixed* — derived from the eval stream, identical
  across every trial and replicate;
- **training** placements *vary per replicate* — derived from that
  replicate's domain seed, so each run trains on different randomization.

Intended end-to-end flow::

    master → { eval suite (fixed), sampler, K replicates × (agent ⊥ domain) }
           → Optuna maximizes the rliable IQM over the pooled eval suite,
             with variance read across the K replicates.

#FUTURE — not yet wireable into this pipeline
---------------------------------------------
The manager already derives every seed stream, but several consumers don't
exist yet, so the corresponding hooks are marked ``#FUTURE`` below:

- **Domain randomization / ``place()``** — there is no scenario-placement
  function in the env yet (no obstacle/track/start-pose randomizer keyed by a
  seed). :meth:`build_eval_suite` accepts a ``place`` callable so the suite
  can be built the moment one exists; the *training* domain seed
  (:attr:`ReplicateSeeds.domain`) is derived but nothing consumes it yet —
  ``ExperimentConfig.seed`` is a single global seed feeding both policy and
  env, so the agent/domain split can't be plumbed end-to-end until the env
  takes a separate domain seed.
- **Track switching** — choosing/rotating tracks deterministically from the
  domain stream is part of the same future ``place()`` surface.
- **rliable IQM aggregation** — the manager produces the K replicate seeds and
  the fixed eval suite that feed an rliable IQM, but the aggregation itself
  (and the Optuna objective that maximizes it) lives in analysis code, not
  here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator, SeedSequence


# Fixed role order. Append new roles at the END only: spawn keys for earlier
# roles are positional, so inserting a role in the middle would silently
# re-derive every downstream stream and break reproducibility of past runs.
_ROLE_EVAL = 0
_ROLE_SAMPLER = 1
_ROLE_REPLICATE = 2
_N_ROLES = 3

# Sub-stream order within a single replicate (agent first, then domain).
_SUB_AGENT = 0
_SUB_DOMAIN = 1


def _as_seed_int(seq: "SeedSequence") -> int:
    """Collapse a ``SeedSequence`` into a 32-bit int seed.

    Most libraries (SB3, Optuna, ``random.seed``, ``np.random.seed``) want a
    plain integer in ``[0, 2**32)``. ``generate_state`` runs the same mixing
    used by ``default_rng``, so the int is a faithful, independent draw from
    the stream — not just the raw spawn key.
    """
    return int(seq.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class ReplicateSeeds:
    """The two independent sub-seeds for one training replicate.

    ``agent`` and ``domain`` are derived from *independent* spawns of the same
    replicate stream, so seeding the policy from ``agent`` and the scenario
    randomizer from ``domain`` keeps the two sources of stochasticity from
    perturbing each other.

    ``agent`` is ready to use today (feed it to ``ExperimentConfig.seed`` /
    SB3). ``domain`` is derived but currently has no consumer — see the
    module ``#FUTURE`` notes.
    """

    index: int
    agent: int
    domain: int
    _agent_seq: "SeedSequence"
    _domain_seq: "SeedSequence"

    def agent_seed_sequence(self) -> "SeedSequence":
        """The raw agent ``SeedSequence`` (for callers wanting a Generator)."""
        return self._agent_seq

    def domain_seed_sequence(self) -> "SeedSequence":
        """The raw domain ``SeedSequence``. #FUTURE: feed to ``place()``."""
        return self._domain_seq

    def domain_rng(self) -> "Generator":
        """A fresh NumPy ``Generator`` for *training* scenario randomization.

        #FUTURE — the placement/track randomizer that would consume this does
        not exist yet. Provided now so wiring it later is a one-liner.
        """
        return np.random.default_rng(self._domain_seq)


class SeedManager:
    """Turn one master seed into all the pipeline's independent seed streams.

    Pure and deterministic: the same ``master_seed`` (and the same
    ``n_eval_seeds`` / ``n_eval_scenarios`` / ``n_replicates``) reproduces
    every derived seed exactly. Access is *named and lazy* — ask for
    "the agent seed for replicate k" or "the eval suite seeds", never juggle
    raw integers.

    Parameters
    ----------
    master_seed:
        The single number you record. Everything else is derived from it.
    n_eval_seeds:
        How many eval seeds the eval stream spawns. Default 5.
    n_eval_scenarios:
        Scenarios placed per eval seed (the ``M`` in ``place(eval_seed, i)``).
        The pooled fixed suite has ``n_eval_seeds * n_eval_scenarios``
        scenarios. #FUTURE consumer (``place``); default 4.
    n_replicates:
        How many training replicates (the ``K`` runs you loop over to read
        pipeline variance). Default 5.
    """

    def __init__(
        self,
        master_seed: int,
        *,
        n_eval_seeds: int = 5,
        n_eval_scenarios: int = 4,
        n_replicates: int = 5,
    ) -> None:
        if n_eval_seeds < 1 or n_eval_scenarios < 1 or n_replicates < 1:
            raise ValueError("n_eval_seeds, n_eval_scenarios, n_replicates must be >= 1")
        self._master_seed = int(master_seed)
        self._n_eval_seeds = int(n_eval_seeds)
        self._n_eval_scenarios = int(n_eval_scenarios)
        self._n_replicates = int(n_replicates)

        root = np.random.SeedSequence(self._master_seed)
        # Fixed role order — see _ROLE_* constants. spawn() gives positional,
        # independent children; appending roles later leaves these unchanged.
        roles = root.spawn(_N_ROLES)
        self._eval_root = roles[_ROLE_EVAL]
        self._sampler_root = roles[_ROLE_SAMPLER]
        self._replicate_root = roles[_ROLE_REPLICATE]

        # Eager spawn (SeedSequences are cheap), cached for stable named
        # access. spawn() is *stateful* — it advances the parent's child
        # counter on every call — so we spawn each stream exactly once here
        # and cache the result. Re-spawning lazily on each accessor would hand
        # out a different child every time and break purity.
        self._eval_seqs = list(self._eval_root.spawn(self._n_eval_seeds))
        replicate_seqs = self._replicate_root.spawn(self._n_replicates)
        # Each replicate splits into two independent sub-streams: agent ⊥
        # domain. Spawn them now so repeated replicate(k) calls are pure.
        self._replicates: List[ReplicateSeeds] = []
        for k, rep_seq in enumerate(replicate_seqs):
            agent_seq, domain_seq = rep_seq.spawn(2)
            self._replicates.append(
                ReplicateSeeds(
                    index=k,
                    agent=_as_seed_int(agent_seq),
                    domain=_as_seed_int(domain_seq),
                    _agent_seq=agent_seq,
                    _domain_seq=domain_seq,
                )
            )

        # Built once, never regenerated.
        self._eval_suite: Optional[Tuple[Any, ...]] = None

    # ----------------------------- properties ----------------------------- #

    @property
    def master_seed(self) -> int:
        return self._master_seed

    @property
    def n_eval_seeds(self) -> int:
        return self._n_eval_seeds

    @property
    def n_eval_scenarios(self) -> int:
        return self._n_eval_scenarios

    @property
    def n_replicates(self) -> int:
        return self._n_replicates

    # ------------------------------- eval --------------------------------- #

    def eval_seeds(self) -> List[int]:
        """The fixed eval seeds (one per eval-stream spawn).

        Identical across every trial and replicate — that's the point: the
        evaluation suite must not move between runs being compared.
        """
        return [_as_seed_int(s) for s in self._eval_seqs]

    def build_eval_suite(
        self, place: Callable[[int, int], Any]
    ) -> Tuple[Any, ...]:
        """Build (once) and return the pooled, fixed evaluation suite.

        For each eval seed, ``n_eval_scenarios`` scenarios are placed via
        ``place(eval_seed, scenario_index)`` and pooled into a single
        heterogeneous suite. The result is cached, so repeated calls return
        the *same* suite object — it is built one time and never regenerated.

        Parameters
        ----------
        place:
            ``place(eval_seed: int, scenario_index: int) -> scenario``. The
            scenario type is whatever the (future) randomizer returns.

        #FUTURE — no ``place`` implementation exists in the pipeline yet
        (no seed-keyed obstacle/track/start-pose randomizer). This method is
        the integration point: the day ``place`` lands, the fixed suite is
        one call away. The seed *grid* it will consume is already available
        today via :meth:`eval_seeds` × ``range(n_eval_scenarios)``.
        """
        if self._eval_suite is None:
            suite: List[Any] = []
            for eval_seed in self.eval_seeds():
                for scenario_index in range(self._n_eval_scenarios):
                    suite.append(place(eval_seed, scenario_index))
            self._eval_suite = tuple(suite)
        return self._eval_suite

    # ------------------------------ sampler ------------------------------- #

    def sampler_seed(self) -> int:
        """Seed for Optuna's ``TPESampler`` — makes the search reproducible.

        Replaces the previous ``base.seed + worker_idx`` scheme: derive the
        sampler seed from the master here instead of offsetting a user seed by
        the worker index (adjacent integers are not guaranteed independent).
        """
        return _as_seed_int(self._sampler_root)

    # ---------------------------- replicates ------------------------------ #

    def replicate(self, k: int) -> ReplicateSeeds:
        """The ``(agent ⊥ domain)`` seeds for training replicate ``k``."""
        if not 0 <= k < self._n_replicates:
            raise IndexError(
                f"replicate index {k} out of range [0, {self._n_replicates})"
            )
        return self._replicates[k]

    def replicates(self) -> List[ReplicateSeeds]:
        """All ``n_replicates`` replicate seed pairs, in order."""
        return list(self._replicates)

    def agent_seed(self, k: int) -> int:
        """Shortcut for ``replicate(k).agent`` — the policy/rollout seed."""
        return self.replicate(k).agent

    def domain_seed(self, k: int) -> int:
        """Shortcut for ``replicate(k).domain``. #FUTURE consumer."""
        return self.replicate(k).domain

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SeedManager(master_seed={self._master_seed}, "
            f"n_eval_seeds={self._n_eval_seeds}, "
            f"n_eval_scenarios={self._n_eval_scenarios}, "
            f"n_replicates={self._n_replicates})"
        )
