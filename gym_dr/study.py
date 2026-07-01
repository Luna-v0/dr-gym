"""``Study`` — one interface for training *and* hyperparameter search.

A study is *always* defined over a hyperparameter space (``params``). A plain
training run is simply a study whose hyperparameters are all
:class:`~gym_dr.search.Fixed` — there is no separate ``train`` vs ``study`` API::

    from gym_dr import Study, Float

    # single run — every hyperparameter fixed
    Study(experiment, params={"trainer.kwargs.learning_rate": 3e-4}).run()

    # HPO — one dimension searched (same interface, add n_trials)
    Study(experiment,
          params={"trainer.kwargs.learning_rate": Float(1e-5, 1e-3, log=True)},
          master_seed=42, n_trials=40, n_replicates=3).run()

Reproducibility from one number
-------------------------------
Everything stochastic derives from ``master_seed`` via
:class:`~gym_dr.seeding.SeedManager`: replicate ``k`` trains from
``replicate(k).agent`` (with ``replicate(k).domain`` reserved for env
randomization), and the HPO sampler is seeded from the same root. ``master_seed``
is the single source of truth — it overrides ``ExperimentConfig.seed``.

Facade over the container orchestration
---------------------------------------
``Study.run()`` is *mode-dispatched* exactly like the previous ``train``/``study``
entrypoints, and delegates to the proven host/container/worker machinery in
``gym_dr.app`` (Docker spawn, runtime world-rotation, crash recovery, Optuna
workers) rather than reimplementing it:

- **HPO worker** (``GYM_DR_WORKER``): pull trials from the shared Optuna study.
- **In container** (``GYM_DR_IN_CONTAINER``): run exactly one replicate/chunk;
  the container applies its ``CHUNK_NAME``/``SEED`` env overrides itself.
- **Host, single run**: loop ``n_replicates`` — one container per replicate,
  each seeded from ``SeedManager``.
- **Host, search space**: spawn ``n_parallel`` Optuna workers over ``n_trials``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional

from gym_dr.config import ExperimentConfig
from gym_dr.search import Hyperparam, SearchSpace
from gym_dr.seeding import SeedManager


class _ImperativeSearch:
    """Adapter wrapping a legacy imperative ``search_space(trial) -> overrides``
    callable so a :class:`Study` treats it as a (never-single-run) search space.

    Lets HPO experiments pass their existing Optuna-style ``search_space``
    function straight to ``Study(experiment, params=search_space, ...)`` — the
    declarative :class:`~gym_dr.search.SearchSpace` is the clean alternative.
    """

    __slots__ = ("_fn",)

    def __init__(self, fn) -> None:
        self._fn = fn

    @property
    def is_single_run(self) -> bool:
        return False

    def overrides(self, trial: Any) -> "dict[str, Any]":
        return self._fn(trial)

    def fixed_overrides(self) -> "dict[str, Any]":
        return {}


@dataclass
class StudyResult:
    """Outcome of :meth:`Study.run`.

    ``run_paths`` holds one entry per replicate for a single-run study (the host
    path of each run's ``latest_model.zip``); for an HPO study the best params
    live in the Optuna storage and ``exit_code`` is the worker-spawn return code.
    """

    run_paths: List[Any] = field(default_factory=list)
    n_trials: int = 1
    n_replicates: int = 1
    best_params: dict = field(default_factory=dict)
    best_value: float = float("nan")
    exit_code: int = 0


class Study:
    """A reproducible study over an :class:`ExperimentConfig` and a hyperparameter
    space. See the module docstring for the interface and dispatch model.

    Parameters
    ----------
    experiment:
        The base experiment to train/search over.
    params:
        A :class:`~gym_dr.search.SearchSpace`, or a mapping of dotted
        ``ExperimentConfig`` keys to :class:`~gym_dr.search.Hyperparam` (or bare
        constants, coerced to ``Fixed``). All-``Fixed`` ⇒ a single training run;
        any search distribution ⇒ HPO. Empty/omitted ⇒ a single run of the
        experiment as-authored.
    master_seed:
        The one recorded seed; every stochastic stream derives from it.
    n_replicates:
        Independent training repeats (from independent agent seeds) — the runs
        used to read pipeline variance for rliable-style analysis.
    n_trials:
        HPO trials (ignored for a single run; forced to 1 there).
    n_parallel:
        Concurrent HPO worker containers.
    study_name / storage / image_tag / extra_env:
        HPO plumbing (Optuna study name + storage URL, Docker image, extra
        container env). Default study_name is ``experiment.name``.
    """

    def __init__(
        self,
        experiment: ExperimentConfig,
        params: "SearchSpace | Mapping[str, Hyperparam | Any] | Callable[[Any], dict] | None" = None,
        *,
        master_seed: int = 0,
        n_replicates: int = 1,
        n_trials: int = 1,
        n_parallel: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        image_tag: Optional[str] = None,
        extra_env: "Optional[Mapping[str, str]]" = None,
    ) -> None:
        if not isinstance(experiment, ExperimentConfig):
            raise TypeError(
                f"Study needs an ExperimentConfig, got {type(experiment).__name__}"
            )
        self.experiment = experiment
        if isinstance(params, SearchSpace):
            self.space: Any = params
        elif callable(params) and not isinstance(params, Mapping):
            # Legacy imperative search_space(trial) -> overrides dict.
            self.space = _ImperativeSearch(params)
        else:
            self.space = SearchSpace(params)
        self.master_seed = int(master_seed)
        self.n_replicates = max(1, int(n_replicates))
        self.n_trials = max(1, int(n_trials))
        self.n_parallel = max(1, int(n_parallel))
        self.study_name = study_name or experiment.name
        self.storage = storage
        self.image_tag = image_tag
        self.extra_env = dict(extra_env or {})
        self.seeds = SeedManager(self.master_seed, n_replicates=self.n_replicates)

    @property
    def is_single_run(self) -> bool:
        """True when there is nothing to search (every hyperparameter is Fixed)."""
        return self.space.is_single_run

    # --------------------------------------------------------------- dispatch
    def run(self) -> StudyResult:
        """Run the study — mode-dispatched (see the module docstring)."""
        if os.getenv("GYM_DR_WORKER"):
            return self._run_worker()
        if os.getenv("GYM_DR_IN_CONTAINER"):
            return self._run_in_container()
        if self.is_single_run:
            return self._run_training_host()
        return self._run_hpo_host()

    # ------------------------------------------------------------- internals
    def _search_adapter(self):
        """An Optuna ``objective``-side ``search_space(trial) -> overrides`` fn,
        as ``gym_dr.app.study`` / ``hpo.build_objective`` expect."""
        space = self.space

        def search_space(trial: Any) -> "dict[str, Any]":
            return space.overrides(trial)

        return search_space

    def _run_worker(self) -> StudyResult:
        from gym_dr.app import study as _study_entry

        _study_entry(
            self.experiment,
            self._search_adapter(),
            study_name=self.study_name,
            n_trials=self.n_trials,
            n_parallel=self.n_parallel,
            storage=self.storage,
            image_tag=self.image_tag,
            extra_env=self.extra_env,
        )
        return StudyResult(n_trials=self.n_trials, n_replicates=self.n_replicates)

    def _run_in_container(self) -> StudyResult:
        # One replicate/chunk only — the container applies CHUNK_NAME/SEED/
        # CHUNK_STEPS env overrides itself (do NOT loop replicates here).
        from gym_dr.app import train as _train_entry

        result = _train_entry(self.experiment)
        return StudyResult(run_paths=[result], n_trials=1, n_replicates=1)

    def _run_training_host(self) -> StudyResult:
        from gym_dr.app import train as _train_entry

        fixed = self.space.fixed_overrides()
        paths: List[Any] = []
        for k in range(self.n_replicates):
            seed = self.seeds.replicate(k).agent
            name = (
                self.experiment.name
                if self.n_replicates == 1
                else f"{self.experiment.name}_rep{k}"
            )
            exp = self.experiment.with_overrides(name=name, seed=seed, **fixed)
            paths.append(_train_entry(exp))
        return StudyResult(
            run_paths=paths,
            n_trials=1,
            n_replicates=self.n_replicates,
            best_params=dict(fixed),
        )

    def _run_hpo_host(self) -> StudyResult:
        from gym_dr.app import study as _study_entry

        # Seed the Optuna sampler from the master seed so the search is
        # reproducible (app.study/hpo derive the TPE seed from experiment.seed;
        # master_seed is authoritative). #FUTURE: derive per-worker independence
        # via SeedManager.sampler_seed() through the worker env.
        base = self.experiment
        if base.seed is None:
            base = base.with_overrides(seed=self.master_seed)
        rc = _study_entry(
            base,
            self._search_adapter(),
            study_name=self.study_name,
            n_trials=self.n_trials,
            n_parallel=self.n_parallel,
            storage=self.storage,
            image_tag=self.image_tag,
            extra_env=self.extra_env,
        )
        return StudyResult(
            n_trials=self.n_trials,
            n_replicates=self.n_replicates,
            exit_code=int(rc or 0),
        )

    def __repr__(self) -> str:
        mode = "single-run" if self.is_single_run else f"hpo[{self.n_trials} trials]"
        return (
            f"Study({self.experiment.name!r}, {mode}, "
            f"n_replicates={self.n_replicates}, master_seed={self.master_seed})"
        )
