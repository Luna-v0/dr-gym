"""Unified hyperparameter space for :class:`gym_dr.study.Study`.

A ``Study`` is *always* defined over hyperparameters. A **single training run is
a Study whose hyperparameters are all** :class:`Fixed` — so there is one interface
for "just train" and "search over hyperparameters", the only difference being
whether any dimension is a search distribution.

Vocabulary
----------
- :class:`Fixed` — a constant. A bare Python value in a :class:`SearchSpace` is
  coerced to ``Fixed`` automatically.
- :class:`Float`, :class:`Int`, :class:`Categorical` — search distributions that
  know how to draw themselves from an Optuna trial.
- :class:`SearchSpace` — maps dotted ``ExperimentConfig`` keys (the same keys
  ``ExperimentConfig.with_overrides`` understands, e.g. ``trainer.kwargs.learning_rate``)
  to hyperparameters, reports whether it is a single run, and compiles to a
  per-trial overrides dict.

These are HPO hyperparameters — distinct from ``gym_dr.randomization.{Range, Choice}``,
which randomize the *environment* (domain randomization), not the search.

The module never imports Optuna: a hyperparameter only calls ``trial.suggest_*`` on
whatever trial object it is handed, so it is trivially testable with a fake trial.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Tuple


class Hyperparam(ABC):
    """A single tunable value: either :class:`Fixed` or a search distribution."""

    @abstractmethod
    def suggest(self, trial: Any, name: str) -> Any:
        """Return this hyperparameter's value for ``trial`` (Optuna ``Trial``).

        ``Fixed`` ignores the trial and returns its constant; distributions call
        the matching ``trial.suggest_*(name, ...)``.
        """

    @property
    def is_fixed(self) -> bool:
        return False

    def describe(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class Fixed(Hyperparam):
    """A constant hyperparameter — the single-run case."""

    value: Any

    def suggest(self, trial: Any, name: str) -> Any:
        return self.value

    @property
    def is_fixed(self) -> bool:
        return True

    def describe(self) -> str:
        return f"Fixed({self.value!r})"


@dataclass(frozen=True)
class Float(Hyperparam):
    """A continuous search dimension → ``trial.suggest_float``.

    ``log=True`` samples on a log scale (right for learning rates); ``step`` (if
    set) quantises the range. ``log`` and ``step`` are mutually exclusive, as in
    Optuna.
    """

    low: float
    high: float
    log: bool = False
    step: float | None = None

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Float low ({self.low}) > high ({self.high})")
        if self.log and self.step is not None:
            raise ValueError("Float cannot set both log=True and step")

    def suggest(self, trial: Any, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=self.log, step=self.step)

    def describe(self) -> str:
        scale = " log" if self.log else ""
        return f"Float[{self.low}, {self.high}{scale}]"


@dataclass(frozen=True)
class Int(Hyperparam):
    """A discrete integer search dimension → ``trial.suggest_int``."""

    low: int
    high: int
    log: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Int low ({self.low}) > high ({self.high})")
        if self.log and self.step != 1:
            raise ValueError("Int cannot set both log=True and step != 1")

    def suggest(self, trial: Any, name: str) -> int:
        return trial.suggest_int(name, self.low, self.high, log=self.log, step=self.step)

    def describe(self) -> str:
        scale = " log" if self.log else ""
        return f"Int[{self.low}, {self.high}{scale}]"


@dataclass(frozen=True)
class Categorical(Hyperparam):
    """A categorical search dimension → ``trial.suggest_categorical``.

    ``choices`` is stored as a tuple so the hyperparameter stays hashable/frozen;
    a list is coerced. Optuna requires the choices be JSON-serialisable scalars
    (str/int/float/bool/None).
    """

    choices: Tuple[Any, ...]

    def __post_init__(self) -> None:
        if isinstance(self.choices, (list, tuple)):
            if len(self.choices) == 0:
                raise ValueError("Categorical needs at least one choice")
            object.__setattr__(self, "choices", tuple(self.choices))
        else:
            raise TypeError("Categorical choices must be a list or tuple")

    def suggest(self, trial: Any, name: str) -> Any:
        return trial.suggest_categorical(name, list(self.choices))

    def describe(self) -> str:
        return f"Categorical{list(self.choices)!r}"


def as_hyperparam(value: "Hyperparam | Any") -> Hyperparam:
    """Coerce a bare value to :class:`Fixed`; pass a :class:`Hyperparam` through."""
    return value if isinstance(value, Hyperparam) else Fixed(value)


class SearchSpace:
    """A named collection of hyperparameters keyed by dotted config path.

    ``SearchSpace({"trainer.kwargs.learning_rate": Float(1e-5, 1e-3, log=True),
    "trainer.kwargs.gamma": 0.99})`` mixes a searched and a fixed dimension. A
    space whose entries are *all* :class:`Fixed` :attr:`is_single_run`.
    """

    def __init__(self, params: "Mapping[str, Hyperparam | Any] | None" = None) -> None:
        params = params or {}
        self._params: "dict[str, Hyperparam]" = {
            key: as_hyperparam(val) for key, val in params.items()
        }

    @property
    def is_single_run(self) -> bool:
        """True when there is nothing to search — every dimension is Fixed."""
        return all(p.is_fixed for p in self._params.values())

    @property
    def search_dims(self) -> "list[str]":
        """Keys that are actual search distributions (not Fixed)."""
        return [k for k, p in self._params.items() if not p.is_fixed]

    def overrides(self, trial: Any) -> "dict[str, Any]":
        """Per-trial dotted-key overrides for ``ExperimentConfig.with_overrides``.

        Fixed keys contribute their constant; search keys draw from ``trial``.
        The keys are the config paths, so the result plugs straight into
        ``with_overrides(**space.overrides(trial))``.
        """
        return {key: p.suggest(trial, key) for key, p in self._params.items()}

    def fixed_overrides(self) -> "dict[str, Any]":
        """The Fixed-only overrides — used by the single-run path (no trial)."""
        return {key: p.value for key, p in self._params.items() if isinstance(p, Fixed)}

    def describe(self) -> "dict[str, str]":
        return {key: p.describe() for key, p in self._params.items()}

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __getitem__(self, key: str) -> Hyperparam:
        return self._params[key]

    def __contains__(self, key: str) -> bool:
        return key in self._params

    def __repr__(self) -> str:
        kind = "single-run" if self.is_single_run else f"search[{len(self.search_dims)}]"
        return f"SearchSpace({kind}, {len(self._params)} params)"
