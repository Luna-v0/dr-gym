"""Value specs for domain randomization — ``Range`` (continuous) and ``Choice``
(discrete list).

A DR knob is a ``ParamSpec``: a ``Range(low, high)`` sampled uniformly each episode,
a ``Choice([...])`` that picks one value each episode, or a bare scalar treated as a
constant. This replaces the old flat ``*_std`` scalars (which conflated "the value"
with "the ADR ceiling") — now the bounds are explicit, ADR widens a ``Range`` toward
its ``high`` (``gym_dr.domain_randomization.ADRController``), and HPO can sweep the
bounds. Specs are frozen + hashable so configs still serialise/round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Range:
    """Continuous range; ``sample`` draws ``U[low, high]`` (per episode)."""
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"Range high {self.high} < low {self.low}")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class Choice:
    """Discrete set (the "list" spec); ``sample`` picks one value uniformly."""
    values: Tuple[float, ...]

    def __init__(self, values: Sequence[float]) -> None:
        vals = tuple(float(v) for v in values)
        if not vals:
            raise ValueError("Choice needs at least one value")
        object.__setattr__(self, "values", vals)

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.values[int(rng.integers(len(self.values)))])


# A DR knob: a Range, a Choice, or a bare scalar (constant).
ParamSpec = Union[Range, Choice, float, int]


def sample_spec(spec: ParamSpec, rng: np.random.Generator) -> float:
    """Draw one value from *spec* (Range/Choice sampled; scalar returned as-is)."""
    if isinstance(spec, (Range, Choice)):
        return spec.sample(rng)
    return float(spec)


def spec_bounds(spec: ParamSpec) -> Tuple[float, float]:
    """``(low, high)`` envelope of *spec* — for ADR widening and serialisation."""
    if isinstance(spec, Range):
        return (spec.low, spec.high)
    if isinstance(spec, Choice):
        return (min(spec.values), max(spec.values))
    return (float(spec), float(spec))


def is_randomized(spec: ParamSpec) -> bool:
    """True if *spec* actually varies (a non-degenerate Range or multi-value Choice)."""
    if isinstance(spec, Range):
        return spec.high > spec.low
    if isinstance(spec, Choice):
        return len(set(spec.values)) > 1
    return False


def spec_to_dict(spec: ParamSpec) -> Union[dict, float]:
    """Serialise a spec (scalars stay scalars; Range/Choice tagged by ``type``)."""
    if isinstance(spec, Range):
        return {"type": "Range", "low": spec.low, "high": spec.high}
    if isinstance(spec, Choice):
        return {"type": "Choice", "values": list(spec.values)}
    return float(spec)


__all__ = [
    "Range", "Choice", "ParamSpec",
    "sample_spec", "spec_bounds", "is_randomized", "spec_to_dict",
]
