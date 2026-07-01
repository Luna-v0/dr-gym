"""Pluggable early-stopping strategies (Strategy pattern).

An :class:`EarlyStopStrategy` decides, from an evaluation round's metrics,
whether the current training chunk should stop early. Strategies are **frozen
dataclasses** so they hash and serialise like the rest of the config and can be
swept by HPO via dotted keys (e.g. ``training.early_stop.threshold``).

Separation of concerns
----------------------
- The **strategy** (frozen, pure) answers one question: *does this single eval
  round qualify?* — :meth:`EarlyStopStrategy.met`.
- The **controller** (:class:`EarlyStopController`, stateful) owns the *streak*:
  it requires ``patience`` consecutive qualifying rounds and resets the streak
  when a round fails or when a new chunk starts. The trainer callback owns one
  controller and calls :meth:`EarlyStopController.update` after each eval.

This reproduces the historical behaviour exactly — the old
``early_stop_max_offtrack_rate`` + ``early_stop_patience`` fields are now
``OfftrackRate(max_offtrack_rate=..., patience=...)`` — while making reward-,
completion-, and arbitrary-metric-based stopping first-class and composable.

The metrics mapping
-------------------
``met`` receives the aggregate eval metrics the pipeline already produces (see
``gym_dr.trainers.base._agg_eval``): ``clean_completion_rate``,
``completion_rate``, ``mean_progress``, ``mean_reward``, ``offtrack_rate`` — plus
any extra keys a custom trainer logs (e.g. ``mean_cost``). Strategies read by
key, so a strategy is decoupled from *how* the metric was computed.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

Metrics = Mapping[str, float]


class EarlyStopStrategy(ABC):
    """Interface for early-stop decisions. Subclasses are frozen dataclasses.

    A strategy carries its own ``patience`` (consecutive qualifying eval rounds
    required before stopping). Only :meth:`met` is abstract.
    """

    #: Consecutive qualifying eval rounds required before stopping. Subclasses
    #: redeclare this as a dataclass field so it is sweepable/serialisable.
    patience: int = 1

    @abstractmethod
    def met(self, metrics: Metrics) -> bool:
        """Does *this* eval round satisfy the stopping condition?

        Pure and side-effect-free. Streak/patience is the controller's job.
        """

    def describe(self) -> str:
        """One-line human summary (for status JSON and logs)."""
        return f"{type(self).__name__}(patience={self.patience})"


@dataclass(frozen=True)
class OfftrackRate(EarlyStopStrategy):
    """Track-mastery stop: qualify when the eval off-track rate is low enough.

    This is the historical default. ``OfftrackRate(max_offtrack_rate=0.0,
    patience=1)`` reproduces the old strict behaviour (stop the first eval round
    where the car completes every episode without leaving the track).
    """

    max_offtrack_rate: float = 0.0
    patience: int = 1

    def met(self, metrics: Metrics) -> bool:
        return metrics.get("offtrack_rate", 1.0) <= self.max_offtrack_rate

    def describe(self) -> str:
        return f"OfftrackRate(<= {self.max_offtrack_rate}, patience={self.patience})"


@dataclass(frozen=True)
class MetricThreshold(EarlyStopStrategy):
    """Generic threshold on any eval metric.

    ``mode="max"`` qualifies when ``metrics[metric] >= threshold`` (the metric
    should be *at least* the threshold — reward, completion rate); ``mode="min"``
    qualifies when ``metrics[metric] <= threshold`` (the metric should be *at
    most* the threshold — off-track rate, cost).
    """

    metric: str
    threshold: float
    mode: str = "max"
    patience: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")

    def met(self, metrics: Metrics) -> bool:
        if self.metric not in metrics:
            return False
        value = float(metrics[self.metric])
        if self.mode == "max":
            return value >= self.threshold
        return value <= self.threshold

    def describe(self) -> str:
        op = ">=" if self.mode == "max" else "<="
        return f"MetricThreshold({self.metric} {op} {self.threshold}, patience={self.patience})"


@dataclass(frozen=True)
class RewardThreshold(EarlyStopStrategy):
    """Convenience: stop once the (eval) reward reaches ``min_reward``.

    ``metric`` defaults to ``mean_reward`` (the aggregate eval reward the
    pipeline reports); override to key off a different reward-like metric.
    """

    min_reward: float
    metric: str = "mean_reward"
    patience: int = 1

    def met(self, metrics: Metrics) -> bool:
        return metrics.get(self.metric, -math.inf) >= self.min_reward

    def describe(self) -> str:
        return f"RewardThreshold({self.metric} >= {self.min_reward}, patience={self.patience})"


@dataclass(frozen=True)
class CleanCompletion(EarlyStopStrategy):
    """Convenience: stop once the headline clean-completion rate is high enough.

    ``CleanCompletion(min_rate=1.0, patience=2)`` — the car must complete every
    eval episode cleanly (finish the lap without leaving the track) for two
    consecutive eval rounds. Matches the project's success criterion
    (``docs/eval-protocol.md``).
    """

    min_rate: float = 1.0
    patience: int = 2

    def met(self, metrics: Metrics) -> bool:
        return metrics.get("clean_completion_rate", 0.0) >= self.min_rate

    def describe(self) -> str:
        return f"CleanCompletion(>= {self.min_rate}, patience={self.patience})"


@dataclass(frozen=True)
class AllOf(EarlyStopStrategy):
    """Composite: qualify only when *every* sub-strategy qualifies this round.

    Patience is this composite's own field; each round is "met" iff all children
    are met (children's own patience is ignored — the composite owns the streak).
    Example — stop when the car both drives clean *and* keeps cost under budget::

        AllOf((CleanCompletion(1.0), MetricThreshold("mean_cost", 10.0, "min")), patience=2)
    """

    strategies: tuple[EarlyStopStrategy, ...]
    patience: int = 1

    def met(self, metrics: Metrics) -> bool:
        return all(s.met(metrics) for s in self.strategies)

    def describe(self) -> str:
        inner = " AND ".join(s.describe() for s in self.strategies)
        return f"AllOf([{inner}], patience={self.patience})"


@dataclass(frozen=True)
class AnyOf(EarlyStopStrategy):
    """Composite: qualify when *any* sub-strategy qualifies this round."""

    strategies: tuple[EarlyStopStrategy, ...]
    patience: int = 1

    def met(self, metrics: Metrics) -> bool:
        return any(s.met(metrics) for s in self.strategies)

    def describe(self) -> str:
        inner = " OR ".join(s.describe() for s in self.strategies)
        return f"AnyOf([{inner}], patience={self.patience})"


class EarlyStopController:
    """Stateful streak tracker wrapping an :class:`EarlyStopStrategy`.

    Owned by the trainer's eval callback. After each eval round, call
    :meth:`update` with the aggregate metrics; it returns ``True`` when the
    strategy has qualified for ``patience`` consecutive rounds. Call
    :meth:`reset` at the start of each training chunk so mastering one track does
    not pre-credit the next (matching the historical per-chunk streak reset).

    A ``None`` strategy disables early stopping — :meth:`update` always returns
    ``False`` — so the callback can hold a controller unconditionally.
    """

    __slots__ = ("_strategy", "_streak")

    def __init__(self, strategy: EarlyStopStrategy | None) -> None:
        self._strategy = strategy
        self._streak = 0

    @property
    def strategy(self) -> EarlyStopStrategy | None:
        return self._strategy

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def enabled(self) -> bool:
        return self._strategy is not None

    def update(self, metrics: Metrics) -> bool:
        """Record one eval round; return ``True`` iff training should stop now."""
        if self._strategy is None:
            return False
        if self._strategy.met(metrics):
            self._streak += 1
            return self._streak >= max(1, self._strategy.patience)
        self._streak = 0
        return False

    def reset(self) -> None:
        """Zero the streak — call at each new training chunk."""
        self._streak = 0
