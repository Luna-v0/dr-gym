"""Composable ``Stage`` pipeline — the explicit MDP data-path primitive.

A :class:`Stage` wraps a function ``I -> O``. Stages compose with ``>>`` into a
new stage whose input is the first's input and output is the last's output. The
type checker enforces the seam: one stage's output type must match the next
stage's input type.

Why this exists
---------------
The orchestrator must stay **algorithm-agnostic** (no Stable-Baselines lock-in —
see ``docs/decisions/0004-orchestrator-refactor-interface.md``). ``Stage`` is the
shared, framework-free vocabulary for describing the observation → encode →
policy → action flow of every experiment. A researcher composes their MDP
pipeline explicitly and readably::

    # camera end-to-end
    pipeline = adr_input >> encode_cnn >> policy >> adr_output   # Obs -> Action
    action   = pipeline(obs)

    # feature vector
    pipeline = features >> mlp_policy >> adr_output

    # asymmetric frozen-CNN (decoupled obs-net vs policy)
    pipeline = frozen_cnn >> feature_noise >> asym_split >> asym_policy >> adr_output

Two roles, one primitive (the "hybrid" decision, ADR-0004/0005)
--------------------------------------------------------------
1. **Declarative assembly.** Stages *describe* how an experiment's env-wrappers,
   encoder and policy are wired. A fast, vectorised trainer adapter (e.g. the SB3
   adapter) reads that description and drives its own optimised rollout loop — so
   the per-step cost stays in batched tensor code, not a Python call chain.
2. **Literal data-path.** For the light paths — single-observation inference,
   ONNX export, on-car deployment, and the dissertation's decoupled
   obs-encoder → policy evaluation — the very same composed ``Stage`` is *called*
   directly: ``action = pipeline(obs)``.

Because a composed pipeline flattens and remembers its sub-stages, it is
inspectable (``len(p)``, ``list(p)``, ``repr(p)``) — training loops can be
printed and audited, which is the whole point of making them explicit.

The primitive is deliberately tiny and dependency-free (no torch, no gym). Neural
stages wrap a ``torch.nn.Module`` so a batched adapter can run them efficiently,
but that is the caller's concern — ``Stage`` itself only knows ``I -> O``.
"""
from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

I = TypeVar("I")
O = TypeVar("O")
X = TypeVar("X")

# A stage boundary accepts either a Stage or a bare callable; bare callables are
# auto-wrapped on composition so ``Stage(f) >> g >> h`` reads naturally.
StageLike = "Stage[I, O] | Callable[[I], O]"


class Stage(Generic[I, O]):
    """A named, composable function ``I -> O``.

    Parameters
    ----------
    fn:
        The wrapped callable. Called with the stage's input; returns its output.
    name:
        Human-readable label used in ``repr`` and composed names. Defaults to
        the wrapped callable's ``__name__`` (or ``"stage"``).

    Compose with ``>>``::

        doubled = Stage(lambda x: x + 1) >> (lambda x: x * 2)
        doubled(3)      # (3 + 1) * 2 == 8
        doubled.name    # '<lambda>→<lambda>'  (or explicit names if given)

    A composed stage is a first-class :class:`Stage`; it also flattens and
    retains its constituent sub-stages, so it is introspectable::

        len(pipeline)          # number of leaf stages
        [s.name for s in pipeline]
    """

    __slots__ = ("_fn", "_name", "_sub_stages")

    def __init__(self, fn: Callable[[I], O], name: str | None = None) -> None:
        if not callable(fn):
            raise TypeError(f"Stage needs a callable, got {type(fn).__name__}")
        self._fn = fn
        self._name = name or getattr(fn, "__name__", None) or "stage"
        # A leaf stage's only sub-stage is itself; composition replaces this with
        # the flattened list of both operands' sub-stages (see __rshift__).
        self._sub_stages: tuple["Stage", ...] = (self,)

    # ------------------------------------------------------------------ core
    def __call__(self, x: I) -> O:
        return self._fn(x)

    def __rshift__(self, nxt: "StageLike") -> "Stage[I, X]":
        """Compose: ``self >> nxt`` runs ``self`` then ``nxt``.

        ``nxt`` may be a :class:`Stage` or a bare callable (auto-wrapped). The
        result is a new :class:`Stage[I, X]` whose sub-stages are the flattened
        concatenation of both operands', so a long pipeline stays inspectable.
        """
        nxt_stage = as_stage(nxt)
        composed_name = f"{self._name}→{nxt_stage._name}"
        composed: "Stage[I, X]" = Stage(
            lambda x: nxt_stage(self(x)), name=composed_name
        )
        composed._sub_stages = self._sub_stages + nxt_stage._sub_stages
        return composed

    def __rrshift__(self, prev: Callable[[X], I]) -> "Stage[X, O]":
        """Allow ``bare_callable >> stage`` (left operand is a plain callable)."""
        return as_stage(prev) >> self

    # --------------------------------------------------------- introspection
    @property
    def name(self) -> str:
        return self._name

    @property
    def fn(self) -> Callable[[I], O]:
        """The wrapped callable (e.g. a ``torch.nn.Module`` for neural stages)."""
        return self._fn

    def rename(self, name: str) -> "Stage[I, O]":
        """Return a copy of this stage with a new display name."""
        s: "Stage[I, O]" = Stage(self._fn, name=name)
        s._sub_stages = self._sub_stages
        return s

    def __iter__(self) -> Iterator["Stage"]:
        return iter(self._sub_stages)

    def __len__(self) -> int:
        return len(self._sub_stages)

    def __repr__(self) -> str:
        if len(self._sub_stages) > 1:
            inner = " >> ".join(s._name for s in self._sub_stages)
            return f"Stage({inner})"
        return f"Stage({self._name})"


def as_stage(obj: "StageLike") -> "Stage[I, O]":
    """Coerce a :class:`Stage` or bare callable into a :class:`Stage` (idempotent)."""
    if isinstance(obj, Stage):
        return obj
    if callable(obj):
        return Stage(obj)
    raise TypeError(f"cannot make a Stage from {type(obj).__name__}")


def stage(
    fn: Callable[[I], O] | None = None, *, name: str | None = None
) -> "Stage[I, O] | Callable[[Callable[[I], O]], Stage[I, O]]":
    """Decorator turning a function into a :class:`Stage`.

    Usage::

        @stage
        def grayscale(obs): ...

        @stage(name="adr-input")
        def add_obs_noise(obs): ...
    """
    if fn is not None:
        return Stage(fn, name=name)

    def _wrap(f: Callable[[I], O]) -> "Stage[I, O]":
        return Stage(f, name=name)

    return _wrap


def identity(name: str = "identity") -> "Stage[I, I]":
    """A no-op stage — a neutral element for building pipelines conditionally."""
    return Stage(lambda x: x, name=name)


def compose(*stages: "StageLike") -> "Stage":
    """Left-to-right composition of two or more stages.

    ``compose(a, b, c)`` is ``as_stage(a) >> b >> c``. Handy when the stage list
    is built programmatically (e.g. from config) rather than written with ``>>``.
    """
    if not stages:
        raise ValueError("compose() needs at least one stage")
    head, *tail = stages
    result = as_stage(head)
    for nxt in tail:
        result = result >> nxt
    return result
