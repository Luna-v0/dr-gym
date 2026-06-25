"""World-scheduling strategies — *how* training moves between tracks.

The DeepRacer env can hot-swap its Gazebo track at runtime
(``DeepRacerEnv.set_world``; see ``docs/trace-contract.md`` on the hot-reload
system). A :class:`WorldStrategy` decides the *order* the policy trains across
worlds, and — separately — which worlds it is *evaluated* on. This is a strategy
pattern: the orchestrator and trainer depend only on the
:class:`WorldStrategy` interface, so new schedules (curriculum, randomised,
performance-adaptive, …) drop in without touching them.

Two strategies ship today:

- :class:`FixedWorlds` — the historical behaviour: one ordered list,
  repeated ``rotations`` times; evaluation runs on whatever world training is
  currently on (``evaluation_worlds()`` is empty).
- :class:`OrderedSplit` — train on one ordered list, **evaluate on a different
  ordered (held-out) list**. Training proceeds strictly in ``train_worlds``
  order; at each evaluation the policy is measured on every world in
  ``eval_worlds`` (track generalisation), then training resumes where it left
  off.

A strategy is pure data + plan: it yields the training chunk sequence and the
eval world order. The trainer owns the mechanics of swapping
(``Sb3Trainer``); the host orchestrator reads :meth:`first_world` to pick the
container's initial ``WORLD_NAME``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class WorldChunk:
    """One contiguous training segment: train ``steps`` timesteps on ``world``."""

    world: str
    steps: int


class WorldStrategy(ABC):
    """Interface every world schedule implements.

    Subclasses are frozen dataclasses (so they hash and serialise like the
    trainer config). Only :meth:`training_chunks` is required; the rest have
    sensible defaults.
    """

    @abstractmethod
    def training_chunks(self) -> List[WorldChunk]:
        """Ordered training segments. Chunk ``i`` trains ``steps`` timesteps on
        ``world``; the trainer hot-swaps to chunk ``i+1``'s world between them.
        """

    def evaluation_worlds(self) -> List[str]:
        """Ordered worlds to evaluate on. Empty (default) means "evaluate on the
        current training world" — the legacy single-env behaviour."""
        return []

    def first_world(self) -> str:
        """The world the container loads at startup (``WORLD_NAME``)."""
        chunks = self.training_chunks()
        if not chunks:
            raise ValueError(f"{type(self).__name__} produced no training chunks")
        return chunks[0].world

    @property
    def name(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class FixedWorlds(WorldStrategy):
    """Train through ``names`` in order, ``rotations`` times. The default.

    ``FixedWorlds(["A", "B"], chunk_steps=20_000, rotations=2)`` trains
    A→B→A→B, 20k steps each. ``evaluation_worlds()`` is empty, so evaluation
    uses the current training world (matching the original pipeline).
    """

    names: List[str] = field(default_factory=lambda: ["reinvent_base"])
    chunk_steps: int = 50_000
    rotations: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.names, str):
            object.__setattr__(self, "names", [self.names])

    def training_chunks(self) -> List[WorldChunk]:
        return [
            WorldChunk(w, self.chunk_steps)
            for _ in range(self.rotations)
            for w in self.names
        ]


@dataclass(frozen=True)
class OrderedSplit(WorldStrategy):
    """Train on one ordered list, evaluate on a different ordered (held-out) list.

    The canonical train/eval *track split* for measuring generalisation: the
    policy never trains on the eval worlds, so eval reward reflects transfer to
    unseen tracks.

    ``OrderedSplit(train_worlds=["A", "B", "C"], eval_worlds=["D", "E"],
    chunk_steps=20_000)`` trains A→B→C (20k each), and at every evaluation
    measures the policy on D then E, averaging them into the reported eval
    metric (per-world values are logged too). ``rotations`` repeats the train
    order (A→B→C→A→B→C…); the eval list is independent of rotations.
    """

    train_worlds: List[str] = field(default_factory=lambda: ["reinvent_base"])
    eval_worlds: List[str] = field(default_factory=list)
    chunk_steps: int = 50_000
    rotations: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.train_worlds, str):
            object.__setattr__(self, "train_worlds", [self.train_worlds])
        if isinstance(self.eval_worlds, str):
            object.__setattr__(self, "eval_worlds", [self.eval_worlds])
        if not self.train_worlds:
            raise ValueError("OrderedSplit needs at least one train world")

    def training_chunks(self) -> List[WorldChunk]:
        return [
            WorldChunk(w, self.chunk_steps)
            for _ in range(self.rotations)
            for w in self.train_worlds
        ]

    def evaluation_worlds(self) -> List[str]:
        return list(self.eval_worlds)


@dataclass(frozen=True)
class ACL(WorldStrategy):
    """Spaced-repetition curriculum: sample each chunk's track from an expanding
    window, favouring newer tracks but **always** keeping a chance of revisiting
    older ones — to fight the catastrophic forgetting a strict sequential
    curriculum suffers (see ``docs/open-questions.md`` Q7).

    The window of "unlocked" tracks grows by one every ``unlock_every`` chunks
    (track 0 from the start; track 1 after ``unlock_every`` chunks; ...). Within
    the unlocked window, track *j* is sampled with weight ``recency_weight ** j``
    (``recency_weight > 1`` ⇒ the most recently unlocked is heaviest), so older
    tracks keep a small, non-zero probability — exactly the "always a little bit
    of training on an older track" behaviour requested. Held-out evaluation uses
    ``eval_worlds`` (like :class:`OrderedSplit`).

    The chunk sequence is generated **deterministically** from ``seed`` (a fresh
    ``random.Random(seed)`` each call) so the host and in-container trainer agree
    on the same plan.

    NOTE: unlocking here is *schedule-based* (every ``unlock_every`` chunks) — an
    approximation of true **mastery-gated** progression (advance only once the
    current track is driven cleanly). Mastery-gating needs the trainer to feed
    the eval mastery signal back to the strategy at runtime; that is a follow-up
    that would reuse the early-stop mastery signal in
    ``gym_dr/trainers/sb3/callbacks.py``.
    """

    train_worlds: List[str] = field(default_factory=lambda: ["reinvent_base"])
    eval_worlds: List[str] = field(default_factory=list)
    chunk_steps: int = 50_000
    n_chunks: int = 20
    unlock_every: int = 3
    recency_weight: float = 2.0
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.train_worlds, str):
            object.__setattr__(self, "train_worlds", [self.train_worlds])
        if isinstance(self.eval_worlds, str):
            object.__setattr__(self, "eval_worlds", [self.eval_worlds])
        if not self.train_worlds:
            raise ValueError("ACL needs at least one train world")
        if self.n_chunks < 1:
            raise ValueError("n_chunks must be >= 1")
        if self.recency_weight <= 0:
            raise ValueError("recency_weight must be > 0")

    def training_chunks(self) -> List[WorldChunk]:
        import random

        rng = random.Random(self.seed)
        n_tracks = len(self.train_worlds)
        every = max(1, self.unlock_every)
        chunks: List[WorldChunk] = []
        for i in range(self.n_chunks):
            window = min(n_tracks, 1 + i // every)
            weights = [self.recency_weight ** j for j in range(window)]
            world = rng.choices(self.train_worlds[:window], weights=weights, k=1)[0]
            chunks.append(WorldChunk(world, self.chunk_steps))
        return chunks

    def evaluation_worlds(self) -> List[str]:
        return list(self.eval_worlds)

    def first_world(self) -> str:
        # Chunk 0 always samples from a window of size 1, so this is deterministic.
        return self.train_worlds[0]
