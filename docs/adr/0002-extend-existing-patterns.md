# ADR-0002 — New features reuse the existing patterns (callable / Strategy / Wrapper / ABC)

- **Status:** Proposed
- **Date:** 2026-06-21 · **Tags:** `[BOTH]`

## Context
The upcoming axes (curriculum, domain randomization, safe-RL cost, perception, alternate RL backends) each
need a place to plug in. dr-gym already has consistent extension points: plain-callable `reward`, the
`WorldStrategy` Strategy, gym `Wrapper`s, the `Trainer` ABC, and the trace Adapter. Bespoke config per
feature would fragment the design.

## Decision
Implement each new axis with the existing pattern, not new bespoke machinery:
- **Curriculum** → a `WorldStrategy` subclass.
- **Domain randomization** → composable gym `Wrapper`s + a range Scheduler (ADR for ADR pending if needed).
- **Safe-RL cost** → a plain `cost: Callable[[dict], float]` on `ExperimentConfig` (mirrors `reward`) +
  Lagrangian logic inside a constrained `Trainer`.
- **Perception** → a pluggable obs→features extractor interface (raw-CNN ↔ perception-net ↔ privileged).
- **RL backends** → satisfy the `Trainer` ABC contract (`docs/trainer-contract.md`).

## Consequences
- (+) One mental model; features are swappable and HPO-sweepable the same way rewards already are.
- (+) Keeps `ExperimentConfig` the single declarative surface.
- (−) Some axes (safe-RL) strain the per-step-callable shape; document where a callable isn't enough.
