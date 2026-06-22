# ADR-0003 — Cross-repo seams are explicit, versioned contracts

- **Status:** Proposed
- **Date:** 2026-06-21 · **Tags:** `[BOTH]`

## Context
The repos are coupled only by data formats, but those seams are currently implicit and have already drifted
silently (e.g. the stale base sim image; the `progress_safe` docstring vs constant). A change to any seam
is exactly what can break analysis or the physical car.

## Decision
Add a top-level `CONTRACTS.md` enumerating each cross-repo seam with a `schema_version`, and treat a bump
as a deliberate, reviewed event:
1. **Trace schema** (`docs/trace-contract.md`) — add a `schema_version` field.
2. **`reward_params`** — the 26 keys `deepracer-env` guarantees to the reward callback (owner: deepracer-env).
3. **Action/units + `model_metadata.json`** — engineering units (or `[-1,1]` under `normalize_actions`),
   with one rescale function as the single source of truth (refactor R1).
4. **IR model I/O** — input `(4,120,160)` uint8 grayscale stack; output action-mean (units per #3).

## Consequences
- (+) Silent drift becomes a visible, reviewed change; the on-car and analysis seams are protected.
- (+) Enables the `deepracer-deploy` split (ADR-0001) to depend on a named contract version.
- (−) Small upkeep: bump versions and note changes when a seam moves.
