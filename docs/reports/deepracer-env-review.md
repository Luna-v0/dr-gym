# deepracer-env review + cross-repo re-homing · `[BOTH]` · 2026-06-21

Extends the architecture review (`docs/reports/architecture-review.md`). Proposals only — env changes are
image-gated and need sign-off. Grounded in the earlier code read of `deepracer-env`.

## Headline
The right move is **not** shuffling code between repos — keep training in `dr-gym`, the sim in
`deepracer-env`, analysis in `deepracer-utils`. The leverage is in **tightening the deepracer-env → dr-gym
contract** (a few small env-side additions) and a couple of robustness/feature fixes.

## deepracer-env improvement candidates (ranked by leverage)

1. **Expose `sim_time` (the `/clock`) to `step()` info / reward params** — `[BOTH]`, high leverage.
   Today the reward callback has no sim clock, so dr-gym's Tier-1 trace stamps `sim_time = null` and falls
   back to wall time (the trace contract's *join key* is missing without a ROS bag). Surfacing the current
   `/clock` value in the info/params dict would give the whole analysis pipeline the real join key for free
   — no bag converter needed for Tier-1.

2. **A clean episode-lifecycle config surface** — `[DISS]`, high leverage (Q3).
   `is_continuous` / `number_of_trials` (default **1000 laps**) / `MAX_STEPS` / `penalty_seconds` are read
   from an internal `config_dict`; the dr-gym factory passes none, so episodes inherit opaque defaults.
   Expose a typed, documented config (or accept these via the env `config=` kwarg) so dr-gym controls
   episode length / termination explicitly instead of relying on defaults.

3. **Random valid-start + random direction at reset** — `[BOTH]` (W-dr, the maintainer asked for this).
   Reset supports only deterministic round-robin advance (`CHANGE_START`) + alternating direction
   (`ALT_DIR`). Add a `RANDOM_START` mode that samples `start_ndist` from the env `np_random` each reset,
   and a random-direction option. Small, localized change in `agent_ctrl/rollout_agent_ctrl.py`.

4. **Honor (or document) `rtf_override`** — `[BOTH]`.
   The throughput benchmark shows the sim caps at ~4.5× regardless of the requested RTF
   (`docs/reports/throughput.md`). Investigate the physics `real_time_update_rate` / step size / headless GL
   path; either make the override effective or document the true ceiling. Pairs with #6.

5. **N-cars-in-one-world (multi-robot)** — `[BOTH]`, the throughput lever.
   The env already carries multi-agent scaffolding (`agents_info_map`, per-agent start lanes). Extending to
   N namespaced cars in one world (shared physics/render) is the parallelism path that *separate* containers
   failed to deliver. A real deepracer-env feature; benchmark before committing.

6. **Harden the high-RTF spawn race** — `[REAL]`-ish reliability.
   At `rtf=160` the racecar `spawn_model` service timed out (benign here, but flaky). Add a wait-for-entity
   retry so startup is robust at high RTF.

7. **Object-avoidance `object_in_camera` / FrustumManager** — `[DISS]`.
   `object_in_camera` depends on a frustum registration the env build doesn't do, so it silently stays
   False. Fix or document for OA reward work.

## Re-homing (dr-gym ↔ deepracer-env)
- **Keep in dr-gym:** rewards, trainer, world strategies, metrics, the Tier-1 trace sink, export/optimize —
  all training concerns. Nothing here should move into the sim.
- **Cost interface (`gym_dr/costs.py`) vs the `feat/safety-gymnasium` branch:** the cost is *derived from*
  `is_offtrack`/`is_crashed`, which `deepracer-env` owns. Decide a single canonical home — recommend the
  cost *definitions* stay a dr-gym callable (ADR-0002) and the branch's `SafetyDeepRacerEnv` consume them
  (or vice-versa) rather than duplicating. Resolve during the safety-gym rebase (W-saferl / D9).
- **The real seam is the contract, not code:** `reward_params` keys + `sim_time` + action units +
  `model_metadata`. Make it explicit/versioned (ADR-0003 `CONTRACTS.md`); #1 above adds `sim_time` to it.

## Recommendation / sequencing
Land #1 (`sim_time`) and #3 (random start/direction) first — both are small, high-value, and unblock the
trace/analysis (#1) and W-dr (#3). #4/#5 are the throughput investigation (after the N-cars-in-one-world
benchmark). #2 resolves Q3. Track under the "deepracer-env review" task; each is image-gated + sign-off.
