# Postmortem — D3 hang at 3.5M steps · `[BOTH]` · 2026-06-23

## What happened
The D3 held-out validation (`experiments/p1p3_validation.py`) ran cleanly to **3.5M / 4M steps** (checkpoints
every ~15 min: 22:47 → 23:03 → 23:18), then **hung at 23:18** — no log output and no step progress for ~3.5 h,
while the container `gym-dr-p1p3_validation` kept burning **585% CPU**. Killed manually (`docker kill`, rc=137)
on 2026-06-23. The 3.5M checkpoint is retained; the baseline verdict was already conclusive (fast-crash,
~28% progress, 0 completions — plateaued the whole run).

## Root cause
The last container output was the **held-out eval's world-swap sequence**: `Bowtie_track` →
`jyllandsringen_pro` (track_length 62 m, a long mesh), with the WARN "Could not find <model>…" on each swap,
two clean resets, then silence. Diagnosis:

1. **Trigger:** rapid mesh delete/spawn during the eval world-swaps hit the *documented intermittent Gazebo
   instability on mesh `delete_model`/spawn* (`deepracer_env/environments/world_swap.py` comments) and
   **wedged gzserver** — physics kept spinning (the 585% CPU) but `/clock`/step responses stopped flowing. The
   Python RL loop blocked waiting on the dead sim ⇒ total log silence.
2. **Why it never recovered (the real bug):** the host crash-recovery (`gym_dr/app.py:_train_host`, the
   `while True` at ~L246) keys **entirely on the container EXIT CODE** — it blocks on `spawn_training_chunk`
   until the container exits, then relaunches iff `rc == _SIM_RESTART_RC`. The wedged gzserver **never crashed
   / never exited**, so no exit code was ever returned, so recovery never fired, and the host blocked forever.
   **Recovery handles sim *crashes* (exits) but not sim *hangs* (wedged-but-alive).**

## What it was NOT
- **Not** the ROS service-retry wrapper (`rospy_wrappers.py`): it retries 5× *with a log line each time*, then
  `time.sleep(5)` + `log_and_exit` — it would have logged retries and **exited** (triggering recovery). The
  log shows neither, so this path is exonerated. (This was the initial "retry" hypothesis — close, but the
  culprit is the recovery loop's exit-code-only assumption, not the service retry.)
- **Not** a blocked Python wait (those idle near 0% CPU; we saw 585%, i.e. gzserver actively spinning).

## Fix — liveness watchdog (BUILT 2026-06-23)
1. **Liveness watchdog (primary) — DONE.** The host now monitors *progress*, not just exit code:
   - In-container `HeartbeatCallback` (`gym_dr/trainers/sb3/callbacks.py`) touches `$GYM_DR_HEARTBEAT` every
     ~256 training steps, and the **eval step-callback touches it too** (so a long multi-world eval isn't
     mistaken for a hang).
   - Host (`gym_dr/docker_runner.py`): both `spawn_training_chunk` and `spawn_workers` poll the per-container
     heartbeat; if stale > `GYM_DR_WATCHDOG_TIMEOUT` (600 s) after a boot grace (360 s), `docker kill`.
   - **Single training:** the watchdog returns `SIM_RESTART_RC` (75); `_train_host` relaunches, resuming from
     `rotation_resume.json` if present, else falling back to the **newest checkpoint** (`_newest_checkpoint`)
     so weights are preserved. **HPO/reward search:** `spawn_workers` kills + **relaunches** the hung worker,
     which rejoins the shared Optuna study (bounded by `GYM_DR_MAX_WORKER_RESTARTS`).
   - `SIM_RESTART_RC` is now defined once in `docker_runner` and imported by `app` (no drift).
   - Tests: `tests/test_watchdog.py` (7) — liveness decision, boot grace, heartbeat touch, code sync.
2. **Bound episodes (defense in depth) — DEFERRED.** `MAX_STEPS=10000` default still applies; an explicit
   smaller eval cap (Q3) is a nice-to-have but the watchdog now covers the hang class generally.
3. **Safer world-swap — DEFERRED.** Pause-physics + collision-check on swap would reduce the *trigger*
   probability; lower priority now that hangs are recoverable.

## Status
**Closed.** Root cause understood; baseline captured; the primary fix (watchdog) is built + tested and now
protects the reward search (running) and Phase 2.
