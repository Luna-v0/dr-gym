"""Reward function variants for DeepRacer training.

A reward is a plain ``Callable[[dict], float]`` — there's no registry, just
functions. Users typically pick one of these or write their own in
``app.py`` and pass it to ``ExperimentConfig(reward=my_reward)``.

The ``params`` dict comes from the upstream DeepRacer environment. Full key
list: ``.deepracer-env-upstream/deepracer_env/agent_ctrl/constants.py:108``.
Common ones used below: ``track_width``, ``distance_from_center``,
``progress`` (0-100), ``steps``, ``speed``, ``steering_angle``, ``heading``,
``all_wheels_on_track``, ``is_offtrack``, ``waypoints``, ``closest_waypoints``.

For HPO over reward variants, ``REWARD_VARIANTS`` at the bottom maps a
name (sweepable as an Optuna categorical) to the corresponding callable.

The variants are grounded in well-known DeepRacer community patterns:
- AWS Developer Guide reward-function examples
- USYD F1 team (Suntup, 12/1291 in 2020) — the centerline_quadratic shape
- dgnzlz Capstone + cdthompson K1999 — racing-line ideas (not packaged here
  because they require a per-track precomputed racing line)

Hard rules every variant follows:
- Off-track is *actively* punished (negative number per step). The upstream
  DeepRacer env does NOT terminate the episode on excursion, so a small
  positive off-track reward leaves the agent free to drive off-track and
  accumulate return — which is exactly what trials in the first study did.
- Gate speed bonuses on ``all_wheels_on_track`` so agents can't game raw
  speed by adding off-track steps.
- Use ``.get(...)`` defensively so a variant works against a partial
  ``params`` dict (e.g. test stubs).
"""
from __future__ import annotations

import math


# Per-step penalty applied when the car is off-track. Sized to overwhelm
# the *best* per-step on-track reward across the training variants below,
# so any off-track step strictly worsens episode return. The eval-only
# reward (``progress_safe``) uses a much larger penalty (-1000) because
# it's summed over the whole episode for Optuna ranking; here we want a
# value small enough that PPO's value head stays well-conditioned.
OFFTRACK_STEP_PENALTY = -5.0


# --------------------------------------------------------------------------- #
# Existing rewards (the running study uses these; do not silently rewrite).
# --------------------------------------------------------------------------- #

def center_line(params: dict) -> float:
    """Reward staying near the centre of the track lane.

    Three concentric bands measured as a fraction of ``track_width``:
    inside 10% gets the strongest reward, inside 25% a small reward, inside
    50% a tiny one. Off-track is actively punished. Mirrors the canonical
    AWS DeepRacer starter reward.
    """
    if not params.get("all_wheels_on_track", True):
        return OFFTRACK_STEP_PENALTY

    track_width = float(params.get("track_width", 1.0))
    distance_from_center = float(params.get("distance_from_center", 0.0))

    if distance_from_center <= 0.1 * track_width:
        multiplier = 1.0
    elif distance_from_center <= 0.25 * track_width:
        multiplier = 0.5
    elif distance_from_center <= 0.5 * track_width:
        multiplier = 0.1
    else:
        multiplier = 0.01

    base = max(params.get("progress", 0.0) * params.get("speed", 0.0) / 4.0, 1e-3)
    return float(base) * multiplier



def progress_and_speed(params: dict) -> float:
    """Maximize forward progress weighted by speed; floor when off-track.

    Mirrors the example in ``.deepracer-env-upstream/examples/train.py:21``.
    Encourages the policy to finish laps fast rather than crawl along the
    centre line.
    """
    if not params.get("all_wheels_on_track", True):
        return OFFTRACK_STEP_PENALTY
    progress = float(params.get("progress", 0.0))
    speed = float(params.get("speed", 0.0))
    return float(max(progress * speed / 4.0, 1e-3))


# --------------------------------------------------------------------------- #
# New variants (research-grounded; see module docstring for citations).
# --------------------------------------------------------------------------- #

def progress_per_step(params: dict) -> float:
    """Lap-pace baseline: ``progress/steps * 100 + speed^2``.

    The simplest reward that is monotone in "did you finish a fast lap"
    without much other shaping. ``progress/steps`` is the canonical
    "fast laps" signal that the agent cannot game by slowing down (slower
    → more steps → smaller term). ``speed^2`` rewards committing on
    straights once the agent is on-track.

    Excellent as a sanity baseline and as the *evaluation* reward — being
    invariant to training-reward choice makes cross-trial comparison fair.
    """
    if not params.get("all_wheels_on_track", True):
        return OFFTRACK_STEP_PENALTY
    progress = float(params.get("progress", 0.0))
    steps = max(int(params.get("steps", 1) or 1), 1)
    speed = float(params.get("speed", 0.0))
    return float((progress / steps) * 100.0 + speed ** 2)


def centerline_quadratic(params: dict) -> float:
    """Smooth centerline reward + lap-pace + steering smoothness.

    ``1 - (d / (w/2))^2`` gives a gradient near the center and tolerates
    corner-cutting — preferred over the tiered baseline by the USYD F1
    team. Adds ``progress/steps`` for lap pace and a soft penalty for
    sharp steering.
    """
    if not params.get("all_wheels_on_track", True) or params.get("is_offtrack", False):
        return OFFTRACK_STEP_PENALTY
    d = float(params.get("distance_from_center", 0.0))
    half_w = float(params.get("track_width", 1.0)) / 2.0
    if half_w <= 0:
        return OFFTRACK_STEP_PENALTY
    reward = max(1.0 - (d / half_w) ** 2, 1e-3)
    steps = max(int(params.get("steps", 1) or 1), 1)
    reward += float(params.get("progress", 0.0)) / steps
    if abs(float(params.get("steering_angle", 0.0))) > 15.0:
        reward *= 0.8
    return float(reward)


def anti_zigzag(params: dict) -> float:
    """Tiered centerline × steering-jerk penalty (AWS Example 3).

    Same tiers as ``center_line``, multiplicatively dampened by 0.8 when
    ``|steering_angle| > 15``. Drop-in modifier for "stop the car
    swerving" — used as-is by many DeepRacer League entrants.
    """
    if not params.get("all_wheels_on_track", True) or params.get("is_offtrack", False):
        return OFFTRACK_STEP_PENALTY
    d = float(params.get("distance_from_center", 0.0))
    tw = float(params.get("track_width", 1.0))
    if d <= 0.1 * tw:
        reward = 1.0
    elif d <= 0.25 * tw:
        reward = 0.5
    elif d <= 0.5 * tw:
        reward = 0.1
    else:
        reward = 1e-3
    if abs(float(params.get("steering_angle", 0.0))) > 15.0:
        reward *= 0.8
    return float(reward)


def waypoint_anticipation(params: dict) -> float:
    """Corner-aware: fast & straight on straights, slow into corners.

    Reads ``waypoints`` + ``closest_waypoints`` to estimate the heading
    angle of the track ``LOOKAHEAD`` waypoints ahead. If the upcoming
    section is straight (turn angle < 10°), reward going fast and not
    steering. If a corner is imminent, reward slowing down. Builds on a
    quadratic centerline base so the agent doesn't just hug the wall.

    Source: MatthewSuntup/DeepRacer ``identify_corner``.
    """
    LOOKAHEAD = 5
    TURN_THRESHOLD = 10.0
    SPEED_THRESHOLD = 2.0

    if not params.get("all_wheels_on_track", True) or params.get("is_offtrack", False):
        return OFFTRACK_STEP_PENALTY

    d = float(params.get("distance_from_center", 0.0))
    half_w = float(params.get("track_width", 1.0)) / 2.0
    if half_w <= 0:
        return OFFTRACK_STEP_PENALTY
    reward = max(1.0 - (d / half_w) ** 2, 1e-3)

    waypoints = params.get("waypoints") or []
    closest = params.get("closest_waypoints") or [0, 0]
    if len(waypoints) >= 2 and len(closest) >= 2:
        n = len(waypoints)
        idx_now = int(closest[1]) % n
        idx_fut = (idx_now + LOOKAHEAD) % n
        p_now = waypoints[idx_now]
        p_fut = waypoints[idx_fut]
        track_dir = math.degrees(math.atan2(p_fut[1] - p_now[1], p_fut[0] - p_now[0]))
        heading = float(params.get("heading", 0.0))
        diff = abs(track_dir - heading)
        if diff > 180:
            diff = 360 - diff
    else:
        # No waypoint info — fall back to centerline-only.
        diff = 0.0

    speed = float(params.get("speed", 0.0))
    steering_abs = abs(float(params.get("steering_angle", 0.0)))
    if diff < TURN_THRESHOLD:           # straight section ahead
        if speed > SPEED_THRESHOLD:
            reward += 0.5
        if steering_abs < 5.0:
            reward += 0.3
    else:                                # corner ahead
        if speed < SPEED_THRESHOLD:
            reward += 0.3

    return float(reward)


# --------------------------------------------------------------------------- #
# Object Avoidance variants. Only meaningful when
# ``ExperimentConfig.object_avoidance`` is set (so the env populates
# ``is_crashed``, ``closest_objects`` and ``objects_location`` in params).
# Safe to call without OA enabled — the relevant keys default to
# "no obstacle" / "not crashed" via .get(...).
# --------------------------------------------------------------------------- #

CRASH_PENALTY = -10.0
"""Per-step penalty when the car hits a spawned obstacle. Sized larger
than ``OFFTRACK_STEP_PENALTY`` because a collision is a harder failure
mode — in the AWS Object Avoidance race it ends the lap, and in our
fork (with ``terminate_on_collision=True``, the default) it terminates
the episode. Reduce if you set ``terminate_on_collision=False`` and
want the per-step cost to persist across the rest of the trajectory
without overwhelming the on-track learning signal."""


def object_avoidance_aware(params: dict) -> float:
    """Progress × speed with crash penalty + lane-commit bonus.

    Mirrors the example reward in the deepracer-env fork
    (``examples/train_object_avoidance.py``): a hard penalty when the car
    crashes into an obstacle, and a small bonus for committing to one
    side of the track when an obstacle is *ahead* — encouraging the
    policy to pick a lane before reaching the obstacle rather than
    swerving at the last moment.

    Reward params consumed (in addition to the standard set):
      - ``is_crashed``       (bool) — collision flag from the OA-aware controller.
      - ``closest_objects``  (``[prev_idx, next_idx]``; ``-1`` = none ahead).

    With OA disabled both default-getter to safe values, so this reduces
    to a centerline-agnostic progress × speed reward.
    """
    if params.get("is_crashed", False):
        return CRASH_PENALTY
    if not params.get("all_wheels_on_track", True):
        return OFFTRACK_STEP_PENALTY

    closest_objects = params.get("closest_objects", [-1, -1])
    next_obj_idx = int(closest_objects[1]) if len(closest_objects) >= 2 else -1
    has_obstacle_ahead = next_obj_idx >= 0

    if has_obstacle_ahead:
        # Reward committing to one side of the track before reaching the
        # obstacle. >10 cm off the centerline counts as "committed".
        bonus = 1.5 if float(params.get("distance_from_center", 0.0)) > 0.1 else 0.7
    else:
        bonus = 1.0

    progress = float(params.get("progress", 0.0))
    speed = float(params.get("speed", 0.0))
    return float(max(progress * speed * bonus / 4.0, 1e-3))


# --------------------------------------------------------------------------- #
# Eval-only reward. NOT in REWARD_VARIANTS — never sampled as a training
# reward because the off-track penalty is large and negative, which would
# destabilize PPO gradients. Used by ExperimentConfig.eval_reward to score
# trials trained with different rewards on a single "stay-on-track + fast"
# axis. Comparable across trials because every trial sees the same eval
# reward regardless of its training reward.
# --------------------------------------------------------------------------- #

OFFTRACK_PENALTY = -1.0
"""Per-step penalty applied (in the eval reward) when off-track. Sized to
be a small constant negative so that the eval reward stays in the same
order of magnitude as the on-track reward (~progress/steps*100 + speed^2,
roughly 10–30/step at racing pace).

Originally -1000, which made *any* off-track excursion dominate the
episode total and crushed Optuna's ability to distinguish a fast lap
with one excursion from a slow but clean crawl. With -1 the off-track
penalty is a soft, comparable bias: a few off-track steps modestly
reduce the score, while persistent off-track driving still ranks below
on-track driving — but not catastrophically so.

Combine with the training-reward off-track penalty (``OFFTRACK_STEP_PENALTY``)
which is sized differently because it drives PPO's value head directly."""


def progress_safe(params: dict) -> float:
    """Eval-only: progress-per-step, heavily penalised on any off-track step.

    On-track: ``(progress/steps) * 100 + speed^2`` — same shape as
    :func:`progress_per_step`, monotone in lap pace, can't be gamed.

    Off-track: returns :data:`OFFTRACK_PENALTY` (-100). Summed over an
    episode, even a brief excursion meaningfully discounts the score, so
    a model that finishes the lap *without* leaving the track will always
    rank above one that drives faster but exits the lane.
    """
    offtrack = (
        (not params.get("all_wheels_on_track", True))
        or params.get("is_offtrack", False)
    )
    if offtrack:
        return OFFTRACK_PENALTY
    progress = float(params.get("progress", 0.0))
    steps = max(int(params.get("steps", 1) or 1), 1)
    speed = float(params.get("speed", 0.0))
    return float((progress / steps) * 100.0 + speed ** 2)


# Per-step penalty (eval-only) for an off-track step in ``clean_completion``.
# Larger than progress_safe's -1.0 so that ANY off-track excursion clearly sinks
# the episode score below a clean lap — matching the maintainer's success
# criterion that the car must finish *without leaving the track*.
CLEAN_OFFTRACK_PENALTY = -10.0

# One-off bonus (eval-only) added on the step the lap completes (progress ~100),
# so a finished lap dominates an unfinished one of the same pace.
COMPLETION_BONUS = 100.0


def clean_completion(params: dict) -> float:
    """Eval-only reward aligned with the success criterion: *finish every track
    without leaving it, at a reasonable (non-minimum) speed.*

    On-track step: lap pace ``(progress/steps) * 100`` (cannot be gamed by
    crawling — slower ⇒ more steps ⇒ smaller term) **plus the step ``speed``
    linearly** (so faster-but-clean ranks above slow-but-clean, *without* the
    ``speed²`` domination that let :func:`progress_safe` reward reckless speed).
    On the completing step (``progress >= 100``) a one-off
    :data:`COMPLETION_BONUS`. Any off-track step returns
    :data:`CLEAN_OFFTRACK_PENALTY` — large enough that a lap with *any* excursion
    scores below a clean one.

    Eval-only (NOT in ``REWARD_VARIANTS``), like :func:`progress_safe`. The
    authoritative generalization measure is the ``dr/ep_completed_clean`` episode
    metric surfaced per held-out world as ``eval/<world>_clean_completion_rate``;
    this reward is the per-step proxy that drives ``best_model`` selection toward
    clean, reasonably-fast laps.
    """
    offtrack = (
        (not params.get("all_wheels_on_track", True))
        or params.get("is_offtrack", False)
    )
    if offtrack:
        return CLEAN_OFFTRACK_PENALTY
    progress = float(params.get("progress", 0.0))
    steps = max(int(params.get("steps", 1) or 1), 1)
    speed = float(params.get("speed", 0.0))
    pace = (progress / steps) * 100.0
    bonus = COMPLETION_BONUS if progress >= 99.999 else 0.0
    return float(pace + speed + bonus)


# --------------------------------------------------------------------------- #
# Registry — for HPO sweep over reward variants.
# --------------------------------------------------------------------------- #

REWARD_VARIANTS: dict = {
    "center_line": center_line,
    "progress_and_speed": progress_and_speed,
    "progress_per_step": progress_per_step,
    "centerline_quadratic": centerline_quadratic,
    "anti_zigzag": anti_zigzag,
    "waypoint_anticipation": waypoint_anticipation,
    "object_avoidance_aware": object_avoidance_aware,
}
"""Name -> callable map. Used by HPO to sample a training reward via Optuna
``suggest_categorical`` (which only accepts hashable scalars, not function
objects) — the search space picks a name, then looks up the callable here."""
