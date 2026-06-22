"""Cost functions for constrained / safe RL (CMDP) — the W-saferl interface.

**Semantics (maintainer-specified):** a cost is a graded signal of *getting close
to a bad state* — approaching the track edge, nearing an object, driving
erratically — **not** the catastrophe itself. The terminal events (fully leaving
the track / smashing into an object) are already handled by the env's termination
and the reward's off-track penalty; the *cost* is the early-warning, risk-exposure
metric a constrained-RL algorithm keeps under a budget so the policy learns to
**stay away from the boundary**.

So every cost here returns a value in **[0, 1]** that is 0 while the car is in a
comfortable state and ramps toward 1 as it approaches danger (saturating at 1 once
the boundary is reached). A cost is a plain ``Callable[[dict], float]`` over the
DeepRacer reward-params dict — same shape as a reward (ADR-0002), tapped the same
way (`gym_dr/metrics.py`).

This is the **canonical** cost definition for the project; it intentionally
differs from the `deepracer-env` `feat/safety-gymnasium` branch's binary
off-track/crash levels (which fire on the *terminal* event). When we integrate
that branch (W-saferl) the env should consume these graded costs. See
``docs/reports/safety-gymnasium.md``.
"""
from __future__ import annotations

from typing import Callable, Dict

# Graded composite cost term names.
COMPOSITE_TERMS = ("near_edge", "near_collision", "steering_jerk")


def cost_near_edge(params: dict, *, onset: float = 0.5) -> float:
    """Off-track **risk**: 0 while comfortably inside the lane, ramping to 1 as the
    car nears the edge. ``f = |distance_from_center| / (track_width/2)`` (0 at the
    centre, 1 at the edge); cost is 0 until ``f == onset`` then ramps linearly to 1.

    Signals *getting close to leaving the track* — not the (terminal) excursion,
    which the env termination + reward off-track penalty handle.
    """
    tw = float(params.get("track_width", 0.0))
    if tw <= 0:
        return 0.0
    f = abs(float(params.get("distance_from_center", 0.0))) / (tw / 2.0)
    onset = min(max(onset, 0.0), 0.99)
    return float(min(max((f - onset) / (1.0 - onset), 0.0), 1.0))


def cost_near_collision(params: dict, *, threshold_m: float = 0.75) -> float:
    """Collision **risk**: 0 when no object is within ``threshold_m``, ramping to 1
    as the nearest object approaches.

    NOTE: uses ``objects_distance`` (per-object arc-length on the centerline) as a
    proximity proxy; the precise car→object gap (using the car's own arc position +
    ``closest_objects``) is a refinement. Only meaningful with object avoidance on.
    """
    dists = params.get("objects_distance") or []
    if not dists:
        return 0.0
    try:
        d = min(abs(float(x)) for x in dists)
    except (TypeError, ValueError):
        return 0.0
    if d >= threshold_m:
        return 0.0
    return float(min(max((threshold_m - d) / threshold_m, 0.0), 1.0))


def make_composite_cost(
    weights: Dict[str, float],
    *,
    near_edge_onset: float = 0.5,
    near_collision_threshold_m: float = 0.75,
) -> Callable[[dict], float]:
    """Weighted sum of the graded risk terms in :data:`COMPOSITE_TERMS`.

    ``weights`` maps any subset of ``("near_edge", "near_collision",
    "steering_jerk")`` to non-negative floats; missing terms contribute 0. Returns
    a stateful closure (it tracks the previous steering angle for the jerk term).
    Each term is in [0,1]; the result is their weighted sum (cap it by choosing
    weights that sum to 1 if you want [0,1]).
    """
    if not weights or any(w < 0 for w in weights.values()):
        raise ValueError("composite cost needs non-negative weights")
    unknown = set(weights) - set(COMPOSITE_TERMS)
    if unknown:
        raise ValueError(f"unknown cost term(s): {sorted(unknown)}; valid: {COMPOSITE_TERMS}")

    prev = {"steer": None}

    def cost(params: dict) -> float:
        terms = {k: 0.0 for k in COMPOSITE_TERMS}
        if "near_edge" in weights:
            terms["near_edge"] = cost_near_edge(params, onset=near_edge_onset)
        if "near_collision" in weights:
            terms["near_collision"] = cost_near_collision(
                params, threshold_m=near_collision_threshold_m)
        if "steering_jerk" in weights and prev["steer"] is not None:
            try:
                jerk = abs(float(params.get("steering_angle", 0.0)) - prev["steer"]) / 60.0
                terms["steering_jerk"] = min(max(jerk, 0.0), 1.0)
            except (TypeError, ValueError):
                pass
        try:
            prev["steer"] = float(params.get("steering_angle", 0.0))
        except (TypeError, ValueError):
            prev["steer"] = None
        return float(sum(weights.get(k, 0.0) * v for k, v in terms.items()))

    cost.__name__ = "composite_cost"
    return cost


# Name -> callable, for config/HPO selection of the cost. Graded risk costs only;
# terminal off-track/crash are NOT costs (handled by termination + reward).
COST_VARIANTS: dict = {
    "near_edge": cost_near_edge,
    "near_collision": cost_near_collision,
}
