"""Host-importable Object Avoidance configuration.

Upstream ``deepracer_env.object_avoidance.ObjectAvoidanceConfig`` lives
inside the simapp Python package, which is only installed in the training
container — importing it on the host (where ``app.py`` is *parsed* before
any container is spawned) would fail.

This module exposes a frozen mirror dataclass with the same fields, so
users can declare OA settings in ``app.py``::

    from gym_dr import ExperimentConfig, ObjectAvoidanceConfig

    experiment = ExperimentConfig(
        ...,
        object_avoidance=ObjectAvoidanceConfig(n_obstacles=3),
    )

The env factory (``gym_dr.envs.time_trial``) translates this mirror into
the upstream type at env-construction time inside the container.

Mirrors ``deepracer_env/object_avoidance/config.py`` from the user's
deepracer-env fork (commit ``e7e2cec``). Keep these fields in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


PLACEMENT_RANDOM = "random_on_waypoints"
PLACEMENT_FIXED = "fixed"
PLACEMENT_CALLABLE = "callable"

LANE_ANY = "any"
LANE_INNER = "inner"
LANE_OUTER = "outer"
LANE_CENTER = "center"


@dataclass(frozen=True)
class ObjectAvoidanceConfig:
    """Static-obstacle Object Avoidance settings.

    Set ``ExperimentConfig.object_avoidance`` to an instance of this to
    enable obstacle spawning each episode. Leave it ``None`` (the default)
    to keep training pure time-trial.

    Fields mirror upstream one-for-one. Off-track and crash signals are
    surfaced into the reward function's ``params`` dict as ``is_crashed``,
    ``objects_location`` and ``closest_objects`` — see
    :func:`gym_dr.rewards.object_avoidance_aware` for an example reward
    that consumes them.
    """

    enabled: bool = True
    """Master switch. If ``False`` the env factory skips OA setup
    entirely, even if this config is set."""

    n_obstacles: int = 2
    """Number of obstacles to spawn per episode."""

    placement: str = PLACEMENT_RANDOM
    """One of ``"random_on_waypoints"``, ``"fixed"``, or ``"callable"``."""

    fixed_positions: list[tuple[float, float]] | None = None
    """For ``placement="fixed"``: world-frame (x, y) positions. Length
    should match ``n_obstacles``."""

    placement_fn: Callable | None = None
    """For ``placement="callable"``: a function ``(track_data, np_random)
    -> list[(x, y)]`` returning ``n_obstacles`` positions."""

    min_spacing_m: float = 2.0
    """Minimum centerline-projected spacing (meters) between any two
    obstacles. Placement retries up to ``max_placement_attempts`` to
    satisfy this; if it can't, it gives up and places fewer."""

    lane: str = LANE_ANY
    """Where on the track cross-section to place obstacles. One of
    ``"any"``, ``"inner"``, ``"outer"``, ``"center"``."""

    terminate_on_collision: bool = True
    """AWS DeepRacer Object Avoidance default. Set ``False`` if you want
    the per-step ``is_crashed`` cost signal to keep firing across the rest
    of the episode (safety-style training)."""

    seed: int | None = None
    """Placement RNG seed. ``None`` defers to the env's ``np_random`` from
    ``reset(seed=...)``, so deterministic runs (with a fixed
    ``ExperimentConfig.seed``) produce reproducible obstacle layouts."""

    obstacle_sdf_path: str | None = None
    """Override path to the obstacle SDF. ``None`` uses the bundled
    ``deepracer_env/object_avoidance/sdf/obstacle_box.sdf``."""

    name_prefix: str = "obstacle"
    """Spawned Gazebo model name prefix. Must contain ``"obstacle"`` so
    the upstream RolloutCtrl mercy-reset / off-track logic recognises
    them."""

    max_placement_attempts: int = 200
    """Per-obstacle rejection-sampling cap. Increase if placements
    routinely fail to satisfy ``min_spacing_m`` / ``lane`` on tight
    tracks."""

    def to_upstream(self) -> Any:
        """Translate this mirror to upstream
        ``deepracer_env.object_avoidance.ObjectAvoidanceConfig``.

        Only callable inside the training container, where
        ``deepracer_env`` is importable. The env factory invokes this.
        """
        from deepracer_env.object_avoidance import ObjectAvoidanceConfig as _Upstream

        return _Upstream(
            enabled=self.enabled,
            n_obstacles=self.n_obstacles,
            placement=self.placement,
            fixed_positions=(
                list(self.fixed_positions) if self.fixed_positions is not None else None
            ),
            placement_fn=self.placement_fn,
            min_spacing_m=self.min_spacing_m,
            lane=self.lane,
            terminate_on_collision=self.terminate_on_collision,
            seed=self.seed,
            obstacle_sdf_path=self.obstacle_sdf_path,
            name_prefix=self.name_prefix,
            max_placement_attempts=self.max_placement_attempts,
        )
