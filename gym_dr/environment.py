"""``EnvironmentConfig`` — the single, typed environment-building API.

Everything about *what world the agent learns in* lives here, composed of typed
strategy objects rather than loose flags:

  * ``observation``  — ``CameraObs`` (pixels) or ``FeatureObs`` (the reward-element
    vector, camera-off). Replaces the old ``camera_obs: bool`` + the
    ``GYM_DR_FEATURE_SET`` env-var hack.
  * ``action_space`` — continuous bounds / discrete list.
  * ``curriculum``   — a ``WorldStrategy``: ``FixedWorlds`` / ``OrderedSplit`` /
    ``ACL`` (Automatic Curriculum Learning).
  * ``domain_randomization`` — ``DomainRandomization`` / ``ADR`` (Automatic Domain
    Randomization), knobs as ``Range``/``Choice``.
  * ``object_avoidance`` — static-obstacle config (or ``None``).
  * ``safe_rl``      — ``SafeRL`` (cost + budget); presence routes training to the
    constrained (FSRL/Lagrangian) backend.
  * ``n_cars``       — N agents in one Gazebo (multi-agent VecEnv).

``ExperimentConfig`` (``gym_dr/config.py``) composes one of these plus the
training/tracking/trainer concerns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

from gym_dr.action_space import ActionSpaceConfig, ContinuousActionSpaceConfig
from gym_dr.domain_randomization import DomainRandomization
from gym_dr.object_avoidance import ObjectAvoidanceConfig
from gym_dr.worlds import FixedWorlds, WorldStrategy


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraObs:
    """Pixel observation (vision). ``sensors`` are deepracer-env sensor names."""
    sensors: Tuple[str, ...] = ("FRONT_FACING_CAMERA",)


@dataclass(frozen=True)
class FeatureObs:
    """Camera-off observation: the low-dim feature vector (the reward elements).

    ``features`` selects the actor vector (default the 11-feature ACTOR_FEATURES);
    nothing is rendered, so the sim step is much cheaper."""
    features: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.features:
            from gym_dr.perception import ACTOR_FEATURES
            object.__setattr__(self, "features", tuple(ACTOR_FEATURES))


Observation = Union[CameraObs, FeatureObs]


# --------------------------------------------------------------------------- #
# Safe RL
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SafeRL:
    """Constrained-RL (CMDP) settings. When present on an ``EnvironmentConfig`` the
    orchestrator builds the constrained (FSRL/Lagrangian) trainer with this budget;
    ``cost`` is the per-step graded risk (``gym_dr/costs.py``)."""
    cost: Callable[[dict], float]
    cost_limit: float = 10.0          # CMDP budget d (E[discounted cost] ≤ d)
    gamma: float = 0.99


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def _default_reward() -> Callable[[dict], float]:
    from gym_dr.rewards import centerline_quadratic
    return centerline_quadratic


def _default_eval_reward() -> Callable[[dict], float]:
    from gym_dr.rewards import clean_completion
    return clean_completion


@dataclass(frozen=True)
class EnvironmentConfig:
    """The composed environment definition (see module docstring)."""

    observation: Observation = field(default_factory=CameraObs)
    action_space: ActionSpaceConfig = field(default_factory=ContinuousActionSpaceConfig)
    curriculum: WorldStrategy = field(default_factory=FixedWorlds)
    domain_randomization: Optional[DomainRandomization] = None
    object_avoidance: Optional[ObjectAvoidanceConfig] = None
    safe_rl: Optional[SafeRL] = None
    n_cars: int = 1
    reward: Callable[[dict], float] = field(default_factory=_default_reward)
    eval_reward: Callable[[dict], float] = field(default_factory=_default_eval_reward)
    enable_gui: bool = False

    # ---- convenience derived views (used by the env factory / orchestrator) ----
    @property
    def camera_obs(self) -> bool:
        return isinstance(self.observation, CameraObs)

    @property
    def is_safe_rl(self) -> bool:
        return self.safe_rl is not None


__all__ = [
    "EnvironmentConfig", "Observation", "CameraObs", "FeatureObs", "SafeRL",
]
