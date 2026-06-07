"""Top-level package re-exports.

The user's ``app.py`` typically only needs symbols from here:

    from gym_dr import (
        ExperimentConfig, Sb3Trainer, TrainingConfig, TrackingConfig,
        ContinuousActionSpaceConfig, WorldsConfig,
        time_trial, train, study,
    )
    from gym_dr.rewards import center_line   # or write your own

For deeper extension points (custom Trainer, env factory), see
``gym_dr.trainers.base`` and ``gym_dr.envs``.
"""
from gym_dr.action_space import (
    ContinuousActionSpaceConfig,
    DiscreteAction,
    DiscreteActionSpaceConfig,
)
from gym_dr.app import inspect, study, train
from gym_dr.config import (
    ExperimentConfig,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    load_config,
    load_search_space,
)
from gym_dr.envs import time_trial
from gym_dr.object_avoidance import ObjectAvoidanceConfig
from gym_dr.seeding import ReplicateSeeds, SeedManager
from gym_dr.rewards import (
    REWARD_VARIANTS,
    anti_zigzag,
    center_line,
    centerline_quadratic,
    object_avoidance_aware,
    progress_and_speed,
    progress_per_step,
    progress_safe,
    waypoint_anticipation,
)
from gym_dr.tracks import ALL_TRACKS, TRACKS, display_name, existing_tracks
from gym_dr.trainers import Sb3Trainer, Trainer, TrainingContext, TrainResult

__all__ = [
    "ALL_TRACKS",
    "ContinuousActionSpaceConfig",
    "DiscreteAction",
    "DiscreteActionSpaceConfig",
    "ExperimentConfig",
    "ObjectAvoidanceConfig",
    "ReplicateSeeds",
    "Sb3Trainer",
    "SeedManager",
    "TRACKS",
    "TrackingConfig",
    "Trainer",
    "TrainingConfig",
    "TrainingContext",
    "TrainResult",
    "WorldsConfig",
    "REWARD_VARIANTS",
    "anti_zigzag",
    "center_line",
    "centerline_quadratic",
    "display_name",
    "existing_tracks",
    "inspect",
    "load_config",
    "load_search_space",
    "object_avoidance_aware",
    "progress_and_speed",
    "progress_per_step",
    "progress_safe",
    "waypoint_anticipation",
    "study",
    "time_trial",
    "train",
]
