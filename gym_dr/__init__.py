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
# Clean break (ADR-0004): `Study` is the single user-facing entrypoint. The
# legacy `train`/`study` functions live on as internal orchestration in
# `gym_dr.app` (Study delegates to them) but are no longer part of the public
# surface. `inspect` stays — it's a dry-run helper, not an entrypoint.
from gym_dr.app import inspect
from gym_dr.config import (
    ExperimentConfig,
    TraceConfig,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    load_config,
    load_search_space,
)
from gym_dr.envs import time_trial
from gym_dr.data_store import (
    DataStore,
    NullDataStore,
    SQLiteDataStore,
    make_data_store,
)
from gym_dr.domain_randomization import ADR, DomainRandomization
from gym_dr.early_stopping import (
    AllOf,
    AnyOf,
    CleanCompletion,
    EarlyStopStrategy,
    MetricThreshold,
    OfftrackRate,
    RewardThreshold,
)
from gym_dr.environment import CameraObs, EnvironmentConfig, FeatureObs, SafeRL
from gym_dr.pipeline import Stage, compose, stage
from gym_dr.randomization import Choice, Range
from gym_dr.search import Categorical, Fixed, Float, Int, SearchSpace
from gym_dr.study import Study, StudyResult
from gym_dr.object_avoidance import ObjectAvoidanceConfig
from gym_dr.seeding import ReplicateSeeds, SeedManager
from gym_dr.rewards import (
    REWARD_VARIANTS,
    anti_zigzag,
    center_line,
    centerline_quadratic,
    clean_completion,
    object_avoidance_aware,
    progress_and_speed,
    progress_per_step,
    progress_safe,
    waypoint_anticipation,
)
from gym_dr.tracks import ALL_TRACKS, TRACKS, display_name, existing_tracks
from gym_dr.trainers import Sb3Trainer, Trainer, TrainingContext, TrainResult
from gym_dr.worlds import (
    OrderedSplit,
    FixedWorlds,
    ACL,
    WorldChunk,
    WorldStrategy,
)

__all__ = [
    "ALL_TRACKS",
    "ContinuousActionSpaceConfig",
    "DiscreteAction",
    "DiscreteActionSpaceConfig",
    "DomainRandomization",
    "ADR",
    # Pipeline + early-stopping + hyperparameter-search primitives (Task-1/8 refactor)
    "Stage",
    "compose",
    "stage",
    "EarlyStopStrategy",
    "OfftrackRate",
    "MetricThreshold",
    "RewardThreshold",
    "CleanCompletion",
    "AllOf",
    "AnyOf",
    "SearchSpace",
    "Fixed",
    "Float",
    "Int",
    "Categorical",
    "Study",
    "StudyResult",
    "DataStore",
    "SQLiteDataStore",
    "NullDataStore",
    "make_data_store",
    "EnvironmentConfig",
    "CameraObs",
    "FeatureObs",
    "SafeRL",
    "Range",
    "Choice",
    "ExperimentConfig",
    "ObjectAvoidanceConfig",
    "ReplicateSeeds",
    "Sb3Trainer",
    "SeedManager",
    "TRACKS",
    "TraceConfig",
    "TrackingConfig",
    "Trainer",
    "OrderedSplit",
    "FixedWorlds",
    "ACL",
    "WorldChunk",
    "WorldStrategy",
    "TrainingConfig",
    "TrainingContext",
    "TrainResult",
    "WorldsConfig",
    "REWARD_VARIANTS",
    "anti_zigzag",
    "center_line",
    "centerline_quadratic",
    "clean_completion",
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
    "time_trial",
]
