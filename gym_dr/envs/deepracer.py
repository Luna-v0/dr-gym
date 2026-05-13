"""DeepRacer env factories.

An env factory is any callable `(experiment) -> gym.Env`. The experiment
is the full `ExperimentConfig`, so the factory can read `reward`,
`action_space`, `world_name`, or any other field it needs.

Add new env versions by writing a new factory and pointing
`ExperimentConfig.env_factory` at it.
"""
from __future__ import annotations

from typing import Any


def deepracer_env_v1(experiment: Any):
    from deepracer_env.environments.deepracer_env import DeepRacerEnv

    from gym_dr.reward import make_reward

    return DeepRacerEnv(reward_fn=make_reward(experiment.reward))
