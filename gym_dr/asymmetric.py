"""Asymmetric actor-critic for feature-vector policies.

The actor (deployed) sees a **noised** feature vector; the critic (training-only,
discarded at deployment) sees the **true** feature vector. This is the classic
privileged-critic / asymmetric-information setup: the value function gets a clean,
low-variance learning signal while the policy is forced to be robust to feature
noise — exactly the robustness study the maintainer asked for. The noise lives
under domain randomization (``DomainRandomization.feature_noise``) so it's variable.

The observation is a Dict ``{"actor": noised (F,), "critic": true (F,)}`` (built by
``gym_dr.envs.feature_obs.FeatureObsWrapper`` in asymmetric mode). SB3's
``share_features_extractor=False`` path already routes a *pi* feature tensor to
``mlp_extractor.forward_actor`` and a *vf* tensor to ``forward_critic`` — so all we
need is a per-key feature extractor: pi reads ``obs["actor"]``, vf reads
``obs["critic"]``. Both keys have the same dim F, so the shared ``mlp_extractor``
input width is unchanged.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class KeyExtractor(BaseFeaturesExtractor):
    """Flatten a single key of a Dict observation (identity for a 1-D Box)."""

    def __init__(self, observation_space: gym.spaces.Dict, key: str = "actor") -> None:
        sub = observation_space.spaces[key]
        super().__init__(observation_space, int(gym.spaces.utils.flatdim(sub)))
        self._key = key

    def forward(self, observations: dict) -> th.Tensor:
        return th.flatten(observations[self._key], start_dim=1)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """PPO policy whose actor reads ``obs["actor"]`` and critic reads ``obs["critic"]``.

    Pass as ``Sb3Trainer(policy=AsymmetricActorCriticPolicy)``. Requires a Dict obs
    with "actor" and "critic" keys of equal shape (the FeatureObsWrapper asym mode).
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args: Any,
                 **kwargs: Any) -> None:
        if not (isinstance(observation_space, gym.spaces.Dict)
                and {"actor", "critic"} <= set(observation_space.spaces)):
            raise ValueError(
                "AsymmetricActorCriticPolicy needs a Dict obs with 'actor' + 'critic' "
                f"keys; got {observation_space}")
        # pi feature extractor reads the "actor" key; the critic extractor is swapped
        # to "critic" in _build. Separate extractors => SB3 uses forward_actor/critic.
        kwargs["features_extractor_class"] = KeyExtractor
        kwargs["features_extractor_kwargs"] = {"key": "actor"}
        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # super() built BOTH extractors on the "actor" key; point the value tower at
        # the TRUE feature vector instead. Same dim, so mlp_extractor is unaffected.
        self.vf_features_extractor = KeyExtractor(self.observation_space, key="critic")
        # Re-create the optimizer so the swapped extractor's params are registered.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **(self.optimizer_kwargs or {}))
