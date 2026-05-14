"""CNN feature extractor for the DeepRacer policy.

Grounded in the *real* AWS DeepRacer training stack (the RoboMaker `markov`
bundle + Intel rl-coach), not the community sim:

- AWS's clipped-PPO uses ``use_separate_networks_per_head=True`` — the actor
  and critic each get their **own** CNN tower (same spec, independent
  weights), not a shared trunk. We reproduce that in SB3 by setting
  ``policy_kwargs["share_features_extractor"] = False``: SB3 then builds two
  ``DeepRacerCNN`` instances, one per head.
- The named DeepRacer architectures map to concrete conv stacks
  (``[filters, kernel, stride]``):
    SHALLOW  : (32,8,4) (64,4,2) (64,3,1)            -- the classic 3-conv net
    STANDARD : (32,5,2) (32,3,1) (64,3,2) (64,3,1)
    DEEP     : (32,8,4) (32,4,2) (64,4,2) (64,3,1)
  exposed below as ``DEEPRACER_CONV_PRESETS``.
- AWS feeds the network raw uint8 [0,255] grayscale — **no /255**. Match that
  with ``policy_kwargs["normalize_images"] = False`` (SB3 then skips the
  divide) and the grayscale observation wrapper in ``gym_dr/envs/wrappers.py``.

Padding note: AWS's rl-coach convs use TF ``'same'`` padding, which PyTorch
doesn't support for strided convs. Since we train from scratch (not loading
AWS weights), exact equivalence is irrelevant — we use valid (padding 0) for
strided/downsampling layers like SB3's stock ``NatureCNN``, and
``kernel // 2`` padding for stride-1 refinement layers so depth stacks
freely. The architecture is faithful in *shape*, not bit-for-bit.

The FC middleware that sits between this CNN and the action/value heads is
*not* in here — it's SB3's ``net_arch=dict(pi=..., vf=...)``, sized
independently per head. So one policy/value tower is
``DeepRacerCNN`` (this file) + ``net_arch`` FC (SB3), and there are two
independent towers.
"""
from __future__ import annotations

from typing import Any

# (filters, kernel_size, stride) per conv layer.
ConvSpec = tuple[tuple[int, int, int], ...]

DEEPRACER_CONV_PRESETS: dict[str, ConvSpec] = {
    "shallow": ((32, 8, 4), (64, 4, 2), (64, 3, 1)),
    "standard": ((32, 5, 2), (32, 3, 1), (64, 3, 2), (64, 3, 1)),
    "deep": ((32, 8, 4), (32, 4, 2), (64, 4, 2), (64, 3, 1)),
}
DEFAULT_CONV: ConvSpec = DEEPRACER_CONV_PRESETS["shallow"]


def _build_deepracer_cnn():
    """Build the DeepRacerCNN class lazily (torch import is deferred)."""
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from stable_baselines3.common.preprocessing import (
        is_image_space,
        is_image_space_channels_first,
    )
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class DeepRacerCNN(BaseFeaturesExtractor):
        """Config-driven CNN feature extractor for Dict observations.

        Each image key in the obs Dict runs through the ``conv_layers`` conv
        stack; non-image keys are flattened. Per-key outputs are concatenated
        and projected to ``features_dim``.

        Follows SB3's feature-extractor contract — SB3 hands ``forward`` the
        observation already preprocessed: image keys channels-first, and
        (because we set ``normalize_images=False`` in policy_kwargs to match
        the physical car) raw uint8-valued float32. This class does no
        transposing or normalization itself.

        Args:
            observation_space: a ``gymnasium.spaces.Dict``. Image subspaces
                are expected channels-first (``(C, H, W)``) — what SB3 passes
                after ``VecTransposeImage``.
            features_dim: width of the final projection — the size the
                ``net_arch`` MLP head receives. Default 512.
            conv_layers: ``((filters, kernel, stride), ...)`` per conv layer.
                Default ``DEFAULT_CONV`` (the DeepRacer "shallow" stack).
                Pass any preset from ``DEEPRACER_CONV_PRESETS`` or a custom
                tuple.
            activation: ``"relu"`` (default) or ``"tanh"`` — AWS's STANDARD
                arch uses tanh; SHALLOW/DEEP use relu.
        """

        def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim: int = 512,
            conv_layers: ConvSpec = DEFAULT_CONV,
            activation: str = "relu",
        ) -> None:
            super().__init__(observation_space, features_dim)
            act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation.lower()]

            self._image_keys: list[str] = []
            extractors: dict[str, nn.Module] = {}
            total_concat = 0

            for key, subspace in observation_space.spaces.items():
                if is_image_space(subspace):
                    self._image_keys.append(key)
                    in_channels = _image_channels(subspace)
                    cnn = self._make_cnn(in_channels, conv_layers, act_cls)
                    with torch.no_grad():
                        sample = torch.as_tensor(subspace.sample()[None]).float()
                        n_flat = cnn(sample).shape[1]
                    extractors[key] = cnn
                    total_concat += n_flat
                else:
                    extractors[key] = nn.Flatten()
                    total_concat += int(gym.spaces.flatdim(subspace))

            self._extractors = nn.ModuleDict(extractors)
            self._projection = nn.Sequential(
                nn.Linear(total_concat, features_dim),
                act_cls(),
            )

        @staticmethod
        def _make_cnn(in_channels: int, conv_layers: ConvSpec, act_cls) -> "nn.Module":
            layers: list[nn.Module] = []
            prev = in_channels
            for filters, kernel, stride in conv_layers:
                # Strided layers downsample with valid padding (NatureCNN
                # style); stride-1 layers pad so they don't shrink the map,
                # letting depth/kernel sweep freely.
                padding = kernel // 2 if stride == 1 else 0
                layers += [
                    nn.Conv2d(prev, filters, kernel_size=kernel, stride=stride, padding=padding),
                    act_cls(),
                ]
                prev = filters
            layers.append(nn.Flatten())
            return nn.Sequential(*layers)

        def forward(self, observations: dict) -> "torch.Tensor":
            parts = [self._extractors[k](observations[k]) for k in self._extractors]
            return self._projection(torch.cat(parts, dim=1))

    def _image_channels(subspace) -> int:
        """Channel count of an image subspace, layout-agnostic."""
        if is_image_space_channels_first(subspace):
            return int(subspace.shape[0])
        return int(subspace.shape[-1])

    return DeepRacerCNN


# Public name — built once on first access (so importing this module doesn't
# pull in torch) and memoized so every access returns the SAME class object
# (SB3 / HPO compare it by identity).
_DEEPRACER_CNN: Any = None


def __getattr__(name: str) -> Any:
    if name == "DeepRacerCNN":
        global _DEEPRACER_CNN
        if _DEEPRACER_CNN is None:
            _DEEPRACER_CNN = _build_deepracer_cnn()
        return _DEEPRACER_CNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
