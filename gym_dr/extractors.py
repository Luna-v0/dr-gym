"""Custom SB3 feature extractors — the *CNN* side of the network.

Background: ``Sb3Trainer.kwargs["policy_kwargs"]["net_arch"]`` only controls
the *MLP head* that sits after the feature extractor. For DeepRacer's image
observations the heavy lifting is done by a CNN that runs *before* that
head. SB3 picks the extractor automatically:

- ``MultiInputPolicy`` + Dict obs → ``CombinedExtractor``, which routes each
  image key through a fixed 3-layer ``NatureCNN`` and concatenates.

Three levers to change the CNN, from cheapest to most flexible:

1. **Embedding width (no custom class).** ``CombinedExtractor`` takes a
   ``cnn_output_dim`` kwarg. Bump it via
   ``policy_kwargs["features_extractor_kwargs"]["cnn_output_dim"]``. This is
   what ``app.py``'s search space sweeps by default.

2. **Conv stack shape (custom class, this module).** ``DeepImageExtractor``
   lets you specify the whole conv stack — channels, **kernel size**, and
   **stride** per layer — via the ``conv_layers`` kwarg. Plug it in with
   ``policy_kwargs["features_extractor_class"]``.

3. **Anything else.** Subclass ``BaseFeaturesExtractor`` yourself.

Usage in a config::

    from gym_dr.extractors import DeepImageExtractor

    trainer = Sb3Trainer(
        name="ppo", policy="MultiInputPolicy",
        kwargs={
            "policy_kwargs": {
                "features_extractor_class": DeepImageExtractor,
                "features_extractor_kwargs": {
                    "features_dim": 512,
                    # (out_channels, kernel_size, stride) per conv layer:
                    "conv_layers": ((32, 8, 4), (64, 4, 2), (64, 3, 1),
                                    (128, 3, 1), (128, 3, 1)),
                },
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            },
        },
    )

``features_extractor_class`` is a class object, not JSON-serializable, so
it can't be swept by Optuna directly — pick it in the base config. The
``features_extractor_kwargs`` dict (``features_dim``, ``conv_layers``) *is*
sweepable: build the ``conv_layers`` tuple inside ``search_space(trial)``
from sampled scalars and put it in the ``policy_kwargs`` override.
"""
from __future__ import annotations

from typing import Any

# (out_channels, kernel_size, stride) — one per conv layer.
ConvSpec = tuple[tuple[int, int, int], ...]

# NatureCNN's classic Atari stack, as an explicit ConvSpec for reference / default.
NATURE_CNN_CONV: ConvSpec = ((32, 8, 4), (64, 4, 2), (64, 3, 1))

# A deeper default for DeepRacer's 120x160 camera.
DEEP_CONV: ConvSpec = ((32, 8, 4), (64, 4, 2), (64, 3, 1), (128, 3, 1), (128, 3, 1))


def _build_deep_image_extractor():
    """Build the DeepImageExtractor class lazily (torch import is deferred)."""
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from stable_baselines3.common.preprocessing import is_image_space
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    from stable_baselines3.common.preprocessing import is_image_space_channels_first

    class DeepImageExtractor(BaseFeaturesExtractor):
        """A fully-configurable CNN feature extractor for Dict observations.

        Each image-shaped key in the observation Dict runs through a conv
        stack you specify layer-by-layer (channels, kernel size, stride);
        non-image keys are flattened. Per-key embeddings are concatenated
        and projected to ``features_dim``.

        Follows SB3's feature-extractor contract: SB3 hands ``forward`` the
        observation *already preprocessed* — image keys arrive channels-first
        and float-normalized to [0, 1] (SB3 inserts ``VecTransposeImage`` and
        runs ``preprocess_obs`` before the extractor). This class therefore
        does no transposing or normalization itself; it just runs the conv
        stack. (SB3's stock ``NatureCNN`` works the same way.)

        Args:
            observation_space: a ``gymnasium.spaces.Dict``. Image subspaces
                are expected channels-first (``(C, H, W)``) — which is what
                SB3 passes after ``VecTransposeImage``.
            features_dim: width of the final projection — the number the
                MLP head in ``net_arch`` receives. Default 512.
            conv_layers: ``((out_channels, kernel_size, stride), ...)``,
                one tuple per conv layer. Default is a 5-layer deep stack
                (``DEEP_CONV``). Pass ``NATURE_CNN_CONV`` to mimic SB3's
                stock CNN, or any tuple of your own.
            activation: ``"relu"`` (default) or ``"gelu"``.
        """

        def __init__(
            self,
            observation_space: gym.spaces.Dict,
            features_dim: int = 512,
            conv_layers: ConvSpec = DEEP_CONV,
            activation: str = "relu",
        ) -> None:
            super().__init__(observation_space, features_dim)
            act_cls = {"relu": nn.ReLU, "gelu": nn.GELU}[activation.lower()]

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
                    flat = int(gym.spaces.flatdim(subspace))
                    extractors[key] = nn.Flatten()
                    total_concat += flat

            self._extractors = nn.ModuleDict(extractors)
            self._projection = nn.Sequential(
                nn.Linear(total_concat, features_dim),
                act_cls(),
            )

        @staticmethod
        def _make_cnn(in_channels: int, conv_layers: ConvSpec, act_cls) -> "nn.Module":
            layers: list[nn.Module] = []
            prev = in_channels
            for out_ch, kernel, stride in conv_layers:
                # Stride-1 layers pad to preserve spatial size — that's how
                # you stack depth without collapsing the feature map (and it
                # keeps the stack robust across input resolutions). Strided
                # layers do the downsampling, unpadded.
                padding = kernel // 2 if stride == 1 else 0
                layers += [
                    nn.Conv2d(prev, out_ch, kernel_size=kernel, stride=stride, padding=padding),
                    act_cls(),
                ]
                prev = out_ch
            layers.append(nn.Flatten())
            return nn.Sequential(*layers)

        def forward(self, observations: dict) -> "torch.Tensor":
            # SB3 has already preprocessed: image keys are channels-first
            # float tensors in [0, 1]. Just run the stack.
            parts: list[torch.Tensor] = []
            for key, extractor in self._extractors.items():
                parts.append(extractor(observations[key]))
            return self._projection(torch.cat(parts, dim=1))

    def _image_channels(subspace) -> int:
        """Channel count of an image subspace, layout-agnostic.

        SB3 passes channels-first spaces (``(C, H, W)``) after
        ``VecTransposeImage``; a raw gym space is channels-last (``(H, W, C)``).
        """
        if is_image_space_channels_first(subspace):
            return int(subspace.shape[0])
        return int(subspace.shape[-1])

    return DeepImageExtractor


# Public name — built on first access so importing this module doesn't pull
# in torch unless the extractor is actually used. Memoized so every access
# returns the *same* class object: SB3 and HPO compare it by identity, and
# rebuilding it per-access would break `is` checks and bloat memory.
_DEEP_IMAGE_EXTRACTOR: Any = None


def __getattr__(name: str) -> Any:
    if name == "DeepImageExtractor":
        global _DEEP_IMAGE_EXTRACTOR
        if _DEEP_IMAGE_EXTRACTOR is None:
            _DEEP_IMAGE_EXTRACTOR = _build_deep_image_extractor()
        return _DEEP_IMAGE_EXTRACTOR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
