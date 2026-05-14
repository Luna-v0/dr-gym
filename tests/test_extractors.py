"""Unit tests for the custom CNN feature extractor (no sim, no training).

The extractor follows SB3's contract: it receives observations *already
preprocessed* — image keys channels-first (``(B, C, H, W)``), float. So the
standalone tests here construct channels-first spaces and feed CHW tensors,
matching what SB3 hands the extractor inside a real policy.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch


def _chw_dict_space(c=3, h=64, w=64):
    # Channels-first image space — what SB3 passes after VecTransposeImage.
    return gym.spaces.Dict(
        {"FRONT_FACING_CAMERA": gym.spaces.Box(0, 255, (c, h, w), np.uint8)}
    )


def test_deep_image_extractor_default_stack():
    from gym_dr.extractors import DeepImageExtractor

    ext = DeepImageExtractor(_chw_dict_space(), features_dim=512)
    out = ext({"FRONT_FACING_CAMERA": torch.zeros(2, 3, 64, 64)})
    assert out.shape == (2, 512)


def test_deep_image_extractor_real_camera_size():
    """The default DEEP_CONV stack must survive the real 120x160 DeepRacer obs."""
    from gym_dr.extractors import DeepImageExtractor

    ext = DeepImageExtractor(_chw_dict_space(3, 120, 160), features_dim=256)
    out = ext({"FRONT_FACING_CAMERA": torch.zeros(1, 3, 120, 160)})
    assert out.shape == (1, 256)


def test_deep_image_extractor_custom_kernels():
    """conv_layers fully controls channels / kernel size / stride per layer."""
    from gym_dr.extractors import DeepImageExtractor

    ext = DeepImageExtractor(
        _chw_dict_space(),
        features_dim=128,
        conv_layers=((16, 5, 2), (32, 3, 1), (64, 3, 1)),
    )
    out = ext({"FRONT_FACING_CAMERA": torch.zeros(3, 3, 64, 64)})
    assert out.shape == (3, 128)


def test_deep_image_extractor_gelu():
    from gym_dr.extractors import DeepImageExtractor

    ext = DeepImageExtractor(_chw_dict_space(), features_dim=64, activation="gelu")
    out = ext({"FRONT_FACING_CAMERA": torch.zeros(1, 3, 64, 64)})
    assert out.shape == (1, 64)


def test_extractor_plugs_into_sb3_policy_kwargs(tmp_path, monkeypatch):
    """End-to-end: a Sb3Trainer using DeepImageExtractor trains on the stub env.

    This is the real contract check — SB3 builds the policy, inserts
    VecTransposeImage, preprocesses obs, and calls the extractor. If the
    extractor's channels-first assumptions are wrong this fails here.
    """
    pytest.importorskip("stable_baselines3")
    from gym_dr import (
        ContinuousActionSpaceConfig,
        ExperimentConfig,
        Sb3Trainer,
        TrackingConfig,
        TrainingConfig,
        WorldsConfig,
        center_line,
        train,
    )
    from gym_dr.extractors import DeepImageExtractor
    from tests.test_smoke import stub_env_factory

    monkeypatch.setenv("GYM_DR_IN_CONTAINER", "1")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    exp = ExperimentConfig(
        name="extractor_e2e",
        env_factory=stub_env_factory,
        trainer=Sb3Trainer(
            name="ppo",
            policy="MultiInputPolicy",
            kwargs={
                "n_steps": 64,
                "batch_size": 32,
                "learning_rate": 3e-4,
                "ent_coef": 0.01,
                "policy_kwargs": {
                    "features_extractor_class": DeepImageExtractor,
                    "features_extractor_kwargs": {
                        "features_dim": 128,
                        "conv_layers": ((16, 5, 2), (32, 3, 1)),
                    },
                    "net_arch": dict(pi=[64, 64], vf=[64, 64]),
                },
            },
            device="cpu",
        ),
        reward=center_line,
        action_space=ContinuousActionSpaceConfig(),
        worlds=WorldsConfig(names=["stub"], chunk_steps=128, rotations=1),
        training=TrainingConfig(total_timesteps=128, checkpoint_freq=128, eval_freq=64, n_eval_episodes=1),
        tracking=TrackingConfig(
            mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
            mlflow_experiment="extractor-e2e",
        ),
    )
    result = train(exp)
    assert isinstance(result, float)
    assert (tmp_path / "artifacts" / "extractor_e2e" / "final_model.zip").exists()
