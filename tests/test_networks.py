"""Unit + end-to-end tests for the DeepRacer CNN and the grayscale wrapper.

The extractor follows SB3's contract: it receives observations already
preprocessed — image keys channels-first. With ``normalize_images=False``
(which app.py sets, to match the physical car) SB3 feeds raw 0-255 floats.
The standalone tests construct channels-first spaces accordingly.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch


def _chw_dict_space(c=1, h=120, w=160):
    """Channels-first image space — what SB3 passes after VecTransposeImage."""
    return gym.spaces.Dict(
        {"FRONT_FACING_CAMERA": gym.spaces.Box(0, 255, (c, h, w), np.uint8)}
    )


# --------------------------------------------------------------------------- #
# DeepRacerCNN
# --------------------------------------------------------------------------- #

def test_deepracer_cnn_named_presets():
    """Each named DeepRacer arch builds and runs on the real 120x160 obs."""
    from gym_dr.networks import DEEPRACER_CONV_PRESETS, DeepRacerCNN

    space = _chw_dict_space(c=1, h=120, w=160)
    for name, conv in DEEPRACER_CONV_PRESETS.items():
        ext = DeepRacerCNN(space, features_dim=512, conv_layers=conv)
        out = ext({"FRONT_FACING_CAMERA": torch.zeros(2, 1, 120, 160)})
        assert out.shape == (2, 512), (name, out.shape)


def test_deepracer_cnn_custom_stack():
    """Arbitrary (filters, kernel, stride) stacks work."""
    from gym_dr.networks import DeepRacerCNN

    ext = DeepRacerCNN(
        _chw_dict_space(),
        features_dim=256,
        conv_layers=((16, 5, 2), (32, 3, 1), (64, 3, 1)),
    )
    out = ext({"FRONT_FACING_CAMERA": torch.zeros(3, 1, 120, 160)})
    assert out.shape == (3, 256)


def test_deepracer_cnn_memoized_identity():
    """DeepRacerCNN must be the SAME class object across imports — SB3/HPO
    compare it by identity, and rebuilding per-access would break that."""
    from gym_dr.networks import DeepRacerCNN as A
    from gym_dr.networks import DeepRacerCNN as B

    assert A is B


def test_deepracer_cnn_channel_agnostic():
    """Handles stacked (multi-channel) grayscale obs, e.g. frame_stack=3."""
    from gym_dr.networks import DeepRacerCNN

    ext = DeepRacerCNN(_chw_dict_space(c=3), features_dim=128)
    out = ext({"FRONT_FACING_CAMERA": torch.zeros(1, 3, 120, 160)})
    assert out.shape == (1, 128)


def test_separate_towers_e2e(tmp_path, monkeypatch):
    """End-to-end: Sb3Trainer with DeepRacerCNN + share_features_extractor=False
    + normalize_images=False trains on the stub env. Validates the full
    AWS-faithful policy_kwargs path through real SB3."""
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
    from gym_dr.networks import DeepRacerCNN
    from tests.test_smoke import stub_env_factory

    monkeypatch.setenv("GYM_DR_IN_CONTAINER", "1")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    exp = ExperimentConfig(
        name="towers_e2e",
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
                    "share_features_extractor": False,
                    "normalize_images": False,
                    "features_extractor_class": DeepRacerCNN,
                    "features_extractor_kwargs": {
                        "conv_layers": ((16, 5, 2), (32, 3, 1)),
                        "features_dim": 128,
                    },
                    "net_arch": dict(pi=[64], vf=[128, 64]),  # heads sized differently
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
            mlflow_experiment="towers-e2e",
        ),
    )
    result = train(exp)
    assert isinstance(result, float)
    assert (tmp_path / "artifacts" / "towers_e2e" / "final_model.zip").exists()


# --------------------------------------------------------------------------- #
# GrayscaleObs wrapper
# --------------------------------------------------------------------------- #

class _StubRGBEnv(gym.Env):
    """Minimal Dict-obs env with an RGB camera key, for wrapper tests."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {"FRONT_FACING_CAMERA": gym.spaces.Box(0, 255, (8, 8, 3), np.uint8)}
        )
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"FRONT_FACING_CAMERA": np.full((8, 8, 3), 100, np.uint8)}, {}

    def step(self, action):
        obs = {"FRONT_FACING_CAMERA": np.full((8, 8, 3), 100, np.uint8)}
        return obs, 0.0, True, False, {}


def test_grayscale_wrapper_space_and_values():
    from gym_dr.envs.wrappers import GrayscaleObs

    env = GrayscaleObs(_StubRGBEnv())
    cam_space = env.observation_space["FRONT_FACING_CAMERA"]
    assert cam_space.shape == (8, 8, 1), cam_space.shape
    assert cam_space.dtype == np.uint8

    obs, _ = env.reset()
    cam = obs["FRONT_FACING_CAMERA"]
    assert cam.shape == (8, 8, 1)
    assert cam.dtype == np.uint8
    # All channels were 100; BT.601 luma of (100,100,100) is 100.
    assert np.all(cam == 100)


def test_grayscale_wrapper_luma_weights():
    """Pure-red input -> 0.299 * 255 ~= 76; confirms BT.601 weighting, not a
    plain channel average."""
    from gym_dr.envs.wrappers import GrayscaleObs

    class _RedEnv(_StubRGBEnv):
        def reset(self, *, seed=None, options=None):
            rgb = np.zeros((8, 8, 3), np.uint8)
            rgb[..., 0] = 255  # red channel
            return {"FRONT_FACING_CAMERA": rgb}, {}

    env = GrayscaleObs(_RedEnv())
    obs, _ = env.reset()
    val = int(obs["FRONT_FACING_CAMERA"][0, 0, 0])
    assert val == round(0.299 * 255), val  # 76


def test_grayscale_wrapper_rejects_non_dict():
    from gym_dr.envs.wrappers import GrayscaleObs

    class _BoxEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (8, 8, 3), np.uint8)
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    with pytest.raises(TypeError, match="Dict observation space"):
        GrayscaleObs(_BoxEnv())
