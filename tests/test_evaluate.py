"""Tests for the view-mode evaluation runner.

Exercises ``gym_dr.evaluate.run_evaluation`` against the stub env from
``test_smoke.py`` — no Docker, no sim. Covers a plain model and a
frame-stacked model (the frame_stack must be re-applied at eval time or
the obs shape won't match the trained policy).
"""
from __future__ import annotations

import json

import pytest

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
from gym_dr.evaluate import run_evaluation
from tests.test_smoke import stub_env_factory  # noqa: F401


def _experiment(name, tmp_path, *, frame_stack=1):
    return ExperimentConfig(
        name=name,
        env_factory=stub_env_factory,
        trainer=Sb3Trainer(
            name="ppo",
            policy="MultiInputPolicy",
            kwargs={"n_steps": 64, "batch_size": 32, "learning_rate": 3e-4, "ent_coef": 0.01},
            device="cpu",
            frame_stack=frame_stack,
        ),
        reward=center_line,
        action_space=ContinuousActionSpaceConfig(),
        worlds=WorldsConfig(names=["stub"], chunk_steps=200, rotations=1),
        training=TrainingConfig(total_timesteps=200, checkpoint_freq=200, eval_freq=100, n_eval_episodes=1),
        tracking=TrackingConfig(
            mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
            mlflow_experiment="eval-test",
        ),
        seed=7,
    )


@pytest.fixture
def container_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GYM_DR_IN_CONTAINER", "1")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("MLFLOW_RUN_GROUP", raising=False)
    return tmp_path


def test_run_evaluation_plain(container_mode):
    """Train a tiny model, then evaluate it for a couple of episodes."""
    tmp_path = container_mode
    exp = _experiment("eval_plain", tmp_path)
    train(exp)

    model = tmp_path / "artifacts" / "eval_plain" / "final_model.zip"
    assert model.exists()

    summaries = run_evaluation(exp, model, n_episodes=2, step_log_every=5)
    assert len(summaries) == 2
    for s in summaries:
        # The metrics wrapper should have produced a dr_episode summary.
        assert "dr/ep_reward" in s
        assert "dr/ep_length" in s


def test_run_evaluation_frame_stacked(container_mode):
    """A frame_stack>1 model must still evaluate — run_evaluation re-applies
    VecFrameStack so the observation shape matches the trained policy."""
    tmp_path = container_mode
    exp = _experiment("eval_fs", tmp_path, frame_stack=3)
    train(exp)

    run_dir = tmp_path / "artifacts" / "eval_fs"
    model = run_dir / "final_model.zip"
    assert model.exists()
    # run_config.json should record the frame_stack so eval can recover it.
    cfg = json.loads((run_dir / "run_config.json").read_text())
    assert cfg["trainer"]["frame_stack"] == 3

    # No explicit frame_stack arg — it must be read from run_config.json.
    summaries = run_evaluation(exp, model, n_episodes=1, step_log_every=5)
    assert len(summaries) == 1
    assert "dr/ep_reward" in summaries[0]


def test_frame_stack_override_arg(container_mode):
    """Explicit frame_stack arg wins over run_config.json."""
    tmp_path = container_mode
    exp = _experiment("eval_override", tmp_path, frame_stack=2)
    train(exp)
    model = tmp_path / "artifacts" / "eval_override" / "final_model.zip"

    # Passing the matching value explicitly should work identically.
    summaries = run_evaluation(exp, model, n_episodes=1, frame_stack=2, step_log_every=10)
    assert len(summaries) == 1
