"""Tests for the framework-agnostic trainer toolkit (TrainingContext services)
that let a custom (non-SB3) algorithm reuse the pipeline plumbing."""
from __future__ import annotations

import glob

import pytest

from gym_dr.action_space import ContinuousActionSpaceConfig
from gym_dr.config import TrainingConfig
from gym_dr.trainers.base import Trainer, TrainingContext, TrainResult


def _ctx(tmp_path, eval_worlds=None):
    return TrainingContext(
        run_dir=tmp_path, action_space=ContinuousActionSpaceConfig(),
        training=TrainingConfig(), name_prefix="ppo", eval_worlds=eval_worlds,
    )


def test_save_model_and_checkpoint(tmp_path):
    ctx = _ctx(tmp_path)
    p = ctx.save_model(lambda path: path.write_bytes(b"x"), name="initial_model")
    assert p.exists() and p.with_suffix(".model_metadata.json").exists()
    c = ctx.save_checkpoint(lambda path: path.write_bytes(b"x"), step=100)
    assert c.exists() and "checkpoints" in str(c)
    assert c.with_suffix(".model_metadata.json").exists()


def test_log_metrics_writes_tensorboard(tmp_path):
    pytest.importorskip("tensorboard")
    ctx = _ctx(tmp_path)
    ctx.log_metrics({"train/loss": 1.5, "train/kl": 0.01}, step=0)
    ctx.log_metrics({"train/loss": 1.2}, step=1)
    events = glob.glob(str(tmp_path / "tensorboard" / "**" / "events.*"), recursive=True)
    assert events, "expected a TensorBoard event file"


def test_record_episode(tmp_path):
    ctx = _ctx(tmp_path)
    s = {"dr/ep_completed": 1.0, "dr/ep_completed_clean": 1.0, "dr/ep_max_progress": 100.0}
    assert ctx.record_episode({"dr_episode": s}, 0) == s
    assert ctx.record_episode({}, 0) is None


class _StubEnv:
    def __init__(self, summary):
        self._s = summary
        self.set_world_calls = []

    def reset(self, **kw):
        return {"o": 0}, {}

    def step(self, action):
        return {"o": 0}, 1.0, True, False, {"dr_episode": self._s}

    def set_world(self, world):
        self.set_world_calls.append(world)


def test_evaluate_current_world(tmp_path):
    s = {"dr/ep_completed": 1.0, "dr/ep_completed_clean": 1.0,
         "dr/ep_max_progress": 80.0, "dr/ep_eval_reward": 1234.0, "dr/ep_offtrack_rate": 0.1}
    env = _StubEnv(s)
    agg = _ctx(tmp_path).evaluate(lambda obs: 0, env, n_episodes=2, step=10)
    assert agg["clean_completion_rate"] == 1.0
    assert agg["mean_reward"] == 1234.0
    assert env.set_world_calls == []  # no swap when eval_worlds is None


def test_evaluate_held_out_swaps_worlds(tmp_path):
    s = {"dr/ep_completed": 1.0, "dr/ep_completed_clean": 1.0, "dr/ep_eval_reward": 5.0}
    env = _StubEnv(s)
    agg = _ctx(tmp_path, eval_worlds=["WA", "WB"]).evaluate(lambda obs: 0, env, n_episodes=1)
    assert "WA" in env.set_world_calls and "WB" in env.set_world_calls
    assert agg["clean_completion_rate"] == 1.0


def test_custom_trainer_satisfies_protocol(tmp_path):
    class MyTrainer:
        def fit(self, env, ctx):
            ctx.save_model(lambda p: p.write_bytes(b"x"), name="initial_model")
            return TrainResult(final_eval_reward=1.0)

    assert isinstance(MyTrainer(), Trainer)  # runtime_checkable Protocol
    r = MyTrainer().fit(_StubEnv({}), _ctx(tmp_path))
    assert r.final_eval_reward == 1.0
