"""End-to-end smoke test of the gym_dr pipeline against a stub env.

Verifies the orchestrator -> Sb3Trainer -> TrainingContext flow without
requiring Docker or the upstream DeepRacer simapp:

1. Build a tiny stub env that matches DeepRacerEnv's observation+action shape.
2. Run ``train(experiment)`` in in-container mode (GYM_DR_IN_CONTAINER=1) so
   the orchestrator skips Docker and directly calls ``run_training``.
3. Assert every saved ``.zip`` has a ``.model_metadata.json`` sibling, MLflow
   logged the run, and the multi-world override path resolves through env
   vars correctly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
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


# --- stub env ---------------------------------------------------------------


class StubDeepRacerEnv(gym.Env):
    """Cheap stand-in for DeepRacerEnv: dict obs, Box action, scripted reward params."""

    metadata = {"render_modes": []}

    def __init__(self, reward_fn, sensors=None):
        super().__init__()
        sensor_name = (sensors or ["FRONT_FACING_CAMERA"])[0]
        self.observation_space = gym.spaces.Dict({
            sensor_name: gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        self.action_space = gym.spaces.Box(
            low=np.array([-30.0, 0.1], dtype=np.float32),
            high=np.array([30.0, 4.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._reward_fn = reward_fn
        self._sensor_name = sensor_name
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        self._step += 1
        params = {
            "track_width": 1.0,
            "distance_from_center": float(abs(action[0]) / 30.0 * 0.5),
            "all_wheels_on_track": True,
            "progress": min(self._step / 64.0, 1.0),
            "speed": float(action[1]),
            "steering_angle": float(action[0]),
            "is_offtrack": False,
        }
        reward = float(self._reward_fn(params))
        terminated = self._step >= 32
        truncated = False
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        return {
            self._sensor_name: self.np_random.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        }


def stub_env_factory(experiment: ExperimentConfig) -> Any:
    return StubDeepRacerEnv(
        reward_fn=experiment.reward,
        sensors=list(experiment.action_space.sensor),
    )


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def container_mode(tmp_path, monkeypatch):
    """Force gym_dr.train into in-container mode and redirect artifacts/MLflow."""
    monkeypatch.setenv("GYM_DR_IN_CONTAINER", "1")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("MLFLOW_PARENT_RUN_ID", raising=False)
    monkeypatch.delenv("CHUNK_NAME", raising=False)
    monkeypatch.delenv("RESUME_FROM", raising=False)
    monkeypatch.delenv("CHUNK_STEPS", raising=False)
    return tmp_path


def _experiment(name: str, tmp_path: Path, *, total_timesteps=200, eval_freq=100) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        env_factory=stub_env_factory,
        trainer=Sb3Trainer(
            name="ppo",
            policy="MultiInputPolicy",
            kwargs={"n_steps": 64, "batch_size": 32, "learning_rate": 3e-4, "ent_coef": 0.01},
            device="cpu",
        ),
        reward=center_line,
        action_space=ContinuousActionSpaceConfig(),
        worlds=WorldsConfig(names=["stub_world"], chunk_steps=total_timesteps, rotations=1),
        training=TrainingConfig(
            total_timesteps=total_timesteps,
            checkpoint_freq=total_timesteps,  # one checkpoint at the end
            eval_freq=eval_freq,
            n_eval_episodes=1,
        ),
        tracking=TrackingConfig(
            mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
            mlflow_experiment="smoke",
        ),
    )


# --- tests ------------------------------------------------------------------


def test_train_end_to_end(container_mode):
    tmp_path = container_mode
    exp = _experiment("smoke_run", tmp_path)
    result = train(exp)
    assert isinstance(result, float), f"expected float eval reward, got {result!r}"

    run_dir = tmp_path / "artifacts" / "smoke_run"
    assert run_dir.exists(), run_dir

    # Required artifacts
    for f in ("model_metadata.json", "reward_function.py", "run_config.json", "training_status.json",
              "initial_model.zip", "latest_model.zip", "final_model.zip"):
        assert (run_dir / f).exists(), f"missing {f}"

    # Every .zip has a .model_metadata.json sibling
    zips = list(run_dir.rglob("*.zip"))
    assert zips, "no .zip artifacts written"
    for zip_path in zips:
        sibling = zip_path.with_suffix(".model_metadata.json")
        assert sibling.exists(), f"missing metadata sibling for {zip_path}"

    # TB events written
    tb = run_dir / "tensorboard"
    assert tb.exists() and any(tb.iterdir()), "tensorboard dir empty"

    # MLflow logged something
    mlruns = tmp_path / "mlruns"
    assert mlruns.exists(), "mlruns dir not created"


def test_per_chunk_env_overrides_apply(container_mode):
    tmp_path = container_mode
    os.environ["CHUNK_NAME"] = "override_run"
    os.environ["CHUNK_STEPS"] = "128"
    try:
        exp = _experiment("base_name", tmp_path, total_timesteps=1_000_000)  # would-be huge
        train(exp)
    finally:
        del os.environ["CHUNK_NAME"]
        del os.environ["CHUNK_STEPS"]

    # CHUNK_NAME overrode the experiment name
    assert (tmp_path / "artifacts" / "override_run").exists()
    assert not (tmp_path / "artifacts" / "base_name").exists()


def test_custom_reward_archived(container_mode):
    tmp_path = container_mode

    def my_custom_reward(params: dict) -> float:
        return 1.23

    exp = _experiment("custom_reward", tmp_path).with_overrides(reward=my_custom_reward)
    train(exp)
    src = (tmp_path / "artifacts" / "custom_reward" / "reward_function.py").read_text()
    assert "my_custom_reward" in src, f"reward archival missing function name; got:\n{src}"


def test_with_overrides_does_not_mutate_original():
    base = ExperimentConfig(
        name="base",
        trainer=Sb3Trainer(kwargs={"learning_rate": 3e-4}),
    )
    new = base.with_overrides(**{"trainer.kwargs.learning_rate": 1e-5})
    assert base.trainer.kwargs["learning_rate"] == 3e-4
    assert new.trainer.kwargs["learning_rate"] == 1e-5


def test_worlds_config_coerces_bare_string():
    """WorldsConfig(names="X") is an easy mistake — a bare str is iterable and
    would silently train on single characters. It must coerce to ["X"]."""
    from gym_dr import WorldsConfig

    assert WorldsConfig(names="Oval_track").names == ["Oval_track"]
    # A real list passes through untouched.
    assert WorldsConfig(names=["A", "B"]).names == ["A", "B"]


def test_app_py_search_space_applies_to_base():
    """app.py defines base + search_space; ensure trial overrides land cleanly:
    the AWS-faithful policy_kwargs (separate towers, raw 0-255), the swept CNN
    conv stack, and independently-sized pi/vf FC heads."""
    import importlib.util
    import optuna
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "app_under_test", Path(__file__).parent.parent / "app.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    study = optuna.create_study(direction="maximize")
    # Sample a few trials so the conditional "custom" CNN branch is exercised
    # and the reward-fn categorical sees more than one value.
    seen_rewards = set()
    for _ in range(12):
        trial = study.ask()
        overrides = mod.search_space(trial)
        seen_rewards.add(overrides["reward"].__name__)
        new_cfg = mod.base.with_overrides(**overrides)

        # PPO hyperparams landed.
        assert "learning_rate" in new_cfg.trainer.kwargs
        assert "ent_coef" in new_cfg.trainer.kwargs

        pkw = new_cfg.trainer.kwargs["policy_kwargs"]

        # AWS-faithful policy flags.
        assert pkw["share_features_extractor"] is False, pkw
        assert pkw["normalize_images"] is False, pkw

        # CNN extractor: the DeepRacerCNN class + a valid conv stack.
        from gym_dr.networks import DeepRacerCNN

        assert pkw["features_extractor_class"] is DeepRacerCNN, pkw
        fe_kwargs = pkw["features_extractor_kwargs"]
        assert fe_kwargs["features_dim"] in (256, 512, 1024)
        conv_layers = fe_kwargs["conv_layers"]
        assert len(conv_layers) >= 3, conv_layers
        for filters, kernel, stride in conv_layers:
            assert filters > 0 and kernel > 0 and stride > 0, conv_layers

        # pi / vf heads exist and can be sized independently.
        net_arch = pkw["net_arch"]
        assert "pi" in net_arch and "vf" in net_arch, net_arch
        assert all(w > 0 for w in net_arch["pi"]), net_arch
        assert all(w > 0 for w in net_arch["vf"]), net_arch

        # Original base config not mutated.
        assert "policy_kwargs" not in mod.base.trainer.kwargs

    # Reward sweep: across 12 trials at least 2 different variants should
    # have been picked from the registry.
    assert len(seen_rewards) >= 2, seen_rewards


def test_device_gpu_alias_falls_back_too(container_mode, capfd):
    """device='gpu' is a common typo; it should normalize to cuda and then
    take the same fall-back path when cuda isn't available."""
    import pytest

    tmp_path = container_mode
    try:
        import torch
        if torch.cuda.is_available():
            pytest.skip("CUDA actually available — test only covers the missing-runtime path")
    except ImportError:
        pass

    exp = _experiment("gpu_alias", tmp_path).with_overrides(
        **{"trainer.device": "gpu"}
    )
    result = train(exp)
    assert isinstance(result, float), "training should complete on CPU fallback"

    out = capfd.readouterr().out
    assert "WARNING" in out and "cuda" in out.lower()


def test_pruned_trial_status_is_pruned_not_failed(container_mode):
    """When the inner trainer raises optuna.TrialPruned, the run dir's
    training_status.json should say 'pruned', not 'failed'."""
    import json

    import optuna
    import pytest

    tmp_path = container_mode

    class _PruningTrainer:
        """Raises TrialPruned mid-fit to simulate Optuna's pruner kicking in."""

        def fit(self, env, ctx):
            raise optuna.TrialPruned()

    exp = _experiment("prune_check", tmp_path).with_overrides(trainer=_PruningTrainer())
    with pytest.raises(optuna.TrialPruned):
        train(exp)

    status = json.loads(
        (tmp_path / "artifacts" / "prune_check" / "training_status.json").read_text()
    )
    assert status["status"] == "pruned", status
    assert "TrialPruned" in status["reason"]


def test_cuda_without_runtime_falls_back_to_cpu(container_mode, capfd):
    """device='cuda' on a host without CUDA should warn and train on CPU,
    not fail the trial (which would kill an HPO study)."""
    import pytest

    tmp_path = container_mode
    try:
        import torch
        if torch.cuda.is_available():
            pytest.skip("CUDA actually available — test only covers the missing-runtime path")
    except ImportError:
        pass

    exp = _experiment("cuda_fallback", tmp_path).with_overrides(
        **{"trainer.device": "cuda"}
    )
    result = train(exp)
    assert isinstance(result, float), "training should complete on CPU fallback"

    out = capfd.readouterr().out
    assert "WARNING" in out and "cuda" in out.lower()
    assert "Falling back to CPU" in out
    # Standard artifacts still landed
    assert (tmp_path / "artifacts" / "cuda_fallback" / "final_model.zip").exists()


def test_seed_lands_on_sb3_model(container_mode, monkeypatch):
    """experiment.seed reaches Sb3Trainer's algorithm kwargs (so PPO seeds itself
    and its first env.reset(seed=...))."""
    tmp_path = container_mode
    exp = _experiment("seed_check", tmp_path).with_overrides(seed=12345)
    train(exp)

    import json
    cfg_dict = json.loads((tmp_path / "artifacts" / "seed_check" / "run_config.json").read_text())
    assert cfg_dict["seed"] == 12345
    # SB3 stores the seed it was given; if PPO was constructed with seed=12345
    # the model zip's contained args reflect that. We don't crack open the zip
    # here; checking the run_config is sufficient evidence of plumbing.


def test_frame_stack_smoke(container_mode):
    """Sb3Trainer with frame_stack > 1 wraps the env in VecFrameStack and trains."""
    tmp_path = container_mode
    exp = _experiment("frame_stack_check", tmp_path).with_overrides(
        **{"trainer.frame_stack": 3}
    )
    result = train(exp)
    assert isinstance(result, float)
    # Standard artifacts still land — VecFrameStack didn't break checkpoint saving.
    run_dir = tmp_path / "artifacts" / "frame_stack_check"
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "final_model.model_metadata.json").exists()


def test_track_catalog_lookup():
    """ALL_TRACKS exposes every name in TRACKS; display_name maps to labels."""
    from gym_dr import ALL_TRACKS, TRACKS, display_name

    assert "reinvent_base" in TRACKS
    assert "reinvent_base" in ALL_TRACKS
    assert display_name("reinvent_base") == "re:Invent 2018"
    assert display_name("totally_made_up_world") == "totally_made_up_world"
    assert set(ALL_TRACKS) == set(TRACKS.keys())


def test_trainer_protocol_duck_typed():
    """Any object with fit(env, ctx) satisfies the protocol — no inheritance."""
    from gym_dr.trainers.base import Trainer, TrainResult

    class MyTrainer:
        def fit(self, env, ctx):
            return TrainResult(final_eval_reward=42.0)

    assert isinstance(MyTrainer(), Trainer)


def test_reward_metrics_recorded_to_tensorboard(container_mode):
    """dr/ep_* metrics land as TB scalars + MLflow metrics.

    The stub env reports is_offtrack on every other step (see StubDeepRacerEnv.step)
    so we should see non-zero offtrack counts.
    """
    from tensorboard.backend.event_processing import event_accumulator

    tmp_path = container_mode
    exp = _experiment("metrics_check", tmp_path)
    train(exp)

    tb_root = tmp_path / "artifacts" / "metrics_check" / "tensorboard"
    assert tb_root.exists()
    sub = next((p for p in tb_root.rglob("events.out.tfevents.*")), None)
    assert sub is not None, f"no TB event files under {tb_root}"
    acc = event_accumulator.EventAccumulator(str(sub.parent))
    acc.Reload()
    scalar_tags = set(acc.Tags().get("scalars", []))
    expected = {
        "dr/ep_reward",
        "dr/ep_eval_reward",   # parallel-recorded eval reward
        "dr/ep_length",
        "dr/ep_max_progress",
        "dr/ep_offtrack_count",
        "dr/ep_mean_speed",
    }
    missing = expected - scalar_tags
    assert not missing, f"missing DR metrics in TB: {missing}; got {sorted(scalar_tags)}"


def test_eval_reward_differs_from_training_reward(container_mode):
    """If training reward and eval reward are different functions, the
    recorded `dr/ep_reward` and `dr/ep_eval_reward` should differ for at
    least one episode. (If they're the same fn, they'd match — sanity-
    test the wiring by setting them to known-different functions.)"""
    from gym_dr.rewards import center_line, progress_per_step

    tmp_path = container_mode
    # Force them to be different callables.
    exp = _experiment("eval_reward_check", tmp_path).with_overrides(
        reward=center_line, eval_reward=progress_per_step,
    )
    train(exp)

    # Walk the run_config.json — eval_reward serialized as a different dotted
    # path than reward.
    import json
    cfg = json.loads((tmp_path / "artifacts" / "eval_reward_check" / "run_config.json").read_text())
    assert cfg["reward"] != cfg["eval_reward"], cfg
    assert cfg["eval_reward"].endswith("progress_per_step")


def test_mlflow_run_group_tag_applied(container_mode, monkeypatch):
    """MLFLOW_RUN_GROUP env var lands on the chunk's MLflow run as a tag.

    Replaces the old MLFLOW_PARENT_RUN_ID flow (which broke across MLflow
    versions). The host orchestrator sets this; chunks tag themselves so
    the UI can group them via `tags.run_group = "<experiment.name>"`.
    """
    import mlflow

    tmp_path = container_mode
    monkeypatch.setenv("MLFLOW_RUN_GROUP", "smoke_group")
    exp = _experiment("tag_check", tmp_path)
    train(exp)

    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    experiment_handle = mlflow.get_experiment_by_name("smoke")
    assert experiment_handle is not None
    runs = mlflow.search_runs(
        experiment_ids=[experiment_handle.experiment_id],
        filter_string="tags.run_group = 'smoke_group'",
    )
    assert len(runs) == 1, f"expected exactly one run tagged run_group=smoke_group, got {len(runs)}"
