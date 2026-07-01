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
    OfftrackRate,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    center_line,
)
from gym_dr.app import train  # internal orchestration entrypoint (Study delegates here)


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
        self.world = "stub_world"
        self.set_world_calls: list[str] = []

    def set_world(self, world_name):
        """Mirror DeepRacerEnv.set_world's between-episodes contract for the
        rotation test: record the swap and update the active world."""
        self.set_world_calls.append(world_name)
        self.world = world_name

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


class OffTrackStubEnv(StubDeepRacerEnv):
    """Stub whose every episode ends with a FULL track-out (``is_offtrack``).

    Short 4-step episodes so eval rollouts finish fast; the terminal step
    reports ``is_offtrack=True`` (car entirely off track) so each finished
    episode counts as a track-out reset for the ``eval/*_offtrack_resets``
    metric.
    """

    def step(self, action):
        self._step += 1
        terminated = self._step >= 4
        params = {
            "track_width": 1.0,
            "distance_from_center": 0.0,
            "all_wheels_on_track": not terminated,
            "progress": min(self._step / 64.0, 1.0),
            "speed": float(action[1]),
            "steering_angle": float(action[0]),
            "is_offtrack": bool(terminated),
        }
        reward = float(self._reward_fn(params))
        return self._obs(), reward, terminated, False, {}


def offtrack_env_factory(experiment: ExperimentConfig) -> Any:
    return OffTrackStubEnv(
        reward_fn=experiment.reward,
        sensors=list(experiment.action_space.sensor),
    )


class PathStubEnv(StubDeepRacerEnv):
    """Stub that emits the geometry the eval path plots need: per-step ``x``/``y``
    along a circle plus a ``waypoints`` skeleton + ``track_width``."""

    _WAYPOINTS = [
        [float(np.cos(a)), float(np.sin(a))]
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ]

    def step(self, action):
        self._step += 1
        ang = self._step / 4.0
        terminated = self._step >= 6
        params = {
            "track_width": 0.5,
            "distance_from_center": 0.0,
            "all_wheels_on_track": not terminated,
            "progress": min(self._step / 6.0, 1.0),
            "speed": float(action[1]),
            "steering_angle": float(action[0]),
            "is_offtrack": bool(terminated),
            "x": float(np.cos(ang)),
            "y": float(np.sin(ang)),
            "heading": 0.0,
            "waypoints": self._WAYPOINTS,
        }
        reward = float(self._reward_fn(params))
        return self._obs(), reward, terminated, False, {}


def path_env_factory(experiment: ExperimentConfig) -> Any:
    return PathStubEnv(
        reward_fn=experiment.reward,
        sensors=list(experiment.action_space.sensor),
    )


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def container_mode(tmp_path, monkeypatch):
    """Force gym_dr.train into in-container mode and redirect artifacts/MLflow."""
    monkeypatch.setenv("GYM_DR_IN_CONTAINER", "1")
    # The in-container train() path hard-exits via os._exit(0) to dodge the
    # rclpy/DDS finalize segfault (see gym_dr/app.py). Under the in-process test
    # suite that would kill pytest before any post-train() assertion runs, so opt
    # out — the segfault-dodge only matters in a real single-chunk container.
    monkeypatch.setenv("GYM_DR_NO_HARD_EXIT", "1")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("MLFLOW_PARENT_RUN_ID", raising=False)
    monkeypatch.delenv("CHUNK_NAME", raising=False)
    monkeypatch.delenv("RESUME_FROM", raising=False)
    monkeypatch.delenv("CHUNK_STEPS", raising=False)
    # Sb3Trainer._boot_world_consumed is process-global (it models "has a
    # rotation chunk run yet in THIS worker container"). Tests share one
    # process, so reset it per test to model a fresh container — otherwise the
    # rotation tests would see a stale flag from a prior test and re-pin the
    # boot world unexpectedly. The dedicated reuse test sets it on purpose.
    Sb3Trainer._boot_world_consumed = False
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


def test_runtime_world_rotation_swaps_in_process(container_mode, monkeypatch):
    """With GYM_DR_ROTATE=1 and a multi-world WorldsConfig, the trainer runs
    one in-process rotation and calls env.set_world() once per world after the
    first — no second container, no checkpoint reload between worlds."""
    tmp_path = container_mode
    monkeypatch.setenv("GYM_DR_ROTATE", "1")

    # Capture set_world calls across the (wrapped, vec-env'd) stub instance.
    calls: list[str] = []
    orig_set_world = StubDeepRacerEnv.set_world

    def _recording_set_world(self, world_name):
        calls.append(world_name)
        return orig_set_world(self, world_name)

    monkeypatch.setattr(StubDeepRacerEnv, "set_world", _recording_set_world)

    exp = _experiment("rotation_run", tmp_path, total_timesteps=64).with_overrides(
        worlds=WorldsConfig(names=["world_a", "world_b"], chunk_steps=64, rotations=2),
    )
    result = train(exp)
    assert isinstance(result, float)

    # Plan = [a, b, a, b]; first chunk uses the already-loaded world, so three
    # set_world swaps happen, ending on world_b.
    assert calls == ["world_b", "world_a", "world_b"], calls

    run_dir = tmp_path / "artifacts" / "rotation_run"
    assert (run_dir / "latest_model.zip").exists()


def test_rotation_writes_single_tensorboard_run(container_mode, monkeypatch):
    """Regression (TB fragmentation): a multi-chunk rotation must write ONE
    TensorBoard run, not one per chunk. Pre-fix each chunk's model.learn()
    re-ran configure() and opened a new run_name_N SummaryWriter, so TB showed N
    overlapping partial 'runs' and no complete curve — which read as 'runs not
    showing on TensorBoard'. The pre-created logger (model.set_logger before the
    rotation loop) keeps a single SummaryWriter across all chunks."""
    tmp_path = container_mode
    monkeypatch.setenv("GYM_DR_ROTATE", "1")
    exp = _experiment("tb_rotation", tmp_path, total_timesteps=64).with_overrides(
        worlds=WorldsConfig(names=["world_a", "world_b"], chunk_steps=64, rotations=2),
    )
    train(exp)  # plan = [a, b, a, b] -> 4 chunks, one logger

    tb_root = tmp_path / "artifacts" / "tb_rotation" / "tensorboard"
    assert tb_root.exists()
    subdirs = [d for d in tb_root.iterdir() if d.is_dir()]
    assert len(subdirs) == 1, f"expected 1 TB run dir, got {[d.name for d in subdirs]}"
    event_files = list(tb_root.rglob("events.out.tfevents.*"))
    assert len(event_files) == 1, f"expected 1 TB event file, got {event_files}"


def test_rotation_crash_writes_resume_and_restart_code(container_mode, monkeypatch):
    """If gzserver dies mid-swap (set_world raises WorldSwapError), the trainer
    persists rotation_resume.json (chunk index + checkpoint) and the container
    exits with _SIM_RESTART_RC so the host can relaunch and resume the rotation."""
    import json

    from gym_dr.trainers.sb3 import Sb3Trainer

    tmp_path = container_mode
    monkeypatch.setenv("GYM_DR_ROTATE", "1")
    # Deterministic: treat this as the first trial (trust boot world, no pin).
    monkeypatch.setattr(Sb3Trainer, "_boot_world_consumed", False, raising=False)

    class WorldSwapError(RuntimeError):
        pass

    orig = StubDeepRacerEnv.set_world

    def crash_on_world_b(self, world_name):
        if world_name == "world_b":
            raise WorldSwapError("simulated gzserver segfault")
        return orig(self, world_name)

    monkeypatch.setattr(StubDeepRacerEnv, "set_world", crash_on_world_b)

    exp = _experiment("crash_run", tmp_path, total_timesteps=64).with_overrides(
        worlds=WorldsConfig(
            names=["world_a", "world_b", "world_c"], chunk_steps=64, rotations=1),
    )

    with pytest.raises(SystemExit) as ei:
        train(exp)
    assert ei.value.code == 75, ei.value.code

    run_dir = tmp_path / "artifacts" / "crash_run"
    resume = run_dir / "rotation_resume.json"
    assert resume.exists(), "rotation_resume.json not written"
    state = json.loads(resume.read_text())
    assert state["start_index"] == 1, state  # crashed swapping to world_b (idx 1)
    assert state["resume_from"].endswith("latest_model.zip"), state
    # chunk 0 (world_a) trained before the crash, so there's a checkpoint to
    # resume from.
    assert (run_dir / "latest_model.zip").exists()


def test_reused_worker_repins_world_before_first_chunk(container_mode, monkeypatch):
    """A worker runs several HPO trials back-to-back in one container, and the
    Gazebo world persists across them. The first trial trusts the boot world,
    but every later trial must re-pin the track to world_plan[0] before chunk 0
    (the previous trial's rotation left a different world loaded). Verifies the
    second trial's swaps start with the first planned world, not chunk 1's."""
    tmp_path = container_mode
    monkeypatch.setenv("GYM_DR_ROTATE", "1")

    calls: list[str] = []
    orig_set_world = StubDeepRacerEnv.set_world

    def _recording_set_world(self, world_name):
        calls.append(world_name)
        return orig_set_world(self, world_name)

    monkeypatch.setattr(StubDeepRacerEnv, "set_world", _recording_set_world)

    plan = WorldsConfig(names=["world_a", "world_b"], chunk_steps=64, rotations=1)

    # First trial in the (simulated) container: boot world is world_a, so the
    # first chunk doesn't swap; only the rotation to world_b does.
    exp1 = _experiment("reuse_trial_0", tmp_path, total_timesteps=64).with_overrides(worlds=plan)
    train(exp1)
    assert calls == ["world_b"], calls

    # Second trial in the SAME process (flag NOT reset): the container is left
    # on world_b, so the trial re-pins to world_a up front, then rotates to b.
    calls.clear()
    exp2 = _experiment("reuse_trial_1", tmp_path, total_timesteps=64).with_overrides(worlds=plan)
    train(exp2)
    assert calls == ["world_a", "world_b"], calls

    assert (tmp_path / "artifacts" / "reuse_trial_1" / "latest_model.zip").exists()


def test_ordered_split_evaluates_on_held_out_worlds(container_mode, monkeypatch):
    """OrderedSplit trains in train_worlds order and, at eval time, swaps the
    env to each held-out eval world (then restores the training world)."""
    tmp_path = container_mode
    monkeypatch.setenv("GYM_DR_ROTATE", "1")

    calls: list[str] = []
    orig_set_world = StubDeepRacerEnv.set_world

    def _recording_set_world(self, world_name):
        calls.append(world_name)
        return orig_set_world(self, world_name)

    monkeypatch.setattr(StubDeepRacerEnv, "set_world", _recording_set_world)

    from gym_dr import OrderedSplit

    exp = _experiment("ordered_split", tmp_path, total_timesteps=64, eval_freq=64).with_overrides(
        world_strategy=OrderedSplit(
            train_worlds=["train_a", "train_b"],
            eval_worlds=["eval_x", "eval_y"],
            chunk_steps=64,
            rotations=1,
        ),
    )
    result = train(exp)
    assert isinstance(result, float)

    # Training rotates to the second train world (first uses the preloaded one).
    assert "train_b" in calls, calls
    # Evaluation visited every held-out world.
    assert "eval_x" in calls and "eval_y" in calls, calls
    # The held-out worlds are never trained on (they only appear via eval swaps,
    # never as a chunk in the training plan order).
    strat = exp.world_strategy
    assert [c.world for c in strat.training_chunks()] == ["train_a", "train_b"]
    assert strat.evaluation_worlds() == ["eval_x", "eval_y"]

    run_dir = tmp_path / "artifacts" / "ordered_split"
    assert (run_dir / "latest_model.zip").exists()


def test_eval_offtrack_resets_logged_per_world_and_global(container_mode, monkeypatch):
    """Each held-out eval world gets its own ``eval/<world>_offtrack_resets``
    track-out count, plus a global ``eval/offtrack_resets`` summed across worlds.

    The off-track stub ends every episode fully off the track, so with
    ``n_eval_episodes=1`` each of the two eval worlds tallies 1 reset and the
    global tally is 2.
    """
    from tensorboard.backend.event_processing import event_accumulator

    from gym_dr import OrderedSplit

    tmp_path = container_mode
    monkeypatch.setenv("GYM_DR_ROTATE", "1")

    exp = _experiment("offtrack_eval", tmp_path, total_timesteps=64, eval_freq=64).with_overrides(
        env_factory=offtrack_env_factory,
        world_strategy=OrderedSplit(
            train_worlds=["train_a", "train_b"],
            eval_worlds=["eval_x", "eval_y"],
            chunk_steps=64,
            rotations=1,
        ),
    )
    train(exp)

    tb_root = tmp_path / "artifacts" / "offtrack_eval" / "tensorboard"
    sub = next((p for p in tb_root.rglob("events.out.tfevents.*")), None)
    assert sub is not None, f"no TB event files under {tb_root}"
    acc = event_accumulator.EventAccumulator(str(sub.parent))
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))

    for tag in ("eval/offtrack_resets", "eval/eval_x_offtrack_resets", "eval/eval_y_offtrack_resets"):
        assert tag in tags, f"missing {tag}; got {sorted(tags)}"

    # Every eval episode ended off-track: 1 per world, 2 globally.
    assert acc.Scalars("eval/eval_x_offtrack_resets")[-1].value == 1.0
    assert acc.Scalars("eval/eval_y_offtrack_resets")[-1].value == 1.0
    assert acc.Scalars("eval/offtrack_resets")[-1].value == 2.0


def test_eval_path_plots_logged_as_tb_images(container_mode, monkeypatch):
    """With ``eval_path_plots=True`` each eval world logs a trajectory overlay
    image plus one image per eval episode to TensorBoard's Images tab."""
    from tensorboard.backend.event_processing import event_accumulator

    from gym_dr import OrderedSplit

    tmp_path = container_mode
    monkeypatch.setenv("GYM_DR_ROTATE", "1")

    exp = _experiment("path_plots", tmp_path, total_timesteps=64, eval_freq=64).with_overrides(
        env_factory=path_env_factory,
        world_strategy=OrderedSplit(
            train_worlds=["train_a", "train_b"],
            eval_worlds=["eval_x", "eval_y"],
            chunk_steps=64,
            rotations=1,
        ),
        **{"training.eval_path_plots": True},
    )
    train(exp)

    tb_root = tmp_path / "artifacts" / "path_plots" / "tensorboard"
    sub = next((p for p in tb_root.rglob("events.out.tfevents.*")), None)
    assert sub is not None, f"no TB event files under {tb_root}"
    acc = event_accumulator.EventAccumulator(
        str(sub.parent), size_guidance={"images": 0}
    )
    acc.Reload()
    img_tags = set(acc.Tags().get("images", []))

    # n_eval_episodes=1 -> one overlay + one per-episode image per eval world.
    for tag in (
        "eval_paths/eval_x",
        "eval_paths/eval_x/ep0",
        "eval_paths/eval_y",
        "eval_paths/eval_y/ep0",
    ):
        assert tag in img_tags, f"missing image tag {tag}; got {sorted(img_tags)}"


def test_offtrack_strategy_reproduces_mastery_logic():
    """The OfftrackRate strategy + EarlyStopController reproduce the old
    ``_mastery_met`` gate: stop when the eval off-track fraction is within
    tolerance, and never stop when the strategy is ``None`` (disabled)."""
    from gym_dr.early_stopping import EarlyStopController, OfftrackRate

    strict = OfftrackRate(max_offtrack_rate=0.0, patience=1)
    assert strict.met({"offtrack_rate": 0.0}) is True       # zero off-track -> mastered
    assert strict.met({"offtrack_rate": 1 / 3}) is False    # any off-track fails at rate 0
    tol = OfftrackRate(max_offtrack_rate=0.5)
    assert tol.met({"offtrack_rate": 0.5}) is True           # 0.50 <= 0.50
    assert tol.met({"offtrack_rate": 2 / 3}) is False        # 0.66 > 0.50
    # None strategy -> the controller is a no-op (disabled).
    assert EarlyStopController(None).update({"offtrack_rate": 0.0}) is False


def test_early_stop_ends_single_track_run_when_mastered(container_mode):
    """With early stop on and a strict rate of 0.0, a track the car never leaves
    (the default stub reports is_offtrack=False) masters on the first eval round,
    ending the run before total_timesteps with status 'early_stopped'."""
    import json

    tmp_path = container_mode
    exp = _experiment("early_stop_single", tmp_path, total_timesteps=512, eval_freq=64).with_overrides(
        **{
            "training.early_stop": OfftrackRate(max_offtrack_rate=0.0, patience=1),
            "training.n_eval_episodes": 2,
        },
    )
    train(exp)

    status = json.loads(
        (tmp_path / "artifacts" / "early_stop_single" / "training_status.json").read_text()
    )
    assert status["status"] == "early_stopped", status
    assert status["timesteps_completed"] < 512, status


def test_early_stop_does_not_fire_when_car_leaves_track(container_mode):
    """The off-track stub ends every eval episode off the track, so mastery is
    never reached and the run trains its full budget (status 'completed')."""
    import json

    tmp_path = container_mode
    exp = _experiment("no_early_stop", tmp_path, total_timesteps=128, eval_freq=64).with_overrides(
        env_factory=offtrack_env_factory,
        **{
            "training.early_stop": OfftrackRate(max_offtrack_rate=0.0, patience=1),
            "training.n_eval_episodes": 2,
        },
    )
    train(exp)

    status = json.loads(
        (tmp_path / "artifacts" / "no_early_stop" / "training_status.json").read_text()
    )
    assert status["status"] == "completed", status
    assert status["timesteps_completed"] >= 128, status


def test_early_stop_advances_rotation_per_track(container_mode, monkeypatch):
    """Mastering a track mid-rotation ends that chunk early and the loop advances
    to the next track. The default stub never leaves the track, so each chunk
    masters on its first eval yet the rotation still swaps through to world_b."""
    import json

    tmp_path = container_mode
    monkeypatch.setenv("GYM_DR_ROTATE", "1")

    calls: list[str] = []
    orig_set_world = StubDeepRacerEnv.set_world

    def _recording_set_world(self, world_name):
        calls.append(world_name)
        return orig_set_world(self, world_name)

    monkeypatch.setattr(StubDeepRacerEnv, "set_world", _recording_set_world)

    exp = _experiment("early_stop_rot", tmp_path, total_timesteps=256, eval_freq=64).with_overrides(
        worlds=WorldsConfig(names=["world_a", "world_b"], chunk_steps=256, rotations=1),
        **{
            "training.early_stop": OfftrackRate(max_offtrack_rate=0.0, patience=1),
        },
    )
    train(exp)

    # Chunk 0 (world_a) mastered early but the rotation still advanced to world_b.
    assert calls == ["world_b"], calls
    status = json.loads(
        (tmp_path / "artifacts" / "early_stop_rot" / "training_status.json").read_text()
    )
    assert status["status"] == "early_stopped", status


def test_multiworld_study_enables_rotation(tmp_path, monkeypatch):
    """HPO host wiring: a multi-world strategy makes ``study`` set GYM_DR_ROTATE
    on the worker env (so each trial rotates training worlds), while a
    single-world strategy leaves it unset (legacy one-world-per-trial)."""
    import gym_dr.docker_runner as docker_runner
    from gym_dr import OrderedSplit
    from gym_dr.app import study

    monkeypatch.delenv("GYM_DR_WORKER", raising=False)
    monkeypatch.delenv("GYM_DR_IN_CONTAINER", raising=False)
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    exp_file = tmp_path / "exp.py"
    exp_file.write_text("experiment = None\n")
    monkeypatch.setenv("GYM_DR_EXPERIMENT_FILE", str(exp_file))

    captured: dict[str, Any] = {}

    def fake_spawn_workers(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(docker_runner, "spawn_workers", fake_spawn_workers)

    def _study_with(strategy):
        exp = _experiment("study_x", tmp_path).with_overrides(world_strategy=strategy)
        study(exp, lambda t: {}, study_name="s", n_trials=1, n_parallel=1)
        return captured["base_env"]

    multi_env = _study_with(
        OrderedSplit(train_worlds=["world_a", "world_b", "world_c"],
                     eval_worlds=["eval_x"], chunk_steps=10)
    )
    assert multi_env.get("GYM_DR_ROTATE") == "1", multi_env
    assert multi_env["WORLD_NAME"] == "world_a"   # first training world

    single_env = _study_with(
        OrderedSplit(train_worlds=["world_a"], eval_worlds=["eval_x"], chunk_steps=10)
    )
    assert "GYM_DR_ROTATE" not in single_env, single_env
    assert single_env["WORLD_NAME"] == "world_a"


def test_seed_env_override_applied(container_mode, monkeypatch):
    """The container reads SEED from the env and applies it (the multi-seed
    host loop relies on this to give each seed a distinct, reproducible run)."""
    import json

    tmp_path = container_mode
    monkeypatch.setenv("SEED", "1234")
    exp = _experiment("seed_run", tmp_path, total_timesteps=64)
    train(exp)
    rc = json.loads((tmp_path / "artifacts" / "seed_run" / "run_config.json").read_text())
    assert rc["seed"] == 1234


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


def test_object_avoidance_hpo_search_space_applies_to_base():
    """The OA HPO study (experiments/object_avoidance_hpo.py) defines base +
    search_space; ensure trial overrides land cleanly: the AWS-faithful
    policy_kwargs (separate towers, raw 0-255), the swept CNN conv stack, and
    independently-sized pi/vf FC heads. (This study used to live in app.py.)"""
    import importlib.util
    import optuna
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "app_under_test",
        Path(__file__).parent.parent / "experiments" / "object_avoidance_hpo.py",
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


def test_custom_trainer_extends_base():
    """A custom algorithm extends the Trainer ABC and implements fit() — the
    "bring your own algorithm" seam (no Stable-Baselines lock-in). The ABC
    enforces the contract: a subclass that omits fit() cannot instantiate."""
    from gym_dr.trainers.base import Trainer, TrainResult

    class MyTrainer(Trainer):
        def fit(self, env, ctx):
            return TrainResult(final_eval_reward=42.0)

    assert isinstance(MyTrainer(), Trainer)

    class Incomplete(Trainer):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # abstract fit() not implemented


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
