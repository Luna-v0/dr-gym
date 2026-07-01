"""Tests for the EnvironmentConfig authoring API + ExperimentConfig composition."""
from __future__ import annotations

import numpy as np

from gym_dr import (ACL, ADR, CameraObs, EnvironmentConfig, ExperimentConfig,
                    FeatureObs, Range, Choice, SafeRL)
from gym_dr.randomization import sample_spec, spec_bounds, is_randomized


def test_range_choice_sampling_and_bounds():
    rng = np.random.default_rng(0)
    r = Range(0.7, 1.0)
    assert 0.7 <= sample_spec(r, rng) <= 1.0 and spec_bounds(r) == (0.7, 1.0)
    c = Choice([0.8, 1.0, 1.5])
    assert sample_spec(c, rng) in (0.8, 1.0, 1.5) and spec_bounds(c) == (0.8, 1.5)
    assert sample_spec(2.5, rng) == 2.5 and not is_randomized(2.5)   # scalar = constant


def test_environment_unpacks_into_experiment():
    env = EnvironmentConfig(
        observation=FeatureObs(),
        curriculum=ACL(train_worlds=["Spain_track", "Monaco"],
                       eval_worlds=["Bowtie_track"], n_chunks=5),
        domain_randomization=ADR(steering_noise=Range(0, 3), drag=Range(0.7, 1.0),
                                 friction=Range(0.8, 1.5), random_start=True,
                                 random_direction=True),
        n_cars=2)
    exp = ExperimentConfig.from_environment(env, name="t")
    assert exp.camera_obs is False            # FeatureObs -> camera_obs False
    assert exp.n_cars == 2
    assert type(exp.world_strategy).__name__ == "ACL"
    assert type(exp.effective_strategy()).__name__ == "ACL"
    assert exp.domain_randomization.has_friction
    assert exp.to_dict()["domain_randomization"] is not None


def test_camera_obs_default_and_safe_rl_flag():
    env = EnvironmentConfig(observation=CameraObs())
    assert env.camera_obs is True and env.is_safe_rl is False
    env2 = EnvironmentConfig(safe_rl=SafeRL(cost=lambda p: 0.0, cost_limit=5.0))
    assert env2.is_safe_rl is True
    assert ExperimentConfig.from_environment(env2, name="t2").cost is not None


def test_adr_is_adr_flag():
    assert ADR().is_adr is True
    from gym_dr import DomainRandomization
    assert DomainRandomization().is_adr is False


def test_with_overrides_preserves_reward_override():
    """``from_environment`` reads the reward ONCE and nothing re-unpacks, so a later
    ``with_overrides`` (exactly how ``install_metrics`` injects the metrics-wrapped
    reward) can never be silently undone — the reward-clobber bug is gone by
    construction (previously ``__post_init__`` re-ran on every ``replace`` and could
    overwrite the wrapped reward, emptying all ``dr/*`` metrics / trace / eval)."""
    from gym_dr.rewards import centerline_quadratic

    env = EnvironmentConfig(observation=FeatureObs(), reward=centerline_quadratic, n_cars=1)
    exp = ExperimentConfig.from_environment(env, name="t")
    assert exp.reward is centerline_quadratic            # taken from the environment

    sentinel = lambda p: 42.0                             # noqa: E731 — stand-in wrapped reward
    exp2 = exp.with_overrides(reward=sentinel)
    assert exp2.reward is sentinel                        # override SURVIVES the replace()
    # an override that touches an unrelated field must also leave reward intact
    exp3 = exp2.with_overrides(name="t2")
    assert exp3.reward is sentinel and exp3.name == "t2"


def test_install_metrics_wrapped_reward_records_steps():
    """The full chain: install_metrics -> with_overrides keeps the tap, and calling
    the resulting reward records a step into the metrics state."""
    from gym_dr.metrics import install_metrics
    from gym_dr.rewards import centerline_quadratic

    env = EnvironmentConfig(observation=FeatureObs(), reward=centerline_quadratic)
    exp = ExperimentConfig.from_environment(env, name="t")
    wrapped_exp, _wrap, state = install_metrics(exp, run_dir=None)
    assert hasattr(wrapped_exp.reward, "__wrapped__")    # metrics tap present
    state.capture_path = True
    state.reset()
    wrapped_exp.reward({"x": 1.0, "y": 2.0, "progress": 5.0, "speed": 3.0})
    assert state.steps == 1 and state.max_progress == 5.0 and state.path_x == [1.0]
