"""Time-trial env factory.

DeepRacer upstream supports multiple race types, selected via the ``config``
kwarg to ``DeepRacerEnv``:

- ``TIME_TRIAL`` (default) — what this factory builds.
- ``OBJECT_AVOIDANCE``
- ``HEAD_TO_BOT``
- ``HEAD_TO_MODEL``
- ``F1``

Reference: ``.deepracer-env-upstream/deepracer_env/reset/constants.py:21``.

Static-obstacle Object Avoidance is not a separate race type in our fork —
it's a feature toggle on the env (``object_avoidance=`` kwarg on
``DeepRacerEnv``). This factory enables it when
``experiment.object_avoidance`` is set, otherwise the env runs pure
time-trial.

To add support for another upstream race type (head-to-head, F1), write a
sibling factory under ``gym_dr/envs/`` that passes the appropriate
``config={'race_type': '...'}`` to ``DeepRacerEnv``, and re-export it
from ``gym_dr/envs/__init__.py``.

``world_name`` is **not** a kwarg to the env. The simapp loads
``WORLD_NAME`` once at container startup
(``.deepracer-env-upstream/deepracer_env/track_geom/track_data.py:186``), so
this factory builds the env on the *first* world. To change tracks afterwards,
call ``env.set_world(name)`` at runtime — upstream now swaps the Gazebo track
in place without restarting the container (the trainer does this between
chunks for multi-world rotation). ``set_world`` is reachable straight through
the ``ActionBounds`` / ``GrayscaleObs`` wrappers this factory returns, since
gymnasium wrappers forward unknown attributes to the base env.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gym_dr.config import ExperimentConfig


def time_trial(experiment: "ExperimentConfig") -> Any:
    """Build a single-agent time-trial ``DeepRacerEnv`` from the experiment.

    Pulls:
      - ``experiment.reward`` (callable) as the reward function;
      - ``experiment.action_space.sensor`` (list of sensor names) for the
        observation dict keys.

    The camera observation is converted to single-channel grayscale via
    ``GrayscaleObs`` — matching what the physical AWS DeepRacer car feeds its
    model (its inference node does BGR->gray before the network). This keeps
    frame-stacking and the ONNX/.pb export consistently grayscale.

    Returns a ``gymnasium.Env`` instance. The caller is responsible for
    closing it.
    """
    from deepracer_env.environments.deepracer_env import DeepRacerEnv

    from gym_dr.action_space import ContinuousActionSpaceConfig
    from gym_dr.envs.wrappers import (
        ActionBounds,
        ActuatorNoise,
        GrayscaleObs,
        NormalizeActions,
        ObservationNoise,
    )

    oa_cfg = experiment.object_avoidance
    upstream_oa = (
        oa_cfg.to_upstream() if oa_cfg is not None and oa_cfg.enabled else None
    )
    env = DeepRacerEnv(
        reward_fn=experiment.reward,
        sensors=list(experiment.action_space.sensor),
        object_avoidance=upstream_oa,
    )
    # Enforce ``ContinuousActionSpaceConfig`` bounds at the wrapper level.
    # Upstream's default action space is ``Box([-30, 0.1], [30, 4.0])`` and its
    # rollout controller hardcodes ``MIN_SPEED=0.1`` — passing a tighter
    # ``speed_low`` only flows into ``model_metadata.json`` otherwise. The
    # wrapper makes the bound real for both PPO's action distribution and the
    # commanded action that reaches Gazebo.
    dr = getattr(experiment, "domain_randomization", None)
    adr_state = adr_controller = None
    if dr is not None and getattr(dr, "adr", False):
        from gym_dr.domain_randomization import ADRController, ADRState

        adr_state = ADRState()
        adr_controller = ADRController(dr, adr_state)
    cfg = experiment.action_space
    if isinstance(cfg, ContinuousActionSpaceConfig):
        env = ActionBounds(
            env,
            steering_low=cfg.steering_low,
            steering_high=cfg.steering_high,
            speed_low=cfg.speed_low,
            speed_high=cfg.speed_high,
        )
        # Actuator-noise DR (engineering units) sits between ActionBounds (inner
        # clip, re-bounds the noisy command) and NormalizeActions (outer
        # [-1,1]->eng map applied first). See docs/reports/domain-randomization.md.
        if dr is not None and dr.has_action_noise:
            env = ActuatorNoise(
                env, steering_std=dr.actuator_steering_std,
                speed_std=dr.actuator_speed_std, seed=dr.seed, adr_state=adr_state,
            )
        # Optionally let the policy act in a symmetric [-1, 1] space (mapped back
        # to engineering units for the env). Keeps the ONNX/on-car interface in
        # engineering units while giving PPO's unit Gaussian comparable
        # exploration on every action dim. See docs/reports/q1-generalization.md.
        if getattr(cfg, "normalize_actions", False):
            env = NormalizeActions(env)
    env = GrayscaleObs(env)
    # Observation-noise DR perturbs the grayscale frames the policy sees, so it
    # wraps OUTSIDE GrayscaleObs.
    if dr is not None and dr.has_obs_noise:
        env = ObservationNoise(
            env, gaussian_std=dr.obs_gaussian_std,
            brightness_jitter=dr.obs_brightness_jitter, seed=dr.seed, adr_state=adr_state,
        )
    if dr is not None and (dr.random_start or dr.random_direction):
        import warnings

        warnings.warn(
            "domain_randomization.random_start/random_direction need a "
            "deepracer-env reset change (see docs/reports/domain-randomization.md) "
            "— ignored until that lands.",
            stacklevel=2,
        )
    if adr_controller is not None:
        # Expose for the eval hook (SB3 callback via vec.get_attr, or ctx.evaluate)
        # to call adr_controller.update(clean_completion_rate) after each eval.
        env.adr_controller = adr_controller
    return env
