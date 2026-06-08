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
    from gym_dr.envs.wrappers import ActionBounds, GrayscaleObs

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
    cfg = experiment.action_space
    if isinstance(cfg, ContinuousActionSpaceConfig):
        env = ActionBounds(
            env,
            steering_low=cfg.steering_low,
            steering_high=cfg.steering_high,
            speed_low=cfg.speed_low,
            speed_high=cfg.speed_high,
        )
    return GrayscaleObs(env)
