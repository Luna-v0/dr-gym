"""Time-trial env factory.

DeepRacer upstream supports multiple race types, selected via the ``config``
kwarg to ``DeepRacerEnv``:

- ``TIME_TRIAL`` (default) — what this factory builds.
- ``OBJECT_AVOIDANCE``
- ``HEAD_TO_BOT``
- ``HEAD_TO_MODEL``
- ``F1``

Reference: ``.deepracer-env-upstream/deepracer_env/reset/constants.py:21``.

To add support for another race type, write a sibling factory under
``gym_dr/envs/`` that passes the appropriate ``config={'race_type': '...'}``
to ``DeepRacerEnv``, and re-export it from ``gym_dr/envs/__init__.py``.

``world_name`` is **not** a kwarg to the env. The simapp reads it once at
container startup from the ``WORLD_NAME`` environment variable
(``.deepracer-env-upstream/deepracer_env/track_geom/track_data.py:186``). This
factory cannot change the world; the host orchestrator does that by
respawning the container with a different ``WORLD_NAME``.
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

    from gym_dr.envs.wrappers import GrayscaleObs

    env = DeepRacerEnv(
        reward_fn=experiment.reward,
        sensors=list(experiment.action_space.sensor),
    )
    return GrayscaleObs(env)
