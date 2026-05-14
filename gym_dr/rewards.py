"""Example reward functions.

A reward is just a plain ``Callable[[dict], float]`` — there is no registry.
Users typically write their own in ``app.py`` and pass it to
``ExperimentConfig(reward=my_reward)``. The functions here are starting
points.

The ``params`` dict passed to a reward function comes from the upstream
DeepRacer environment. See
``.deepracer-env-upstream/deepracer_env/agent_ctrl/constants.py:108`` for the
full key list. Common keys: ``track_width``, ``distance_from_center``,
``progress``, ``speed``, ``steering_angle``, ``all_wheels_on_track``,
``is_offtrack``, ``waypoints``, ``closest_waypoints``.
"""
from __future__ import annotations


def center_line(params: dict) -> float:
    """Reward staying near the centre of the track lane.

    Three concentric bands measured as a fraction of ``track_width``:
    inside 10% gets the strongest reward, inside 25% a small reward, inside
    50% a tiny one, off-lane a near-zero floor. Mirrors the canonical AWS
    DeepRacer starter reward.
    """
    track_width = params["track_width"]
    distance_from_center = params["distance_from_center"]
    all_wheels_on_track = params["all_wheels_on_track"]

    if not all_wheels_on_track: return 0.1



    if distance_from_center <= 0.1 * track_width:
        multiplier = 100.0
    if distance_from_center <= 0.25 * track_width:
        multiplier =  0.5
    if distance_from_center <= 0.5 * track_width:
        multiplier = 0.1
    else:
        multiplier = 0.01
    
    return float(max(params["progress"] * params["speed"] / 4.0, 1e-3)) * multiplier



def progress_and_speed(params: dict) -> float:
    """Maximize forward progress weighted by speed; floor when off-track.

    Mirrors the example in ``.deepracer-env-upstream/examples/train.py:21``.
    Encourages the policy to finish laps fast rather than crawl along the
    centre line.
    """
    if not params["all_wheels_on_track"]:
        return 1e-3
    return float(max(params["progress"] * params["speed"] / 4.0, 1e-3))
