"""Action-space configs and the ``model_metadata.json`` writer.

DeepRacer's ``model_metadata.json`` is the sidecar that tells the physical
car how to interpret a saved policy. Two schemas are valid:

- **Continuous** — Gaussian policy outputs are clipped to
  ``[steering_low, steering_high]`` and ``[speed_low, speed_high]``.
- **Discrete**   — the policy outputs an action index into a fixed list of
  ``(steering_angle, speed)`` pairs.

``write_model_metadata`` serializes either schema. The trainer writes this
file once per run (at ``artifacts/<chunk>/model_metadata.json``) AND a
sibling next to every saved model ``.zip`` so any single checkpoint is
shippable to the physical car on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


@dataclass(frozen=True)
class DiscreteAction:
    """One row in a discrete action list."""

    steering_angle: float
    """Degrees. Positive = right, negative = left. Conventional range
    ``[-30, 30]``."""

    speed: float
    """Meters/second. Conventional range ``[0.1, 4.0]``."""


@dataclass(frozen=True)
class ContinuousActionSpaceConfig:
    """Continuous action space.

    The gym env exposes a 2D Box action: ``[steering_angle, speed]``. The
    upstream env clips agent outputs to ``[steering_low, steering_high]``
    and ``[speed_low, speed_high]``.
    """

    steering_low: float = -30.0
    """Steering lower bound, degrees. Negative = left."""

    steering_high: float = 30.0
    """Steering upper bound, degrees. Positive = right."""

    speed_low: float = 0.1
    """Minimum speed, m/s. Must be > 0 so the car can move; the upstream env
    floors near-zero outputs."""

    speed_high: float = 4.0
    """Maximum speed, m/s. Higher = harder to control, especially on tight
    tracks."""

    normalize_actions: bool = True
    """When True (the **default**), the env factory wraps the env so the *policy*
    sees a symmetric ``[-1, 1]`` action space (mapped back to these
    engineering-unit bounds for the sim). PPO's unit-init Gaussian then explores
    steering and speed comparably; the raw ``Box([-30,30]×[low,high])`` otherwise
    gives steering only ~±1° of exploration — the trial_18 failure root cause
    (see ``docs/reports/q1-generalization.md``). The sim still receives
    engineering units, but note the **exported ONNX now outputs ``[-1, 1]``**, so
    the on-car node must rescale it (see ``docs/physical-car-integration-notes.md``).
    Set False to reproduce the old raw-action behaviour."""

    sensor: list[str] = field(default_factory=lambda: ["FRONT_FACING_CAMERA"])
    """Active sensors. Each becomes a key in the observation dict. Valid
    values (from upstream): ``CAMERA`` / ``FRONT_FACING_CAMERA``,
    ``LEFT_CAMERA``, ``STEREO``, ``LIDAR``, ``SECTOR_LIDAR``,
    ``DISCRETIZED_SECTOR_LIDAR``."""

    neural_network: str = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
    """Network architecture name the physical car expects. The simapp does
    not use this — it's metadata for the car's model loader."""

    version: float = 5.0
    """``model_metadata.json`` schema version. Match the car firmware."""

    training_algorithm: str = "clipped_ppo"
    """String identifier written to ``model_metadata.json``. Independent of
    the actual trainer in use; it's a label for the car."""

    @property
    def action_space_type(self) -> str:
        return "continuous"

    def to_model_metadata_dict(self) -> dict[str, Any]:
        """Render to the DeepRacer-compatible JSON shape (continuous schema)."""
        return {
            "sensor": list(self.sensor),
            "neural_network": self.neural_network,
            "version": self.version,
            "training_algorithm": self.training_algorithm,
            "action_space_type": "continuous",
            "action_space": {
                "steering_angle": {"low": float(self.steering_low), "high": float(self.steering_high)},
                "speed": {"low": float(self.speed_low), "high": float(self.speed_high)},
            },
        }


@dataclass(frozen=True)
class DiscreteActionSpaceConfig:
    """Discrete action space.

    The policy outputs an index into ``actions``; the env converts to the
    corresponding ``(steering_angle, speed)``. Matches the schema used by
    the AWS DeepRacer console exporter.
    """

    actions: list[DiscreteAction] = field(default_factory=list)
    """Ordered list of (steering_angle, speed) pairs. ``index`` is
    auto-assigned (0-based, list order) on serialization. Must be
    non-empty."""

    sensor: list[str] = field(default_factory=lambda: ["FRONT_FACING_CAMERA"])
    """See ``ContinuousActionSpaceConfig.sensor``."""

    neural_network: str = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
    """See ``ContinuousActionSpaceConfig.neural_network``."""

    version: float = 5.0
    """See ``ContinuousActionSpaceConfig.version``."""

    training_algorithm: str = "clipped_ppo"
    """See ``ContinuousActionSpaceConfig.training_algorithm``."""

    @property
    def action_space_type(self) -> str:
        return "discrete"

    def to_model_metadata_dict(self) -> dict[str, Any]:
        """Render to the DeepRacer-compatible JSON shape (discrete schema).

        Auto-assigns ``index`` from list order. Raises if ``actions`` is empty.
        """
        if not self.actions:
            raise ValueError("DiscreteActionSpaceConfig.actions must be non-empty")
        return {
            "sensor": list(self.sensor),
            "neural_network": self.neural_network,
            "version": self.version,
            "training_algorithm": self.training_algorithm,
            "action_space_type": "discrete",
            "action_space": [
                {"steering_angle": float(a.steering_angle), "speed": float(a.speed), "index": i}
                for i, a in enumerate(self.actions)
            ],
        }


ActionSpaceConfig = Union[ContinuousActionSpaceConfig, DiscreteActionSpaceConfig]


def write_model_metadata(path: str | Path, cfg: ActionSpaceConfig) -> Path:
    """Write the DeepRacer-compatible ``model_metadata.json`` at ``path``.

    Creates parent dirs as needed. Returns the final path. Used both at
    chunk start (to seed ``/workspace/model_metadata.json`` for the simapp
    to pick up) and on every ``.zip`` save (the per-zip sidecar that makes
    individual checkpoints shippable to the physical car).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_model_metadata_dict(), indent=2) + "\n", encoding="utf-8")
    return p
