from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


@dataclass(frozen=True)
class DiscreteAction:
    steering_angle: float
    speed: float


@dataclass(frozen=True)
class ContinuousActionSpaceConfig:
    steering_low: float = -30.0
    steering_high: float = 30.0
    speed_low: float = 0.1
    speed_high: float = 4.0
    sensor: list[str] = field(default_factory=lambda: ["FRONT_FACING_CAMERA"])
    neural_network: str = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
    version: float = 5.0
    training_algorithm: str = "clipped_ppo"

    @property
    def action_space_type(self) -> str:
        return "continuous"

    def to_model_metadata_dict(self) -> dict[str, Any]:
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
    actions: list[DiscreteAction] = field(default_factory=list)
    sensor: list[str] = field(default_factory=lambda: ["FRONT_FACING_CAMERA"])
    neural_network: str = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
    version: float = 5.0
    training_algorithm: str = "clipped_ppo"

    @property
    def action_space_type(self) -> str:
        return "discrete"

    def to_model_metadata_dict(self) -> dict[str, Any]:
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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_model_metadata_dict(), indent=2) + "\n", encoding="utf-8")
    return p
