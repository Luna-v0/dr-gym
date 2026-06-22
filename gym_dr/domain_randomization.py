"""Domain-randomization config (W-dr).

Domain randomization (DR) is applied as a stack of opt-in gym wrappers (ADR-0002):
``ActuatorNoise`` / ``ObservationNoise`` (`gym_dr/envs/wrappers.py`), wired by the
env factory when ``ExperimentConfig.domain_randomization`` is set. DR targets
**environmental robustness** (a separate axis from track generalization, which is
the curriculum's job — see ``docs/glossary.md``).

Two of the requested knobs (``random_start``, ``random_direction``) are
**episode-reset** behaviours owned by ``deepracer-env``. Its reset supports only
deterministic round-robin start advance (``CHANGE_START``) and alternating
direction (``ALT_DIR``) — not *random* — so true random valid-start + random
direction need a small env-side change (sample ``start_ndist`` from the env
``np_random`` each reset; randomise the lane/direction). Until that lands the
factory warns if these are set. Design: ``docs/reports/domain-randomization.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Opt-in domain randomization. All-zero / all-False = no randomization."""

    # --- Actuator noise (engineering units), added to the commanded action each
    # step after the [-1,1]→eng mapping and before the action-space clip. ---
    actuator_steering_std: float = 0.0   # degrees
    actuator_speed_std: float = 0.0      # m/s

    # --- Observation noise on the grayscale uint8 camera the policy sees. ---
    obs_gaussian_std: float = 0.0        # additive, 0–255 scale
    obs_brightness_jitter: float = 0.0   # per-step multiplicative, fraction (0.1 ⇒ ±10%)

    # --- Episode-reset randomization (REQUIRES a deepracer-env change; the
    # factory warns if set until then). See the module docstring. ---
    random_start: bool = False
    random_direction: bool = False

    seed: Optional[int] = None

    @property
    def has_action_noise(self) -> bool:
        return self.actuator_steering_std > 0 or self.actuator_speed_std > 0

    @property
    def has_obs_noise(self) -> bool:
        return self.obs_gaussian_std > 0 or self.obs_brightness_jitter > 0
