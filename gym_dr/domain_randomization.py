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

    # --- Automatic DR (ADR): grow ranges as the agent succeeds (dactyl-style). ---
    adr: bool = False
    """Enable ADR: start every range near 0 and **expand toward the configured
    maxima** (the ``actuator_*`` / ``obs_*`` values above act as the *ceilings*)
    when the held-out clean-completion rate clears ``adr_promote``; shrink when it
    drops below ``adr_demote``. So robustness grows automatically to the hardest
    level the policy can handle. See ``docs/reports/domain-randomization.md``."""
    adr_step: float = 0.1
    """Fraction of each ceiling added/removed per ADR update (per eval)."""
    adr_promote: float = 0.7
    """Widen ranges when held-out clean-completion rate ≥ this."""
    adr_demote: float = 0.3
    """Narrow ranges when held-out clean-completion rate ≤ this."""

    @property
    def has_action_noise(self) -> bool:
        return self.adr or self.actuator_steering_std > 0 or self.actuator_speed_std > 0

    @property
    def has_obs_noise(self) -> bool:
        return self.adr or self.obs_gaussian_std > 0 or self.obs_brightness_jitter > 0


# Per-dimension randomization knobs ADR scales (the names match both
# DomainRandomizationConfig fields and ADRState attributes).
ADR_DIMS = (
    "actuator_steering_std",
    "actuator_speed_std",
    "obs_gaussian_std",
    "obs_brightness_jitter",
)


@dataclass
class ADRState:
    """Mutable *current* std per DR dimension, read **live** by the noise wrappers
    each step so the ADR controller can widen/narrow ranges without rebuilding the
    env. Starts at 0 (no randomization) and grows toward the config ceilings."""

    actuator_steering_std: float = 0.0
    actuator_speed_std: float = 0.0
    obs_gaussian_std: float = 0.0
    obs_brightness_jitter: float = 0.0


class ADRController:
    """Automatic Domain Randomization controller (dactyl-style).

    Holds the ceilings (the ``DomainRandomizationConfig`` std values) and a shared
    :class:`ADRState`. Each evaluation, :meth:`update` nudges *every* dimension's
    current value up (toward its ceiling) when the held-out clean-completion rate
    is high, down (toward 0) when it's low, and leaves it when in between. Extensive
    by construction: it scales all of actuator steering+speed and observation
    gaussian+brightness together (and, once the env-side reset DR lands, start
    position / direction can be added as further dims).
    """

    def __init__(self, cfg: DomainRandomizationConfig, state: ADRState) -> None:
        self.state = state
        self._ceil = {d: float(getattr(cfg, d)) for d in ADR_DIMS}
        self._step = float(cfg.adr_step)
        self._promote = float(cfg.adr_promote)
        self._demote = float(cfg.adr_demote)

    def update(self, success_rate: float) -> "dict[str, float]":
        """Adjust ranges from a held-out success rate; returns ``adr/<dim>`` values."""
        if success_rate >= self._promote:
            sign = 1.0
        elif success_rate <= self._demote:
            sign = -1.0
        else:
            sign = 0.0
        if sign != 0.0:
            for d, ceil in self._ceil.items():
                cur = getattr(self.state, d) + sign * self._step * ceil
                setattr(self.state, d, min(max(cur, 0.0), ceil))
        return self.ranges()

    def ranges(self) -> "dict[str, float]":
        return {f"adr/{d}": getattr(self.state, d) for d in ADR_DIMS}
