"""Domain randomization config (W-dr) — ``DomainRandomization`` + ``ADR``.

Each DR knob is a :class:`gym_dr.randomization.ParamSpec` — a ``Range(low, high)``
sampled per episode, a ``Choice([...])``, or a bare scalar constant — applied as a
stack of opt-in gym wrappers (`gym_dr/envs/wrappers.py`) wired by the env factory
when ``EnvironmentConfig.domain_randomization`` is set. DR targets **environmental
robustness** (a separate axis from track generalization — the curriculum/ACL's job).

Knobs
-----
* ``steering_noise`` / ``speed_noise`` — actuator/calibration noise (deg / m/s).
* ``obs_gaussian`` / ``obs_brightness`` — camera observation noise.
* ``drag`` — per-episode throttle→speed factor (sim2real motor/drag mismatch); 1.0 off.
* ``friction`` — per-episode wheel-μ multiplier, applied **sim-side** at reset; 1.0 off.
* ``random_start`` / ``random_direction`` — deepracer-env reset modes (random valid
  start location / CW-CCW direction). Require the patched sim.

ADR (Automatic Domain Randomization)
------------------------------------
``ADR`` subclasses ``DomainRandomization`` and, each held-out eval, **widens the
noise knobs** — for each noise ``Range`` it keeps a live ``cur_high`` that starts at
``low`` (≈ no randomization) and grows toward ``high`` by ``step·(high−low)`` when
clean-completion ≥ ``promote`` (shrinks ≤ ``demote``). ``drag``/``friction`` sample
their full ``Range`` every episode regardless (their "easy" anchor is 1.0, not 0, so
naive low→high widening would run backwards — ADR-ramped drag/friction is a follow-up).
Design: ``docs/reports/domain-randomization.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from gym_dr.randomization import ParamSpec, is_randomized, spec_bounds

# Noise knobs ADR ramps (the value is a magnitude/std with a natural "0 = easy" end).
ADR_NOISE_DIMS: Tuple[str, ...] = (
    "steering_noise", "speed_noise", "obs_gaussian", "obs_brightness",
)


@dataclass(frozen=True)
class DomainRandomization:
    """Static domain randomization — every knob a ``Range``/``Choice``/scalar."""

    steering_noise: ParamSpec = 0.0     # deg  (±30 steering range)
    speed_noise: ParamSpec = 0.0        # m/s  (1–4 speed range)
    obs_gaussian: ParamSpec = 0.0       # 0–255 grayscale additive
    obs_brightness: ParamSpec = 0.0     # per-step multiplicative fraction
    obs_contrast: ParamSpec = 0.0       # per-step contrast jitter around mid-gray
    obs_gamma: ParamSpec = 0.0          # per-step gamma (luminance curve) jitter
    feature_noise: ParamSpec = 0.0      # additive Gaussian on the FEATURE obs vector
                                        # (camera-off path) — actor-robustness DR; the
                                        # asymmetric critic still sees the TRUE vector
    drag: ParamSpec = 1.0               # throttle→speed factor (e.g. Range(0.7,1.0))
    friction: ParamSpec = 1.0           # wheel-μ multiplier (Range or Choice of surfaces)
    random_start: bool = False
    random_direction: bool = False
    seed: Optional[int] = None

    # ---- capability flags (env factory wires only what's active) ----
    @property
    def has_action_noise(self) -> bool:
        return spec_bounds(self.steering_noise)[1] > 0 or spec_bounds(self.speed_noise)[1] > 0

    @property
    def has_feature_noise(self) -> bool:
        return spec_bounds(self.feature_noise)[1] > 0

    @property
    def has_obs_noise(self) -> bool:
        return (spec_bounds(self.obs_gaussian)[1] > 0 or spec_bounds(self.obs_brightness)[1] > 0
                or spec_bounds(self.obs_contrast)[1] > 0 or spec_bounds(self.obs_gamma)[1] > 0)

    @property
    def has_drag(self) -> bool:
        return spec_bounds(self.drag)[0] < 1.0 or is_randomized(self.drag)

    @property
    def has_friction(self) -> bool:
        return spec_bounds(self.friction) != (1.0, 1.0)

    @property
    def is_adr(self) -> bool:
        return False


@dataclass(frozen=True)
class ADR(DomainRandomization):
    """Automatic Domain Randomization — widens the noise knobs as the agent succeeds."""
    step: float = 0.1       # fraction of each (high−low) added/removed per eval
    promote: float = 0.7    # widen when held-out clean-completion ≥ this
    demote: float = 0.3     # narrow when ≤ this

    @property
    def is_adr(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# ADR runtime: a shared mutable state read live by the noise wrappers, advanced
# by the controller from the held-out success rate each eval.
# ---------------------------------------------------------------------------
@dataclass
class ADRState:
    """Live effective upper bound (``cur_high``) per noise dim, read by the wrappers
    each step. Starts at each dim's ``low`` (≈ no randomization), grows toward ``high``."""
    cur_high: Dict[str, float] = field(default_factory=dict)

    # convenience: the wrappers read e.g. ``state.steering_noise`` as the live std
    def __getattr__(self, name: str) -> float:
        if name in ADR_NOISE_DIMS:
            return self.cur_high.get(name, 0.0)
        raise AttributeError(name)


class ADRController:
    """Advances an :class:`ADRState` from held-out clean-completion (dactyl-style)."""

    def __init__(self, cfg: DomainRandomization, state: ADRState) -> None:
        self.state = state
        self._bounds = {d: spec_bounds(getattr(cfg, d)) for d in ADR_NOISE_DIMS}
        for d, (lo, _hi) in self._bounds.items():
            state.cur_high.setdefault(d, lo)          # start narrow (≈ no noise)
        self._step = float(getattr(cfg, "step", 0.1))
        self._promote = float(getattr(cfg, "promote", 0.7))
        self._demote = float(getattr(cfg, "demote", 0.3))

    def update(self, success_rate: float) -> Dict[str, float]:
        sign = 1.0 if success_rate >= self._promote else (-1.0 if success_rate <= self._demote else 0.0)
        if sign != 0.0:
            for d, (lo, hi) in self._bounds.items():
                cur = self.state.cur_high[d] + sign * self._step * (hi - lo)
                self.state.cur_high[d] = min(max(cur, lo), hi)
        return self.ranges()

    def ranges(self) -> Dict[str, float]:
        return {f"adr/{d}_high": self.state.cur_high[d] for d in ADR_NOISE_DIMS}


__all__ = [
    "DomainRandomization", "ADR", "ADRState", "ADRController", "ADR_NOISE_DIMS",
]
