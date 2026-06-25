"""Supervised perception for DeepRacer — camera/stack -> *frame-local* driving
features, trained on free sim ground truth (`reward_params`).

Why this exists (W-perception, `[REAL]`)
----------------------------------------
The deployed policy on the physical car sees **only the camera**. In sim we have
privileged state (pose, waypoints, track width, off-track flags) for free in the
26-key ``reward_params`` dict. Two ways to use it:

1. **Asymmetric actor-critic** — the *critic* may read privileged state (it only
   runs at train time), but the *actor* consumes only what the car will have.
   See ``AsymmetricCritic`` notes below.
2. **A supervised perception head** — learn a small net that maps the camera
   stack to the *frame-local* driving features a controller actually needs, with
   the privileged state as the regression label. The actor then drives off these
   features (or off the camera with this net as a pretrained, optionally frozen,
   front-end). This module is that head.

What we regress — and what we DON'T
-----------------------------------
We deliberately regress **frame-local, ego-relative** quantities that are in
principle observable from a single forward camera, and that don't alias:

    lateral_offset   signed position across the lane, +1 = at the right edge,
                     -1 = at the left edge, 0 = centerline  (distance_from_center
                     / (track_width/2), signed by is_left_of_center)
    heading_error    car heading minus the local track tangent, normalized by
                     180 deg; + = pointing right of the track direction
    dist_left_edge   distance to the left lane edge / track_width   (in [0,1])
    dist_right_edge  distance to the right lane edge / track_width   (in [0,1])
    speed_mps        raw speed in m/s (NOT normalised — proprioceptive, sim2real-stable)
    yaw_rate         finite-difference of heading per step, normalized

We do **NOT** regress global pose (``x``/``y``/``heading`` absolute) or
``progress``: two different places on a track look identical to a forward camera
(perceptual aliasing), so those labels are unlearnable from one frame and would
just train the net to hallucinate. Keep the targets ego-relative.

torch is imported lazily (like ``gym_dr/networks.py``) so importing this module
stays cheap.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from gym_dr.networks import DEFAULT_CONV, ConvSpec

# Ordered output features. The collection, training, and net all key off this —
# change it in one place. Index = column in the label/prediction vector.
PERCEPTION_FEATURES: Tuple[str, ...] = (
    "lateral_offset",
    "heading_error",
    "dist_left_edge",
    "dist_right_edge",
    "speed_mps",
    "yaw_rate",
)

# Normalisers. speed_max matches ContinuousActionSpaceConfig's ceiling (4.0 m/s);
# yaw_rate is normalised by a generous per-step heading swing (deg).
_SPEED_MAX = 4.0       # action-space speed ceiling (used by the reward's [0,1] gating)
_SPEED_CLIP = 8.0      # generous physical bound for the RAW speed_mps feature (no
                       # sim-max normalisation — see perception_targets)
_YAW_RATE_NORM = 30.0  # deg/step; clips fast spins to +-1


def _wrap_deg(angle: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""
    return (angle + 180.0) % 360.0 - 180.0


def _track_tangent_deg(params: dict) -> Optional[float]:
    """Local track direction (deg) from the two closest waypoints, or None."""
    wps = params.get("waypoints")
    cw = params.get("closest_waypoints")
    if not wps or not cw or len(cw) < 2:
        return None
    try:
        prev_i, next_i = int(cw[0]), int(cw[1])
        x0, y0 = wps[prev_i]
        x1, y1 = wps[next_i]
    except (IndexError, TypeError, ValueError):
        return None
    if (x1 - x0) == 0.0 and (y1 - y0) == 0.0:
        return None
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def perception_targets(params: dict, prev_params: Optional[dict] = None) -> np.ndarray:
    """Build the frame-local label vector from a ``reward_params`` dict.

    ``prev_params`` (the previous step's params) is only needed for ``yaw_rate``
    (a finite difference of ``heading``); pass ``None`` on the first step and the
    yaw_rate label is 0.

    Returns a float32 array of shape ``(len(PERCEPTION_FEATURES),)``. Missing
    keys fall back to safe defaults (centerline / zero), so a partial params dict
    never raises.
    """
    track_width = float(params.get("track_width", 0.0)) or 1e-6
    half = track_width / 2.0
    dist_center = float(params.get("distance_from_center", 0.0))
    left_of_center = bool(params.get("is_left_of_center", False))

    # signed lateral: + = right of centerline, normalised so +-1 = the edges.
    signed = -dist_center if left_of_center else dist_center
    lateral_offset = float(np.clip(signed / half, -1.0, 1.0))

    # edge distances (fractions of full track width, clipped to [0,1]).
    if left_of_center:
        d_left = half - dist_center
        d_right = half + dist_center
    else:
        d_left = half + dist_center
        d_right = half - dist_center
    dist_left_edge = float(np.clip(d_left / track_width, 0.0, 1.0))
    dist_right_edge = float(np.clip(d_right / track_width, 0.0, 1.0))

    # heading error vs local track tangent.
    heading = float(params.get("heading", 0.0))
    tangent = _track_tangent_deg(params)
    if tangent is None:
        heading_error = 0.0
    else:
        heading_error = float(np.clip(_wrap_deg(heading - tangent) / 180.0, -1.0, 1.0))

    # Raw speed in m/s — NOT normalised by the sim's action-space max. Speed is
    # proprioceptive on the real car (motor/IMU) and reported in physical units, so
    # dividing by the sim max (4.0) would mean different things in sim vs real if the
    # real car's throttle->speed map differs (the drag mismatch we also randomize).
    # Clipped only to a generous physical bound to keep the NN input finite.
    speed_mps = float(np.clip(float(params.get("speed", 0.0)), 0.0, _SPEED_CLIP))

    if prev_params is None:
        yaw_rate = 0.0
    else:
        dh = _wrap_deg(heading - float(prev_params.get("heading", heading)))
        yaw_rate = float(np.clip(dh / _YAW_RATE_NORM, -1.0, 1.0))

    return np.array(
        [lateral_offset, heading_error, dist_left_edge, dist_right_edge, speed_mps, yaw_rate],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Privileged state — the EXTRA signals the asymmetric value/cost critic (or a
# privileged teacher) may use that the deployable actor must NOT. See
# docs/reports/asymmetric-architecture.md for the full study; the short version:
#   * The critic only runs at TRAIN time and is discarded at deployment, so it
#     may read ground-truth map/contact/object state that no camera can recover
#     (global progress, exact off-track/crash flags, full object geometry). This
#     lowers value-estimation variance (Pinto et al. 2017, asymmetric AC).
#   * The COST critic benefits most: our graded costs (gym_dr/costs.py) are
#     DEFINED on near-edge / near-object distances, so giving the cost critic the
#     exact privileged distances makes the constraint estimate far less noisy.
#   * `perception_targets` above is the deployable, camera-distillable part; the
#     critic input is the SUPERSET concat(perception_targets, privileged_state).
# Of the actor's six, `speed_mps` and `yaw_rate` are PROPRIOCEPTIVE on the real
# car (wheel encoders + IMU), so the actor keeps them at deploy without vision;
# the genuinely vision-distilled ones are lateral_offset / heading_error / edges.
# --------------------------------------------------------------------------- #
PRIVILEGED_EXTRA_FEATURES: Tuple[str, ...] = (
    "progress_frac",          # global lap progress [0,1] — aliased from a single camera frame
    "curvature_ahead",        # signed mean turn over the next K waypoints, /90deg — map knowledge
    "nearest_object_dist",    # min objects_distance, normalised (1.0 = no/very-far object)
    "offtrack",               # exact terminal flag (1.0 = all four wheels off)
    "crashed",                # exact collision flag
    "wheels_on_track",        # exact contact (1.0 = all wheels on)
)

_CURV_AHEAD_K = 5             # waypoints to look ahead for curvature — REWARD / critic
                              # (privileged sim-side: can anticipate further)
_ACTOR_CURV_K = 3             # the ACTOR feature uses a SHORTER lookahead so the
                              # curvature stays inside the camera FOV (CNN-learnable;
                              # 5 reached too far ahead to be visible in one frame)
_OBJECT_DIST_NORM = 5.0       # m; clips far objects to "no risk"


def _track_curvature_ahead(params: dict, k: int = _CURV_AHEAD_K) -> float:
    """Signed mean heading change (deg) over the next ``k`` waypoint segments,
    normalised by 90deg. + = the track bends left ahead. 0 if unavailable.

    This is *map knowledge* — the upcoming bend the car can't yet see — so it is
    a privileged signal for the critic (the actor only gets what the camera shows
    of the road ahead, which the perception net must learn to read)."""
    wps = params.get("waypoints")
    cw = params.get("closest_waypoints")
    if not wps or not cw or len(cw) < 2:
        return 0.0
    try:
        n = len(wps)
        start = int(cw[1])
        diffs = []
        for i in range(k):
            a = wps[(start + i) % n]
            b = wps[(start + i + 1) % n]
            c = wps[(start + i + 2) % n]
            h1 = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            h2 = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]))
            diffs.append(_wrap_deg(h2 - h1))
    except (IndexError, TypeError, ValueError):
        return 0.0
    if not diffs:
        return 0.0
    return float(np.clip((sum(diffs) / len(diffs)) / 90.0, -1.0, 1.0))


def privileged_state(params: dict, prev_params: Optional[dict] = None) -> np.ndarray:
    """The EXTRA privileged feature vector for the asymmetric critic / teacher.

    Returns ``(len(PRIVILEGED_EXTRA_FEATURES),)`` float32. Combine with the
    actor's deployable features for the critic input:
    ``np.concatenate([perception_targets(p), privileged_state(p)])``.

    Never feed this to the deployed actor — it is ground truth the car can't
    sense (the deploy guardrail, docs/reports/perception.md)."""
    progress = float(params.get("progress", 0.0))
    # deepracer-env progress is [0,1] in reward_params; tolerate 0-100 too.
    progress_frac = float(np.clip(progress if progress <= 1.0 else progress / 100.0, 0.0, 1.0))

    curvature_ahead = _track_curvature_ahead(params)

    objs = params.get("objects_distance") or []
    try:
        nearest = min(float(d) for d in objs) if objs else _OBJECT_DIST_NORM
    except (TypeError, ValueError):
        nearest = _OBJECT_DIST_NORM
    nearest_object_dist = float(np.clip(nearest / _OBJECT_DIST_NORM, 0.0, 1.0))

    offtrack = 1.0 if params.get("is_offtrack") else 0.0
    crashed = 1.0 if params.get("is_crashed") else 0.0
    wheels_on = 1.0 if params.get("all_wheels_on_track", True) else 0.0

    return np.array(
        [progress_frac, curvature_ahead, nearest_object_dist, offtrack, crashed, wheels_on],
        dtype=np.float32,
    )


def critic_state(params: dict, prev_params: Optional[dict] = None) -> np.ndarray:
    """Full asymmetric-critic input = deployable features ⊕ privileged extras."""
    return np.concatenate(
        [perception_targets(params, prev_params), privileged_state(params, prev_params)]
    )


# --------------------------------------------------------------------------- #
# Dynamic (temporal-derivative) features — CANDIDATES the 4-frame stack makes
# learnable. The LABEL is a cheap finite difference of `reward_params`; the point
# is that *observing* them needs the stack (2 frames to see speed, 3 to see
# acceleration, 4 to see jerk). These are deployable (a real car has the same
# frame history / IMU), so they sit on the ACTOR side like speed/yaw_rate.
# Treat them as candidates: collect labels for all, then keep whichever the
# held-out per-feature MAE (experiments/train_perception.py) shows are actually
# learnable. The core six (perception_targets) stay the validated set.
# --------------------------------------------------------------------------- #
DYNAMIC_FEATURES: Tuple[str, ...] = (
    "long_accel",          # Δ speed per step (signed) — momentum / throttle response
    "lateral_velocity",    # Δ lateral_offset per step (signed) — drifting toward an edge
    "edge_closing_rate",   # −Δ(nearest-edge distance) per step: + = approaching the edge (RISK)
)

# All actor-side features (validated core ⊕ dynamic candidates).
ALL_FEATURES: Tuple[str, ...] = PERCEPTION_FEATURES + DYNAMIC_FEATURES

# Extended actor set (maintainer-selected): the 9 above PLUS two CNN-learnable
# privileged extras — short-lookahead curvature (anticipate the visible corner) and
# nearest-object distance (visible obstacle). progress_frac is deliberately EXCLUDED
# (perceptual aliasing — not recoverable from one frame, stays critic-only).
ACTOR_EXTRA_FEATURES: Tuple[str, ...] = ("curvature_ahead", "nearest_object_dist")
ACTOR_FEATURES: Tuple[str, ...] = ALL_FEATURES + ACTOR_EXTRA_FEATURES
def actor_targets(params: dict, prev_params: Optional[dict] = None) -> np.ndarray:
    """ACTOR_FEATURES vector: the 9 ALL_FEATURES ⊕ the selected CNN-learnable extras.

    curvature_ahead uses the SHORT actor lookahead (``_ACTOR_CURV_K``, FOV-sized) —
    NOT the reward/critic's longer ``_CURV_AHEAD_K``. nearest_object_dist reuses the
    privileged builder (index 2 of privileged_state)."""
    base = all_targets(params, prev_params)
    curvature_ahead = _track_curvature_ahead(params, k=_ACTOR_CURV_K)
    nearest_object_dist = float(privileged_state(params)[2])
    return np.concatenate(
        [base, np.array([curvature_ahead, nearest_object_dist], dtype=np.float32)]
    ).astype(np.float32)

# Features whose range is signed [-1,1] (tanh head); the rest are [0,1] (sigmoid).
SIGNED_FEATURES = frozenset({
    "lateral_offset", "heading_error", "yaw_rate",
    "long_accel", "lateral_velocity", "edge_closing_rate",
})


def signed_indices_for(features: Sequence[str]) -> Tuple[int, ...]:
    """Output indices that are signed (tanh), given an ordered feature list — so
    the net's head matches whatever target set the dataset was built with."""
    return tuple(i for i, name in enumerate(features) if name in SIGNED_FEATURES)


_STEP_DELTA_NORM = 0.1     # a 10%-of-range change per step saturates a derivative to ±1


def dynamic_targets(params: dict, prev_params: Optional[dict] = None) -> np.ndarray:
    """Finite-difference dynamic labels (``DYNAMIC_FEATURES``) from this step and
    the previous one. Returns zeros on the first step (no ``prev_params``)."""
    if prev_params is None:
        return np.zeros(len(DYNAMIC_FEATURES), dtype=np.float32)
    cur = perception_targets(params)
    prev = perception_targets(prev_params)
    # indices into perception_targets: 0=lateral_offset, 2=dist_left, 3=dist_right, 4=speed_mps
    long_accel = float(np.clip((cur[4] - prev[4]) / _STEP_DELTA_NORM, -1.0, 1.0))
    lateral_velocity = float(np.clip((cur[0] - prev[0]) / _STEP_DELTA_NORM, -1.0, 1.0))
    nearest_cur = min(float(cur[2]), float(cur[3]))
    nearest_prev = min(float(prev[2]), float(prev[3]))
    edge_closing_rate = float(np.clip((nearest_prev - nearest_cur) / _STEP_DELTA_NORM, -1.0, 1.0))
    return np.array([long_accel, lateral_velocity, edge_closing_rate], dtype=np.float32)


def all_targets(params: dict, prev_params: Optional[dict] = None) -> np.ndarray:
    """Full actor label vector = ``perception_targets ⊕ dynamic_targets`` — the
    candidate set the learnability (MAE) study scores."""
    return np.concatenate(
        [perception_targets(params, prev_params), dynamic_targets(params, prev_params)]
    )


def enrich_reward_params(params: dict, prev_params: Optional[dict] = None) -> dict:
    """Return ``params`` with the derived feature keys (``ALL_FEATURES``) added.

    Makes the distilled / dynamic features (``lateral_offset``, ``edge_closing_rate``,
    ``long_accel``, …) available as **reward-function arguments**, and is the same
    vector a *feature-based* policy observes (the oracle-feature test, Test 1 in
    `docs/reports/feature-based-policy.md`). Original keys are preserved; the new
    keys are normalised to the same ranges the perception net predicts, so a reward
    or a state-based policy sees exactly what the camera extractor will later
    estimate."""
    return {**params, **dict(zip(ALL_FEATURES, all_targets(params, prev_params).tolist()))}


# --------------------------------------------------------------------------- #
# The supervised net (lazy torch, mirrors networks.py's pattern).
# --------------------------------------------------------------------------- #
def _build_perception_net():
    import torch
    import torch.nn as nn

    class PerceptionNet(nn.Module):
        """Camera stack -> frame-local driving features (regression).

        Input: a channels-first float tensor ``(N, C, H, W)`` — the same
        grayscale frame stack the policy sees (``normalize_images=False`` world,
        so feed raw 0-255 floats; the net divides by 255 internally so it can be
        used standalone, off the policy's preprocessing path).

        Output: ``(N, n_outputs)`` with a tanh on the signed channels (lateral
        offset / heading error / yaw rate / the dynamic derivatives) and a sigmoid
        on the bounded-positive channels (edge distances / speed), so predictions
        stay in the same ranges as the targets. ``signed_indices`` defaults to the
        core six's signed channels; pass ``signed_indices_for(features)`` when
        training on the extended (dynamic) set so the head matches.
        """

        def __init__(
            self,
            in_channels: int = 4,
            conv_layers: ConvSpec = DEFAULT_CONV,
            features_dim: int = 256,
            n_outputs: int = len(PERCEPTION_FEATURES),
            input_hw: Tuple[int, int] = (120, 160),
            signed_indices: Tuple[int, ...] = (0, 1, 5),  # core: lateral, heading, yaw_rate
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = in_channels
            for filters, kernel, stride in conv_layers:
                padding = kernel // 2 if stride == 1 else 0
                layers += [
                    nn.Conv2d(prev, filters, kernel_size=kernel, stride=stride, padding=padding),
                    nn.ReLU(),
                ]
                prev = filters
            layers.append(nn.Flatten())
            self.encoder = nn.Sequential(*layers)
            with torch.no_grad():
                sample = torch.zeros(1, in_channels, *input_hw)
                n_flat = self.encoder(sample).shape[1]
            self.head = nn.Sequential(
                nn.Linear(n_flat, features_dim),
                nn.ReLU(),
                nn.Linear(features_dim, n_outputs),
            )
            signed_mask = torch.zeros(n_outputs)
            for i in signed_indices:
                if i < n_outputs:
                    signed_mask[i] = 1.0
            self.register_buffer("_signed_mask", signed_mask)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = self.head(self.encoder(x / 255.0))
            signed = torch.tanh(z)
            bounded = torch.sigmoid(z)
            return self._signed_mask * signed + (1.0 - self._signed_mask) * bounded

    return PerceptionNet


_PERCEPTION_NET: Any = None


def __getattr__(name: str) -> Any:
    if name == "PerceptionNet":
        global _PERCEPTION_NET
        if _PERCEPTION_NET is None:
            _PERCEPTION_NET = _build_perception_net()
        return _PERCEPTION_NET
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
