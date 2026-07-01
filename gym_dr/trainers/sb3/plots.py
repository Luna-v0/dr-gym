"""Evaluation trajectory charts: the car's driven path over a skeleton of the
track.

Rendered to matplotlib figures and wrapped in SB3's ``Figure`` so they land in
TensorBoard's *Images* tab. Imported lazily (only when
``TrainingConfig.eval_path_plots`` is set) so matplotlib stays an optional
dependency.

Geometry comes straight from the env's reward params — ``waypoints`` (the
centerline polyline) + ``track_width`` for the skeleton, and the per-step
``x``/``y`` for the path — so nothing in DeepRacerEnv has to change.

Two chart kinds per eval world:

- :func:`render_overlay` — all ``n_eval_episodes`` traces on one skeleton,
  colour + legend per episode (``ep0: off-track 73%`` …). This answers "who is
  which iteration": the TB image step-slider scrubs eval rounds (each logged at
  ``num_timesteps``), the legend identifies each episode within the round.
- :func:`render_episode` — one episode on its own, start (green ●) / stop
  (red ✗) markers.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Dict, List, Optional, Tuple


def _episode_label(idx: int, ep: Dict[str, Any]) -> str:
    return f"ep{idx}: {ep.get('status', '?')} {ep.get('progress', 0.0):.0f}%"


@functools.lru_cache(maxsize=None)
def _load_route_borders(world: str) -> Optional[Tuple[Any, Any]]:
    """Return ``(inner, outer)`` border polylines for *world* from its route
    asset, or ``None`` if the asset can't be found.

    DeepRacer route ``.npy`` files are ``(N, 6)``:
    ``[center_x, center_y, inner_x, inner_y, outer_x, outer_y]`` — so this gives
    the *true* track edges (vs. offsetting the centerline by ±half-width, which
    pinches on sharp corners). Searched, in order: a ``GYM_DR_ROUTES_DIR``
    override, the installed ``deepracer_simulation_environment`` package, then the
    dev upstream clone. Memoised per world (the geometry is constant)."""
    import numpy as np

    candidates = []
    env_dir = os.getenv("GYM_DR_ROUTES_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, f"{world}.npy"))
    try:
        import importlib

        pkg = importlib.import_module("deepracer_simulation_environment")
        candidates.append(os.path.join(os.path.dirname(pkg.__file__), "routes", f"{world}.npy"))
    except Exception:
        pass
    candidates.append(
        os.path.join(".deepracer-env-upstream", "simulation", "routes", f"{world}.npy")
    )

    for path in candidates:
        try:
            if not os.path.exists(path):
                continue
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                return arr[:, 2:4], arr[:, 4:6]
        except Exception:
            continue
    return None


def _draw_skeleton(ax: Any, ep: Dict[str, Any], world: Optional[str] = None) -> None:
    """Draw the track as a backdrop: the *real* inner/outer borders from the
    route asset when available, otherwise the centerline offset by
    ±``track_width``/2 (or just the centerline if width is unknown)."""
    import numpy as np

    def _close(v):
        v = np.asarray(v, dtype=float)
        return np.vstack([v, v[:1]]) if v.ndim == 2 else np.append(v, v[:1])

    borders = _load_route_borders(world) if world else None
    wp_x = ep.get("wp_x") or []
    wp_y = ep.get("wp_y") or []

    if borders is not None:
        inner, outer = borders
        ax.plot(_close(inner)[:, 0], _close(inner)[:, 1], color="0.45", lw=1.1, zorder=1)
        ax.plot(_close(outer)[:, 0], _close(outer)[:, 1], color="0.45", lw=1.1, zorder=1)
        if wp_x:
            ax.plot(
                list(wp_x) + list(wp_x[:1]), list(wp_y) + list(wp_y[:1]),
                color="0.8", lw=0.8, ls="--", zorder=1,
            )
    elif wp_x:
        # No route asset (e.g. running off-host): approximate the edges by
        # offsetting the centerline by ±half-width along its local normal.
        xs = _close(wp_x)
        ys = _close(wp_y)
        width = float(ep.get("track_width") or 0.0)
        if width > 0.0:
            dx, dy = np.gradient(xs), np.gradient(ys)
            norm = np.hypot(dx, dy)
            norm[norm == 0.0] = 1.0
            nx, ny = -dy / norm, dx / norm
            half = width / 2.0
            ax.plot(xs + nx * half, ys + ny * half, color="0.45", lw=1.1, zorder=1)
            ax.plot(xs - nx * half, ys - ny * half, color="0.45", lw=1.1, zorder=1)
            ax.plot(xs, ys, color="0.8", lw=0.8, ls="--", zorder=1)
        else:
            ax.plot(xs, ys, color="0.6", lw=1.0, ls="--", zorder=1)

    # Frame the axes to the FULL track extent (borders or centerline), NOT the
    # driven path — else a car that barely moves (a tiny/zero-length trajectory)
    # autoscales the view into a dot and the skeleton drops out of frame, making the
    # panel look empty/"broken". With explicit limits the whole track always shows and
    # the short path reads as a dot at the start.
    geom = None
    if borders is not None:
        geom = np.vstack([np.asarray(borders[0], float), np.asarray(borders[1], float)])
    elif wp_x:
        geom = np.column_stack([np.asarray(wp_x, float), np.asarray(wp_y, float)])
    if geom is not None and len(geom):
        xmin, ymin = geom.min(axis=0)
        xmax, ymax = geom.max(axis=0)
        mx = (xmax - xmin) * 0.05 or 1.0
        my = (ymax - ymin) * 0.05 or 1.0
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_overlay(world: str, timestep: int, episodes: List[Dict[str, Any]]) -> Any:
    """All eval episodes for *world* overlaid on the skeleton (one colour each)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    skeleton = next((e for e in episodes if e.get("wp_x")), episodes[0] if episodes else {})
    _draw_skeleton(ax, skeleton, world)

    cmap = plt.get_cmap("turbo")
    n = max(len(episodes), 1)
    for i, ep in enumerate(episodes):
        colour = cmap(i / n)
        ax.plot(ep.get("x", []), ep.get("y", []), color=colour, lw=1.6,
                zorder=2, label=_episode_label(i, ep))
        if ep.get("x"):
            ax.plot(ep["x"][0], ep["y"][0], "o", color=colour, ms=5, zorder=3)
            ax.plot(ep["x"][-1], ep["y"][-1], "x", color=colour, ms=7, mew=2, zorder=3)

    ax.set_title(f"{world} — eval @ {timestep:,} steps")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    return fig


def render_episode(world: str, timestep: int, idx: int, ep: Dict[str, Any]) -> Any:
    """A single eval episode on its own skeleton, with start/stop markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    _draw_skeleton(ax, ep, world)
    xs, ys = ep.get("x", []), ep.get("y", [])
    speeds = ep.get("speed") or []
    if xs and len(speeds) == len(xs):
        # Speed-coloured trajectory + colourbar (deepracer-utils-style: shows
        # *where* the car is fast/slow along the lap, not just the path shape).
        sc = ax.scatter(xs, ys, c=speeds, cmap="turbo", s=7, zorder=2)
        fig.colorbar(sc, ax=ax, label="speed (m/s)", fraction=0.046, pad=0.04)
    else:
        ax.plot(xs, ys, color="C0", lw=1.6, zorder=2)
    if xs:
        ax.plot(xs[0], ys[0], "o", color="green", ms=6, zorder=3, label="start")
        ax.plot(xs[-1], ys[-1], "x", color="red", ms=8, mew=2, zorder=3, label="stop")
    status = ep.get("status", "?")
    progress = ep.get("progress", 0.0)
    ax.set_title(f"{world} ep{idx} — {status} {progress:.0f}%  @ {timestep:,} steps")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    return fig
