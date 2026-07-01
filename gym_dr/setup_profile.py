"""Per-machine optimal training config (Task 2) — recommendation logic.

``scripts/optimized_setup.py`` probes the current machine and writes a JSON of the
best training configuration for it. This module holds the two *pure* pieces so the
policy is unit-testable without a simulator:

- :func:`detect_machine` — cores / RAM / GPU (name + VRAM + CUDA availability);
- :func:`recommend` — pick the optimal ``n_cars`` from benchmark candidates using
  the maintainer's rule, and :func:`build_profile` to assemble the JSON.

The maintainer's rule (TASKS.md Task 2 / BLOCKERS B4): the score is aggregate
throughput = ``steps_per_s`` (already ``per_car_steps × n_cars``); **favour more
cars over raw speed** — a config within ~80% of the best achievable throughput is
acceptable if it trains more cars — subject to a **per-car sample-quality floor** so
PPO still gets enough steps per car per rollout.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

# PPO stays stable when each car contributes enough steps to a rollout; below this
# the per-car sample budget gets too thin. Tunable.
DEFAULT_PER_CAR_FLOOR = 16
# "80% of max perf is reasonable if more cars are training" (maintainer).
DEFAULT_THROUGHPUT_TOL = 0.8


def detect_machine() -> "dict[str, Any]":
    """Best-effort machine profile: cores, RAM (GiB), GPU name/VRAM, CUDA flag."""
    import os

    info: "dict[str, Any]" = {
        "cores": os.cpu_count() or 1,
        "ram_gib": None,
        "cuda": False,
        "gpu_name": None,
        "gpu_vram_gib": None,
    }
    try:  # RAM (Linux)
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["ram_gib"] = round(int(line.split()[1]) / (1024 ** 2), 1)
                    break
    except OSError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["cuda"] = True
            info["gpu_name"] = props.name
            info["gpu_vram_gib"] = round(props.total_memory / (1024 ** 3), 1)
    except Exception:  # noqa: BLE001 — torch missing or no GPU
        pass
    return info


def _steps_per_car(c: "Mapping[str, Any]") -> float:
    return float(c["steps_per_s"]) / max(1, int(c["n_cars"]))


def recommend(
    candidates: "List[Mapping[str, Any]]",
    *,
    per_car_floor: int = DEFAULT_PER_CAR_FLOOR,
    throughput_tol: float = DEFAULT_THROUGHPUT_TOL,
) -> "Optional[dict]":
    """Choose the optimal candidate config (or ``None`` if there are none).

    Each candidate is ``{"n_cars": int, "steps_per_s": float, ...}`` where
    ``steps_per_s`` is the *aggregate* agent-steps/s (already ×n_cars). Selection:

    1. keep candidates whose per-car steps ≥ ``per_car_floor`` (PPO stability); if
       none qualify, relax and consider all;
    2. of those, keep the ones within ``throughput_tol`` of the best achievable
       throughput;
    3. among those, pick the **most cars** (throughput breaks ties).
    """
    cands = [dict(c) for c in candidates if c.get("n_cars") and c.get("steps_per_s") is not None]
    if not cands:
        return None
    max_tput = max(c["steps_per_s"] for c in cands)
    stable = [c for c in cands if _steps_per_car(c) >= per_car_floor]
    pool = stable or cands
    threshold = throughput_tol * max_tput
    within = [c for c in pool if c["steps_per_s"] >= threshold] or pool
    best = max(within, key=lambda c: (int(c["n_cars"]), float(c["steps_per_s"])))
    out = dict(best)
    out["per_car_steps_s"] = round(_steps_per_car(best), 2)
    out["relaxed_floor"] = not stable  # flag when nothing met the per-car floor
    return out


def rule_of_thumb(machine: "Mapping[str, Any]") -> "dict[str, Any]":
    """Conservative per-machine defaults WITHOUT a benchmark (clearly labelled
    ``source="heuristic"``). Feature obs scales with cores (cap 18); camera obs is
    render-bound and reset-storms past ~4 cars on an 8-core box (D10), so it caps
    low and prefers a GPU render when there's enough VRAM. Encodes the maintainer's
    two-machine intent: 8-core+16GB → feature n=6 (CPU) / camera n=2 (GPU);
    22-core+8GB → feature n=18 (CPU) / camera n=4 (GPU)."""
    cores = int(machine.get("cores", 1))
    vram = float(machine.get("gpu_vram_gib") or 0.0)
    cuda = bool(machine.get("cuda"))
    gpu_render = cuda and vram >= 8.0
    return {
        "feature_obs": {
            "n_cars": min(max(cores - 2, 2), 18),
            "device": "cpu",  # feature obs is CPU-rollout bound; cores are the lever
            "source": "heuristic",
        },
        "camera_obs": {
            "n_cars": 2 if cores <= 8 else 4,  # D10: >4 reset-storms on 8 cores
            "device": "cuda" if gpu_render else "cpu",
            "render": "gpu" if gpu_render else "software",
            "source": "heuristic",
            "note": "camera n>2 needs the generalised racecar_2..N launch + reset-seam work",
        },
    }


def heuristic_profile(
    machine: "Mapping[str, Any]", *, timestamp: "Optional[str]" = None
) -> "dict[str, Any]":
    """The ``optimized_setup.json`` structure from :func:`rule_of_thumb` (no benchmark)."""
    return {
        "machine": dict(machine),
        "recommendations": rule_of_thumb(machine),
        "policy": {"source": "heuristic",
                   "per_car_floor": DEFAULT_PER_CAR_FLOOR,
                   "throughput_tol": DEFAULT_THROUGHPUT_TOL},
        "generated_at": timestamp,
    }


def build_profile(
    machine: "Mapping[str, Any]",
    *,
    feature_candidates: "Optional[List[Mapping[str, Any]]]" = None,
    camera_candidates: "Optional[List[Mapping[str, Any]]]" = None,
    per_car_floor: int = DEFAULT_PER_CAR_FLOOR,
    throughput_tol: float = DEFAULT_THROUGHPUT_TOL,
    timestamp: "Optional[str]" = None,
) -> "dict[str, Any]":
    """Assemble the ``optimized_setup.json`` structure from a machine profile and
    (optional) per-observation-type benchmark candidates."""
    def rec(cands):
        return recommend(cands or [], per_car_floor=per_car_floor,
                         throughput_tol=throughput_tol)

    return {
        "machine": dict(machine),
        "recommendations": {
            "feature_obs": rec(feature_candidates),
            "camera_obs": rec(camera_candidates),
        },
        "policy": {"per_car_floor": per_car_floor, "throughput_tol": throughput_tol},
        "generated_at": timestamp,
    }
