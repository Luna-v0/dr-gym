"""Tests for the per-machine config recommendation (gym_dr.setup_profile, Task 2)."""
from __future__ import annotations

from gym_dr.setup_profile import (
    build_profile,
    detect_machine,
    heuristic_profile,
    recommend,
    rule_of_thumb,
)


def _c(n, tput):
    return {"n_cars": n, "steps_per_s": tput}


def test_favors_more_cars_within_throughput_tolerance():
    # feature-obs shape: n8 best throughput, n6 within 80%, n4 also. Per-car steps:
    # n4=24, n6=18, n8=14. Floor 16 excludes n8 -> pick n6 (most cars that stays
    # within the floor and within 80% of max).
    cands = [_c(4, 96), _c(6, 108), _c(8, 112)]
    best = recommend(cands, per_car_floor=16, throughput_tol=0.8)
    assert best["n_cars"] == 6
    assert best["per_car_steps_s"] == 18.0
    assert best["relaxed_floor"] is False


def test_per_car_floor_can_pick_fewer_cars():
    # If only n=4 keeps the per-car budget, it wins despite lower throughput.
    cands = [_c(4, 96), _c(8, 100)]  # spc: n4=24, n8=12.5
    best = recommend(cands, per_car_floor=20)
    assert best["n_cars"] == 4


def test_relaxes_floor_when_nothing_qualifies():
    # All candidates below the floor -> relax and pick most cars within 80%.
    cands = [_c(8, 112), _c(12, 120)]  # spc: 14, 10 -> both < 16
    best = recommend(cands, per_car_floor=16, throughput_tol=0.8)
    assert best["relaxed_floor"] is True
    assert best["n_cars"] == 12  # within 80% of 120 and most cars


def test_throughput_tolerance_excludes_weak_configs():
    # n=10 has more cars but only 50% of the best throughput -> excluded by the 80% rule.
    cands = [_c(4, 100), _c(10, 55)]  # spc 25, 5.5 ; 55 < 0.8*100
    best = recommend(cands, per_car_floor=1, throughput_tol=0.8)
    assert best["n_cars"] == 4


def test_empty_candidates_returns_none():
    assert recommend([]) is None
    assert recommend([{"n_cars": 0, "steps_per_s": None}]) is None


def test_build_profile_shape():
    machine = {"cores": 8, "cuda": True, "gpu_name": "RTX 4060 Ti"}
    prof = build_profile(
        machine,
        feature_candidates=[_c(4, 96), _c(6, 108)],
        camera_candidates=[_c(2, 51)],
        timestamp="2026-07-01T00:00:00",
    )
    assert prof["machine"]["cores"] == 8
    assert prof["recommendations"]["feature_obs"]["n_cars"] == 6
    assert prof["recommendations"]["camera_obs"]["n_cars"] == 2
    assert prof["policy"]["throughput_tol"] == 0.8
    assert prof["generated_at"] == "2026-07-01T00:00:00"


def test_detect_machine_has_cores():
    m = detect_machine()
    assert m["cores"] >= 1
    assert "cuda" in m and "gpu_vram_gib" in m


def test_rule_of_thumb_pc1_8core_16gb():
    # PC1: 8 cores + RTX 4060 Ti 16 GB -> feature n=6 CPU, camera n=6 GPU render
    # (storm fixed by B7 §1.1 → the n>=6 target is viable, no longer capped at 2).
    rot = rule_of_thumb({"cores": 8, "cuda": True, "gpu_vram_gib": 16.0})
    assert rot["feature_obs"]["n_cars"] == 6
    assert rot["feature_obs"]["device"] == "cpu"
    assert rot["camera_obs"]["n_cars"] == 6
    assert rot["camera_obs"]["render"] == "gpu"


def test_rule_of_thumb_pc2_22core_8gb():
    # PC2: 22 cores + RTX 4070 Laptop 8 GB -> feature n=18, camera n=8 (launch cap) GPU.
    rot = rule_of_thumb({"cores": 22, "cuda": True, "gpu_vram_gib": 8.0})
    assert rot["feature_obs"]["n_cars"] == 18
    assert rot["camera_obs"]["n_cars"] == 8
    assert rot["camera_obs"]["render"] == "gpu"


def test_rule_of_thumb_no_gpu_uses_software_render():
    rot = rule_of_thumb({"cores": 4, "cuda": False, "gpu_vram_gib": None})
    assert rot["feature_obs"]["n_cars"] == 2
    assert rot["camera_obs"]["device"] == "cpu"
    assert rot["camera_obs"]["render"] == "software"


def test_heuristic_profile_shape():
    prof = heuristic_profile({"cores": 8, "cuda": True, "gpu_vram_gib": 16.0})
    assert prof["policy"]["source"] == "heuristic"
    assert prof["recommendations"]["feature_obs"]["n_cars"] == 6
