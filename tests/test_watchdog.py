"""Tests for the host liveness watchdog (docker_runner) + the in-container
heartbeat callback — the D3-hang fix (docs/reports/d3-hang-postmortem.md).
The docker spawn loops need Docker; the liveness *decision* and the heartbeat
touch do not, and they're the load-bearing logic."""
from __future__ import annotations

import time

import gym_dr.docker_runner as dr


class _FakeProc:
    """Stand-in for a Popen that never exits (a hung container)."""

    def poll(self):
        return None


def test_is_hung_fresh_heartbeat_is_alive(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "_WATCHDOG_TIMEOUT", 600)
    hb = tmp_path / "hb"
    hb.touch()  # just now
    assert dr._is_hung(_FakeProc(), hb, time.monotonic()) is False


def test_is_hung_stale_heartbeat_is_hung(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "_WATCHDOG_TIMEOUT", 5)
    hb = tmp_path / "hb"
    hb.touch()
    # backdate the heartbeat well past the timeout
    old = time.time() - 60
    import os
    os.utime(hb, (old, old))
    assert dr._is_hung(_FakeProc(), hb, time.monotonic()) is True


def test_is_hung_boot_grace_before_first_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "_WATCHDOG_BOOT_GRACE", 300)
    hb = tmp_path / "missing"          # no heartbeat yet (still booting)
    just_started = time.monotonic()
    assert dr._is_hung(_FakeProc(), hb, just_started) is False
    # past the boot grace with still no heartbeat -> hung
    long_ago = time.monotonic() - 1000
    assert dr._is_hung(_FakeProc(), hb, long_ago) is True


def test_heartbeat_paths_match_mount():
    host, container = dr._heartbeat_paths(__import__("pathlib").Path("/a/art"), "gym-dr-z")
    assert str(host) == "/a/art/.heartbeat-gym-dr-z"
    assert container == "/workspace/artifacts/.heartbeat-gym-dr-z"


def test_heartbeat_callback_touches(tmp_path, monkeypatch):
    hb = tmp_path / "beat"
    monkeypatch.setenv("GYM_DR_HEARTBEAT", str(hb))
    from gym_dr.trainers.sb3.callbacks import HeartbeatCallback

    cb = HeartbeatCallback(interval_steps=2)
    cb._on_training_start()
    assert hb.exists()                 # touched on start
    first = hb.stat().st_mtime
    time.sleep(0.01)
    cb.num_timesteps = 5               # >= interval since last (0)
    assert cb._on_step() is True
    assert hb.stat().st_mtime >= first


def test_heartbeat_callback_noop_without_env(monkeypatch):
    monkeypatch.delenv("GYM_DR_HEARTBEAT", raising=False)
    from gym_dr.trainers.sb3.callbacks import HeartbeatCallback

    cb = HeartbeatCallback()
    cb._on_training_start()            # must not raise
    cb.num_timesteps = 1000
    assert cb._on_step() is True


def test_restart_code_synced_with_app():
    from gym_dr.app import _SIM_RESTART_RC
    assert dr.SIM_RESTART_RC == _SIM_RESTART_RC
