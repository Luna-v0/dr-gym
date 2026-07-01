"""Tests for the Ctrl-C pipeline teardown in gym_dr.docker_runner (Task 8)."""
from __future__ import annotations

import signal

import pytest

import gym_dr.docker_runner as dr


@pytest.fixture(autouse=True)
def _clean_registry():
    dr._ACTIVE_CONTAINERS.clear()
    yield
    dr._ACTIVE_CONTAINERS.clear()


def test_register_unregister():
    dr._register("a")
    dr._register("b")
    assert dr._ACTIVE_CONTAINERS == {"a", "b"}
    dr._unregister("a")
    assert dr._ACTIVE_CONTAINERS == {"b"}
    dr._unregister("missing")  # idempotent


def test_kill_all_active_kills_and_clears(monkeypatch):
    killed = []
    monkeypatch.setattr(dr, "_docker_kill", lambda n: killed.append(n))
    dr._ACTIVE_CONTAINERS.update({"c1", "c2"})
    dr._kill_all_active()
    assert set(killed) == {"c1", "c2"}
    assert dr._ACTIVE_CONTAINERS == set()


def test_signal_teardown_restores_handlers(monkeypatch):
    monkeypatch.setattr(dr, "_docker_kill", lambda n: None)
    before = signal.getsignal(signal.SIGINT)
    with dr._signal_teardown():
        assert signal.getsignal(signal.SIGINT) is not before  # our handler installed
    assert signal.getsignal(signal.SIGINT) is before          # restored on exit


def test_signal_handler_kills_all_and_reraises(monkeypatch):
    """The installed handler docker-kills every active container and re-raises
    KeyboardInterrupt, so one Ctrl-C stops the containers AND the host."""
    killed = []
    monkeypatch.setattr(dr, "_docker_kill", lambda n: killed.append(n))
    dr._ACTIVE_CONTAINERS.update({"a", "b"})
    with pytest.raises(KeyboardInterrupt):
        with dr._signal_teardown():
            handler = signal.getsignal(signal.SIGINT)
            handler(signal.SIGINT, None)  # simulate the kernel delivering SIGINT
    assert set(killed) == {"a", "b"}
    assert dr._ACTIVE_CONTAINERS == set()


def test_spawn_training_chunk_registers_then_cleans(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(dr, "_docker_kill", lambda n: None)
    monkeypatch.setattr(dr, "_docker_rm_f", lambda n: None)

    seen = {}

    class FakeProc:
        def __init__(self, *a, **k):
            seen["spawned"] = True

        def wait(self, timeout=None):
            # The container must be registered while it is running.
            seen["registered_during"] = "gym-dr-x" in dr._ACTIVE_CONTAINERS
            return 0

    monkeypatch.setattr(dr.subprocess, "Popen", FakeProc)

    rc = dr.spawn_training_chunk("img:cpu", "gym-dr-x", {"A": "B"})
    assert rc == 0
    assert seen["registered_during"] is True
    assert "gym-dr-x" not in dr._ACTIVE_CONTAINERS  # unregistered after exit
