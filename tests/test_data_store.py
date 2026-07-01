"""Tests for the provenance data store (gym_dr.data_store, Task 5)."""
from __future__ import annotations

from gym_dr.data_store import (
    NullDataStore,
    SQLiteDataStore,
    make_data_store,
)


def test_sqlite_round_trip(tmp_path):
    db = tmp_path / "run" / "provenance.sqlite"
    with SQLiteDataStore(db) as store:
        store.register_run("run1", experiment="oracle", config={"n_cars": 4})
        uid = store.start_episode("run1", episode=0, car=2, world="Oval", phase="eval")
        store.record_dr(uid, {"friction_mu": 1.35, "reverse_dir": True, "track_color": "blue"})
        store.record_metrics(uid, {"dr/ep_reward": 12.5, "dr/ep_offtrack_rate": 0.0,
                                   "note": "not-a-number"})

        assert store.episode_count("run1") == 1
        dr = store.read_dr(uid)
        assert dr["friction_mu"] == 1.35
        assert dr["reverse_dir"] == 1.0          # bool -> numeric
        assert dr["track_color"] == "blue"       # categorical -> text
        metrics = store.read_metrics(uid)
        assert metrics["dr/ep_reward"] == 12.5
        assert "note" not in metrics             # non-numeric metric dropped


def test_read_episodes_merges_dr_and_metrics(tmp_path):
    store = SQLiteDataStore(tmp_path / "p.sqlite")
    store.register_run("r")
    u0 = store.start_episode("r", episode=0, world="A")
    u1 = store.start_episode("r", episode=1, world="B")
    store.record_dr(u0, {"friction_mu": 1.0})
    store.record_metrics(u1, {"dr/ep_reward": 3.0})

    rows = store.read_episodes("r")
    assert [r["episode"] for r in rows] == [0, 1]
    assert rows[0]["dr"] == {"friction_mu": 1.0}
    assert rows[0]["metrics"] == {}
    assert rows[1]["metrics"] == {"dr/ep_reward": 3.0}
    assert rows[0]["episode_uuid"] != rows[1]["episode_uuid"]  # unique join keys
    store.close()


def test_episode_uuids_are_unique(tmp_path):
    store = SQLiteDataStore(tmp_path / "p.sqlite")
    store.register_run("r")
    uids = {store.start_episode("r", episode=i) for i in range(50)}
    assert len(uids) == 50
    store.close()


def test_persists_across_reopen(tmp_path):
    db = tmp_path / "p.sqlite"
    s1 = SQLiteDataStore(db)
    s1.register_run("r")
    s1.start_episode("r", episode=0)
    s1.close()
    s2 = SQLiteDataStore(db)          # reopen the same file
    assert s2.episode_count("r") == 1
    s2.close()


def test_null_store_is_noop():
    store = NullDataStore()
    store.register_run("r", experiment="x")
    uid = store.start_episode("r", episode=0)
    assert uid == ""
    store.record_dr(uid, {"a": 1})
    store.record_metrics(uid, {"b": 2})
    store.close()


def test_factory_selects_adapter(tmp_path):
    assert isinstance(make_data_store(None), NullDataStore)
    assert isinstance(make_data_store(""), NullDataStore)
    s = make_data_store(tmp_path / "p.sqlite")
    assert isinstance(s, SQLiteDataStore)
    s.close()
