"""Tests for the physical-car bundle exporter.

Two paths exercised end-to-end:

1. ``.pb`` round-trip: a small fake .pb file is packaged verbatim; we untar
   the bundle and assert the tree layout matches the contract.
2. SB3 .zip → ONNX: a tiny PPO is trained briefly against the stub env from
   ``test_smoke.py``, the resulting .zip is exported, and the ONNX inside
   the bundle is validated with ``onnx.checker.check_model``.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from gym_dr import (
    ContinuousActionSpaceConfig,
    ExperimentConfig,
    Sb3Trainer,
    TrackingConfig,
    TrainingConfig,
    WorldsConfig,
    center_line,
)
from gym_dr.app import train
from gym_dr.export import export_bundle, sb3_to_onnx
from tests.test_smoke import StubDeepRacerEnv, stub_env_factory  # noqa: F401


def _untar(tar_path: Path, dest: Path) -> list[str]:
    """Extract a tar.gz and return the sorted member list."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        names = sorted(m.name for m in tf.getmembers())
        tf.extractall(dest)
    return names


def _minimal_metadata(tmp_path: Path) -> Path:
    p = tmp_path / "meta.json"
    p.write_text(json.dumps({
        "sensor": ["FRONT_FACING_CAMERA"],
        "neural_network": "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW",
        "version": 5.0,
        "training_algorithm": "clipped_ppo",
        "action_space_type": "continuous",
        "action_space": {
            "steering_angle": {"low": -30.0, "high": 30.0},
            "speed": {"low": 0.1, "high": 4.0},
        },
    }))
    return p


def test_export_pb_verbatim(tmp_path):
    """A .pb file is copied verbatim into agent/agent.pb; metadata at tar root."""
    fake_pb = tmp_path / "fake_model.pb"
    fake_pb.write_bytes(b"\x08\x01")  # plausible protobuf header bytes
    meta = _minimal_metadata(tmp_path)

    bundle = tmp_path / "bundle.tar.gz"
    export_bundle(model_path=fake_pb, output_path=bundle, metadata_path=meta)

    assert bundle.exists()
    members = _untar(bundle, tmp_path / "unpacked")
    assert members == ["agent", "agent/agent.pb", "model_metadata.json"]

    extracted_pb = tmp_path / "unpacked" / "agent" / "agent.pb"
    assert extracted_pb.read_bytes() == fake_pb.read_bytes()

    extracted_meta = json.loads((tmp_path / "unpacked" / "model_metadata.json").read_text())
    assert extracted_meta["action_space_type"] == "continuous"
    assert extracted_meta["version"] == 5.0


def test_export_onnx_verbatim(tmp_path):
    """An .onnx file is copied verbatim into agent/agent.onnx."""
    fake_onnx = tmp_path / "fake_model.onnx"
    fake_onnx.write_bytes(b"\x08\x01")  # placeholder bytes
    meta = _minimal_metadata(tmp_path)

    bundle = tmp_path / "bundle.tar.gz"
    export_bundle(model_path=fake_onnx, output_path=bundle, metadata_path=meta)

    members = _untar(bundle, tmp_path / "unpacked")
    assert members == ["agent", "agent/agent.onnx", "model_metadata.json"]


def test_bundle_filename_override(tmp_path):
    """--bundle-filename overrides the in-tar filename."""
    fake = tmp_path / "fake.pb"
    fake.write_bytes(b"\x08\x01")
    meta = _minimal_metadata(tmp_path)

    bundle = tmp_path / "bundle.tar.gz"
    export_bundle(
        model_path=fake,
        output_path=bundle,
        metadata_path=meta,
        bundle_filename="model.pb",  # what AWS Console uses sometimes
    )
    members = _untar(bundle, tmp_path / "unpacked")
    assert members == ["agent", "agent/model.pb", "model_metadata.json"]


def test_metadata_from_sibling(tmp_path):
    """If <model>.model_metadata.json sits next to the model, use it."""
    fake = tmp_path / "model.pb"
    fake.write_bytes(b"\x08\x01")
    sibling = tmp_path / "model.model_metadata.json"
    sibling.write_text((_minimal_metadata(tmp_path)).read_text())

    bundle = tmp_path / "bundle.tar.gz"
    export_bundle(model_path=fake, output_path=bundle)

    members = _untar(bundle, tmp_path / "unpacked")
    assert "model_metadata.json" in members


def test_missing_metadata_fails_clearly(tmp_path):
    fake = tmp_path / "model.pb"
    fake.write_bytes(b"\x08\x01")
    with pytest.raises(FileNotFoundError, match="metadata"):
        export_bundle(model_path=fake, output_path=tmp_path / "out.tar.gz")


def test_sb3_zip_to_onnx_bundle(tmp_path, monkeypatch):
    """End-to-end: train a tiny PPO, export to bundle, verify the ONNX inside."""
    onnx = pytest.importorskip("onnx")  # noqa: F841

    monkeypatch.setenv("GYM_DR_IN_CONTAINER", "1")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    experiment = ExperimentConfig(
        name="export_test",
        env_factory=stub_env_factory,
        trainer=Sb3Trainer(
            name="ppo",
            policy="MultiInputPolicy",
            kwargs={"n_steps": 64, "batch_size": 32, "learning_rate": 3e-4, "ent_coef": 0.01},
            device="cpu",
        ),
        reward=center_line,
        action_space=ContinuousActionSpaceConfig(),
        worlds=WorldsConfig(names=["stub"], chunk_steps=200, rotations=1),
        training=TrainingConfig(total_timesteps=200, checkpoint_freq=200, eval_freq=100, n_eval_episodes=1),
        tracking=TrackingConfig(
            mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
            mlflow_experiment="export-test",
        ),
        seed=42,
    )
    train(experiment)

    run_dir = tmp_path / "artifacts" / "export_test"
    model_zip = run_dir / "final_model.zip"
    assert model_zip.exists()
    # The sibling metadata gets written by Sb3Trainer's checkpoint flow.
    assert (run_dir / "final_model.model_metadata.json").exists()

    bundle = tmp_path / "bundle.tar.gz"
    export_bundle(model_path=model_zip, output_path=bundle)

    members = _untar(bundle, tmp_path / "unpacked")
    assert members == ["agent", "agent/agent.onnx", "model_metadata.json"]

    # Validate the ONNX is structurally well-formed.
    import onnx as _onnx
    model = _onnx.load(str(tmp_path / "unpacked" / "agent" / "agent.onnx"))
    _onnx.checker.check_model(model)
    # Sanity-check the input names: one per obs dict key.
    input_names = {i.name for i in model.graph.input}
    assert input_names == {"FRONT_FACING_CAMERA"}
    # One output named per our default.
    output_names = {o.name for o in model.graph.output}
    assert output_names == {"action"}


def test_sb3_to_onnx_helper_directly(tmp_path, monkeypatch):
    """sb3_to_onnx() is callable on its own without going through export_bundle."""
    pytest.importorskip("onnx")

    monkeypatch.setenv("GYM_DR_IN_CONTAINER", "1")
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    experiment = ExperimentConfig(
        name="onnx_helper",
        env_factory=stub_env_factory,
        trainer=Sb3Trainer(
            name="ppo", policy="MultiInputPolicy",
            kwargs={"n_steps": 64, "batch_size": 32, "learning_rate": 3e-4, "ent_coef": 0.01},
            device="cpu",
        ),
        reward=center_line,
        action_space=ContinuousActionSpaceConfig(),
        worlds=WorldsConfig(names=["stub"], chunk_steps=200, rotations=1),
        training=TrainingConfig(total_timesteps=200, checkpoint_freq=200, eval_freq=100, n_eval_episodes=1),
        tracking=TrackingConfig(
            mlflow_tracking_uri=f"file://{tmp_path / 'mlruns'}",
            mlflow_experiment="onnx-helper",
        ),
    )
    train(experiment)

    out = tmp_path / "policy.onnx"
    sb3_to_onnx(tmp_path / "artifacts" / "onnx_helper" / "final_model.zip", out)
    assert out.exists() and out.stat().st_size > 0
