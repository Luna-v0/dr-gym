"""Package a trained model + DeepRacer metadata into a physical-car upload tar.

Two input paths produce the same on-disk contract (so the on-device loader
treats them identically):

- a TF frozen-graph ``.pb`` from elsewhere → packaged verbatim as ``agent/agent.pb``.
- an SB3 ``.zip`` from our training pipeline → ``model.policy`` exported via
  ``torch.onnx.export`` → packaged as ``agent/agent.onnx``.

Output bundle (all paths)::

    <name>.tar.gz
    ├── model_metadata.json
    └── agent/
        └── agent.{pb,onnx}

The on-device loader (``aws-deepracer/aws-deepracer-systems-pkg``) dispatches
on the file extension, so we keep extensions truthful: ``.pb`` for TF
protobufs, ``.onnx`` for ONNX. The script's ``--bundle-filename`` flag lets
the user override the in-tar filename if their target expects something
specific.

``model_metadata.json``'s schema already matches what the upstream loader
expects (cross-checked against
``.deepracer-env-upstream/deepracer_env/boto/s3/files/model_metadata.py``).
This module never mutates the metadata's ``version`` — that's a contract
with the on-device loader.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def export_bundle(
    model_path: Path,
    output_path: Path,
    *,
    metadata_path: Path | None = None,
    app_path: Path | None = None,
    bundle_filename: str | None = None,
    opset_version: int = 11,
    input_name: str = "input",
    output_name: str = "action",
) -> Path:
    """Build a DeepRacer physical-car upload tar.gz from a trained model.

    Args:
        model_path: ``.pb``, ``.onnx``, or SB3 ``.zip``. Extension drives the
            bundle path:
            - ``.pb``   → ``agent/agent.pb``
            - ``.onnx`` → ``agent/agent.onnx``
            - ``.zip``  → exported to ONNX → ``agent/agent.onnx``
        output_path: where to write the ``.tar.gz``.
        metadata_path: explicit ``model_metadata.json`` to embed (highest
            priority).
        app_path: ``app.py`` (or any module exporting ``experiment``/``base``);
            metadata is rendered from ``experiment.action_space``.
        bundle_filename: override the in-tar model filename. Defaults track
            the input extension (``agent.pb`` or ``agent.onnx``).
        opset_version: ONNX opset for SB3 → ONNX export. 11 is broadly
            compatible.
        input_name / output_name: ONNX tensor names — the on-device loader's
            contract dictates these; the defaults are pragmatic placeholders.

    Returns the output path.
    """
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_bytes = _resolve_metadata(model_path, metadata_path, app_path)
    metadata = json.loads(metadata_bytes)
    LOG.info(
        "metadata: version=%s sensor=%s action_space_type=%s",
        metadata.get("version"),
        metadata.get("sensor"),
        metadata.get("action_space_type"),
    )

    with tempfile.TemporaryDirectory(prefix="dr_export_") as tmp:
        staging = Path(tmp)
        agent_dir = staging / "agent"
        agent_dir.mkdir()

        # Metadata at tar root.
        (staging / "model_metadata.json").write_bytes(metadata_bytes)

        # Materialize the model file inside agent/.
        suffix = model_path.suffix.lower()
        if suffix == ".pb":
            target_name = bundle_filename or "agent.pb"
            shutil.copy2(model_path, agent_dir / target_name)
            LOG.info("packaged frozen-graph .pb verbatim as agent/%s", target_name)
        elif suffix == ".onnx":
            target_name = bundle_filename or "agent.onnx"
            shutil.copy2(model_path, agent_dir / target_name)
            LOG.info("packaged ONNX verbatim as agent/%s", target_name)
        elif suffix == ".zip":
            target_name = bundle_filename or "agent.onnx"
            sb3_to_onnx(
                model_path,
                agent_dir / target_name,
                opset_version=opset_version,
                input_name=input_name,
                output_name=output_name,
            )
            LOG.info("exported SB3 zip → agent/%s (opset %d)", target_name, opset_version)
        else:
            raise ValueError(
                f"unsupported model extension {suffix!r}; expected .pb, .onnx, or .zip"
            )

        pack_tarball(staging, output_path)

    LOG.info("wrote bundle: %s", output_path)
    return output_path


# --------------------------------------------------------------------------- #
# Helpers (also importable from tests)
# --------------------------------------------------------------------------- #

def pack_tarball(source_dir: Path, output_path: Path) -> None:
    """Pack ``source_dir``'s contents (not the dir itself) into a gzipped tar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tf:
        for child in sorted(source_dir.iterdir()):
            tf.add(child, arcname=child.name)


def sb3_to_onnx(
    sb3_zip: Path,
    output_onnx: Path,
    *,
    opset_version: int = 11,
    input_name: str = "input",
    output_name: str = "action",
) -> None:
    """Load an SB3 .zip and export its policy to ONNX (deterministic action only).

    The exported graph takes the same observation shape SB3 trained against
    and returns the deterministic action — argmax for discrete heads, mean
    for continuous heads, matching ``model.predict(obs, deterministic=True)``.

    For dict observation spaces (DeepRacer's default), each obs key becomes a
    named ONNX input; the on-device loader needs to feed them as a dict.
    """
    import torch
    import torch.nn as nn

    model = load_sb3_zip(sb3_zip)
    policy = model.policy
    policy.eval()

    is_dict_obs = hasattr(policy.observation_space, "spaces")
    dummy_obs = _dummy_obs(policy.observation_space)

    if is_dict_obs:
        obs_keys = sorted(dummy_obs.keys())

        class _DictExporter(nn.Module):
            """Accepts one positional tensor per obs key; reassembles dict."""

            def __init__(self, p, keys):
                super().__init__()
                self.policy = p
                self._keys = keys

            def forward(self, *tensors):
                with torch.no_grad():
                    obs = {k: t for k, t in zip(self._keys, tensors)}
                    actions, _, _ = self.policy(obs, deterministic=True)
                return actions

        wrapper = _DictExporter(policy, obs_keys).eval()
        ordered_inputs = tuple(dummy_obs[k] for k in obs_keys)

        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            ordered_inputs,
            str(output_onnx),
            input_names=list(obs_keys),
            output_names=[output_name],
            opset_version=opset_version,
            dynamic_axes={
                **{k: {0: "batch"} for k in obs_keys},
                output_name: {0: "batch"},
            },
            # SB3 policies don't trace cleanly under the new dynamo exporter
            # (the action distribution sampling has dynamic control flow).
            # The legacy TorchScript-based exporter handles them fine.
            dynamo=False,
        )
    else:

        class _BoxExporter(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.policy = p

            def forward(self, obs):
                with torch.no_grad():
                    actions, _, _ = self.policy(obs, deterministic=True)
                return actions

        wrapper = _BoxExporter(policy).eval()

        output_onnx.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            wrapper,
            dummy_obs,
            str(output_onnx),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset_version,
            dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
            dynamo=False,
        )


# --------------------------------------------------------------------------- #
# Internal: metadata resolution
# --------------------------------------------------------------------------- #

def _resolve_metadata(
    model_path: Path,
    metadata_path: Path | None,
    app_path: Path | None,
) -> bytes:
    if metadata_path is not None:
        return Path(metadata_path).read_bytes()
    if app_path is not None:
        return _metadata_from_app(Path(app_path))
    sibling = model_path.with_suffix(".model_metadata.json")
    if sibling.exists():
        return sibling.read_bytes()
    raise FileNotFoundError(
        "no metadata source: pass --metadata, --app, or place a "
        "<model>.model_metadata.json next to the model file."
    )


def _metadata_from_app(app_path: Path) -> bytes:
    spec = importlib.util.spec_from_file_location(app_path.stem, app_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    experiment = getattr(module, "experiment", None) or getattr(module, "base", None)
    if experiment is None:
        raise ValueError(
            f"{app_path} must export `experiment` or `base` (an ExperimentConfig)"
        )
    return (
        json.dumps(experiment.action_space.to_model_metadata_dict(), indent=2) + "\n"
    ).encode("utf-8")


# --------------------------------------------------------------------------- #
# Internal: SB3 glue
# --------------------------------------------------------------------------- #

def load_sb3_zip(path: Path):
    """Load an SB3 ``.zip`` by trying each known algorithm class."""
    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

    last_err: Exception | None = None
    for cls in (PPO, SAC, TD3, A2C, DDPG):
        try:
            return cls.load(str(path), device="cpu")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"could not load SB3 model from {path}: {last_err}")


def _dummy_obs(space):
    if hasattr(space, "spaces"):  # gym Dict
        return {k: _box_dummy(v) for k, v in space.spaces.items()}
    return _box_dummy(space)


def _box_dummy(box):
    import torch

    shape = (1, *box.shape)
    return torch.zeros(shape, dtype=_torch_dtype_for(box.dtype))


def _torch_dtype_for(np_dtype):
    import numpy as np
    import torch

    if np.issubdtype(np_dtype, np.floating):
        return torch.float32
    if np.issubdtype(np_dtype, np.integer):
        return torch.int64 if np_dtype == np.int64 else torch.uint8
    return torch.float32
