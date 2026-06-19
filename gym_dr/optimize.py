"""Convert an ONNX model to OpenVINO IR (``.xml``/``.bin``) — ROS-free.

This is the standalone, importable mirror of the ``if input_model.endswith(".onnx")``
branch that the AWS device-stack ``model_optimizer_node`` would carry. Keeping it
here (rather than inside a ROS node we don't have in this checkout) lets us validate
the conversion plumbing on a workstation and port the exact logic on-device later.

Two backends, deliberately:

- ``"legacy"`` — OpenVINO 2021.x ``mo`` (the Model Optimizer the physical car actually
  runs, OpenVINO 2021.1.110 era). Invoked as a subprocess so its old NumPy pin stays
  isolated in its own venv.
- ``"modern"`` — current OpenVINO (``openvino.convert_model`` / ``save_model``), the
  ``ovc`` successor to ``mo.py``.

We expose both so a caller can convert with each and compare the resulting IR — the
"try both and report the difference" gate. ``"auto"`` picks whichever is importable in
the running interpreter.

The correctness gate uses FP32 (``data_type="FP32"``); FP16 (for the iGPU) is deferred.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

Backend = str  # "legacy" | "modern" | "auto"


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #

def detect_backend() -> Backend:
    """Return ``"modern"`` or ``"legacy"`` based on the importable OpenVINO.

    OpenVINO >= 2022 ships the ``openvino.convert_model`` API; 2021.x ships only the
    Inference Engine (``openvino.inference_engine``) + the ``mo`` CLI.
    """
    try:
        import openvino  # noqa: F401

        if hasattr(openvino, "convert_model") or hasattr(openvino, "Core"):
            return "modern"
    except Exception:  # noqa: BLE001
        pass
    if _find_mo() is not None:
        return "legacy"
    try:
        from openvino.inference_engine import IECore  # noqa: F401

        return "legacy"
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "no usable OpenVINO backend found in this interpreter "
            f"({sys.executable}); install `openvino` (modern) or "
            "`openvino-dev==2021.4.*` (legacy)"
        ) from e


def _find_mo() -> Optional[str]:
    """Locate the legacy ``mo`` Model Optimizer entry point."""
    candidate = Path(sys.executable).with_name("mo")
    if candidate.exists():
        return str(candidate)
    return shutil.which("mo")


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def onnx_to_ir(
    onnx_path: Path,
    out_dir: Path,
    *,
    data_type: str = "FP32",
    input_shape: Optional[Sequence[int]] = None,
    reverse_input_channels: bool = False,
    backend: Backend = "auto",
    model_name: str = "model",
) -> Tuple[Path, Path]:
    """Convert ``onnx_path`` to OpenVINO IR; return ``(xml_path, bin_path)``.

    Args:
        onnx_path: source ``.onnx`` model.
        out_dir: directory to write ``<model_name>.xml`` / ``.bin``.
        data_type: ``"FP32"`` (correctness gate) or ``"FP16"`` (deferred, GPU).
        input_shape: pin a static input shape, e.g. ``[1, 1, 120, 160]``. Mirrors
            ``mo --input_shape`` / fixes "shape is not fully defined" errors.
        reverse_input_channels: BGR<->RGB swap. No-op for 1-channel inputs (OpenVINO
            only reverses 4-D, 3-channel inputs). Honored on the legacy backend
            (``mo --reverse_input_channels``); on modern it is applied via the
            pre/post processor when the single input is 4-D/3-channel, else skipped
            with a warning.
        backend: ``"legacy"``, ``"modern"``, or ``"auto"``.
        model_name: output basename.

    Returns the IR ``(xml, bin)`` paths.
    """
    onnx_path = Path(onnx_path).resolve()
    out_dir = Path(out_dir).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if backend == "auto":
        backend = detect_backend()
    LOG.info("onnx_to_ir: %s -> IR (backend=%s, data_type=%s)", onnx_path.name, backend, data_type)

    if backend == "legacy":
        xml, bin_ = _onnx_to_ir_legacy(
            onnx_path, out_dir, data_type, input_shape, reverse_input_channels, model_name
        )
    elif backend == "modern":
        xml, bin_ = _onnx_to_ir_modern(
            onnx_path, out_dir, data_type, input_shape, reverse_input_channels, model_name
        )
    else:
        raise ValueError(f"unknown backend {backend!r}; expected legacy/modern/auto")

    if not xml.exists() or not bin_.exists():
        raise RuntimeError(f"IR not produced: {xml}, {bin_}")
    LOG.info("wrote IR: %s (+ .bin)", xml)
    return xml, bin_


def _onnx_to_ir_legacy(onnx_path, out_dir, data_type, input_shape, reverse_input_channels, model_name):
    mo = _find_mo()
    if mo is None:
        raise RuntimeError("legacy backend requested but `mo` not found on PATH / next to python")
    cmd: List[str] = [
        mo,
        "--input_model", str(onnx_path),
        "--output_dir", str(out_dir),
        "--model_name", model_name,
        "--data_type", data_type.upper(),
    ]
    if input_shape is not None:
        cmd += ["--input_shape", "[" + ",".join(str(int(d)) for d in input_shape) + "]"]
    if reverse_input_channels:
        cmd += ["--reverse_input_channels"]
    LOG.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mo failed (exit {proc.returncode}):\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return out_dir / f"{model_name}.xml", out_dir / f"{model_name}.bin"


def _onnx_to_ir_modern(onnx_path, out_dir, data_type, input_shape, reverse_input_channels, model_name):
    import openvino as ov

    ov_model = ov.convert_model(str(onnx_path))

    if input_shape is not None:
        if len(ov_model.inputs) != 1:
            raise ValueError(
                "input_shape pinning supported only for single-input models on the "
                f"modern backend; model has {len(ov_model.inputs)} inputs"
            )
        ov_model.reshape(list(int(d) for d in input_shape))

    if reverse_input_channels:
        _apply_reverse_channels_modern(ov, ov_model)

    xml = out_dir / f"{model_name}.xml"
    ov.save_model(ov_model, str(xml), compress_to_fp16=(data_type.upper() == "FP16"))
    return xml, out_dir / f"{model_name}.bin"


def _apply_reverse_channels_modern(ov, ov_model) -> None:
    """Best-effort BGR<->RGB swap on the modern backend (mirrors mo flag)."""
    try:
        inp = ov_model.input(0)
        shape = inp.get_partial_shape()
        is_4d_3ch = len(shape) == 4 and shape[1].is_static and shape[1].get_length() == 3
        if not is_4d_3ch:
            LOG.warning(
                "reverse_input_channels requested but input is not 4-D/3-channel; "
                "skipping (matches OpenVINO's own no-op behavior for grayscale)"
            )
            return
        from openvino.preprocess import ColorFormat, PrePostProcessor

        ppp = PrePostProcessor(ov_model)
        ppp.input().tensor().set_color_format(ColorFormat.BGR)
        ppp.input().preprocess().convert_color(ColorFormat.RGB)
        ppp.build()
    except Exception as e:  # noqa: BLE001
        LOG.warning("could not apply reverse_input_channels on modern backend: %s", e)


# --------------------------------------------------------------------------- #
# IR inference (both APIs) — reused by the smoke tests
# --------------------------------------------------------------------------- #

def run_ir(
    xml_path: Path,
    feeds: Dict[str, "object"],
    *,
    backend: Backend = "auto",
    device: str = "CPU",
    force_fp32: bool = True,
) -> Dict[str, "object"]:
    """Run an IR model on ``feeds`` (name -> ndarray); return name -> ndarray.

    Handles both the 2021.x Inference Engine (``IECore``) and the modern
    ``ov.Core`` runtime so callers don't branch on version.

    ``force_fp32`` (default True) disables OpenVINO's automatic bf16 inference on
    bf16-capable CPUs (AVX512_BF16). Without it, an FP32 IR is silently executed in
    bf16 (~0.4% error, outputs snap to a ~1/64 grid), which breaks numeric-equivalence
    gates. Set False to measure the device's real (bf16) runtime behavior.
    """
    xml_path = Path(xml_path).resolve()
    if backend == "auto":
        backend = detect_backend()
    if backend == "legacy":
        return _run_ir_legacy(xml_path, feeds, device, force_fp32)
    return _run_ir_modern(xml_path, feeds, device, force_fp32)


_IE_PREC_TO_NP = {
    "U8": "uint8", "FP32": "float32", "FP16": "float16",
    "I32": "int32", "I64": "int64",
}


def _run_ir_legacy(xml_path, feeds, device, force_fp32):
    import numpy as np
    from openvino.inference_engine import IECore

    ie = IECore()
    net = ie.read_network(model=str(xml_path), weights=str(xml_path.with_suffix(".bin")))
    # Cast each feed to the IR's declared input precision (mo may keep U8 or FP32).
    cast = {}
    for name, info in net.input_info.items():
        arr = np.asarray(feeds[name])
        np_dt = _IE_PREC_TO_NP.get(info.precision)
        cast[name] = arr.astype(np_dt) if np_dt else arr
    config = {"ENFORCE_BF16": "NO"} if force_fp32 else {}
    exec_net = ie.load_network(network=net, device_name=device, config=config)
    result = exec_net.infer(inputs=cast)
    return {k: v for k, v in result.items()}


def _run_ir_modern(xml_path, feeds, device, force_fp32):
    import numpy as np
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(xml_path))
    cast = {}
    for inp in model.inputs:
        name = inp.get_any_name()
        arr = np.asarray(feeds[name])
        try:
            np_dt = inp.get_element_type().to_dtype()
            arr = arr.astype(np_dt)
        except Exception:  # noqa: BLE001
            pass
        cast[name] = arr
    props = {"INFERENCE_PRECISION_HINT": "f32"} if force_fp32 else {}
    compiled = core.compile_model(model, device, props)
    result = compiled(cast)
    out: Dict[str, object] = {}
    for port, value in result.items():
        try:
            name = port.get_any_name()
        except Exception:  # noqa: BLE001
            name = str(port)
        out[name] = value
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Convert ONNX -> OpenVINO IR (FP32 by default).")
    p.add_argument("onnx", type=Path, help="input .onnx model")
    p.add_argument("--out", type=Path, default=Path("ir"), help="output dir for .xml/.bin")
    p.add_argument("--name", default="model", help="output basename")
    p.add_argument("--data-type", default="FP32", choices=["FP32", "FP16"])
    p.add_argument("--input-shape", default=None, help='e.g. "1,1,120,160"')
    p.add_argument("--reverse-input-channels", action="store_true")
    p.add_argument("--backend", default="auto", choices=["auto", "legacy", "modern"])
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    shape = [int(x) for x in args.input_shape.split(",")] if args.input_shape else None
    xml, bin_ = onnx_to_ir(
        args.onnx,
        args.out,
        data_type=args.data_type,
        input_shape=shape,
        reverse_input_channels=args.reverse_input_channels,
        backend=args.backend,
        model_name=args.name,
    )
    print(f"IR written:\n  {xml}\n  {bin_}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
