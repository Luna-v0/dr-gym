"""Tests for gym_dr.optimize (ONNX -> OpenVINO IR).

These are guarded with ``importorskip``: the OpenVINO runtimes live in the dedicated
``.venv-ov-*`` venvs, not in the dr-gym dev venv, so on a normal ``pytest`` run (no
openvino) every test here skips cleanly rather than failing.

The round-trip test builds a tiny ONNX with torch, converts it to IR via whichever
backend is importable, and checks the IR output matches onnxruntime within FP32 tolerance
(bf16 disabled via ``force_fp32`` — see optimize.run_ir).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load gym_dr/optimize.py *by path* so this test runs both in the dr-gym dev venv
# (where it skips for lack of openvino) AND in the .venv-ov-* venvs (which have
# openvino but not stable-baselines3, so importing the gym_dr package would fail).
_OPT = Path(__file__).resolve().parents[1] / "gym_dr" / "optimize.py"
_spec = importlib.util.spec_from_file_location("dr_optimize", _OPT)
optimize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(optimize)


def _tiny_onnx(path) -> str:
    """Build a tiny Conv->Relu ONNX with the onnx helper (no torch needed).

    Using onnx.helper (not torch.onnx.export) keeps this runnable in the OV venvs,
    which have onnx/onnxruntime/openvino but not torch.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    w = np.random.default_rng(0).standard_normal((2, 1, 3, 3)).astype(np.float32)
    w_init = numpy_helper.from_array(w, name="W")
    conv = helper.make_node("Conv", ["input", "W"], ["c"], pads=[1, 1, 1, 1])
    relu = helper.make_node("Relu", ["c"], ["out"])
    graph = helper.make_graph(
        [conv, relu], "tiny",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 8, 8])],
        initializer=[w_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7  # OpenVINO 2021.4 reads up to IR v7
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return "input"


def test_detect_backend_or_skip():
    """detect_backend returns a known value, or there's no OpenVINO (skip)."""
    pytest.importorskip("openvino")
    assert optimize.detect_backend() in ("legacy", "modern")


def test_onnx_to_ir_roundtrip(tmp_path):
    pytest.importorskip("openvino")
    ort = pytest.importorskip("onnxruntime")

    onnx_path = tmp_path / "tiny.onnx"
    in_name = _tiny_onnx(onnx_path)
    backend = optimize.detect_backend()

    xml, bin_ = optimize.onnx_to_ir(
        onnx_path, tmp_path / "ir", data_type="FP32",
        input_shape=[1, 1, 8, 8],  # legacy mo needs a static batch
        backend=backend,
    )
    assert xml.exists() and bin_.exists()

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {in_name: x})[0]

    ir_out = optimize.run_ir(xml, {in_name: x}, backend=backend, force_fp32=True)
    got = ir_out[next(iter(ir_out))]

    np.testing.assert_allclose(ref, got, rtol=1e-3, atol=1e-4)


def test_input_shape_pinning_single_input(tmp_path):
    """Modern backend rejects input_shape pinning only for multi-input; single is fine."""
    pytest.importorskip("openvino")
    onnx_path = tmp_path / "tiny.onnx"
    _tiny_onnx(onnx_path)
    xml, _ = optimize.onnx_to_ir(
        onnx_path, tmp_path / "ir2", input_shape=[2, 1, 8, 8],
        backend=optimize.detect_backend(),
    )
    assert xml.exists()
