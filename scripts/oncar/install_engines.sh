#!/usr/bin/env bash
# Install the inference engines for the on-car comparison (run ON the Pi).
# Target: Raspberry Pi 4, aarch64, Ubuntu 24.04, Python 3.12.
# Uses a venv so the system Python / DeepRacer stack stays untouched.
#
#   bash install_engines.sh
#
# Some engines may not ship an aarch64/py3.12 wheel; failures are tolerated
# (the benchmark just skips engines it can't import).
set -u

VENV="${VENV:-$HOME/oncar_bench/venv}"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

echo "== core =="
pip install numpy psutil

echo "== onnxruntime (primary; runs the exported ONNX directly) =="
pip install onnxruntime || echo "[warn] onnxruntime install failed"

echo "== openvino (ARM CPU plugin; also reads the ONNX) =="
pip install "openvino>=2024" || echo "[warn] openvino install failed"

echo "== tflite / LiteRT (needs a .tflite model — conversion comes separately) =="
# tflite-runtime has no aarch64/py3.12 wheel; it's superseded by ai-edge-litert (LiteRT).
pip install ai-edge-litert || pip install tflite-runtime || pip install tensorflow \
  || echo "[warn] no tflite/LiteRT runtime (a system tflite may already exist on the Pi)"

echo "== executorch (needs a .pte model — conversion comes separately) =="
pip install executorch || echo "[warn] executorch wheel unavailable for aarch64/py3.12 — ok to skip"

echo
echo "Installed into $VENV. Now run:"
echo "  source $VENV/bin/activate && python $(dirname "$0")/bench_engines.py --model-dir $HOME/oncar_bench"
