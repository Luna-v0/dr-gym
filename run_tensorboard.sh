#!/usr/bin/env bash
# Launch TensorBoard on the host (NOT inside the container).
#
# The simapp container ships TensorBoard 2.14, which trips a known protobuf
# incompatibility on modern hosts (`MessageToJson() got an unexpected keyword
# argument 'including_default_value_fields'`). uv installs a modern TB in the
# host venv; using it sidesteps the bug entirely.
#
# Usage:
#   ./run_tensorboard.sh                                 # all runs under ./artifacts
#   ./run_tensorboard.sh quick_test_rot0_reinvent_base   # one specific chunk
#   PORT=6007 ./run_tensorboard.sh                       # different port
#
# Open: http://localhost:${PORT:-6006}
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${PROJECT_DIR}/artifacts}"
PORT="${PORT:-6006}"

RUN_NAME="${1:-}"
if [[ -n "${RUN_NAME}" ]]; then
  LOGDIR="${ARTIFACTS_DIR}/${RUN_NAME}/tensorboard"
else
  LOGDIR="${ARTIFACTS_DIR}"
fi

if [[ ! -d "${LOGDIR}" ]]; then
  echo "TensorBoard logdir not found: ${LOGDIR}" >&2
  echo "Existing run dirs under ${ARTIFACTS_DIR}:" >&2
  ls "${ARTIFACTS_DIR}" 2>/dev/null | sed 's/^/  /' >&2 || true
  exit 1
fi

echo "Serving TensorBoard from ${LOGDIR} on http://localhost:${PORT}"
uv run --project "${PROJECT_DIR}" tensorboard \
  --logdir "${LOGDIR}" \
  --host 0.0.0.0 \
  --port "${PORT}"
