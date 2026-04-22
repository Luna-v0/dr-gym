#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-my-deepracer-project:cpu}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$(pwd)/artifacts}"
PORT="${PORT:-6006}"
RUN_NAME="${RUN_NAME:-}"

mkdir -p "${ARTIFACTS_DIR}"

if [[ -n "${RUN_NAME}" ]]; then
  LOGDIR="/workspace/artifacts/${RUN_NAME}/tensorboard"
else
  LOGDIR="/workspace/artifacts"
fi

echo "Serving TensorBoard from ${LOGDIR} on http://localhost:${PORT}"

docker run --rm \
  -p "${PORT}:6006" \
  -v "${ARTIFACTS_DIR}:/workspace/artifacts" \
  --entrypoint python3 \
  "${IMAGE_TAG}" \
  -m tensorboard.main \
  --logdir "${LOGDIR}" \
  --host 0.0.0.0 \
  --port 6006
