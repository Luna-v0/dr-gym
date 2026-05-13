#!/usr/bin/env bash
# Run an experiment script inside the deepracer-env container.
#
# Usage:
#   ./run_cpu_training.sh experiments/quick.py
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-my-deepracer-project:cpu}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${PROJECT_DIR}/artifacts}"
MLRUNS_DIR="${MLRUNS_DIR:-${PROJECT_DIR}/mlruns}"
CHOWN_ARTIFACTS="${CHOWN_ARTIFACTS:-1}"

EXPERIMENT_PATH_HOST="${1:-app.py}"
if [[ ! -f "${EXPERIMENT_PATH_HOST}" ]]; then
  echo "Experiment script not found: ${EXPERIMENT_PATH_HOST}" >&2
  exit 1
fi
ABS_EXPERIMENT="$(realpath "${EXPERIMENT_PATH_HOST}")"
case "${ABS_EXPERIMENT}" in
  "${PROJECT_DIR}"/*) ;;
  *)
    echo "Experiment script must live inside project dir (${PROJECT_DIR}): ${ABS_EXPERIMENT}" >&2
    exit 1
    ;;
esac
EXPERIMENT_PATH_CONTAINER="/workspace${ABS_EXPERIMENT#${PROJECT_DIR}}"

mkdir -p "${ARTIFACTS_DIR}" "${MLRUNS_DIR}"

# Pre-generate model_metadata.json from the experiment's action_space so the
# simapp picks up the right action space at container start.
if command -v uv >/dev/null 2>&1; then
  uv run --project "${PROJECT_DIR}" python -m gym_dr.cli prepare-metadata \
    "${EXPERIMENT_PATH_HOST}" --output "${PROJECT_DIR}/model_metadata.json"
else
  python3 -m gym_dr.cli prepare-metadata \
    "${EXPERIMENT_PATH_HOST}" --output "${PROJECT_DIR}/model_metadata.json"
fi

docker_args=(
  docker run --rm
  -v "${PROJECT_DIR}:/workspace:rw"
  -v "${ARTIFACTS_DIR}:/workspace/artifacts"
  -v "${MLRUNS_DIR}:/workspace/mlruns"
  -e WORLD_NAME="${WORLD_NAME:-reinvent_base}"
  -e ENABLE_GUI="${ENABLE_GUI:-False}"
  -e EXPERIMENT_PATH="${EXPERIMENT_PATH_CONTAINER}"
)

if [[ -n "${RTF_OVERRIDE:-}" ]]; then
  docker_args+=(-e RTF_OVERRIDE="${RTF_OVERRIDE}")
fi

docker_args+=("${IMAGE_TAG}")

echo "Project dir:    ${PROJECT_DIR}"
echo "Artifacts dir:  ${ARTIFACTS_DIR}"
echo "MLflow dir:     ${MLRUNS_DIR}"
echo "Experiment:     ${EXPERIMENT_PATH_HOST}"
echo

set +e
"${docker_args[@]}"
TRAINING_EXIT=$?
set -e

if [[ "${CHOWN_ARTIFACTS}" == "1" ]]; then
  docker run --rm \
    --entrypoint chown \
    -v "${ARTIFACTS_DIR}:/workspace/artifacts" \
    -v "${MLRUNS_DIR}:/workspace/mlruns" \
    "${IMAGE_TAG}" \
    -R "$(id -u):$(id -g)" /workspace/artifacts /workspace/mlruns || true
fi

exit "${TRAINING_EXIT}"
