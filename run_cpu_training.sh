#!/usr/bin/env bash
# Launch a PPO training run inside the deepracer-env container.
#
# Usage:
#   ./run_cpu_training.sh configs/quick.yaml          # config-driven (recommended)
#   TOTAL_TIMESTEPS=100 ./run_cpu_training.sh configs/quick.yaml   # env vars override YAML
#   ./run_cpu_training.sh                              # legacy: env vars only, no YAML
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-my-deepracer-project:cpu}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${PROJECT_DIR}/artifacts}"
# After the container exits, chown the artifacts back to the host user so you
# don't end up with root-owned files on the host. Set CHOWN_ARTIFACTS=0 to skip.
# (We can't just run the container with --user $(id -u): the in-container ROS
# stack assumes a writable HOME and fails fast as a non-root user.)
CHOWN_ARTIFACTS="${CHOWN_ARTIFACTS:-1}"

# Optional first arg: YAML config path. Translated to its in-container path.
CONFIG_PATH_HOST="${1:-}"
CONFIG_PATH_CONTAINER=""
if [[ -n "${CONFIG_PATH_HOST}" ]]; then
  if [[ ! -f "${CONFIG_PATH_HOST}" ]]; then
    echo "Config file not found: ${CONFIG_PATH_HOST}" >&2
    exit 1
  fi
  ABS_CONFIG="$(realpath "${CONFIG_PATH_HOST}")"
  case "${ABS_CONFIG}" in
    "${PROJECT_DIR}"/*)
      CONFIG_PATH_CONTAINER="/workspace${ABS_CONFIG#${PROJECT_DIR}}"
      ;;
    *)
      echo "Config file must live inside project dir (${PROJECT_DIR}): ${ABS_CONFIG}" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "${ARTIFACTS_DIR}"

docker_args=(
  docker run --rm
  -v "${PROJECT_DIR}:/workspace:ro"
  -v "${ARTIFACTS_DIR}:/workspace/artifacts"
  -e WORLD_NAME="${WORLD_NAME:-reinvent_base}"
  -e ENABLE_GUI="${ENABLE_GUI:-False}"
)

if [[ -n "${CONFIG_PATH_CONTAINER}" ]]; then
  docker_args+=(-e CONFIG_PATH="${CONFIG_PATH_CONTAINER}")
fi

# Forward parameter env vars only when the user set them. When unset, the
# container falls back to YAML, then to train.py defaults.
for var in RUN_NAME TOTAL_TIMESTEPS CHECKPOINT_FREQ SB3_DEVICE RESUME_FROM \
           RTF_OVERRIDE MAX_TRAIN_SECONDS STATUS_UPDATE_STEPS \
           STATUS_UPDATE_SECONDS N_STEPS BATCH_SIZE LEARNING_RATE ENT_COEF; do
  if [[ -n "${!var:-}" ]]; then
    docker_args+=(-e "${var}=${!var}")
  fi
done

docker_args+=("${IMAGE_TAG}")

echo "Project dir:    ${PROJECT_DIR}"
echo "Artifacts dir:  ${ARTIFACTS_DIR}"
[[ -n "${CONFIG_PATH_HOST}" ]] && echo "Config:         ${CONFIG_PATH_HOST}"
echo

set +e
"${docker_args[@]}"
TRAINING_EXIT=$?
set -e

if [[ "${CHOWN_ARTIFACTS}" == "1" ]]; then
  docker run --rm \
    --entrypoint chown \
    -v "${ARTIFACTS_DIR}:/workspace/artifacts" \
    "${IMAGE_TAG}" \
    -R "$(id -u):$(id -g)" /workspace/artifacts || true
fi

exit "${TRAINING_EXIT}"
