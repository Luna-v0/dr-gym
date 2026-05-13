#!/usr/bin/env bash
# First-time setup for a fresh machine.
#
# Builds (idempotently):
#   1. The upstream simulator base image  awsdeepracercommunity/deepracer-env:0.1-<arch>
#      (cloned and built from github.com/seresheim/deepracer-env)
#   2. The project training image          my-deepracer-project:<arch>
#
# Already-present images and source checkouts are reused, so re-running is fast.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH_DEFAULT="cpu"
UPSTREAM_REPO_DEFAULT="https://github.com/seresheim/deepracer-env.git"
UPSTREAM_DIR_DEFAULT="${PROJECT_DIR}/.deepracer-env-upstream"
# Pinned upstream commit — a known-good ref. Override with -r or UPSTREAM_REF.
UPSTREAM_REF_DEFAULT="979b095"
MIN_FREE_GB_WARN=30
MIN_FREE_GB_FAIL=20

ARCH="${ARCH:-${ARCH_DEFAULT}}"
UPSTREAM_REPO="${UPSTREAM_REPO:-${UPSTREAM_REPO_DEFAULT}}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${UPSTREAM_DIR_DEFAULT}}"
UPSTREAM_REF="${UPSTREAM_REF:-${UPSTREAM_REF_DEFAULT}}"

usage() {
  cat <<EOF
Usage: ./bootstrap.sh [-a cpu|gpu] [-u UPSTREAM_DIR] [-r UPSTREAM_REF] [-h]

Builds the upstream deepracer-env simulator image (from
${UPSTREAM_REPO_DEFAULT}) and then the project training image. Idempotent —
present images and source dirs are reused.

Options:
  -a ARCH            Architecture: cpu (default) or gpu.
  -u UPSTREAM_DIR    Where to clone/find the upstream source.
                     Default: ${UPSTREAM_DIR_DEFAULT}
  -r UPSTREAM_REF    Upstream git ref to check out after clone.
                     Default: ${UPSTREAM_REF_DEFAULT}
  -h                 Show this help.

Environment variables (overridden by the flags above):
  ARCH, UPSTREAM_DIR, UPSTREAM_REF, UPSTREAM_REPO, PROJECT_IMAGE

Disk: the upstream build needs ~50 GB free in Docker's storage location.
EOF
}

while getopts ":a:u:r:h" opt; do
  case "${opt}" in
    a) ARCH="${OPTARG}" ;;
    u) UPSTREAM_DIR="${OPTARG}" ;;
    r) UPSTREAM_REF="${OPTARG}" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -${OPTARG}" >&2; usage; exit 2 ;;
    :)  echo "Option -${OPTARG} requires a value" >&2; usage; exit 2 ;;
  esac
done

case "${ARCH}" in
  cpu|gpu) ;;
  *) echo "Invalid ARCH '${ARCH}'. Must be 'cpu' or 'gpu'." >&2; exit 2 ;;
esac

BASE_IMAGE="awsdeepracercommunity/deepracer-env:0.1-${ARCH}"
PROJECT_IMAGE="${PROJECT_IMAGE:-my-deepracer-project:${ARCH}}"

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
fail() { printf "\n\033[1;31m==> %s\033[0m\n" "$*" >&2; exit 1; }

# ---------- Preflight ----------

step "Preflight checks"

command -v docker >/dev/null 2>&1 || \
  fail "docker not found in PATH. Install Docker first: https://docs.docker.com/engine/install/"

docker info >/dev/null 2>&1 || \
  fail "Docker daemon not reachable. Start it (e.g. 'sudo systemctl start docker' or
launch Docker Desktop) and confirm your user is in the 'docker' group."

docker buildx version >/dev/null 2>&1 || \
  fail "docker buildx not available. Upstream build.sh requires it.
Install: https://github.com/docker/buildx#installing"

command -v git >/dev/null 2>&1 || \
  fail "git not found in PATH. Install git first."

DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
DISK_CHECK_PATH="${DOCKER_ROOT}"
[[ -d "${DISK_CHECK_PATH}" ]] || DISK_CHECK_PATH="/"
FREE_GB="$(df -BG --output=avail "${DISK_CHECK_PATH}" 2>/dev/null | tail -n1 | tr -dc '0-9')"
if [[ -z "${FREE_GB}" ]]; then
  echo "Warning: could not determine free disk space on ${DISK_CHECK_PATH}."
elif (( FREE_GB < MIN_FREE_GB_FAIL )); then
  fail "Only ${FREE_GB} GB free on ${DISK_CHECK_PATH}. Need at least ${MIN_FREE_GB_FAIL} GB; full build wants ~50 GB."
elif (( FREE_GB < MIN_FREE_GB_WARN )); then
  echo "Warning: ${FREE_GB} GB free on ${DISK_CHECK_PATH} — full build wants ~50 GB."
else
  echo "Disk: ${FREE_GB} GB free on ${DISK_CHECK_PATH}."
fi

echo "Docker daemon, buildx, git: OK."

# ---------- Base image (upstream) ----------

step "Checking base image: ${BASE_IMAGE}"
if docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "Base image already present, skipping upstream build."
else
  if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
    step "Cloning upstream into ${UPSTREAM_DIR}"
    git clone "${UPSTREAM_REPO}" "${UPSTREAM_DIR}"
  else
    echo "Upstream source already at ${UPSTREAM_DIR}."
  fi

  step "Pinning upstream to ${UPSTREAM_REF}"
  if ! git -C "${UPSTREAM_DIR}" cat-file -e "${UPSTREAM_REF}^{commit}" 2>/dev/null; then
    git -C "${UPSTREAM_DIR}" fetch --all --tags
  fi
  CURRENT_HEAD="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
  PINNED_SHA="$(git -C "${UPSTREAM_DIR}" rev-parse "${UPSTREAM_REF}^{commit}")"
  if [[ "${CURRENT_HEAD}" != "${PINNED_SHA}" ]]; then
    git -C "${UPSTREAM_DIR}" -c advice.detachedHead=false checkout "${PINNED_SHA}"
  fi
  echo "Upstream HEAD: $(git -C "${UPSTREAM_DIR}" rev-parse --short HEAD)"

  step "Building base image via upstream build.sh -a ${ARCH} (this takes a while)"
  (cd "${UPSTREAM_DIR}" && ./build.sh -a "${ARCH}")
fi

# ---------- Project image ----------

step "Building project image: ${PROJECT_IMAGE}"
docker build \
  --build-arg "SIMAPP_TAG=0.1-${ARCH}" \
  -t "${PROJECT_IMAGE}" \
  "${PROJECT_DIR}"

# ---------- Post-build sanity check ----------

step "Sanity-checking ${PROJECT_IMAGE}"
SANITY_OUT="$(docker run --rm --entrypoint python3 "${PROJECT_IMAGE}" \
  -c "import stable_baselines3 as sb3, mlflow, optuna; print('OK sb3', sb3.__version__, 'mlflow', mlflow.__version__, 'optuna', optuna.__version__)" 2>&1)" || {
  echo "${SANITY_OUT}" >&2
  fail "Project image built but failed the import sanity check. Inspect the output above."
}
echo "Image OK: ${SANITY_OUT}"

# ---------- Ready ----------

step "Done"
cat <<MSG
Ready.

Single training run — edit app.py, then:

  ./run_cpu_training.sh                        # uses ./app.py
  ./run_cpu_training.sh path/to/other_app.py   # or an explicit path

HPO with parallel containers (same script in host and worker mode):

  uv run python experiments/hpo_example.py
MSG
