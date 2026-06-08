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
UPSTREAM_REPO_DEFAULT="https://github.com/Luna-v0/deepracer-env.git"
UPSTREAM_DIR_DEFAULT="${PROJECT_DIR}/.deepracer-env-upstream"

# Upstream branch to track. Fresh clones use its latest tip; re-runs offer to
# pull when it has advanced. Override with -b or UPSTREAM_BRANCH.
UPSTREAM_BRANCH_DEFAULT="main"

MIN_FREE_GB_WARN=30
MIN_FREE_GB_FAIL=20

ARCH="${ARCH:-${ARCH_DEFAULT}}"
UPSTREAM_REPO="${UPSTREAM_REPO:-${UPSTREAM_REPO_DEFAULT}}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${UPSTREAM_DIR_DEFAULT}}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-${UPSTREAM_BRANCH_DEFAULT}}"

usage() {
  cat <<EOF
Usage: ./bootstrap.sh [-a cpu|gpu] [-u UPSTREAM_DIR] [-b UPSTREAM_BRANCH] [-h|--help]

Builds the upstream deepracer-env simulator image (from
${UPSTREAM_REPO_DEFAULT}) and then the project training image. Idempotent —
present images and source dirs are reused.

Options:
  -a ARCH            Architecture: cpu (default) or gpu.
  -u UPSTREAM_DIR    Where to clone/find the upstream source.
                     Default: ${UPSTREAM_DIR_DEFAULT}
  -b UPSTREAM_BRANCH Upstream branch to track (its latest tip is used).
                     Default: ${UPSTREAM_BRANCH_DEFAULT}
  -h, --help         Show this help.

Environment variables (overridden by the flags above):
  ARCH, UPSTREAM_DIR, UPSTREAM_BRANCH, UPSTREAM_REPO, PROJECT_IMAGE

Disk: the upstream build needs ~50 GB free in Docker's storage location.
EOF
}

# getopts handles short flags only; translate the long --help alias first.
for arg in "$@"; do
  case "${arg}" in
    --help) usage; exit 0 ;;
    --) break ;;
  esac
done

while getopts ":a:u:b:h" opt; do
  case "${opt}" in
    a) ARCH="${OPTARG}" ;;
    u) UPSTREAM_DIR="${OPTARG}" ;;
    b) UPSTREAM_BRANCH="${OPTARG}" ;;
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

# ---------- Upstream source + update check ----------
#
# Fresh machine: clone the upstream repo and build from the latest tip of the
# tracked branch. Re-run: if the branch has advanced on the remote, offer to
# pull it and rebuild the base image; otherwise reuse what's there.

REBUILD_BASE=0
BASE_PRESENT=0
docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 && BASE_PRESENT=1

step "Checking upstream (${UPSTREAM_REPO}, branch ${UPSTREAM_BRANCH})"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  if (( BASE_PRESENT == 1 )); then
    echo "Base image present and no local checkout — nothing to update."
  else
    step "Cloning upstream into ${UPSTREAM_DIR}"
    git clone --branch "${UPSTREAM_BRANCH}" "${UPSTREAM_REPO}" "${UPSTREAM_DIR}"
    echo "Cloned ${UPSTREAM_BRANCH} at $(git -C "${UPSTREAM_DIR}" rev-parse --short HEAD)."
  fi
elif ! git -C "${UPSTREAM_DIR}" fetch --quiet origin; then
  echo "Warning: 'git fetch' on upstream failed (offline?) — using current checkout."
else
  REMOTE_REF="origin/${UPSTREAM_BRANCH}"
  LOCAL_SHA="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
  REMOTE_SHA="$(git -C "${UPSTREAM_DIR}" rev-parse --verify --quiet "${REMOTE_REF}^{commit}" 2>/dev/null || true)"
  if [[ -z "${REMOTE_SHA}" ]]; then
    echo "Could not resolve ${REMOTE_REF} — using current checkout."
  elif [[ "${LOCAL_SHA}" == "${REMOTE_SHA}" ]]; then
    echo "Upstream is up to date with ${REMOTE_REF} ($(git -C "${UPSTREAM_DIR}" rev-parse --short HEAD))."
  else
    BEHIND="$(git -C "${UPSTREAM_DIR}" rev-list --count "${LOCAL_SHA}..${REMOTE_SHA}" 2>/dev/null || echo "?")"
    echo "Upstream ${REMOTE_REF} ($(git -C "${UPSTREAM_DIR}" rev-parse --short "${REMOTE_SHA}")) is ${BEHIND} commit(s) ahead of your checkout ($(git -C "${UPSTREAM_DIR}" rev-parse --short HEAD))."
    DO_UPDATE=0
    if [[ ! -t 0 ]]; then
      echo "Non-interactive shell — keeping current checkout."
    else
      read -r -p "Pull the latest ${UPSTREAM_BRANCH} and rebuild the base image? [y/N] " REPLY
      [[ "${REPLY}" =~ ^[Yy]$ ]] && DO_UPDATE=1
    fi
    if (( DO_UPDATE == 1 )); then
      step "Updating upstream to ${REMOTE_REF}"
      git -C "${UPSTREAM_DIR}" checkout -B "${UPSTREAM_BRANCH}" "${REMOTE_REF}"
      echo "Upstream now at $(git -C "${UPSTREAM_DIR}" rev-parse --short HEAD)."
      REBUILD_BASE=1
    else
      echo "Keeping current checkout."
    fi
  fi
fi

# ---------- Base image (upstream) ----------

step "Checking base image: ${BASE_IMAGE}"
if (( BASE_PRESENT == 1 )) && (( REBUILD_BASE == 0 )); then
  echo "Base image already present, skipping upstream build."
else
  if (( REBUILD_BASE == 1 )); then
    echo "Rebuilding base image to pick up the updated upstream source."
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
  -c "import stable_baselines3 as sb3, mlflow, optuna, pandas, pyarrow; print('OK sb3', sb3.__version__, 'mlflow', mlflow.__version__, 'optuna', optuna.__version__, 'pandas', pandas.__version__, 'pyarrow', pyarrow.__version__)" 2>&1)" || {
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
