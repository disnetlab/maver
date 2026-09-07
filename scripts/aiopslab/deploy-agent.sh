#!/usr/bin/env bash
# Start the agent container an AIOpsLab host grades, holding no cluster credential.
#
# The agent gets no kubeconfig and no shell on this host. Its only cluster authority is
# the loopback bridge the driver serves for the lifetime of one problem, and the bearer
# token for it arrives as a mounted file rather than an environment variable, so it is
# not in `docker inspect` output or in this process's environment.
#
#   make image                                    # builds kira:local
#   KIRA_MODEL_ENDPOINT=http://vllm.internal:8000/v1 \
#     scripts/aiopslab/deploy-agent.sh
#
# The AIOpsLab adapter needs no packaging extra: its transport speaks HTTP to the bridge
# with the client library the core already depends on. `make image-eval` is for SREGym.
set -euo pipefail

IMAGE="${AIOPSLAB_AGENT_IMAGE:-kira:local}"
CONTAINER="${AIOPSLAB_AGENT_CONTAINER:-kira-aiopslab}"
DATA_DIR="${AIOPSLAB_AGENT_DATA_DIR:-/var/lib/aiopslab/agent-workspace}"
CONFIG_DIR="${AIOPSLAB_AGENT_CONFIG_DIR:-/var/lib/aiopslab/agent-config}"
TOKEN_FILE="${CONFIG_DIR}/bridge-token"
ENV_FILE="${CONFIG_DIR}/kira.env"
AGENT_URL="${AIOPSLAB_AGENT_URL:-http://127.0.0.1:8080}"
# One arm name and one model profile, rather than the prototype's thirty-three flat
# variables. Both files ship in the image, and the arm is what the run records.
ARM="${AIOPSLAB_ARM:-ma-v}"
MODEL_PROFILE="${AIOPSLAB_MODEL_PROFILE:-qwen3.6-35b-a3b}"
MODEL_ENDPOINT="${KIRA_MODEL_ENDPOINT:-}"
MODEL_API_KEY="${KIRA_MODEL_API_KEY:-EMPTY}"
BRIDGE_URL="${AIOPSLAB_BRIDGE_URL:-http://127.0.0.1:1819}"
BRIDGE_TIMEOUT_S="${AIOPSLAB_BRIDGE_TIMEOUT_S:-180}"
RUNNER_UID="${SUDO_UID:-$(id -u)}"
RUNNER_GID="${SUDO_GID:-$(id -g)}"

if [[ -z "${MODEL_ENDPOINT}" ]]; then
  printf 'set KIRA_MODEL_ENDPOINT to the private model endpoint, e.g. http://vllm.internal:8000/v1\n' >&2
  exit 2
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  printf 'no such image: %s. Build it with `make image`.\n' "${IMAGE}" >&2
  exit 2
fi

install -d -m 700 -o "${RUNNER_UID}" -g "${RUNNER_GID}" "${CONFIG_DIR}"
# The image runs as uid 1000 on a read-only root filesystem, so the workspace is the one
# writable path it has: sessions, the event log and the retrieval store all live there.
install -d -m 755 "${DATA_DIR}"
chown 1000:1000 "${DATA_DIR}"

if [[ ! -s "${TOKEN_FILE}" ]]; then
  openssl rand -hex 32 |
    install -m 600 -o "${RUNNER_UID}" -g "${RUNNER_GID}" /dev/stdin "${TOKEN_FILE}"
fi
chown "${RUNNER_UID}:${RUNNER_GID}" "${TOKEN_FILE}"
chmod 600 "${TOKEN_FILE}"

TEMP_ENV="$(mktemp "${CONFIG_DIR}/kira.env.XXXXXX")"
trap 'rm -f "${TEMP_ENV}"' EXIT
chmod 600 "${TEMP_ENV}"
{
  printf '%s\n' \
    'KIRA_SERVER__WORKDIR=/workspace' \
    "KIRA_ARM=${ARM}" \
    "KIRA_MODEL__PROFILE=${MODEL_PROFILE}" \
    "KIRA_MODEL__ENDPOINT_OVERRIDE=${MODEL_ENDPOINT}" \
    "KIRA_MODEL__API_KEY=${MODEL_API_KEY}" \
    'KIRA_INTAKE__ADAPTERS=aiopslab' \
    'KIRA_ADAPTERS__AIOPSLAB__ENABLED=true' \
    "KIRA_ADAPTERS__AIOPSLAB__BRIDGE_URL=${BRIDGE_URL}" \
    'KIRA_ADAPTERS__AIOPSLAB__BRIDGE_TOKEN_FILE=/run/secrets/aiopslab-bridge-token' \
    "KIRA_ADAPTERS__AIOPSLAB__BRIDGE_TIMEOUT_S=${BRIDGE_TIMEOUT_S}" \
    'KIRA_PIPELINE__AUTO_SNAPSHOT=true' \
    'KIRA_OBSERVABILITY__LOG_LEVEL=INFO' \
    'KIRA_AUTHORITY__MODE=auto' \
    'KIRA_SERVER__ENVIRONMENT=local' \
    'KIRA_SERVER__HOST=127.0.0.1'
} > "${TEMP_ENV}"
install -m 600 -o "${RUNNER_UID}" -g "${RUNNER_GID}" "${TEMP_ENV}" "${ENV_FILE}"
# The last three lines above belong together. Nobody is waiting to approve a write during
# a sweep, so the approval gate runs in automatic mode and every run's configuration
# snapshot says so; the settings layer refuses `auto` under `production`, which the image
# bakes in, so the host declares that it is not one. And with `--network host` there is
# no published-port boundary left: the API has no authentication, so it binds loopback,
# which is all either process needs.
#
# The snapshot is named although it is the default. The prototype's deploy script asked
# for it to be off under a name the prototype never read, so every archived AIOpsLab run
# had it on; Kira honours that old name, so an agent.env inherited from the prototype
# would now switch it off and the runs would stop being comparable. Naming the canonical
# setting here is what keeps an inherited alias from deciding.

if docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
  docker rm -f "${CONTAINER}" >/dev/null
fi

# --network host because the bridge, the agent and the driver are all on this host's
# loopback. No kubeconfig is mounted, and none is wanted: a mount here would defeat the
# containment the bridge exists to provide.
docker run -d \
  --name "${CONTAINER}" \
  --network host \
  --restart unless-stopped \
  --read-only \
  --tmpfs /tmp \
  --security-opt no-new-privileges \
  --env-file "${ENV_FILE}" \
  --volume "${DATA_DIR}:/workspace" \
  --volume "${TOKEN_FILE}:/run/secrets/aiopslab-bridge-token:ro" \
  "${IMAGE}" >/dev/null

alive=0
for _ in $(seq 1 60); do
  # Both, because they fail differently and the difference is the diagnosis. /healthz is
  # dependency-free, so it answering means the process started and read its
  # configuration; /readyz is the one that means an incident would be accepted.
  if curl -fsS "${AGENT_URL}/healthz" >/dev/null 2>&1; then
    alive=1
    if curl -fsS "${AGENT_URL}/readyz" >/dev/null 2>&1; then
      printf 'agent ready at %s (arm=%s profile=%s)\n' "${AGENT_URL}" "${ARM}" "${MODEL_PROFILE}"
      printf 'bridge token: %s (mode 0600)\n' "${TOKEN_FILE}"
      printf 'environment:  %s\n' "${ENV_FILE}"
      exit 0
    fi
  fi
  sleep 2
done

if [[ "${alive}" -eq 1 ]]; then
  printf 'agent is alive at %s but not ready; its own report:\n' "${AGENT_URL}" >&2
  curl -sS "${AGENT_URL}/readyz" >&2 || true
  printf '\n' >&2
else
  printf 'agent never answered %s/healthz; its last log lines:\n' "${AGENT_URL}" >&2
  docker logs --tail 100 "${CONTAINER}" >&2 || true
fi
exit 1
