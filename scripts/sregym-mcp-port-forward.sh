#!/usr/bin/env bash
# Keep the SREGym MCP server and its Kubernetes filtering proxy exposed on the VM.
#
# This is intentionally a restart loop. `kubectl port-forward` exits whenever the
# selected pod is restarted, the API connection drops, or the local socket is
# disturbed. In SREGym proxy_exec mode, a dead port-forward makes every agent
# kubectl command fail, so the run-level service must be self-healing.
set -euo pipefail

KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
SREGYM_NAMESPACE="${SREGYM_NAMESPACE:-sregym}"
SREGYM_MCP_RESOURCE="${SREGYM_MCP_RESOURCE:-deploy/mcp-server}"
# Loopback alone is not enough for a container-isolated agent. SREGym runs such agents
# (stratus, codex, claudecode) with `--network=host --add-host=host.docker.internal:
# host-gateway` and rewrites their MCP_SERVER_URL to `host.docker.internal`, which resolves
# to the docker bridge — so a forward bound only to 127.0.0.1 refuses their every tool call
# and the driver dies with `httpcore.ConnectError` after passing its own preflight. Binding
# the bridge as well costs nothing for host-side callers and keeps those agents runnable.
SREGYM_MCP_DOCKER_BRIDGE="${SREGYM_MCP_DOCKER_BRIDGE:-$(
  ip -4 addr show docker0 2>/dev/null | awk '/inet /{sub(/\/.*/,"",$2); print $2; exit}')}"
SREGYM_MCP_HOST="${SREGYM_MCP_HOST:-127.0.0.1${SREGYM_MCP_DOCKER_BRIDGE:+,${SREGYM_MCP_DOCKER_BRIDGE}}}"
SREGYM_MCP_PORT="${SREGYM_MCP_PORT:-9954}"
SREGYM_PROXY_HOST_PORT="${SREGYM_PROXY_HOST_PORT:-16444}"
SREGYM_PROXY_POD_PORT="${SREGYM_PROXY_POD_PORT:-16443}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-2}"
HEALTH_INTERVAL_SECONDS="${HEALTH_INTERVAL_SECONDS:-15}"
BIND_GRACE_SECONDS="${BIND_GRACE_SECONDS:-20}"

child_pid=""

# Every address:port this forward is supposed to be serving.
expected_pairs() {
  local hosts host port
  IFS=',' read -ra hosts <<< "${SREGYM_MCP_HOST}"
  for host in "${hosts[@]}"; do
    for port in "${SREGYM_MCP_PORT}" "${SREGYM_PROXY_HOST_PORT}"; do
      printf '%s:%s\n' "${host}" "${port}"
    done
  done
}

listeners_ok() {
  local bound want
  bound="$( (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | awk '{print $4}' )"
  while read -r want; do
    grep -qxF "${want}" <<< "${bound}" || return 1
  done < <(expected_pairs)
  return 0
}

stop_child() {
  if [ -n "${child_pid}" ] && kill -0 "${child_pid}" 2>/dev/null; then
    kill "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
}

trap 'stop_child; exit 0' INT TERM

while true; do
  printf '[%s] port-forward %s/%s: %s:%s %s:%s\n' \
    "$(date -Is)" "${SREGYM_NAMESPACE}" "${SREGYM_MCP_RESOURCE}" \
    "${SREGYM_MCP_PORT}" "${SREGYM_MCP_PORT}" \
    "${SREGYM_PROXY_HOST_PORT}" "${SREGYM_PROXY_POD_PORT}"

  set +e
  "${KUBECTL_BIN}" -n "${SREGYM_NAMESPACE}" port-forward \
    --address "${SREGYM_MCP_HOST}" \
    "${SREGYM_MCP_RESOURCE}" \
    "${SREGYM_MCP_PORT}:${SREGYM_MCP_PORT}" \
    "${SREGYM_PROXY_HOST_PORT}:${SREGYM_PROXY_POD_PORT}" &
  child_pid="$!"

  # Restarting only when the child EXITS is not enough. `kubectl port-forward` binds
  # its ports independently: if one bind fails -- typically "Unable to listen on port
  # 9954: bind: address already in use" while a previous forward still holds the socket
  # -- it logs the error, keeps serving the ports that did bind, and never exits. The
  # bridge then looks alive (the process is up, the other port answers) while one port
  # is simply gone. That is silent and expensive: OpenCode starts its session with the
  # MCP servers unconnected, the agent sees only its built-in tools, has no cluster
  # access, and the cell still exits 0 and writes a grade row -- an instrument failure
  # recorded as an agent failure. Supervise the LISTENERS, not the process.
  for _ in $(seq "${BIND_GRACE_SECONDS}"); do
    kill -0 "${child_pid}" 2>/dev/null || break
    listeners_ok && break
    sleep 1
  done
  if kill -0 "${child_pid}" 2>/dev/null && ! listeners_ok; then
    printf '[%s] port-forward bound only a subset of [%s]; restarting\n' \
      "$(date -Is)" "$(expected_pairs | tr '\n' ' ')" >&2
    kill "${child_pid}" 2>/dev/null || true
  else
    while kill -0 "${child_pid}" 2>/dev/null; do
      sleep "${HEALTH_INTERVAL_SECONDS}"
      kill -0 "${child_pid}" 2>/dev/null || break
      if ! listeners_ok; then
        printf '[%s] port-forward lost a listener from [%s]; restarting\n' \
          "$(date -Is)" "$(expected_pairs | tr '\n' ' ')" >&2
        kill "${child_pid}" 2>/dev/null || true
        break
      fi
    done
  fi

  wait "${child_pid}"
  code="$?"
  child_pid=""
  set -e

  printf '[%s] port-forward exited code=%s; restarting in %ss\n' \
    "$(date -Is)" "${code}" "${RESTART_DELAY_SECONDS}" >&2
  sleep "${RESTART_DELAY_SECONDS}"
done
