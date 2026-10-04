# syntax=docker/dockerfile:1.7
#
# MAVER's runtime image.
#
#   uv-stage      the pinned uv binary
#   kubectl-stage downloads and checksums kubectl, so curl never enters the runtime
#   builder       resolves the locked environment into /opt/venv
#   runtime       the venv, the source, the configs, and nothing else
#
# There is no Node stage: the web UI is dropped. There is no entrypoint script: the
# image never runs as root, so there is nothing to drop privilege from.
#
# Build from the repository root, which is the context the .dockerignore is resolved
# against:  docker buildx build --platform linux/amd64 -f deploy/docker/Dockerfile .

ARG BASE_IMAGE=python:3.12-slim
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.17
# Held at the version every published measurement was taken under. What kubectl prints
# is part of the interface here — the direct transport returns its stdout verbatim and
# the prompts were written against that text — so the binary's version is pinned for the
# same reason the model's is, and moving it is a change to the measurement instrument.
ARG KUBECTL_VERSION=v1.30.1
# "" for the core image; "sregym" for the evaluation image.
ARG EXTRAS=""

FROM ${UV_IMAGE} AS uv-stage

# ---------------------------------------------------------------- kubectl
FROM ${BASE_IMAGE} AS kubectl-stage
ARG KUBECTL_VERSION
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    arch="$(dpkg --print-architecture)"; \
    base="https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}"; \
    curl -fsSL "${base}/kubectl" -o /tmp/kubectl; \
    curl -fsSL "${base}/kubectl.sha256" -o /tmp/kubectl.sha256; \
    echo "$(cat /tmp/kubectl.sha256)  /tmp/kubectl" | sha256sum -c -; \
    install -m 0755 /tmp/kubectl /out-kubectl

# ---------------------------------------------------------------- builder
FROM ${BASE_IMAGE} AS builder
COPY --from=uv-stage /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

# Dependencies first, without the project: a source-only edit reuses this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/

ARG EXTRAS
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable ${EXTRAS:+--extra "${EXTRAS}"}

# ---------------------------------------------------------------- runtime
FROM ${BASE_IMAGE} AS runtime

ARG VERSION=0.0.0+dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="MAVER" \
      org.opencontainers.image.description="Multi-Agent Verification and Experience-informed Recovery" \
      org.opencontainers.image.source="https://github.com/evenodderguy/maver" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

RUN set -eux; \
    groupadd --gid 1000 maver; \
    useradd --uid 1000 --gid 1000 --create-home --home-dir /home/maver --shell /usr/sbin/nologin maver; \
    install -d -o 1000 -g 1000 /workspace

COPY --from=kubectl-stage /out-kubectl /usr/local/bin/kubectl
COPY --from=builder --chown=1000:1000 /opt/venv /opt/venv
COPY --from=builder --chown=1000:1000 /src/configs /app/configs

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAVER_SERVER__HOST=0.0.0.0 \
    MAVER_SERVER__PORT=8080 \
    MAVER_SERVER__WORKDIR=/workspace \
    MAVER_SERVER__ENVIRONMENT=production \
    MAVER_MODEL__PROFILE_DIR=/app/configs/models

# The label's own value, where the running process can read it: /healthz reports it, so
# a deploy can assert that the container answering is the one it just built rather than
# a stale container nobody recreated. Threaded through a second time because a label is
# metadata about the image and not something the process inside it can see.
ARG REVISION
ENV MAVER_REVISION=${REVISION}

WORKDIR /app
USER 1000:1000
EXPOSE 8080

# /readyz, not /healthz: a container is "healthy" when it can take an incident, and
# /healthz is deliberately dependency-free so the evaluation harness's 30 s preflight
# cannot fail on a slow model endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD ["python", "-c", \
       "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz', timeout=4).status == 200 else 1)"]

# No ENTRYPOINT, so `docker run <image> <anything>` still runs that command. A manifest
# that wants to be explicit about this must set both `command` and `args`: Kubernetes
# maps `args` alone onto an empty entrypoint, and the container would try to exec
# "serve".
CMD ["maver", "serve"]
