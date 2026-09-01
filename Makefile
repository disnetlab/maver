.PHONY: help install lint format typecheck test test-all check lock image image-eval clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Sync the virtualenv from uv.lock, including dev dependencies
	uv sync --all-extras

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Apply formatting and safe lint fixes
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Static type check
	uv run mypy

test:  ## Run the unit and contract suites
	uv run pytest -m "not integration"

test-all:  ## Run everything, including tests needing a cluster or database
	uv run pytest

check: lint typecheck test  ## What CI runs

lock:  ## Refresh uv.lock and the exported requirements.txt
	uv lock
	uv export --format requirements-txt --frozen --no-dev -o requirements.txt

# The base image is digest-pinned in a file rather than in the Dockerfile, so bumping
# it is a reviewable one-line commit. --platform is explicit because a plain build on an
# Apple Silicon machine produces an arm64 image whose failure on a deployment host is an
# exec-format error at pod start rather than an error at build time.
# Deferred with `=` rather than `:=`, so that `make check` does not pay for a version
# lookup and a git call it has no use for.
BASE_IMAGE = $(shell cat deploy/docker/base-image.txt)
VERSION    = $(shell uv run kira --version)
REVISION   = $(shell git rev-parse HEAD)

image:  ## Build the runtime image
	docker buildx build --platform linux/amd64 \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  --build-arg VERSION=$(VERSION) \
	  --build-arg REVISION=$(REVISION) \
	  -f deploy/docker/Dockerfile \
	  -t ghcr.io/evenodderguy/kira:$(VERSION) \
	  -t kira:local .

# The same Dockerfile with the sregym extra. k8s-agent:local is an interoperability
# alias for the evaluation harness, which tags that name itself, and nothing else.
image-eval:  ## Build the evaluation image
	docker buildx build --platform linux/amd64 \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  --build-arg VERSION=$(VERSION) \
	  --build-arg REVISION=$(REVISION) \
	  --build-arg EXTRAS=sregym \
	  -f deploy/docker/Dockerfile \
	  -t ghcr.io/evenodderguy/kira-sregym:$(VERSION) \
	  -t k8s-agent:local .

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
