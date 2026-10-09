.PHONY: help install lint format typecheck test test-parity test-postgres test-all check \
        lock requirements image image-eval compose-up compose-eval-up clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

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

# The prototype inputs these checks read are frozen under tests/fixtures/prototype.
# MAVER_PROTOTYPE_ROOT can select a live checkout for a cross-check. Deliberately
# diverged inputs report their skip reasons through the configured pytest -ra option.
test-parity:  ## Run only the checks that guard the published numbers
	uv run pytest -m parity

# The backend every campaign ran on. It skips without a server, and a skipped check
# reports green, so the DSN is required rather than defaulted: refusing here is the
# whole point of the target. --no-skips catches the rest, a missing driver above all,
# which would otherwise skip these at import.
# Named rather than globbed, so adding a file that needs a database is a deliberate
# act. The cost is that forgetting one leaves it out of CI silently, so a new
# database-backed test belongs on this line in the same change that adds it.
POSTGRES_TESTS = tests/unit/test_postgres_backend.py \
                 tests/integration/test_incident_on_postgres.py \
                 tests/integration/test_approval_on_postgres.py

test-postgres:  ## Run the suites needing a live PostgreSQL (set POSTGRES_TEST_DSN)
	@test -n "$(POSTGRES_TEST_DSN)" || { \
	  echo "POSTGRES_TEST_DSN is unset. These tests would skip, and a skipped check reports green."; \
	  echo "  docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=maver -e POSTGRES_USER=maver \\"; \
	  echo "      -e POSTGRES_DB=maver postgres:17-alpine"; \
	  echo "  make test-postgres POSTGRES_TEST_DSN=postgresql://maver:maver@127.0.0.1:55432/maver"; \
	  exit 1; }
	POSTGRES_TEST_DSN=$(POSTGRES_TEST_DSN) \
	  uv run pytest -m integration --no-skips $(POSTGRES_TESTS) $(PYTEST_FLAGS)

test-all:  ## Run everything, including tests needing a cluster or database
	uv run pytest

check: lint typecheck test  ## What CI runs

requirements:  ## Re-export requirements.txt from the lock, for reviewers without uv
	uv export --format requirements-txt --frozen --no-dev -o requirements.txt

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
VERSION    = $(shell uv run maver --version)
REVISION   = $(shell git rev-parse HEAD)

image:  ## Build the runtime image
	docker buildx build --platform linux/amd64 \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  --build-arg VERSION=$(VERSION) \
	  --build-arg REVISION=$(REVISION) \
	  -f deploy/docker/Dockerfile \
	  -t ghcr.io/disnetlab/maver:$(VERSION) \
	  -t maver:local .

# The same Dockerfile with the sregym extra. k8s-agent:local is an interoperability
# alias for the evaluation harness, which tags that name itself, and nothing else.
image-eval:  ## Build the evaluation image
	docker buildx build --platform linux/amd64 \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  --build-arg VERSION=$(VERSION) \
	  --build-arg REVISION=$(REVISION) \
	  --build-arg EXTRAS=sregym \
	  -f deploy/docker/Dockerfile \
	  -t ghcr.io/disnetlab/maver-sregym:$(VERSION) \
	  -t k8s-agent:local .

# Compose cannot run git, so the revision the image records has to be handed to it.
# Without this the container answers /healthz with "unknown" and a deploy loses the one
# thing that distinguishes it from a stale container nobody recreated.
compose-up:  ## Build and start the service container, revision stamped
	REVISION=$(REVISION) VERSION=$(VERSION) \
	  docker compose -f deploy/docker/docker-compose.yml up -d --build

# --force-recreate because compose never recreates a container it did not just build,
# and the harness POSTs to this one by name.
compose-eval-up:  ## Build and start the evaluation container, revision stamped
	REVISION=$(REVISION) VERSION=$(VERSION) \
	  docker compose -f deploy/docker/docker-compose.eval.yml up -d --build --force-recreate

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
