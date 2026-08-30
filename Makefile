.PHONY: help install lint format typecheck test test-all check lock clean

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

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
