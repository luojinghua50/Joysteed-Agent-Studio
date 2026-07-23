.PHONY: help up down build test lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

up: ## Start all services (Docker Compose)
	docker compose up -d

down: ## Stop all services
	docker compose down

build: ## Build all Docker images
	docker compose build

test: test-core test-rag ## Run all tests

test-core: ## Run agent-core tests
	cd agent-core && source .venv/bin/activate && python -m pytest tests/ -v

test-rag: ## Run agent-rag tests
	cd agent-rag && source .venv/bin/activate && python -m pytest tests/ -v

test-web: ## Run agent-web tests
	cd agent-web && npm test

lint: ## Lint all Python projects
	cd agent-core && source .venv/bin/activate && ruff check src/ tests/
	cd agent-rag && source .venv/bin/activate && ruff check src/ tests/

install: install-core install-rag install-web ## Install all dependencies

install-core: ## Install agent-core dependencies
	cd agent-core && uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"

install-rag: ## Install agent-rag dependencies
	cd agent-rag && uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"

install-web: ## Install agent-web dependencies
	cd agent-web && npm install

clean: ## Clean all build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf agent-web/node_modules agent-web/dist
