# Sail-CV Development Makefile

.PHONY: help setup install install-all format lint typecheck test check clean dev \
        test-reconstruction test-tracking \
        docker-build docker-up docker-down \
        pre-commit-install pre-commit-run quick-check update

help: ## Show this help message
	@echo "Sail-CV Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ── Setup ───────────────────────────────────────────────────────────

setup: ## Initial development environment setup
	@echo "Setting up development environment..."
	./scripts/dev-setup.sh

install: ## Install dependencies (main + dev)
	@echo "Installing dependencies..."
	uv sync --group dev

install-all: ## Install all dependencies (main + dev + reconstruction + tracking)
	@echo "Installing all dependencies..."
	uv sync --all-extras --group dev

install-reconstruction: ## Install reconstruction dependencies
	@echo "Installing reconstruction dependencies..."
	uv sync --extra reconstruction --group dev

install-tracking: ## Install tracking dependencies
	@echo "Installing tracking dependencies..."
	uv sync --extra tracking --group dev

# ── Code Quality ────────────────────────────────────────────────────

format: ## Format code with ruff
	@echo "Formatting code..."
	uv run ruff format src/ tests/

lint: ## Lint code with ruff
	@echo "Linting code..."
	uv run ruff check src/ tests/ --fix

typecheck: ## Run type checking with mypy
	@echo "Type checking..."
	uv run mypy --explicit-package-bases src/

# ── Testing ─────────────────────────────────────────────────────────

test: ## Run all tests
	@echo "Running all tests..."
	rm -rf output_tests
	uv run pytest tests/ -v

test-reconstruction: ## Run reconstruction tests only
	@echo "Running reconstruction tests..."
	uv run pytest tests/reconstruction/ -v

test-tracking: ## Run tracking tests only
	@echo "Running tracking tests..."
	uv run pytest tests/tracking/ -v

# ── Combined Checks ─────────────────────────────────────────────────

check: ## Run all checks (format + lint + typecheck + test)
	@echo "Running all checks..."
	./scripts/check-all.sh

quick-check: format lint typecheck ## Quick check (format + lint + typecheck, no tests)
	@echo "Quick check complete!"

# ── Docker ──────────────────────────────────────────────────────────

docker-build: ## Build Docker images
	@echo "Building Docker images..."
	cd docker && docker compose build

docker-build-reconstruction: ## Build reconstruction Docker image
	@echo "Building reconstruction Docker image..."
	cd docker && docker compose build reconstruction

docker-build-tracking: ## Build tracking Docker image
	@echo "Building tracking Docker image..."
	cd docker && docker compose build tracking

docker-up: ## Start Docker containers
	@echo "Starting Docker containers..."
	cd docker && docker compose up

docker-down: ## Stop Docker containers
	@echo "Stopping Docker containers..."
	cd docker && docker compose down

# ── Pre-commit ──────────────────────────────────────────────────────

pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "Running pre-commit hooks..."
	uv run pre-commit run --all-files

# ── Maintenance ─────────────────────────────────────────────────────

clean: ## Clean up temporary files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf output_tests

update: ## Update dependencies
	@echo "Updating dependencies..."
	uv lock --upgrade
	uv sync --group dev

dev: install setup ## Start development mode (install + setup)
