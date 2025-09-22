# MVS App Development Makefile

.PHONY: help setup format lint test check clean install dev

help: ## Show this help message
	@echo "MVS App Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Initial development environment setup
	@echo "🚀 Setting up development environment..."
	./scripts/dev-setup.sh

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	uv sync --group dev

install-all: ## Install all dependencies (main + dev)
	@echo "📦 Installing all dependencies..."
	uv sync

format: ## Format code with ruff
	@echo "🎨 Formatting code..."
	uv run ruff format src/

lint: ## Lint code with ruff
	@echo "🔍 Linting code..."
	uv run ruff check src/ --fix

typecheck: ## Run type checking with mypy
	@echo "🔍 Type checking..."
	uv run mypy --explicit-package-bases src/

test: ## Run tests
	@echo "🧪 Running tests..."
	uv run pytest src/ -v

check: ## Run all checks (format + lint + typecheck + test)
	@echo "🔍 Running all checks..."
	./scripts/check-all.sh

clean: ## Clean up temporary files
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name ".mypy_cache" -delete

dev: ## Start development mode (install + setup)
	@echo "🚀 Starting development mode..."
	$(MAKE) install
	$(MAKE) setup

# Docker commands
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	cd docker && docker-compose build

docker-up: ## Start Docker containers
	@echo "🐳 Starting Docker containers..."
	cd docker && docker-compose up

docker-down: ## Stop Docker containers
	@echo "🐳 Stopping Docker containers..."
	cd docker && docker-compose down

# Pre-commit commands
pre-commit-install: ## Install pre-commit hooks
	@echo "🔧 Installing pre-commit hooks..."
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "🔧 Running pre-commit hooks..."
	uv run pre-commit run
# Quick development workflow
quick-check: format lint typecheck ## Quick check (format + lint + typecheck, no tests)
	@echo "✅ Quick check complete!"

# Update dependencies
update: ## Update dependencies
	@echo "📦 Updating dependencies..."
	uv lock --upgrade
	uv sync --group dev
