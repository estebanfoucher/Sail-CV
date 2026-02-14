# MVS App Development Makefile

.PHONY: help setup format lint test check clean install dev

help: ## Show this help message
	@echo "MVS App Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'


format: ## Format code with ruff src/ tests/
	@echo "🎨 Formatting code..."
	uv run ruff format src/ tests/

lint: ## Lint code with ruff
	@echo "🔍 Linting code..."
	uv run ruff check src/ tests/ --fix

typecheck: ## Run type checking with mypy
	@echo "🔍 Type checking..."
	uv run mypy --explicit-package-bases src/


check-all: ## Run all checks (format + lint + typecheck)
	@echo "🔍 Running all checks..."
	./scripts/check-all.sh

test: ## Run tests
	@echo "🧪 Running tests..." # clear output_tests directory
	rm -rf output_tests
	uv run pytest tests/ -v

dev-setup: ## Start development mode (install + setup)
	@echo "🚀 Setting up development environment..."
	./scripts/dev-setup.sh
