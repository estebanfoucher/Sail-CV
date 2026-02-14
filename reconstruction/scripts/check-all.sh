#!/bin/bash
# Comprehensive check script for MVS App

set -e

echo "🔍 Running comprehensive checks..."

# Format code
echo "1. Formatting code..."
uv run ruff format src/

# Lint and fix
echo "2. Linting and fixing code..."
uv run ruff check src/ --fix

# Type checking
echo "3. Type checking..."
uv run mypy --explicit-package-bases src/

# Run tests
echo "4. Running tests..."
uv run pytest src/ -v

# Run pre-commit hooks
echo "5. Running pre-commit hooks..."
uv run pre-commit run --all-files

echo "✅ All checks passed!"
