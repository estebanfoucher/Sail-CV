#!/bin/bash
# Comprehensive check script for Sail-CV

set -e

echo "Running comprehensive checks..."

# Format code
echo "1. Formatting code..."
uv run ruff format src/ tests/

# Lint and fix
echo "2. Linting and fixing code..."
uv run ruff check src/ tests/ --fix

# Type checking
echo "3. Type checking..."
uv run mypy --explicit-package-bases src/

# Run tests
echo "4. Running tests..."
rm -rf output_tests
uv run pytest tests/ -v

# Run pre-commit hooks
echo "5. Running pre-commit hooks..."
uv run pre-commit run --all-files

echo "All checks passed!"
