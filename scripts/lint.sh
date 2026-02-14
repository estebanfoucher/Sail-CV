#!/bin/bash
# Lint script for Sail-CV

set -e

echo "Running linters..."

echo "Running ruff linter..."
uv run ruff check src/ tests/ --fix

echo "Running mypy type checker..."
uv run mypy --explicit-package-bases src/

echo "Linting complete!"
