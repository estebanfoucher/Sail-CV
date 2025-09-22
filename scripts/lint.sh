#!/bin/bash
# Lint script for MVS App

set -e

echo "🔍 Running linters..."

# Run ruff linting
echo "Running ruff linter..."
uv run ruff check src/ --fix

# Run type checking
echo "Running mypy type checker..."
uv run mypy --explicit-package-bases src/

echo "✅ Linting complete!"
