#!/bin/bash
# Format script for Sail-CV

set -e

echo "Formatting code..."

echo "Running ruff formatter..."
uv run ruff format src/ tests/

echo "Running ruff linter with fixes..."
uv run ruff check src/ tests/ --fix

echo "Formatting complete!"
