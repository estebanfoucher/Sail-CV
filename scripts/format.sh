#!/bin/bash
# Format script for MVS App

set -e

echo "🎨 Formatting code..."

# Run ruff formatting
echo "Running ruff formatter..."
uv run ruff format src/

# Run ruff linting with fixes
echo "Running ruff linter with fixes..."
uv run ruff check src/ --fix

echo "✅ Formatting complete!"
