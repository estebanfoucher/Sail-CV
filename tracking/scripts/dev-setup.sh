#!/bin/bash
# Development setup script for MVS App

set -e

echo "🚀 Setting up MVS App development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies with uv..."
uv sync --group dev

# Install main dependencies for development
echo "📦 Installing main dependencies for development..."
uv sync

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
if uv run pre-commit install; then
    echo "✅ Pre-commit hooks installed successfully"
else
    echo "⚠️  Pre-commit installation failed, but continuing..."
    echo "   You can install them manually later with: uv run pre-commit install"
fi

echo "✅ Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  uv run ruff check          # Lint code"
echo "  uv run ruff check --fix    # Lint and fix code"
echo "  uv run ruff format         # Format code"
echo "  uv run mypy .              # Type checking"
echo "  uv run pytest              # Run tests"
echo "  uv run pre-commit run --all-files  # Run all pre-commit hooks"
