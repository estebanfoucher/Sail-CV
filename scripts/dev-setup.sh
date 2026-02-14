#!/bin/bash
# Development setup script for Sail-CV

set -e

echo "Setting up Sail-CV development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies
echo "Installing dependencies with uv..."
uv sync --group dev

# Install main dependencies for development
echo "Installing main dependencies for development..."
uv sync

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
if uv run pre-commit install; then
    echo "Pre-commit hooks installed successfully"
else
    echo "Pre-commit installation failed, but continuing..."
    echo "   You can install them manually later with: uv run pre-commit install"
fi

echo "Development environment setup complete!"
echo ""
echo "Available commands:"
echo "  make help                  # Show all available commands"
echo "  make format                # Format code"
echo "  make lint                  # Lint code"
echo "  make test                  # Run all tests"
echo "  make test-reconstruction   # Run reconstruction tests"
echo "  make test-tracking         # Run tracking tests"
echo "  make check                 # Run all checks"
