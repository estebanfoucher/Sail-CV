#!/bin/bash
# Test script for MVS App

set -e

echo "🧪 Running tests..."

# Run pytest
echo "Running pytest..."
uv run pytest tests/ -v

echo "✅ Tests complete!"
