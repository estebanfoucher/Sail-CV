#!/bin/bash
# Test script for Sail-CV

set -e

echo "Running tests..."

# Clean test output
rm -rf output_tests

echo "Running pytest..."
uv run pytest tests/ -v

echo "Tests complete!"
