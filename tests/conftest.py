"""Shared test configuration and fixtures for the SailCV test suite."""

from pathlib import Path

import pytest

# Project root: parent of the tests/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> Path:
    """Fixture providing the absolute path to the project root."""
    return PROJECT_ROOT
