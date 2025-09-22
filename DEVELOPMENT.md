# MVS App Development Guide

This guide explains how to develop the MVS (Multi-View Stereo) application using modern Python tooling with `uv` and `ruff`.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Git

## Quick Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Run the development setup script**:
   ```bash
   ./scripts/dev-setup.sh
   ```

This will:
- Install all dependencies (main + dev) using `uv`
- Set up pre-commit hooks
- Configure the development environment

**Note**: The setup installs all main dependencies (OpenCV, PyTorch, etc.) for local development. This allows you to run tests and develop locally without Docker.

## Development Workflow

### Daily Development Commands

```bash
# Format code (equivalent to Black)
uv run ruff format src/

# Lint code with auto-fixes
uv run ruff check src/ --fix

# Type checking
uv run mypy --explicit-package-bases src/

# Run tests
uv run pytest src/ -v

# Run all checks (format + lint + type check + test)
./scripts/check-all.sh
```

### Using the Development Scripts

We provide several convenience scripts in the `scripts/` directory:

- `./scripts/dev-setup.sh` - Initial development environment setup
- `./scripts/format.sh` - Format code using ruff
- `./scripts/lint.sh` - Lint code with fixes
- `./scripts/test.sh` - Run tests
- `./scripts/check-all.sh` - Run all checks

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit:

```bash
# Install pre-commit hooks (done by dev-setup.sh)
uv run pre-commit install

# Run all pre-commit hooks manually
uv run pre-commit run --all-files

# Run hooks on specific files
uv run pre-commit run --files src/process_pairs.py

# Run specific hook
uv run pre-commit run ruff
```

**Pre-commit hooks include:**
- **Trailing whitespace removal**
- **End-of-file fixing**
- **YAML validation**
- **Large file detection**
- **Merge conflict detection**
- **Debug statement detection**
- **Docstring placement**
- **Ruff linting and formatting** (on `src/` only)
- **MyPy type checking** (on `src/` only)

## Code Quality Tools

### Ruff (Linting & Formatting)

Ruff replaces both Black and Flake8, providing:
- **10-100x faster** than traditional tools
- **Unified configuration** in `pyproject.toml`
- **Auto-fixing** for many issues

**Configuration**: See `[tool.ruff]` section in `pyproject.toml`

**Key features**:
- Line length: 88 characters (Black-compatible)
- Import sorting (isort replacement)
- Comprehensive linting rules
- Auto-fixing for most issues

### MyPy (Type Checking)

Static type checking with configuration in `pyproject.toml`. The configuration is set to be lenient initially:

```bash
# Type check all source code
uv run mypy --explicit-package-bases src/

# Type check specific file
uv run mypy --explicit-package-bases src/process_pairs.py
```

**Note**: The codebase currently has some type errors that are expected. You can gradually improve type safety by:
1. Adding type hints to new functions
2. Fixing existing type errors one file at a time
3. Eventually enabling `disallow_untyped_defs = true` in `pyproject.toml`

### Pytest (Testing)

Run tests with pytest:

```bash
# Run all tests
uv run pytest src/

# Run with verbose output
uv run pytest src/ -v

# Run specific test file
uv run pytest src/cameras/test_cameras.py
```

## Project Structure

```
MVS_app/
├── src/                    # Main source code
│   ├── cameras/           # Camera handling
│   ├── mv_utils/          # Multi-view utilities
│   ├── stereo/            # Stereo processing
│   └── unitaries/         # Utility modules
├── scripts/               # Development scripts
├── docker/                # Docker configuration
├── mast3r/                # External MASt3R dependency
├── checkpoints/           # Model checkpoints
├── data/                  # Input data
├── output/                # Generated output
└── pyproject.toml         # Project configuration
```

## Development Best Practices

### Code Style

1. **Use ruff for formatting** - it's faster than Black and fully compatible
2. **Follow the configured rules** in `pyproject.toml`
3. **Use type hints** - MyPy is configured to enforce them
4. **Keep functions small** - complexity limit is set to 10

### Logging

- Use `loguru` for logging (not standard logging)
- Log at appropriate levels (debug, info, warning, error)
- Avoid logging sensitive data

### Error Handling

- **Limit try/except blocks** - let errors raise when appropriate
- Use specific exception types
- Add meaningful error messages

### Dependencies

- Use `uv` instead of `pip` for all package management
- Pin versions in `pyproject.toml`
- Use dependency groups for optional features

## Development Environment

### Local Development (Recommended)

The development setup creates a complete local environment with all dependencies:

1. **Full local development** with `uv` and `ruff`
2. **Run tests locally** with all dependencies installed
3. **Use Docker for deployment** and Jetson-specific testing

### Docker Development

For deployment and Jetson-specific testing:

1. **Test in Docker** before deployment
2. **Use Docker for final testing** on Jetson hardware
3. **Deploy using Docker** for production

### Testing Docker Build

```bash
# Build the Docker image
cd docker
docker-compose build

# Test the container
docker-compose up
```

## IDE Integration

### VS Code

Install these extensions for the best experience:
- Python
- Ruff (official extension)
- MyPy Type Checker

### PyCharm

Configure external tools:
- Ruff formatter: `uv run ruff format`
- Ruff linter: `uv run ruff check`

## Troubleshooting

### Common Issues

1. **Import errors**: Check that `PYTHONPATH` includes the `src/` directory
2. **Type checking errors**: Some external libraries are ignored in MyPy config
3. **Ruff errors**: Most can be auto-fixed with `--fix` flag
4. **Pre-commit installation fails**: The configuration is already simplified to avoid dependency issues

### Pre-commit Issues

If you encounter issues with pre-commit installation:

1. **Skip pre-commit for now**:
   ```bash
   # Just run the tools manually
   make format
   make lint
   make typecheck
   ```

3. **Check pre-commit logs**:
   ```bash
   cat ~/.cache/pre-commit/pre-commit.log
   ```

### Performance

- Ruff is extremely fast, but for very large codebases, you can exclude directories in `pyproject.toml`
- MyPy can be slow on first run due to type checking external libraries

## Contributing

1. **Run all checks** before committing: `./scripts/check-all.sh`
2. **Follow the code style** enforced by ruff
3. **Add type hints** to new functions
4. **Write tests** for new functionality
5. **Update documentation** as needed

## Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
