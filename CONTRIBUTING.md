# Contributing to QuantVec-CLI

Thank you for your interest in contributing to QuantVec-CLI! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/quantvec-cli.git`
3. Install dependencies: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=quantvec --cov-report=html

# Format code
black quantvec/ tests/

# Type check
mypy quantvec/
```

## Pull Request Process

1. Ensure tests pass: `pytest`
2. Update documentation if needed
3. Add tests for new features
4. Follow the existing code style
5. Submit PR with clear description

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public APIs
- Keep functions focused and small

## Reporting Issues

Please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
