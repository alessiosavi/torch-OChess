# Contributing Guide

Thank you for your interest in contributing to Torch o'Chess!

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/torch-ochess.git
cd torch-ochess
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Or install all dependencies
pip install -r requirements-dev.txt
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/data/test_score_encoder.py -v

# Run with coverage
pytest tests/ --cov=ochess --cov-report=html
```

## Development Guidelines

### Code Style

We use:

- **Black** for formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black ochess/ tests/
isort ochess/ tests/

# Lint
flake8 ochess/ tests/
mypy ochess/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Docstrings

Use Google-style docstrings:

```python
def encode(self, score_str: str, white_to_move: bool) -> float:
    """Encode score from side-to-move perspective.

    Args:
        score_str: Score string like "+150" or "#5".
        white_to_move: Whether White is to move.

    Returns:
        Normalized score value.

    Raises:
        ValueError: If score_str is invalid.

    Example:
        >>> encoder = ScoreEncoder()
        >>> encoder.encode("+150", white_to_move=True)
        0.15
    """
```

### Type Hints

Use type hints for all public functions:

```python
from typing import Dict, List, Optional, Tuple

def get_top_moves(
    self,
    board: chess.Board,
    n: int = 5
) -> List[Tuple[chess.Move, float]]:
    ...
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clear, documented code
- Add tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run tests
pytest tests/ -v

# Check formatting
black --check ochess/ tests/
flake8 ochess/ tests/

# Type checking
mypy ochess/
```

### 4. Commit

Write clear commit messages:

```bash
git add .
git commit -m "Add feature: description of feature"
```

Commit message format:

- `Add feature: ...` - New feature
- `Fix: ...` - Bug fix
- `Update: ...` - Update existing feature
- `Docs: ...` - Documentation only
- `Test: ...` - Tests only
- `Refactor: ...` - Code refactoring

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### PR Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Code formatted (`black`, `isort`)
- [ ] Linting passes (`flake8`)
- [ ] Type hints added
- [ ] Docstrings written
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG updated (for significant changes)

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Testing
How did you test these changes?

## Related Issues
Fixes #123
```

## Areas to Contribute

### High Priority

1. **More Model Architectures**
   - EfficientNet-style models
   - Mixture of experts

2. **Improved Data Generation**
   - Better position diversity
   - Curriculum learning support

3. **Performance Optimization**
   - Faster inference
   - Memory optimization

### Medium Priority

1. **Documentation**
   - More examples
   - Tutorial notebooks

2. **Testing**
   - Integration tests
   - Performance benchmarks

3. **Features**
   - Opening book integration
   - Endgame tablebase support

### Good First Issues

Look for issues labeled `good-first-issue`:

- Fix typos in documentation
- Add more test cases
- Improve error messages

## Testing Guidelines

### Writing Tests

```python
import pytest
import torch
from ochess.model import ChessResNet

class TestChessResNet:
    """Tests for ChessResNet."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return ChessResNet()

    def test_forward_shape(self, model):
        """Test output shapes are correct."""
        boards = torch.randint(0, 13, (4, 5, 8, 8))
        colors = torch.randint(0, 2, (4, 5))

        outputs = model(boards, colors)

        assert outputs['move_logits'].shape == (4, 4096)
        assert outputs['score'].shape == (4, 1)
```

### Test Organization

```
tests/
├── conftest.py          # Shared fixtures
├── data/
│   ├── test_fen_parser.py
│   ├── test_move_encoder.py
│   └── test_score_encoder.py
├── model/
│   └── test_chess_net.py
├── datagen/
│   └── test_pipeline.py
└── integration/
    └── test_full_pipeline.py
```

### Running Specific Tests

```bash
# Single test
pytest tests/data/test_score_encoder.py::TestScoreEncoder::test_encode -v

# Tests matching pattern
pytest -k "encoder" -v

# Skip slow tests
pytest -m "not slow"
```

## Documentation

### Building Docs

```bash
cd docs
# If using Sphinx
make html
```

### Documentation Structure

- `README.md` - Main documentation index
- `api/` - API reference
- `guides/` - How-to guides
- `models/` - Model documentation

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release PR
4. After merge, tag release:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn

## Questions?

- Open a GitHub issue
- Join discussions
- Check existing issues/PRs

Thank you for contributing!
