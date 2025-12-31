# Torch o'Chess Documentation

Welcome to the comprehensive documentation for Torch o'Chess, a PyTorch-based chess engine for neural network experimentation.

## Documentation Index

### Getting Started

- [Quick Start Guide](docs/guides/quickstart.md) - Get up and running in 5 minutes
- [Installation](docs/guides/installation.md) - Detailed installation instructions
- [Tutorial](docs/guides/tutorial.md) - Step-by-step walkthrough

### Architecture

- [System Architecture](docs/architecture.md) - High-level system design
- [The Critical Fix](docs/critical-fix.md) - Understanding the score encoding fix

### Neural Network Models

- [Models Overview](docs/models/overview.md) - Comparison of all three architectures
- [ChessResNet](docs/models/resnet.md) - AlphaZero-inspired residual network
- [ChessTransformer](docs/models/transformer.md) - Self-attention architecture
- [ChessHybrid](docs/models/hybrid.md) - CNN + Attention hybrid

### API Reference

- [Data Module](docs/api/data.md) - FEN parsing, move/score encoding, datasets
- [Data Generation](docs/api/datagen.md) - Stockfish wrapper, position generation
- [Models](docs/api/models.md) - Neural network classes and configurations
- [Training](docs/api/training.md) - Trainer, callbacks, metrics
- [Evaluation](docs/api/evaluation.md) - Stockfish evaluation
- [Play](docs/api/play.md) - Engine and CLI interface

### Guides

- [Data Generation Guide](docs/guides/data-generation.md) - Creating training datasets
- [Training Guide](docs/guides/training.md) - Training models effectively
- [Evaluation Guide](docs/guides/evaluation.md) - Testing against Stockfish
- [Playing Guide](docs/guides/playing.md) - Human vs model games

### Reference

- [Configuration Reference](docs/configuration.md) - All configuration options
- [CLI Reference](docs/cli-reference.md) - Command-line interface
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions
- [Contributing](docs/contributing.md) - How to contribute

---

## Quick Links

### Generate Data

```bash
python scripts/generate_data.py --num-positions 10000
```

### Train Model

```bash
python scripts/train.py --data datasets/train_data.parquet --model resnet
```

### Play Against Model

```bash
python scripts/play.py --model checkpoints/best_model.pt
```

---

## Project Overview

Torch o'Chess is a complete chess neural network system featuring:

| Feature | Description |
|---------|-------------|
| **Data Generation** | Stockfish-powered position labeling with good/bad moves |
| **Three Architectures** | ResNet, Transformer, and Hybrid models |
| **Proper Scoring** | Side-to-move perspective (the critical fix) |
| **Multi-task Learning** | Predicts moves, scores, captures, and outcomes |
| **Human Play** | Terminal interface for playing against the model |
| **Stockfish Evaluation** | Automated testing at various skill levels |

## Version

Current version: 0.1.0

## License

MIT License - See LICENSE file for details.
