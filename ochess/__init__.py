"""
Torch o'Chess - PyTorch-based Chess Neural Network

A modular chess engine for experimenting with neural networks
in chess move prediction and evaluation.

Modules:
    data: Data loading and preprocessing (FEN parsing, move encoding)
    datagen: Training data generation using Stockfish
    model: Neural network architectures (ResNet, Transformer, Hybrid)
    training: Training loop and utilities
    evaluation: Model evaluation against Stockfish
    play: Human play interface
"""

__version__ = "1.0.0"
__author__ = "Torch o'Chess Team"

from ochess.config import Config

__all__ = ["Config", "__version__"]
