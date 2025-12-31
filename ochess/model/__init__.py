"""
Neural network models for Torch o'Chess.

Provides three architectures:
    - ChessResNet: ResNet-style with residual blocks (AlphaZero-inspired)
    - ChessTransformer: Transformer-based with self-attention
    - ChessHybrid: Combination of ResNet backbone + attention

All models share the same interface:
    - Input: board tensor [B, T, 8, 8] + colors [B, T]
    - Output: move logits, score, capture, outcome predictions
"""

from ochess.model.chess_hybrid import ChessHybrid
from ochess.model.chess_resnet import ChessResNet
from ochess.model.chess_transformer import ChessTransformer
from ochess.model.losses import ChessLoss

__all__ = [
    "ChessResNet",
    "ChessTransformer",
    "ChessHybrid",
    "ChessLoss",
]
