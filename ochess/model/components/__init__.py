"""
Neural network building blocks for chess models.

Components:
    - embeddings: Piece and positional embeddings
    - residual: Residual blocks for deep networks
    - attention: Self-attention mechanisms
    - heads: Output heads (move, score, capture, outcome)
"""

from ochess.model.components.embeddings import PieceEmbedding, PositionalEmbedding
from ochess.model.components.residual import ResidualBlock
from ochess.model.components.attention import SelfAttention, MultiHeadAttention
from ochess.model.components.heads import MoveHead, ScoreHead, CaptureHead, OutcomeHead

__all__ = [
    "PieceEmbedding",
    "PositionalEmbedding",
    "ResidualBlock",
    "SelfAttention",
    "MultiHeadAttention",
    "MoveHead",
    "ScoreHead",
    "CaptureHead",
    "OutcomeHead",
]
