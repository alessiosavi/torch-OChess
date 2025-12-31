"""
Data handling module for Torch o'Chess.

Provides:
    - FEN string parsing and tensor conversion
    - UCI move encoding/decoding
    - Score encoding with proper perspective handling
    - PyTorch Dataset with efficient caching
"""

from ochess.data.dataset import ChessDataset, create_dataloader
from ochess.data.fen_parser import FenParser
from ochess.data.move_encoder import MoveEncoder
from ochess.data.score_encoder import ScoreEncoder

__all__ = [
    "FenParser",
    "MoveEncoder",
    "ScoreEncoder",
    "ChessDataset",
    "create_dataloader",
]
