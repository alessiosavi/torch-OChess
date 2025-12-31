"""
Pytest configuration and fixtures.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def sample_fen():
    """Standard starting position FEN."""
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def sample_fens():
    """Sample FEN positions."""
    return [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "8/8/8/8/8/8/8/4K2k w - - 0 1",  # Endgame
    ]


@pytest.fixture
def sample_board_tensor():
    """Sample board tensor (starting position)."""
    # Starting position encoded
    board = torch.zeros(8, 8, dtype=torch.long)

    # White back rank
    board[7, :] = torch.tensor([4, 2, 3, 5, 6, 3, 2, 4])  # R N B Q K B N R
    board[6, :] = 1  # Pawns

    # Black back rank
    board[0, :] = torch.tensor([10, 8, 9, 11, 12, 9, 8, 10])  # r n b q k b n r
    board[1, :] = 7  # pawns

    return board
