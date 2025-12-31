"""Tests for FEN parser."""

import pytest
import torch
from ochess.data.fen_parser import FenParser


class TestFenParser:
    """Tests for FEN parsing functionality."""

    @pytest.fixture
    def parser(self):
        return FenParser()

    def test_starting_position(self, parser, sample_fen):
        """Test parsing starting position."""
        tensor = parser.to_tensor(sample_fen)

        assert tensor.shape == (8, 8)
        assert tensor.dtype == torch.long

        # Check white pieces
        assert tensor[7, 4] == 6  # White king at e1
        assert tensor[7, 3] == 5  # White queen at d1
        assert tensor[6, 0] == 1  # White pawn at a2

        # Check black pieces
        assert tensor[0, 4] == 12  # Black king at e8
        assert tensor[0, 3] == 11  # Black queen at d8
        assert tensor[1, 0] == 7   # Black pawn at a7

    def test_empty_board(self, parser):
        """Test parsing empty board."""
        fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        tensor = parser.to_tensor(fen)

        assert (tensor == 0).all()

    def test_roundtrip(self, parser, sample_fen):
        """Test FEN -> tensor -> FEN roundtrip."""
        tensor = parser.to_tensor(sample_fen)
        recovered = parser.to_fen(tensor)

        # Compare board parts
        original_board = sample_fen.split()[0]
        assert recovered == original_board

    def test_get_active_color(self, parser):
        """Test extracting active color."""
        white_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        black_fen = "8/8/8/8/8/8/8/8 b - - 0 1"

        assert parser.get_active_color(white_fen) == 0
        assert parser.get_active_color(black_fen) == 1

    def test_is_white_to_move(self, parser):
        """Test white to move check."""
        white_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        black_fen = "8/8/8/8/8/8/8/8 b - - 0 1"

        assert parser.is_white_to_move(white_fen) is True
        assert parser.is_white_to_move(black_fen) is False

    def test_flip_board(self, parser, sample_fen):
        """Test board flipping."""
        tensor = parser.to_tensor(sample_fen)
        flipped = parser.flip_board(tensor)

        # After flip, black pieces should be at white's side
        assert tensor[0, 0] == flipped[7, 7]
        assert tensor[7, 7] == flipped[0, 0]

    def test_swap_colors(self, parser, sample_fen):
        """Test color swapping."""
        tensor = parser.to_tensor(sample_fen)
        swapped = parser.swap_colors(tensor)

        # White king (6) becomes black king (12)
        white_king_pos = (tensor == 6).nonzero()[0]
        assert swapped[white_king_pos[0], white_king_pos[1]] == 12

        # Black king (12) becomes white king (6)
        black_king_pos = (tensor == 12).nonzero()[0]
        assert swapped[black_king_pos[0], black_king_pos[1]] == 6

    def test_all_piece_types(self, parser):
        """Test all piece types are parsed correctly."""
        # FEN with all pieces
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        tensor = parser.to_tensor(fen)

        # Count each piece type
        for piece, idx in parser.PIECE_TO_INDEX.items():
            if piece != " ":
                count = (tensor == idx).sum().item()
                if piece.upper() in "RNBQK":
                    assert count >= 1, f"Piece {piece} not found"

    def test_normalize_for_player(self, parser, sample_fen):
        """Test position normalization."""
        tensor = parser.to_tensor(sample_fen)

        # White to move - should not change
        white_normalized = parser.normalize_for_player(tensor, white_to_move=True)
        assert torch.equal(tensor, white_normalized)

        # Black to move - should flip and swap
        black_normalized = parser.normalize_for_player(tensor, white_to_move=False)
        assert not torch.equal(tensor, black_normalized)

    def test_display(self, parser, sample_fen):
        """Test board display."""
        tensor = parser.to_tensor(sample_fen)
        display = parser.display(tensor)

        assert isinstance(display, str)
        assert "a b c d e f g h" in display
        assert "K" in display  # White king
        assert "k" in display  # Black king
