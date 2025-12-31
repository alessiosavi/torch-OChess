"""Tests for move encoder."""

import pytest
import torch
import chess
from ochess.data.move_encoder import MoveEncoder


class TestMoveEncoder:
    """Tests for move encoding."""

    @pytest.fixture
    def encoder(self):
        return MoveEncoder()

    def test_basic_encoding(self, encoder):
        """Test basic move encoding."""
        # e2e4
        idx = encoder.encode("e2e4")
        assert 0 <= idx < 4096

        # Decode back
        decoded = encoder.decode(idx)
        assert decoded == "e2e4"

    def test_encode_decode_roundtrip(self, encoder):
        """Test encode/decode roundtrip for various moves."""
        moves = ["e2e4", "d2d4", "g1f3", "a1a8", "h1h8"]

        for uci in moves:
            idx = encoder.encode(uci)
            decoded = encoder.decode(idx)
            assert decoded == uci

    def test_all_indices_unique(self, encoder):
        """Test that all moves produce unique indices."""
        seen = set()
        for from_sq in range(64):
            for to_sq in range(64):
                idx = from_sq * 64 + to_sq
                assert idx not in seen
                seen.add(idx)

        assert len(seen) == 4096

    def test_decode_all_indices(self, encoder):
        """Test decoding all possible indices."""
        for idx in range(4096):
            uci = encoder.decode(idx)
            assert len(uci) == 4
            re_encoded = encoder.encode(uci)
            assert re_encoded == idx

    def test_encode_chess_move(self, encoder):
        """Test encoding chess.Move objects."""
        move = chess.Move.from_uci("e2e4")
        idx = encoder.encode_move(move)

        # Should match string encoding
        assert idx == encoder.encode("e2e4")

    def test_decode_to_move(self, encoder):
        """Test decoding to chess.Move."""
        idx = encoder.encode("e2e4")
        move = encoder.decode_to_move(idx)

        assert move.uci() == "e2e4"

    def test_encode_batch(self, encoder):
        """Test batch encoding."""
        moves = ["e2e4", "d2d4", "g1f3"]
        indices = encoder.encode_batch(moves)

        assert indices.shape == (3,)
        assert indices.dtype == torch.long

        for i, uci in enumerate(moves):
            assert indices[i] == encoder.encode(uci)

    def test_decode_batch(self, encoder):
        """Test batch decoding."""
        indices = torch.tensor([encoder.encode("e2e4"), encoder.encode("d2d4")])
        moves = encoder.decode_batch(indices)

        assert moves == ["e2e4", "d2d4"]

    def test_to_onehot(self, encoder):
        """Test one-hot encoding."""
        idx = encoder.encode("e2e4")
        onehot = encoder.to_onehot(idx)

        assert onehot.shape == (4096,)
        assert onehot.sum() == 1.0
        assert onehot[idx] == 1.0

    def test_from_onehot(self, encoder):
        """Test one-hot decoding."""
        idx = encoder.encode("e2e4")
        onehot = encoder.to_onehot(idx)
        decoded_idx = encoder.from_onehot(onehot)

        assert decoded_idx == idx

    def test_legal_move_mask(self, encoder):
        """Test legal move mask generation."""
        board = chess.Board()
        mask = encoder.get_legal_move_mask(board)

        assert mask.shape == (4096,)
        assert mask.dtype == torch.bool

        # Count legal moves
        num_legal = len(list(board.legal_moves))
        assert mask.sum() == num_legal

    def test_filter_to_legal(self, encoder):
        """Test filtering logits to legal moves."""
        board = chess.Board()
        logits = torch.randn(4096)

        filtered = encoder.filter_to_legal(logits, board)

        # Legal moves should keep their values
        # Illegal moves should be -inf
        mask = encoder.get_legal_move_mask(board)
        assert (filtered[~mask] == float("-inf")).all()

    def test_normalize_move_for_black(self, encoder):
        """Test move normalization for Black."""
        # e2e4 (index for white)
        white_idx = encoder.encode("e2e4")

        # After board flip, this should become d7d5
        black_idx = encoder.normalize_move_for_black(white_idx)

        # Verify it's different
        assert white_idx != black_idx

        # Verify reversibility
        restored = encoder.denormalize_move_for_black(black_idx)
        assert restored == white_idx

    def test_get_from_to_squares(self, encoder):
        """Test extracting squares from index."""
        idx = encoder.encode("e2e4")
        from_sq, to_sq = encoder.get_from_to_squares(idx)

        # e2 = 12, e4 = 28
        assert chess.square_name(from_sq) == "e2"
        assert chess.square_name(to_sq) == "e4"
