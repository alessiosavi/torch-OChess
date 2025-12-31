"""Tests for chess neural networks."""

import pytest
import torch

from ochess.model.chess_hybrid import ChessHybrid, ChessHybridConfig
from ochess.model.chess_resnet import ChessResNet, ChessResNetConfig
from ochess.model.chess_transformer import (ChessTransformer,
                                            ChessTransformerConfig)


class TestChessResNet:
    """Tests for ResNet model."""

    @pytest.fixture
    def model(self):
        config = ChessResNetConfig(num_residual_blocks=2, sequence_length=3)
        return ChessResNet(config)

    def test_forward_shape(self, model):
        """Test output shapes."""
        batch_size = 4
        seq_len = 3

        boards = torch.randint(0, 13, (batch_size, seq_len, 8, 8))
        colors = torch.randint(0, 2, (batch_size, seq_len))

        outputs = model(boards, colors)

        assert outputs["move_logits"].shape == (batch_size, 4096)
        assert outputs["score"].shape == (batch_size, 1)
        assert outputs["capture"].shape == (batch_size, 2)
        assert outputs["outcome"].shape == (batch_size, 3)

    def test_single_position(self, model):
        """Test with single position (no sequence)."""
        boards = torch.randint(0, 13, (2, 8, 8))  # [B, 8, 8]
        colors = torch.randint(0, 2, (2,))  # [B]

        outputs = model(boards, colors)

        assert outputs["move_logits"].shape == (2, 4096)

    def test_return_features(self, model):
        """Test returning intermediate features."""
        boards = torch.randint(0, 13, (2, 3, 8, 8))
        colors = torch.randint(0, 2, (2, 3))

        outputs = model(boards, colors, return_features=True)

        assert "features" in outputs

    def test_predict_move(self, model):
        """Test move prediction method."""
        boards = torch.randint(0, 13, (2, 3, 8, 8))
        colors = torch.randint(0, 2, (2, 3))

        probs = model.predict_move(boards, colors)

        assert probs.shape == (2, 4096)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


class TestChessTransformer:
    """Tests for Transformer model."""

    @pytest.fixture
    def model(self):
        config = ChessTransformerConfig(num_layers=2, sequence_length=3, embed_dim=128)
        return ChessTransformer(config)

    def test_forward_shape(self, model):
        """Test output shapes."""
        batch_size = 2
        seq_len = 3

        boards = torch.randint(0, 13, (batch_size, seq_len, 8, 8))
        colors = torch.randint(0, 2, (batch_size, seq_len))

        outputs = model(boards, colors)

        assert outputs["move_logits"].shape == (batch_size, 4096)
        assert outputs["score"].shape == (batch_size, 1)


class TestChessHybrid:
    """Tests for Hybrid model."""

    @pytest.fixture
    def model(self):
        config = ChessHybridConfig(
            num_residual_blocks=2,
            num_attention_layers=1,
            sequence_length=3,
            hidden_dim=128,
        )
        return ChessHybrid(config)

    def test_forward_shape(self, model):
        """Test output shapes."""
        batch_size = 2
        seq_len = 3

        boards = torch.randint(0, 13, (batch_size, seq_len, 8, 8))
        colors = torch.randint(0, 2, (batch_size, seq_len))

        outputs = model(boards, colors)

        assert outputs["move_logits"].shape == (batch_size, 4096)
        assert outputs["score"].shape == (batch_size, 1)


class TestBoardFlipping:
    """Test board flipping for Black perspective."""

    @pytest.fixture
    def model(self):
        config = ChessResNetConfig(num_residual_blocks=2, flip_board_for_black=True)
        return ChessResNet(config)

    def test_different_outputs_for_different_colors(self, model):
        """Same board, different color should give different outputs."""
        # Same board state
        boards = torch.randint(0, 13, (1, 5, 8, 8))

        # White to move
        white_colors = torch.zeros(1, 5, dtype=torch.long)
        white_out = model(boards.clone(), white_colors)

        # Black to move
        black_colors = torch.ones(1, 5, dtype=torch.long)
        black_out = model(boards.clone(), black_colors)

        # Outputs should be different due to perspective change
        assert not torch.allclose(white_out["move_logits"], black_out["move_logits"])
