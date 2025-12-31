"""
Hybrid chess neural network combining ResNet and Transformer.

Uses a ResNet backbone for local feature extraction followed by
self-attention for global context. This combines the efficiency
of CNNs with the global modeling capability of transformers.

Architecture:
    1. ResNet backbone (reduced depth)
    2. Flatten to sequence
    3. Self-attention layer
    4. Output heads

This is a balanced approach that's faster than pure transformer
but more expressive than pure ResNet.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

from ochess.model.components.embeddings import PieceEmbedding, PositionalEmbedding, ColorEmbedding
from ochess.model.components.residual import ResidualBlock, ResidualTower
from ochess.model.components.attention import SelfAttention, TransformerBlock
from ochess.model.components.heads import MoveHead, ScoreHead, CaptureHead, OutcomeHead


@dataclass
class ChessHybridConfig:
    """Configuration for ChessHybrid."""
    # Embeddings
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # CNN backbone
    hidden_dim: int = 256
    num_residual_blocks: int = 4  # Fewer than pure ResNet

    # Attention
    num_attention_layers: int = 2
    num_heads: int = 8
    attention_dropout: float = 0.1

    # General
    dropout: float = 0.2
    num_moves: int = 4096
    sequence_length: int = 5
    flip_board_for_black: bool = True


class ChessHybrid(nn.Module):
    """
    Hybrid chess neural network.

    Combines the strengths of both architectures:
    - ResNet backbone for efficient local feature extraction
    - Attention layers for global context and long-range dependencies

    This is a good balance between speed and capability.

    Architecture:
        Input -> Embeddings -> ResNet Backbone -> Flatten
             -> Attention Layers -> Output Heads

    Advantages:
        - Faster than pure Transformer
        - More expressive than pure ResNet
        - Good for medium-sized datasets

    Typical use case:
        When you want better than ResNet but can't afford
        full Transformer training time/data.
    """

    def __init__(self, config: Optional[ChessHybridConfig] = None):
        """
        Initialize ChessHybrid.

        Args:
            config: Model configuration
        """
        super().__init__()

        if config is None:
            config = ChessHybridConfig()
        self.config = config

        embed_dim = config.piece_embed_dim + config.position_embed_dim

        # Embeddings
        self.piece_embed = PieceEmbedding(13, config.piece_embed_dim)
        self.pos_embed = PositionalEmbedding(8, config.position_embed_dim)
        self.color_embed = ColorEmbedding(config.hidden_dim)

        # Input projection
        self.input_conv = nn.Conv2d(embed_dim, config.hidden_dim, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(config.hidden_dim)

        # ResNet backbone (fewer blocks than pure ResNet)
        self.residual_tower = ResidualTower(
            channels=config.hidden_dim,
            num_blocks=config.num_residual_blocks,
            block_type="basic",
            dropout=config.dropout
        )

        # Attention layers on flattened features
        # Each position becomes a token
        self.attention_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=2.0,  # Smaller MLP for efficiency
                dropout=config.attention_dropout
            )
            for _ in range(config.num_attention_layers)
        ])

        # Position embeddings for attention (64 squares)
        self.attn_pos_embed = nn.Parameter(torch.zeros(1, 64, config.hidden_dim))

        # Temporal aggregation
        if config.sequence_length > 1:
            self.temporal_attn = nn.MultiheadAttention(
                config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.attention_dropout,
                batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.temporal_attn = None

        # Output heads
        self.move_head = MoveHead(config.hidden_dim, config.num_moves)
        self.score_head = ScoreHead(config.hidden_dim)
        self.capture_head = CaptureHead(config.hidden_dim)
        self.outcome_head = OutcomeHead(config.hidden_dim)

        self.relu = nn.ReLU(inplace=True)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        nn.init.normal_(self.attn_pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        board_tensor: torch.Tensor,
        color_to_move: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            board_tensor: Piece indices [B, T, 8, 8] or [B, 8, 8]
            color_to_move: Side to move [B, T] or [B]
            return_features: Return intermediate features

        Returns:
            Dictionary with move_logits, score, capture, outcome
        """
        # Handle 2D input
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(1)
            color_to_move = color_to_move.unsqueeze(1)

        B, T, H, W = board_tensor.shape

        # Flip for Black
        if self.config.flip_board_for_black:
            board_tensor = self._flip_boards_for_black(board_tensor, color_to_move)

        # Get embeddings
        piece_emb = self.piece_embed(board_tensor)  # [B, T, 8, 8, piece_dim]
        pos_emb = self.pos_embed(board_tensor)       # [8, 8, pos_dim]
        pos_emb = pos_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)

        x = torch.cat([piece_emb, pos_emb], dim=-1)  # [B, T, 8, 8, embed_dim]

        # Reshape for 2D conv: [B*T, embed_dim, 8, 8]
        x = x.view(B * T, H, W, -1).permute(0, 3, 1, 2)

        # CNN backbone
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.relu(x)
        x = self.residual_tower(x)  # [B*T, hidden_dim, 8, 8]

        # Flatten spatial for attention: [B*T, 64, hidden_dim]
        x = x.flatten(2).transpose(1, 2)

        # Add position embeddings for attention
        x = x + self.attn_pos_embed

        # Self-attention layers
        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        # Reshape back: [B, T, 64, hidden_dim]
        x = x.view(B, T, 64, -1)

        # Temporal aggregation
        if T > 1 and self.temporal_attn is not None:
            # Pool spatial dimension first
            x_pooled = x.mean(dim=2)  # [B, T, hidden_dim]
            x_pooled = self.temporal_norm(x_pooled)

            # Self-attention across time
            x_temporal, _ = self.temporal_attn(x_pooled, x_pooled, x_pooled)
            x_temporal = x_temporal[:, -1]  # Take last timestep [B, hidden_dim]

            # Get spatial features from last timestep
            x_spatial = x[:, -1]  # [B, 64, hidden_dim]
        else:
            x_temporal = x[:, -1].mean(dim=1)  # [B, hidden_dim]
            x_spatial = x[:, -1]  # [B, 64, hidden_dim]

        # Add color context
        color_ctx = self.color_embed(color_to_move[:, -1])  # [B, hidden_dim]
        x_temporal = x_temporal + color_ctx

        # Reshape spatial for conv-based heads: [B, hidden_dim, 8, 8]
        x_spatial_2d = x_spatial.transpose(1, 2).view(B, -1, 8, 8)

        # Add temporal context to spatial
        x_spatial_2d = x_spatial_2d + x_temporal.unsqueeze(-1).unsqueeze(-1)

        # Output heads
        outputs = {
            'move_logits': self.move_head(x_spatial_2d),
            'score': self.score_head(x_spatial_2d),
            'capture': self.capture_head(x_spatial_2d),
            'outcome': self.outcome_head(x_spatial_2d)
        }

        if return_features:
            outputs['features'] = x_spatial_2d
            outputs['temporal_features'] = x_temporal

        return outputs

    def _flip_boards_for_black(
        self,
        boards: torch.Tensor,
        colors: torch.Tensor
    ) -> torch.Tensor:
        """Flip boards when Black to move."""
        B, T, H, W = boards.shape
        result = boards.clone()

        black_mask = (colors == 1)

        for b in range(B):
            for t in range(T):
                if black_mask[b, t]:
                    result[b, t] = boards[b, t].flip(0).flip(1)

                    board = result[b, t]
                    white_mask = (board >= 1) & (board <= 6)
                    black_piece_mask = (board >= 7) & (board <= 12)

                    result[b, t][white_mask] = board[white_mask] + 6
                    result[b, t][black_piece_mask] = board[black_piece_mask] - 6

        return result

    def predict_move(
        self,
        board_tensor: torch.Tensor,
        color_to_move: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Get move probabilities."""
        with torch.no_grad():
            outputs = self.forward(board_tensor, color_to_move)
            logits = outputs['move_logits']
            if temperature != 1.0:
                logits = logits / temperature
            return torch.softmax(logits, dim=-1)

    def get_best_move(
        self,
        board_tensor: torch.Tensor,
        color_to_move: torch.Tensor
    ) -> torch.Tensor:
        """Get best move index."""
        probs = self.predict_move(board_tensor, color_to_move, temperature=0.01)
        return torch.argmax(probs, dim=-1)

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_type: str = "hybrid",
    **kwargs
) -> nn.Module:
    """
    Factory function to create chess models.

    Args:
        model_type: "resnet", "transformer", or "hybrid"
        **kwargs: Model-specific configuration

    Returns:
        Initialized model
    """
    from ochess.model.chess_resnet import ChessResNet, ChessResNetConfig
    from ochess.model.chess_transformer import ChessTransformer, ChessTransformerConfig

    if model_type == "resnet":
        config = ChessResNetConfig(**kwargs) if kwargs else None
        return ChessResNet(config)
    elif model_type == "transformer":
        config = ChessTransformerConfig(**kwargs) if kwargs else None
        return ChessTransformer(config)
    elif model_type == "hybrid":
        config = ChessHybridConfig(**kwargs) if kwargs else None
        return ChessHybrid(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
