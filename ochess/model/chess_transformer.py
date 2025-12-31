"""
Transformer-based chess neural network.

Uses self-attention to model long-range dependencies on the chess board.
Each square is treated as a token, allowing the model to learn
complex piece interactions across the entire board.

Key features:
- Treats each of 64 squares as a token
- Multi-head self-attention for piece interactions
- Learnable position embeddings
- Same multi-task outputs as ResNet model
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

from ochess.model.components.embeddings import PieceEmbedding, ColorEmbedding
from ochess.model.components.attention import TransformerEncoder
from ochess.model.components.heads import MoveHead, ScoreHead, CaptureHead, OutcomeHead


@dataclass
class ChessTransformerConfig:
    """Configuration for ChessTransformer."""
    # Embeddings
    embed_dim: int = 256

    # Transformer
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Output
    num_moves: int = 4096

    # Input handling
    sequence_length: int = 5
    flip_board_for_black: bool = True


class ChessTransformer(nn.Module):
    """
    Transformer-based chess neural network.

    This architecture treats each square as a token and uses
    self-attention to learn piece interactions. It's particularly
    good at capturing long-range dependencies.

    Architecture:
        1. Piece + position embeddings per square (64 tokens)
        2. Flatten board to sequence of tokens
        3. Transformer encoder layers
        4. Aggregate temporal information
        5. Output heads for each task

    Advantages:
        - Better at long-range piece interactions
        - More interpretable attention patterns
        - Flexible handling of variable-length sequences

    Disadvantages:
        - Slower than CNN-based models
        - Higher memory usage
        - May require more data to train
    """

    def __init__(self, config: Optional[ChessTransformerConfig] = None):
        """
        Initialize ChessTransformer.

        Args:
            config: Model configuration
        """
        super().__init__()

        if config is None:
            config = ChessTransformerConfig()
        self.config = config

        # Embeddings
        self.piece_embed = nn.Embedding(13, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, config.embed_dim))
        self.color_embed = ColorEmbedding(config.embed_dim)

        # Temporal position embedding (for sequence of boards)
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, config.sequence_length, config.embed_dim)
        )

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Embedding normalization and dropout
        self.embed_norm = nn.LayerNorm(config.embed_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout
        )

        # Output projections (transformer output -> spatial features for heads)
        self.to_spatial = nn.Linear(config.embed_dim, config.embed_dim)

        # Output heads (using pooled CLS token features)
        self.move_proj = nn.Linear(config.embed_dim, 512)
        self.move_out = nn.Linear(512, config.num_moves)

        self.score_proj = nn.Linear(config.embed_dim, 128)
        self.score_out = nn.Linear(128, 1)

        self.capture_out = nn.Linear(config.embed_dim, 2)
        self.outcome_out = nn.Linear(config.embed_dim, 3)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.temporal_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

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

        # Flip boards for Black
        if self.config.flip_board_for_black:
            board_tensor = self._flip_boards_for_black(board_tensor, color_to_move)

        # Get piece embeddings and flatten spatial dimensions
        # [B, T, 8, 8] -> [B, T, 64] -> [B, T, 64, embed_dim]
        board_flat = board_tensor.view(B, T, -1)  # [B, T, 64]
        piece_emb = self.piece_embed(board_flat)   # [B, T, 64, embed_dim]

        # Add position embeddings
        piece_emb = piece_emb + self.pos_embed  # Broadcasting [1, 64, embed_dim]

        # Flatten temporal and spatial: [B, T*64, embed_dim]
        x = piece_emb.view(B, T * 64, -1)

        # Add temporal position embedding
        temporal = self.temporal_embed[:, :T, :].unsqueeze(2)  # [1, T, 1, embed_dim]
        temporal = temporal.expand(B, T, 64, -1).reshape(B, T * 64, -1)
        x = x + temporal

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1 + T*64, embed_dim]

        # Add color embedding to CLS token
        color_emb = self.color_embed(color_to_move[:, -1])  # [B, embed_dim]
        x[:, 0] = x[:, 0] + color_emb

        # Normalize and dropout
        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        # Transformer
        x = self.transformer(x)  # [B, 1 + T*64, embed_dim]

        # Extract CLS token for classification
        cls_out = x[:, 0]  # [B, embed_dim]

        # Extract board tokens (use last time step)
        board_tokens = x[:, 1:].view(B, T, 64, -1)[:, -1]  # [B, 64, embed_dim]

        # Move prediction (from CLS token)
        move_feat = torch.relu(self.move_proj(cls_out))
        move_logits = self.move_out(move_feat)

        # Score prediction
        score_feat = torch.relu(self.score_proj(cls_out))
        score = self.score_out(score_feat)

        # Capture and outcome
        capture = self.capture_out(cls_out)
        outcome = self.outcome_out(cls_out)

        outputs = {
            'move_logits': move_logits,
            'score': score,
            'capture': capture,
            'outcome': outcome
        }

        if return_features:
            outputs['features'] = cls_out
            outputs['board_features'] = board_tokens

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

    def get_attention_weights(
        self,
        board_tensor: torch.Tensor,
        color_to_move: torch.Tensor,
        layer: int = -1
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Args:
            board_tensor: Board positions
            color_to_move: Side to move
            layer: Which layer's attention (-1 for last)

        Returns:
            Attention weights [B, heads, seq_len, seq_len]
        """
        # This would require hooks or modifying the transformer
        # For now, return placeholder
        raise NotImplementedError("Attention visualization not yet implemented")

    @property
    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
