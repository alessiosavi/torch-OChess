"""
ResNet-style chess neural network.

Architecture inspired by AlphaZero:
- Piece + position embeddings
- Initial convolution
- Tower of residual blocks
- Separate policy (move) and value (score/outcome) heads

Key features:
- Board flipping for Black (consistent perspective)
- Multi-task outputs (move, score, capture, outcome)
- Efficient Conv2D-based processing
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

from ochess.model.components.embeddings import PieceEmbedding, PositionalEmbedding, ColorEmbedding
from ochess.model.components.residual import ResidualBlock, ResidualTower
from ochess.model.components.heads import MoveHead, ScoreHead, CaptureHead, OutcomeHead


@dataclass
class ChessResNetConfig:
    """Configuration for ChessResNet."""
    # Embeddings
    piece_embed_dim: int = 64
    position_embed_dim: int = 64

    # Network
    hidden_dim: int = 256
    num_residual_blocks: int = 8
    dropout: float = 0.3

    # Output
    num_moves: int = 4096

    # Input handling
    sequence_length: int = 5  # Number of past positions
    flip_board_for_black: bool = True


class ChessResNet(nn.Module):
    """
    ResNet-style chess neural network.

    This is an AlphaZero-inspired architecture that uses residual
    blocks for deep feature extraction. It's proven to work well
    for chess and is relatively fast to train.

    Architecture:
        1. Piece embeddings (13 types -> embed_dim)
        2. Position embeddings (8x8 grid -> embed_dim)
        3. Initial convolution to hidden_dim channels
        4. Tower of residual blocks
        5. Temporal aggregation (for sequence of positions)
        6. Separate output heads for each task

    Key Features:
        - Board flipping for Black (network always sees from mover's view)
        - Multi-task learning (move, score, capture, outcome)
        - Residual connections for stable training
    """

    def __init__(self, config: Optional[ChessResNetConfig] = None):
        """
        Initialize ChessResNet.

        Args:
            config: Model configuration (uses defaults if None)
        """
        super().__init__()

        if config is None:
            config = ChessResNetConfig()
        self.config = config

        # Calculate embedding dimensions
        embed_dim = config.piece_embed_dim + config.position_embed_dim

        # Embeddings
        self.piece_embed = PieceEmbedding(13, config.piece_embed_dim)
        self.pos_embed = PositionalEmbedding(8, config.position_embed_dim)
        self.color_embed = ColorEmbedding(config.hidden_dim)

        # Initial projection: embed_dim -> hidden_dim
        self.input_conv = nn.Conv2d(embed_dim, config.hidden_dim, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(config.hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Residual tower
        self.residual_tower = ResidualTower(
            channels=config.hidden_dim,
            num_blocks=config.num_residual_blocks,
            block_type="basic",
            dropout=config.dropout
        )

        # Temporal aggregation (if using sequences)
        if config.sequence_length > 1:
            self.temporal_conv = nn.Conv3d(
                config.hidden_dim,
                config.hidden_dim,
                kernel_size=(config.sequence_length, 1, 1),
                bias=False
            )
            self.temporal_bn = nn.BatchNorm3d(config.hidden_dim)
        else:
            self.temporal_conv = None

        # Output heads
        self.move_head = MoveHead(config.hidden_dim, config.num_moves)
        self.score_head = ScoreHead(config.hidden_dim)
        self.capture_head = CaptureHead(config.hidden_dim)
        self.outcome_head = OutcomeHead(config.hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
            color_to_move: Side to move [B, T] or [B] (0=white, 1=black)
            return_features: If True, also return intermediate features

        Returns:
            Dictionary with:
            - 'move_logits': [B, 4096]
            - 'score': [B, 1]
            - 'capture': [B, 2]
            - 'outcome': [B, 3]
            - 'features': [B, C, H, W] (if return_features=True)
        """
        # Handle 2D input (single position)
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(1)  # [B, 1, 8, 8]
            color_to_move = color_to_move.unsqueeze(1)  # [B, 1]

        B, T, H, W = board_tensor.shape

        # Optionally flip boards for Black
        if self.config.flip_board_for_black:
            board_tensor = self._flip_boards_for_black(board_tensor, color_to_move)

        # Get embeddings
        piece_emb = self.piece_embed(board_tensor)  # [B, T, 8, 8, piece_dim]
        pos_emb = self.pos_embed(board_tensor)       # [8, 8, pos_dim]

        # Expand position embeddings
        pos_emb = pos_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)

        # Combine embeddings
        x = torch.cat([piece_emb, pos_emb], dim=-1)  # [B, T, 8, 8, embed_dim]

        # Reshape for 2D convolutions: [B*T, embed_dim, 8, 8]
        x = x.view(B * T, H, W, -1).permute(0, 3, 1, 2)

        # Initial convolution
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.relu(x)

        # Residual tower
        x = self.residual_tower(x)  # [B*T, hidden_dim, 8, 8]

        # Temporal aggregation
        if T > 1 and self.temporal_conv is not None:
            # Reshape: [B, hidden_dim, T, 8, 8]
            x = x.view(B, T, self.config.hidden_dim, H, W).permute(0, 2, 1, 3, 4)
            x = self.temporal_conv(x)  # [B, hidden_dim, 1, 8, 8]
            x = self.temporal_bn(x)
            x = self.relu(x)
            x = x.squeeze(2)  # [B, hidden_dim, 8, 8]
        else:
            # Just take the last position
            x = x.view(B, T, self.config.hidden_dim, H, W)[:, -1]  # [B, hidden_dim, 8, 8]

        # Add color context
        color_ctx = self.color_embed(color_to_move[:, -1])  # [B, hidden_dim]
        color_ctx = color_ctx.unsqueeze(-1).unsqueeze(-1)   # [B, hidden_dim, 1, 1]
        x = x + color_ctx

        # Output heads
        outputs = {
            'move_logits': self.move_head(x),
            'score': self.score_head(x),
            'capture': self.capture_head(x),
            'outcome': self.outcome_head(x)
        }

        if return_features:
            outputs['features'] = x

        return outputs

    def _flip_boards_for_black(
        self,
        boards: torch.Tensor,
        colors: torch.Tensor
    ) -> torch.Tensor:
        """
        Flip boards when Black to move.

        This normalizes the input so the network always sees
        the position from the current player's perspective.

        Args:
            boards: Board tensor [B, T, 8, 8]
            colors: Color tensor [B, T]

        Returns:
            Flipped board tensor
        """
        B, T, H, W = boards.shape
        result = boards.clone()

        # Create mask for Black's turns
        black_mask = (colors == 1)

        for b in range(B):
            for t in range(T):
                if black_mask[b, t]:
                    # Flip board 180 degrees
                    result[b, t] = boards[b, t].flip(0).flip(1)

                    # Swap piece colors (white <-> black)
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
        """
        Get move probabilities.

        Args:
            board_tensor: Board positions
            color_to_move: Side to move
            temperature: Softmax temperature (lower = more deterministic)

        Returns:
            Move probabilities [B, 4096]
        """
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
        """
        Get the best move index.

        Args:
            board_tensor: Board positions
            color_to_move: Side to move

        Returns:
            Best move indices [B]
        """
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
