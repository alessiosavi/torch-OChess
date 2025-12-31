"""
Embedding layers for chess neural networks.

Provides:
    - PieceEmbedding: Learnable embeddings for piece types
    - PositionalEmbedding: Learnable position encodings for squares
"""

import math

import torch
import torch.nn as nn


class PieceEmbedding(nn.Module):
    """
    Learnable embeddings for chess pieces.

    Maps piece indices (0-12) to dense vectors.

    Piece indices:
        0: empty
        1-6: white pieces (P, N, B, R, Q, K)
        7-12: black pieces (p, n, b, r, q, k)
    """

    def __init__(self, num_pieces: int = 13, embed_dim: int = 64):
        """
        Initialize piece embeddings.

        Args:
            num_pieces: Number of piece types (13 = empty + 6 white + 6 black)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.num_pieces = num_pieces
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_pieces, embed_dim)

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for piece indices.

        Args:
            x: Piece indices tensor of shape [..., 8, 8]

        Returns:
            Embeddings of shape [..., 8, 8, embed_dim]
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for board squares.

    Encodes the (row, col) position of each square using
    separate row and column embeddings.
    """

    def __init__(self, board_size: int = 8, embed_dim: int = 64):
        """
        Initialize positional embeddings.

        Args:
            board_size: Size of the board (8 for standard chess)
            embed_dim: Embedding dimension (split between row and col)
        """
        super().__init__()
        self.board_size = board_size
        self.embed_dim = embed_dim

        # Learnable embeddings for rows and columns
        half_dim = embed_dim // 2
        self.row_embed = nn.Embedding(board_size, half_dim)
        self.col_embed = nn.Embedding(board_size, half_dim)

        # Initialize
        nn.init.normal_(self.row_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.col_embed.weight, mean=0, std=0.02)

        # Pre-compute position indices
        # Use clone() to ensure contiguous memory (required for state_dict loading)
        rows = (
            torch.arange(board_size).unsqueeze(1).expand(board_size, board_size).clone()
        )
        cols = (
            torch.arange(board_size).unsqueeze(0).expand(board_size, board_size).clone()
        )
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get positional embeddings.

        Args:
            x: Input tensor of shape [B, ...] (used only for shape/device)

        Returns:
            Positional embeddings of shape [8, 8, embed_dim]
        """
        row_emb = self.row_embed(self.rows)  # [8, 8, half_dim]
        col_emb = self.col_embed(self.cols)  # [8, 8, half_dim]

        return torch.cat([row_emb, col_emb], dim=-1)  # [8, 8, embed_dim]

    def get_expanded(self, batch_size: int, seq_len: int = 1) -> torch.Tensor:
        """
        Get positional embeddings expanded to batch dimensions.

        Args:
            batch_size: Batch size
            seq_len: Sequence length (for temporal models)

        Returns:
            Embeddings of shape [batch_size, seq_len, 8, 8, embed_dim]
        """
        pos_emb = self.forward(None)  # [8, 8, embed_dim]
        return pos_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1, -1)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings (non-learnable).

    Uses the same formula as in Transformer/BERT for positions.
    """

    def __init__(self, board_size: int = 8, embed_dim: int = 64):
        """
        Initialize sinusoidal embeddings.

        Args:
            board_size: Size of the board
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.board_size = board_size
        self.embed_dim = embed_dim

        # Compute sinusoidal embeddings
        pe = self._create_sinusoidal_embeddings(board_size, embed_dim)
        self.register_buffer("pe", pe)

    def _create_sinusoidal_embeddings(
        self, board_size: int, embed_dim: int
    ) -> torch.Tensor:
        """Create sinusoidal position embeddings."""
        pe = torch.zeros(board_size, board_size, embed_dim)

        # Create position indices
        for row in range(board_size):
            for col in range(board_size):
                # Combine row and col into a single position index
                pos = row * board_size + col

                for i in range(0, embed_dim, 2):
                    div_term = math.exp(i * (-math.log(10000.0) / embed_dim))
                    pe[row, col, i] = math.sin(pos * div_term)
                    if i + 1 < embed_dim:
                        pe[row, col, i + 1] = math.cos(pos * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get positional embeddings.

        Returns:
            Embeddings of shape [8, 8, embed_dim]
        """
        return self.pe


class ColorEmbedding(nn.Module):
    """
    Embedding for side to move (color).

    Maps 0 (White) and 1 (Black) to dense vectors.
    """

    def __init__(self, embed_dim: int = 64):
        """
        Initialize color embedding.

        Args:
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, color: torch.Tensor) -> torch.Tensor:
        """
        Get color embeddings.

        Args:
            color: Tensor of 0s and 1s indicating side to move

        Returns:
            Color embeddings
        """
        return self.embedding(color)


class CombinedEmbedding(nn.Module):
    """
    Combined piece and positional embeddings.

    Adds piece embeddings and positional embeddings together.
    """

    def __init__(
        self,
        num_pieces: int = 13,
        embed_dim: int = 128,
        use_learnable_position: bool = True,
    ):
        """
        Initialize combined embeddings.

        Args:
            num_pieces: Number of piece types
            embed_dim: Total embedding dimension
            use_learnable_position: Use learnable vs sinusoidal position embeddings
        """
        super().__init__()

        self.piece_embed = PieceEmbedding(num_pieces, embed_dim)

        if use_learnable_position:
            self.pos_embed = PositionalEmbedding(8, embed_dim)
        else:
            self.pos_embed = SinusoidalPositionalEmbedding(8, embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """
        Get combined embeddings.

        Args:
            board: Board tensor of shape [B, T, 8, 8] or [B, 8, 8]

        Returns:
            Combined embeddings of shape [B, T, 8, 8, embed_dim] or [B, 8, 8, embed_dim]
        """
        # Get piece embeddings
        piece_emb = self.piece_embed(board)

        # Get position embeddings
        pos_emb = self.pos_embed(board)

        # Expand position embeddings to match batch dimensions
        while pos_emb.dim() < piece_emb.dim():
            pos_emb = pos_emb.unsqueeze(0)

        # Broadcast position embeddings
        pos_emb = pos_emb.expand_as(piece_emb)

        # Combine and normalize
        combined = piece_emb + pos_emb
        return self.layer_norm(combined)
