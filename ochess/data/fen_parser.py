"""
FEN (Forsyth-Edwards Notation) parser for chess positions.

Converts between FEN strings and tensor representations.

FEN Format:
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    |--- board ---|  |turn| |cast| |ep| |hm|fullmove|

Tensor Format:
    Shape: (8, 8) with values 0-12 representing:
    0: empty, 1-6: white pieces (P,N,B,R,Q,K), 7-12: black pieces (p,n,b,r,q,k)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import chess


class FenParser:
    """
    Parse FEN strings to tensors and back.

    The parser uses a simple integer encoding:
        0: empty square
        1-6: white pieces (Pawn, Knight, Bishop, Rook, Queen, King)
        7-12: black pieces (pawn, knight, bishop, rook, queen, king)

    Board orientation:
        Row 0 = rank 8 (black's back rank)
        Row 7 = rank 1 (white's back rank)
        Col 0 = file a
        Col 7 = file h
    """

    # Piece to index mapping
    PIECE_TO_INDEX: Dict[str, int] = {
        " ": 0,   # Empty
        "P": 1,   # White pawn
        "N": 2,   # White knight
        "B": 3,   # White bishop
        "R": 4,   # White rook
        "Q": 5,   # White queen
        "K": 6,   # White king
        "p": 7,   # Black pawn
        "n": 8,   # Black knight
        "b": 9,   # Black bishop
        "r": 10,  # Black rook
        "q": 11,  # Black queen
        "k": 12,  # Black king
    }

    # Index to piece mapping (reverse)
    INDEX_TO_PIECE: Dict[int, str] = {v: k for k, v in PIECE_TO_INDEX.items()}

    # Piece symbols for display
    PIECE_SYMBOLS: Dict[int, str] = {
        0: ".",
        1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
        7: "p", 8: "n", 9: "b", 10: "r", 11: "q", 12: "k",
    }

    # Unicode piece symbols for pretty display
    UNICODE_PIECES: Dict[int, str] = {
        0: ".",
        1: "\u2659", 2: "\u2658", 3: "\u2657", 4: "\u2656", 5: "\u2655", 6: "\u2654",
        7: "\u265F", 8: "\u265E", 9: "\u265D", 10: "\u265C", 11: "\u265B", 12: "\u265A",
    }

    NUM_PIECES = 13  # Total piece types including empty

    def __init__(self):
        """Initialize the FEN parser."""
        pass

    def to_tensor(self, fen: str) -> torch.Tensor:
        """
        Convert FEN string to 8x8 tensor of piece indices.

        Args:
            fen: FEN string (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        Returns:
            torch.LongTensor of shape (8, 8) with piece indices 0-12
        """
        # Extract board part (before first space)
        board_part = fen.split()[0]

        # Initialize empty board
        board = torch.zeros(8, 8, dtype=torch.long)

        # Parse ranks (rows) - FEN goes from rank 8 to rank 1
        ranks = board_part.split("/")
        for rank_idx, rank_str in enumerate(ranks):
            file_idx = 0
            for char in rank_str:
                if char.isdigit():
                    # Skip empty squares
                    file_idx += int(char)
                else:
                    # Place piece
                    board[rank_idx, file_idx] = self.PIECE_TO_INDEX[char]
                    file_idx += 1

        return board

    def to_tensor_onehot(self, fen: str) -> torch.Tensor:
        """
        Convert FEN string to one-hot encoded tensor.

        Args:
            fen: FEN string

        Returns:
            torch.FloatTensor of shape (8, 8, 13) with one-hot encoding
        """
        indices = self.to_tensor(fen)
        onehot = torch.zeros(8, 8, self.NUM_PIECES)
        for i in range(8):
            for j in range(8):
                onehot[i, j, indices[i, j]] = 1.0
        return onehot

    def to_fen(self, board: torch.Tensor, include_metadata: bool = False) -> str:
        """
        Convert board tensor back to FEN string.

        Args:
            board: torch.Tensor of shape (8, 8) with piece indices
            include_metadata: If True, append " w - - 0 1" (default game state)

        Returns:
            FEN string representing the position
        """
        fen_parts = []

        for rank_idx in range(8):
            rank_str = ""
            empty_count = 0

            for file_idx in range(8):
                piece_idx = int(board[rank_idx, file_idx].item())

                if piece_idx == 0:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += self.INDEX_TO_PIECE[piece_idx]

            if empty_count > 0:
                rank_str += str(empty_count)

            fen_parts.append(rank_str)

        board_fen = "/".join(fen_parts)

        if include_metadata:
            return f"{board_fen} w - - 0 1"
        return board_fen

    def get_active_color(self, fen: str) -> int:
        """
        Extract active color from FEN.

        Args:
            fen: FEN string

        Returns:
            0 for White to move, 1 for Black to move
        """
        parts = fen.split()
        if len(parts) >= 2:
            return 0 if parts[1] == "w" else 1
        return 0  # Default to white

    def is_white_to_move(self, fen: str) -> bool:
        """Check if it's White's turn to move."""
        return self.get_active_color(fen) == 0

    def flip_board(self, board: torch.Tensor) -> torch.Tensor:
        """
        Flip the board 180 degrees (for viewing from Black's perspective).

        Args:
            board: torch.Tensor of shape (8, 8)

        Returns:
            Flipped board tensor
        """
        return board.flip(0).flip(1)

    def swap_colors(self, board: torch.Tensor) -> torch.Tensor:
        """
        Swap piece colors (white <-> black).

        This is useful for normalizing positions to always be from
        the perspective of the side to move.

        Args:
            board: torch.Tensor of shape (8, 8)

        Returns:
            Board with swapped colors
        """
        result = board.clone()

        # White pieces (1-6) become black (7-12)
        white_mask = (board >= 1) & (board <= 6)
        result[white_mask] = board[white_mask] + 6

        # Black pieces (7-12) become white (1-6)
        black_mask = (board >= 7) & (board <= 12)
        result[black_mask] = board[black_mask] - 6

        return result

    def normalize_for_player(
        self,
        board: torch.Tensor,
        white_to_move: bool
    ) -> torch.Tensor:
        """
        Normalize board so current player's pieces are at the bottom.

        When Black to move:
        1. Flip board 180 degrees
        2. Swap piece colors

        This ensures the network always sees the position from the
        perspective of the player to move.

        Args:
            board: torch.Tensor of shape (8, 8)
            white_to_move: True if White to move

        Returns:
            Normalized board tensor
        """
        if white_to_move:
            return board
        else:
            # Flip and swap colors for Black's perspective
            return self.swap_colors(self.flip_board(board))

    def denormalize_for_player(
        self,
        board: torch.Tensor,
        white_to_move: bool
    ) -> torch.Tensor:
        """
        Reverse the normalization to get original board orientation.

        Args:
            board: Normalized board tensor
            white_to_move: True if White to move in original position

        Returns:
            Original board orientation
        """
        if white_to_move:
            return board
        else:
            # Reverse: swap colors then flip
            return self.flip_board(self.swap_colors(board))

    def display(
        self,
        board: torch.Tensor,
        use_unicode: bool = False,
        flip: bool = False
    ) -> str:
        """
        Create a string representation of the board for display.

        Args:
            board: torch.Tensor of shape (8, 8)
            use_unicode: Use Unicode chess symbols
            flip: Show from Black's perspective

        Returns:
            Multi-line string representation
        """
        symbols = self.UNICODE_PIECES if use_unicode else self.PIECE_SYMBOLS

        if flip:
            board = self.flip_board(board)

        lines = ["   a b c d e f g h", "  +-+-+-+-+-+-+-+-+"]

        ranks = range(8) if not flip else range(7, -1, -1)
        rank_labels = list(range(8, 0, -1)) if not flip else list(range(1, 9))

        for i, rank_idx in enumerate(ranks):
            row = f"{rank_labels[i]} |"
            for file_idx in range(8):
                piece_idx = int(board[rank_idx, file_idx].item())
                row += f"{symbols[piece_idx]}|"
            lines.append(row)
            lines.append("  +-+-+-+-+-+-+-+-+")

        lines.append("   a b c d e f g h")

        return "\n".join(lines)

    @staticmethod
    def from_chess_board(board: chess.Board) -> torch.Tensor:
        """
        Convert python-chess Board to tensor.

        Args:
            board: chess.Board object

        Returns:
            torch.LongTensor of shape (8, 8)
        """
        parser = FenParser()
        return parser.to_tensor(board.fen())

    def validate_fen(self, fen: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a FEN string.

        Args:
            fen: FEN string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            board = chess.Board(fen)
            return True, None
        except ValueError as e:
            return False, str(e)
