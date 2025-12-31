"""
UCI move encoding/decoding for chess.

Encodes chess moves as indices in a flat array:
    index = from_square * 64 + to_square

This gives 4096 possible moves (64 * 64).

Note: This simple encoding treats pawn promotions as regular moves.
All promotions are assumed to be to Queen (the most common choice).

Square numbering (python-chess convention):
    a1=0, b1=1, ..., h1=7
    a2=8, b2=9, ..., h2=15
    ...
    a8=56, b8=57, ..., h8=63
"""

from typing import List, Optional, Tuple

import chess
import torch


class MoveEncoder:
    """
    Encode and decode UCI chess moves.

    Encoding scheme:
        - Each move is represented by (from_square, to_square)
        - Flattened index: from_square * 64 + to_square
        - Total indices: 4096 (0-4095)

    Note on promotions:
        The simple 64x64 encoding doesn't distinguish promotion piece.
        We assume queen promotion (most common in practice).
        For full promotion support, would need ~4600 indices.
    """

    NUM_SQUARES = 64
    NUM_MOVES = NUM_SQUARES * NUM_SQUARES  # 4096

    def __init__(self):
        """Initialize the move encoder."""
        # Pre-compute square names for fast lookup
        self._square_names = [chess.square_name(i) for i in range(64)]

    def encode(self, uci_move: str) -> int:
        """
        Encode a UCI move string to an index.

        Args:
            uci_move: UCI format move (e.g., "e2e4", "e7e8q")

        Returns:
            Integer index in range [0, 4095]

        Example:
            >>> encoder = MoveEncoder()
            >>> encoder.encode("e2e4")  # e2=12, e4=28 -> 12*64+28=796
            796
        """
        # Extract from and to squares (first 4 characters)
        from_sq = chess.parse_square(uci_move[:2])
        to_sq = chess.parse_square(uci_move[2:4])

        return from_sq * self.NUM_SQUARES + to_sq

    def decode(self, index: int) -> str:
        """
        Decode an index back to UCI move string.

        Args:
            index: Integer index in range [0, 4095]

        Returns:
            UCI move string (e.g., "e2e4")

        Example:
            >>> encoder = MoveEncoder()
            >>> encoder.decode(796)  # 796//64=12 (e2), 796%64=28 (e4)
            "e2e4"
        """
        from_sq = index // self.NUM_SQUARES
        to_sq = index % self.NUM_SQUARES

        return self._square_names[from_sq] + self._square_names[to_sq]

    def encode_move(self, move: chess.Move) -> int:
        """
        Encode a python-chess Move object.

        Args:
            move: chess.Move object

        Returns:
            Integer index
        """
        return move.from_square * self.NUM_SQUARES + move.to_square

    def decode_to_move(
        self, index: int, board: Optional[chess.Board] = None
    ) -> chess.Move:
        """
        Decode index to a chess.Move object.

        Args:
            index: Integer index
            board: Optional board to check for promotion

        Returns:
            chess.Move object
        """
        from_sq = index // self.NUM_SQUARES
        to_sq = index % self.NUM_SQUARES

        # Check if this is a pawn promotion
        promotion = None
        if board is not None:
            piece = board.piece_at(from_sq)
            if piece is not None and piece.piece_type == chess.PAWN:
                # Check if pawn is moving to back rank
                to_rank = chess.square_rank(to_sq)
                if to_rank == 0 or to_rank == 7:
                    promotion = chess.QUEEN  # Default to queen

        return chess.Move(from_sq, to_sq, promotion=promotion)

    def encode_batch(self, uci_moves: List[str]) -> torch.Tensor:
        """
        Encode a batch of UCI moves.

        Args:
            uci_moves: List of UCI move strings

        Returns:
            torch.LongTensor of shape (batch_size,)
        """
        return torch.tensor([self.encode(m) for m in uci_moves], dtype=torch.long)

    def decode_batch(self, indices: torch.Tensor) -> List[str]:
        """
        Decode a batch of indices.

        Args:
            indices: torch.Tensor of indices

        Returns:
            List of UCI move strings
        """
        return [self.decode(int(idx)) for idx in indices]

    def to_onehot(self, index: int) -> torch.Tensor:
        """
        Convert move index to one-hot encoding.

        Args:
            index: Move index

        Returns:
            torch.FloatTensor of shape (4096,) with 1.0 at index
        """
        onehot = torch.zeros(self.NUM_MOVES)
        onehot[index] = 1.0
        return onehot

    def from_onehot(self, onehot: torch.Tensor) -> int:
        """
        Convert one-hot encoding to index.

        Args:
            onehot: torch.Tensor of shape (4096,)

        Returns:
            Move index
        """
        return int(torch.argmax(onehot).item())

    def get_legal_move_mask(self, board: chess.Board) -> torch.Tensor:
        """
        Get a mask of legal moves for the given position.

        Args:
            board: chess.Board object

        Returns:
            torch.BoolTensor of shape (4096,) with True for legal moves
        """
        mask = torch.zeros(self.NUM_MOVES, dtype=torch.bool)
        for move in board.legal_moves:
            idx = self.encode_move(move)
            mask[idx] = True
        return mask

    def filter_to_legal(self, logits: torch.Tensor, board: chess.Board) -> torch.Tensor:
        """
        Mask logits to only allow legal moves.

        Args:
            logits: Move logits of shape (4096,)
            board: chess.Board object

        Returns:
            Masked logits with -inf for illegal moves
        """
        mask = self.get_legal_move_mask(board)
        masked_logits = logits.clone()
        masked_logits[~mask] = float("-inf")
        return masked_logits

    def get_from_to_squares(self, index: int) -> Tuple[int, int]:
        """
        Get from and to squares from move index.

        Args:
            index: Move index

        Returns:
            Tuple of (from_square, to_square)
        """
        return index // self.NUM_SQUARES, index % self.NUM_SQUARES

    def normalize_move_for_black(self, index: int) -> int:
        """
        Normalize move index when board is flipped for Black.

        When we flip the board for Black's perspective, we also need
        to flip the move coordinates.

        Args:
            index: Original move index

        Returns:
            Flipped move index
        """
        from_sq, to_sq = self.get_from_to_squares(index)

        # Flip squares: a1 <-> h8, etc.
        # Formula: flipped = 63 - original
        flipped_from = 63 - from_sq
        flipped_to = 63 - to_sq

        return flipped_from * self.NUM_SQUARES + flipped_to

    def denormalize_move_for_black(self, index: int) -> int:
        """
        Reverse the normalization for Black's move.

        Args:
            index: Normalized (flipped) move index

        Returns:
            Original move index
        """
        # Flipping is its own inverse
        return self.normalize_move_for_black(index)

    @staticmethod
    def square_to_coords(square: int) -> Tuple[int, int]:
        """
        Convert square index to board coordinates.

        Args:
            square: Square index (0-63)

        Returns:
            Tuple of (row, col) where row 0 is rank 1, col 0 is file a
        """
        return square // 8, square % 8

    @staticmethod
    def coords_to_square(row: int, col: int) -> int:
        """
        Convert board coordinates to square index.

        Args:
            row: Row (0-7, where 0 is rank 1)
            col: Column (0-7, where 0 is file a)

        Returns:
            Square index (0-63)
        """
        return row * 8 + col

    def is_capture(self, board: chess.Board, index: int) -> bool:
        """
        Check if a move is a capture.

        Args:
            board: Current board position
            index: Move index

        Returns:
            True if the move captures a piece
        """
        move = self.decode_to_move(index, board)
        return board.is_capture(move)

    def is_check(self, board: chess.Board, index: int) -> bool:
        """
        Check if a move gives check.

        Args:
            board: Current board position
            index: Move index

        Returns:
            True if the move gives check
        """
        move = self.decode_to_move(index, board)
        board.push(move)
        is_check = board.is_check()
        board.pop()
        return is_check
