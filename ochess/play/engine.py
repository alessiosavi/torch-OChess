"""
Chess engine wrapper for model inference.
"""

from typing import List, Optional, Tuple

import chess
import torch
import torch.nn as nn

from ochess.data.fen_parser import FenParser
from ochess.data.move_encoder import MoveEncoder


class ChessEngine:
    """
    Chess engine using trained neural network.

    Provides UCI-compatible interface for playing chess.
    """

    def __init__(
        self, model: nn.Module, device: str = "cuda", sequence_length: int = 5
    ):
        """
        Initialize chess engine.

        Args:
            model: Trained neural network
            device: Device for inference
            sequence_length: Number of positions for temporal context
        """
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.sequence_length = sequence_length

        self.fen_parser = FenParser()
        self.move_encoder = MoveEncoder()

        # Game state
        self.board = chess.Board()
        self.history: List[str] = []

    def new_game(self):
        """Start a new game."""
        self.board = chess.Board()
        self.history = []

    def set_position(self, fen: str):
        """Set board to specific position."""
        self.board = chess.Board(fen)
        self.history = [fen]

    def make_move(self, move: chess.Move) -> bool:
        """
        Make a move on the board.

        Args:
            move: Move to make

        Returns:
            True if move was legal
        """
        if move in self.board.legal_moves:
            self.history.append(self.board.fen())
            self.board.push(move)
            return True
        return False

    def get_move(
        self, board: Optional[chess.Board] = None, temperature: float = 0.1
    ) -> Tuple[chess.Move, bool]:
        """
        Get best move for current position.

        Args:
            board: Optional board (uses internal if None)
            temperature: Softmax temperature

        Returns:
            Tuple of (move, was_illegal_prediction)
        """
        if board is None:
            board = self.board

        # Prepare input sequence
        current_fen = board.fen()
        fens = self.history[-(self.sequence_length - 1) :] + [current_fen]
        while len(fens) < self.sequence_length:
            fens = [current_fen] + fens

        # Convert to tensors
        boards = torch.stack(
            [self.fen_parser.to_tensor(fen) for fen in fens]
        ).unsqueeze(0)

        colors = torch.tensor(
            [0 if self.fen_parser.is_white_to_move(fen) else 1 for fen in fens]
        ).unsqueeze(0)

        # Inference
        with torch.no_grad():
            boards = boards.to(self.device)
            colors = colors.to(self.device)
            outputs = self.model(boards, colors)
            logits = outputs["move_logits"][0]

            if temperature != 1.0:
                logits = logits / temperature

        # Select best legal move
        move, was_illegal = self._select_legal_move(board, logits)
        return move, was_illegal

    def _select_legal_move(
        self, board: chess.Board, logits: torch.Tensor
    ) -> Tuple[chess.Move, bool]:
        """Select best legal move from logits."""
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None, True

        # Find best legal move
        best_move = None
        best_score = float("-inf")

        for move in legal_moves:
            idx = self.move_encoder.encode_move(move)
            score = logits[idx].item()
            if score > best_score:
                best_score = score
                best_move = move

        # Check if top prediction was legal
        top_idx = logits.argmax().item()
        top_uci = self.move_encoder.decode(top_idx)
        try:
            top_move = chess.Move.from_uci(top_uci)
            was_illegal = top_move not in legal_moves
        except:
            was_illegal = True

        return best_move, was_illegal

    def get_evaluation(self, board: Optional[chess.Board] = None) -> float:
        """
        Get position evaluation from model.

        Args:
            board: Optional board

        Returns:
            Score (positive = good for side to move)
        """
        if board is None:
            board = self.board

        current_fen = board.fen()
        fens = [current_fen] * self.sequence_length

        boards = torch.stack(
            [self.fen_parser.to_tensor(fen) for fen in fens]
        ).unsqueeze(0)

        colors = torch.tensor(
            [0 if self.fen_parser.is_white_to_move(fen) else 1 for fen in fens]
        ).unsqueeze(0)

        with torch.no_grad():
            boards = boards.to(self.device)
            colors = colors.to(self.device)
            outputs = self.model(boards, colors)
            score = outputs["score"][0].item()

        return score
