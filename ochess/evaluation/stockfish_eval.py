"""
Evaluate chess model by playing against Stockfish.
"""

import chess
import chess.engine
import torch
import torch.nn as nn
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import random
import logging

from ochess.data.fen_parser import FenParser
from ochess.data.move_encoder import MoveEncoder

logger = logging.getLogger(__name__)


class GameResult(Enum):
    WHITE_WIN = "1-0"
    BLACK_WIN = "0-1"
    DRAW = "1/2-1/2"


@dataclass
class GameRecord:
    """Record of a single game."""
    result: GameResult
    moves: List[str]
    termination: str
    model_color: bool  # True = White
    stockfish_level: int
    model_illegal_moves: int
    total_moves: int


class StockfishEvaluator:
    """
    Evaluate model by playing games against Stockfish.
    """

    def __init__(
        self,
        model: nn.Module,
        stockfish_path: str,
        device: str = "cuda",
        sequence_length: int = 5
    ):
        self.model = model
        self.model.eval()
        self.stockfish_path = stockfish_path
        self.device = torch.device(device)
        self.sequence_length = sequence_length

        self.fen_parser = FenParser()
        self.move_encoder = MoveEncoder()

    def play_match(
        self,
        num_games: int = 100,
        stockfish_levels: List[int] = [1, 5, 10, 15, 20],
        model_plays_white: Optional[bool] = None
    ) -> Dict:
        """
        Play a match of games against Stockfish.

        Args:
            num_games: Games per level
            stockfish_levels: Stockfish skill levels to test
            model_plays_white: None = alternate colors

        Returns:
            Results dictionary
        """
        results = {level: [] for level in stockfish_levels}

        for level in stockfish_levels:
            logger.info(f"Playing {num_games} games vs Stockfish level {level}")

            for game_num in range(num_games):
                if model_plays_white is None:
                    model_white = game_num % 2 == 0
                else:
                    model_white = model_plays_white

                record = self._play_single_game(level, model_white)
                results[level].append(record)

            stats = self._compute_stats(results[level])
            logger.info(
                f"  Level {level}: W={stats['wins']} D={stats['draws']} "
                f"L={stats['losses']} ({stats['win_rate']*100:.1f}%)"
            )

        return results

    def _play_single_game(
        self,
        stockfish_level: int,
        model_plays_white: bool
    ) -> GameRecord:
        """Play a single game."""
        board = chess.Board()
        moves = []
        illegal_moves = 0
        history = []

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as sf:
            sf.configure({"Skill Level": stockfish_level})

            while not board.is_game_over() and len(moves) < 500:
                is_model_turn = (board.turn == chess.WHITE) == model_plays_white

                if is_model_turn:
                    move, was_illegal = self._get_model_move(board, history)
                    if was_illegal:
                        illegal_moves += 1
                else:
                    result = sf.play(board, chess.engine.Limit(time=0.1))
                    move = result.move

                moves.append(move.uci())
                history.append(board.fen())
                board.push(move)

        result = self._get_result(board)

        return GameRecord(
            result=result,
            moves=moves,
            termination=str(board.outcome().termination) if board.outcome() else "unknown",
            model_color=model_plays_white,
            stockfish_level=stockfish_level,
            model_illegal_moves=illegal_moves,
            total_moves=len(moves)
        )

    def _get_model_move(
        self,
        board: chess.Board,
        history: List[str]
    ) -> tuple:
        """Get move from model."""
        # Prepare input
        fens = history[-self.sequence_length:] if len(history) >= self.sequence_length else history
        while len(fens) < self.sequence_length:
            fens = [board.fen()] + fens

        # Include current position
        fens = fens[-(self.sequence_length-1):] + [board.fen()]

        # Convert to tensors
        boards = torch.stack([
            self.fen_parser.to_tensor(fen) for fen in fens
        ]).unsqueeze(0)  # [1, seq, 8, 8]

        colors = torch.tensor([
            0 if self.fen_parser.is_white_to_move(fen) else 1
            for fen in fens
        ]).unsqueeze(0)  # [1, seq]

        # Model inference
        with torch.no_grad():
            boards = boards.to(self.device)
            colors = colors.to(self.device)
            outputs = self.model(boards, colors)
            logits = outputs['move_logits'][0]

        # Get best legal move
        move, was_illegal = self._select_legal_move(board, logits)
        return move, was_illegal

    def _select_legal_move(
        self,
        board: chess.Board,
        logits: torch.Tensor
    ) -> tuple:
        """Select best legal move from logits."""
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None, True

        # Score each legal move
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            idx = self.move_encoder.encode_move(move)
            score = logits[idx].item()

            if score > best_score:
                best_score = score
                best_move = move

        # Check if the top predicted move was legal
        top_pred = logits.argmax().item()
        top_move_uci = self.move_encoder.decode(top_pred)
        try:
            top_move = chess.Move.from_uci(top_move_uci)
            was_illegal = top_move not in legal_moves
        except:
            was_illegal = True

        return best_move, was_illegal

    def _get_result(self, board: chess.Board) -> GameResult:
        """Get game result."""
        outcome = board.outcome()
        if outcome is None:
            return GameResult.DRAW
        if outcome.winner is None:
            return GameResult.DRAW
        return GameResult.WHITE_WIN if outcome.winner else GameResult.BLACK_WIN

    def _compute_stats(self, records: List[GameRecord]) -> Dict:
        """Compute statistics from game records."""
        wins = draws = losses = 0

        for record in records:
            if record.result == GameResult.DRAW:
                draws += 1
            elif (record.result == GameResult.WHITE_WIN) == record.model_color:
                wins += 1
            else:
                losses += 1

        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': wins / len(records) if records else 0,
            'total_illegal_moves': sum(r.model_illegal_moves for r in records)
        }


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute prediction accuracy."""
    pred = predictions.argmax(dim=-1)
    return (pred == targets).float().mean().item()


def compute_win_rate(records: List[GameRecord]) -> float:
    """Compute win rate from game records."""
    wins = sum(
        1 for r in records
        if (r.result == GameResult.WHITE_WIN) == r.model_color
    )
    return wins / len(records) if records else 0.0
