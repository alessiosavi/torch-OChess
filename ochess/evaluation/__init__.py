"""
Evaluation module for Torch o'Chess.

Provides tools to evaluate trained models:
    - StockfishEvaluator: Play games against Stockfish
    - Tournament: Run multi-game matches
    - Metrics: Move accuracy, win rate, etc.
"""

from ochess.evaluation.metrics import compute_accuracy, compute_win_rate
from ochess.evaluation.stockfish_eval import (GameRecord, GameResult,
                                              StockfishEvaluator)

__all__ = [
    "StockfishEvaluator",
    "GameResult",
    "GameRecord",
    "compute_accuracy",
    "compute_win_rate",
]
