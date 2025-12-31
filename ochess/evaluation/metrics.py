"""
Evaluation metrics for chess models.
"""

import torch
from typing import List, Dict
from ochess.evaluation.stockfish_eval import GameRecord, GameResult


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute prediction accuracy."""
    pred = predictions.argmax(dim=-1) if predictions.dim() > 1 else predictions
    return (pred == targets).float().mean().item()


def compute_win_rate(records: List[GameRecord]) -> float:
    """Compute win rate from game records."""
    wins = sum(
        1 for r in records
        if (r.result == GameResult.WHITE_WIN) == r.model_color
    )
    return wins / len(records) if records else 0.0


def compute_draw_rate(records: List[GameRecord]) -> float:
    """Compute draw rate from game records."""
    draws = sum(1 for r in records if r.result == GameResult.DRAW)
    return draws / len(records) if records else 0.0


def compute_elo_estimate(win_rate: float, opponent_elo: int = 1500) -> int:
    """
    Estimate Elo rating from win rate against a known opponent.

    Args:
        win_rate: Win rate against opponent (0-1)
        opponent_elo: Opponent's Elo rating

    Returns:
        Estimated Elo rating
    """
    import math

    if win_rate <= 0:
        return opponent_elo - 400
    if win_rate >= 1:
        return opponent_elo + 400

    # Elo formula
    expected_score = win_rate
    elo_diff = -400 * math.log10(1 / expected_score - 1)

    return int(opponent_elo + elo_diff)
