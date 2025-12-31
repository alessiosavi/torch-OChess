"""
Training metrics for chess neural networks.
"""

from typing import Dict, Optional

import torch


class AccuracyMetric:
    """Track move prediction accuracy."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        pred_indices = (
            predictions.argmax(dim=-1) if predictions.dim() > 1 else predictions
        )
        target_indices = targets.argmax(dim=-1) if targets.dim() > 1 else targets

        self.correct += (pred_indices == target_indices).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def compute_move_accuracy(
    outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
) -> float:
    """
    Compute move prediction accuracy.

    Args:
        outputs: Model outputs with 'move_logits'
        targets: Target dict with 'target_move'

    Returns:
        Accuracy as float
    """
    pred = outputs["move_logits"].argmax(dim=-1)
    target = targets["target_move"]

    correct = (pred == target).sum().item()
    total = target.size(0)

    return correct / total if total > 0 else 0.0


def compute_top_k_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Prediction logits [B, num_classes]
        targets: Target indices [B]
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy
    """
    _, top_k_preds = logits.topk(k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=-1).sum().item()

    return correct / targets.size(0) if targets.size(0) > 0 else 0.0
