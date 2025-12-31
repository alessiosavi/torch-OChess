"""
Loss functions for chess neural network training.

Provides multi-task loss functions that combine:
    - Move prediction (CrossEntropy)
    - Score prediction (Huber/MSE)
    - Capture classification (CrossEntropy)
    - Outcome prediction (CrossEntropy)

The losses are weighted to balance the different tasks.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoveLoss(nn.Module):
    """
    Loss for move prediction.

    Uses CrossEntropy with optional label smoothing.
    """

    def __init__(self, label_smoothing: float = 0.0):
        """
        Initialize move loss.

        Args:
            label_smoothing: Amount of label smoothing (0-1)
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute move prediction loss.

        Args:
            logits: Predicted move logits [B, 4096]
            targets: Target move indices [B] or one-hot [B, 4096]

        Returns:
            Scalar loss value
        """
        # Handle one-hot targets
        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=-1)

        return self.ce(logits, targets)


class ScoreLoss(nn.Module):
    """
    Loss for score prediction.

    Uses Huber loss to be robust to outliers (extreme positions).
    Score is from side-to-move perspective, so positive = good.
    """

    def __init__(self, delta: float = 2.0):
        """
        Initialize score loss.

        Args:
            delta: Huber loss delta (transitions from L2 to L1)
        """
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute score prediction loss.

        Args:
            predicted: Predicted scores [B, 1] or [B]
            target: Target scores [B]
            weights: Optional per-sample weights [B]

        Returns:
            Scalar loss value
        """
        predicted = predicted.squeeze(-1)

        if weights is not None:
            # Weighted loss
            loss = F.huber_loss(
                predicted, target, reduction="none", delta=self.huber.delta
            )
            loss = (loss * weights).sum() / weights.sum()
        else:
            loss = self.huber(predicted, target)

        return loss


class CaptureLoss(nn.Module):
    """
    Loss for capture classification.

    Binary classification: is the move a capture?
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute capture classification loss.

        Args:
            logits: Predicted logits [B, 2]
            targets: Target labels [B] (0 or 1)

        Returns:
            Scalar loss value
        """
        # Handle one-hot targets
        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=-1)

        return self.ce(logits, targets.long())


class OutcomeLoss(nn.Module):
    """
    Loss for game outcome prediction.

    Three-class classification: win/draw/loss from current player's view.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute outcome prediction loss.

        Args:
            logits: Predicted logits [B, 3]
            targets: Target labels [B] (0=loss, 1=draw, 2=win)

        Returns:
            Scalar loss value
        """
        # Handle one-hot targets
        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=-1)

        return self.ce(logits, targets.long())


class ChessLoss(nn.Module):
    """
    Combined multi-task loss for chess neural network.

    Combines losses for:
    - Move prediction (most important)
    - Score prediction (auxiliary)
    - Capture classification (auxiliary)
    - Outcome prediction (auxiliary)

    Loss weights control the relative importance of each task.
    Default weights emphasize move prediction (primary objective).
    """

    def __init__(
        self,
        move_weight: float = 3.0,
        score_weight: float = 0.5,
        capture_weight: float = 1.0,
        outcome_weight: float = 1.0,
        label_smoothing: float = 0.0,
        score_delta: float = 2.0,
    ):
        """
        Initialize combined loss.

        Args:
            move_weight: Weight for move prediction loss
            score_weight: Weight for score prediction loss
            capture_weight: Weight for capture classification loss
            outcome_weight: Weight for outcome prediction loss
            label_smoothing: Label smoothing for move prediction
            score_delta: Huber loss delta for score prediction
        """
        super().__init__()

        self.move_weight = move_weight
        self.score_weight = score_weight
        self.capture_weight = capture_weight
        self.outcome_weight = outcome_weight

        self.move_loss = MoveLoss(label_smoothing)
        self.score_loss = ScoreLoss(score_delta)
        self.capture_loss = CaptureLoss()
        self.outcome_loss = OutcomeLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs dict with keys:
                - 'move_logits': [B, 4096]
                - 'score': [B, 1]
                - 'capture': [B, 2]
                - 'outcome': [B, 3]
            targets: Target dict with keys:
                - 'target_move': [B]
                - 'score': [B]
                - 'is_capture': [B]
                - 'outcome': [B] (optional)
            return_components: If True, return dict of individual losses

        Returns:
            Scalar total loss (or dict if return_components=True)
        """
        # Move loss
        move_l = self.move_loss(outputs["move_logits"], targets["target_move"])

        # Score loss
        score_l = self.score_loss(outputs["score"], targets["score"])

        # Capture loss
        capture_l = self.capture_loss(outputs["capture"], targets["is_capture"])

        # Outcome loss (optional)
        if "outcome" in targets and targets["outcome"] is not None:
            outcome_l = self.outcome_loss(outputs["outcome"], targets["outcome"])
        else:
            outcome_l = torch.tensor(0.0, device=move_l.device)

        # Weighted combination
        total_loss = (
            self.move_weight * move_l
            + self.score_weight * score_l
            + self.capture_weight * capture_l
            + self.outcome_weight * outcome_l
        )

        if return_components:
            return {
                "total": total_loss,
                "move": move_l,
                "score": score_l,
                "capture": capture_l,
                "outcome": outcome_l,
            }

        return total_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning to distinguish good vs bad moves.

    Pushes the model to rank good moves higher than bad moves.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.

        Args:
            margin: Minimum margin between good and bad move scores
        """
        super().__init__()
        self.margin = margin

    def forward(
        self, good_move_logits: torch.Tensor, bad_move_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            good_move_logits: Logits for good moves [B, 4096]
            bad_move_logits: Logits for bad moves [B, 4096]

        Returns:
            Scalar loss value
        """
        # Get max logit for each
        good_score = good_move_logits.max(dim=-1).values
        bad_score = bad_move_logits.max(dim=-1).values

        # Hinge loss: good should be margin above bad
        loss = F.relu(self.margin - (good_score - bad_score))

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in move prediction.

    Down-weights well-classified examples to focus on hard cases.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        """
        Initialize focal loss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights [num_classes]
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits [B, C]
            targets: Target indices [B]

        Returns:
            Scalar loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()
