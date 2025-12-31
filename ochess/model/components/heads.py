"""
Output heads for chess neural networks.

Provides specialized output heads for different prediction tasks:
    - MoveHead: Predicts the next move (4096 classes)
    - ScoreHead: Predicts position evaluation
    - CaptureHead: Predicts if move is a capture
    - OutcomeHead: Predicts game outcome (win/draw/loss)
"""

import torch
import torch.nn as nn
from typing import Optional


class MoveHead(nn.Module):
    """
    Output head for move prediction.

    Predicts one of 4096 possible moves (64 from-squares x 64 to-squares).
    Uses a policy network style architecture with convolution + FC.
    """

    def __init__(
        self,
        in_channels: int,
        num_moves: int = 4096,
        hidden_dim: int = 256
    ):
        """
        Initialize move head.

        Args:
            in_channels: Number of input channels
            num_moves: Number of possible moves (default 4096)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.num_moves = num_moves

        # Policy convolution
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Flatten and project
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict move logits.

        Args:
            x: Features tensor [B, C, H, W]

        Returns:
            Move logits [B, num_moves]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ScoreHead(nn.Module):
    """
    Output head for position score prediction.

    Predicts a single scalar value representing the
    position evaluation from side-to-move perspective.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128
    ):
        """
        Initialize score head.

        Args:
            in_channels: Number of input channels
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # MLP for score prediction
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict position score.

        Args:
            x: Features tensor [B, C, H, W]

        Returns:
            Score tensor [B, 1]
        """
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CaptureHead(nn.Module):
    """
    Output head for capture classification.

    Binary classification: is the best move a capture?
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2
    ):
        """
        Initialize capture head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of classes (2 for binary)
        """
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict capture probability.

        Args:
            x: Features tensor [B, C, H, W]

        Returns:
            Capture logits [B, num_classes]
        """
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class OutcomeHead(nn.Module):
    """
    Output head for game outcome prediction.

    Predicts win/draw/loss from current side's perspective.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        hidden_dim: int = 64
    ):
        """
        Initialize outcome head.

        Args:
            in_channels: Number of input channels
            num_classes: Number of outcome classes (3: win/draw/loss)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict game outcome.

        Args:
            x: Features tensor [B, C, H, W]

        Returns:
            Outcome logits [B, num_classes]
        """
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ValueHead(nn.Module):
    """
    Combined value head for score and outcome.

    Similar to AlphaZero's value head, predicts both
    a scalar value and win/draw/loss probabilities.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256
    ):
        """
        Initialize value head.

        Args:
            in_channels: Number of input channels
            hidden_dim: Hidden dimension for shared layers
        """
        super().__init__()

        # Shared convolution
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Flatten
        self.flatten = nn.Flatten()

        # Shared FC
        self.fc_shared = nn.Linear(32 * 8 * 8, hidden_dim)

        # Separate heads
        self.fc_score = nn.Linear(hidden_dim, 1)
        self.fc_outcome = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Predict value and outcome.

        Args:
            x: Features tensor [B, C, H, W]

        Returns:
            Tuple of (score [B, 1], outcome [B, 3])
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.relu(self.fc_shared(x))

        score = torch.tanh(self.fc_score(x))  # Bounded [-1, 1]
        outcome = self.fc_outcome(x)

        return score, outcome


class PolicyHead(nn.Module):
    """
    Policy head with move and capture prediction.

    Combines move prediction with capture classification
    in a shared architecture.
    """

    def __init__(
        self,
        in_channels: int,
        num_moves: int = 4096,
        hidden_dim: int = 256
    ):
        """
        Initialize policy head.

        Args:
            in_channels: Number of input channels
            num_moves: Number of possible moves
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Shared convolution
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Flatten
        self.flatten = nn.Flatten()

        # Shared FC
        self.fc_shared = nn.Linear(64 * 8 * 8, hidden_dim)

        # Move prediction
        self.fc_move = nn.Linear(hidden_dim, num_moves)

        # Capture prediction
        self.fc_capture = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Predict move and capture.

        Args:
            x: Features tensor [B, C, H, W]

        Returns:
            Tuple of (move_logits [B, num_moves], capture_logits [B, 2])
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.relu(self.fc_shared(x))

        move = self.fc_move(x)
        capture = self.fc_capture(x)

        return move, capture


class MultiTaskHead(nn.Module):
    """
    Combined multi-task output head.

    Predicts all outputs in a single forward pass with
    shared feature processing.
    """

    def __init__(
        self,
        in_channels: int,
        num_moves: int = 4096,
        shared_dim: int = 512
    ):
        """
        Initialize multi-task head.

        Args:
            in_channels: Number of input channels
            num_moves: Number of possible moves
            shared_dim: Shared hidden dimension
        """
        super().__init__()

        # Shared processing
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # For policy (uses spatial features)
        self.policy_fc = nn.Linear(64 * 8 * 8, shared_dim)

        # For value (uses pooled features)
        self.value_fc = nn.Linear(64, shared_dim // 2)

        # Output heads
        self.move_out = nn.Linear(shared_dim, num_moves)
        self.score_out = nn.Linear(shared_dim // 2, 1)
        self.capture_out = nn.Linear(shared_dim // 2, 2)
        self.outcome_out = nn.Linear(shared_dim // 2, 3)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Predict all outputs.

        Args:
            x: Features tensor [B, C, H, W]

        Returns:
            Dictionary with 'move_logits', 'score', 'capture', 'outcome'
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Policy path (spatial)
        policy_feat = self.flatten(x)
        policy_feat = self.relu(self.policy_fc(policy_feat))

        # Value path (pooled)
        value_feat = self.global_pool(x).view(x.size(0), -1)
        value_feat = self.relu(self.value_fc(value_feat))

        return {
            'move_logits': self.move_out(policy_feat),
            'score': self.score_out(value_feat),
            'capture': self.capture_out(value_feat),
            'outcome': self.outcome_out(value_feat)
        }
