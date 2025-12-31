"""
Residual blocks for chess neural networks.

Provides various residual block implementations used in
AlphaZero-style architectures.
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """
    Basic residual block with two convolutions.

    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU

    This is the standard ResNet-style residual block used in
    AlphaZero and similar chess networks.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0
    ):
        """
        Initialize residual block.

        Args:
            channels: Number of input/output channels
            kernel_size: Convolution kernel size
            dropout: Dropout probability (0 = no dropout)
        """
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block.

    Architecture:
        x -> 1x1 Conv (reduce) -> BN -> ReLU
          -> 3x3 Conv -> BN -> ReLU
          -> 1x1 Conv (expand) -> BN -> (+x) -> ReLU

    More parameter-efficient than basic residual block for
    large channel counts.
    """

    def __init__(
        self,
        channels: int,
        bottleneck_ratio: float = 0.25,
        dropout: float = 0.0
    ):
        """
        Initialize bottleneck block.

        Args:
            channels: Number of input/output channels
            bottleneck_ratio: Ratio for bottleneck channels
            dropout: Dropout probability
        """
        super().__init__()

        bottleneck_channels = int(channels * bottleneck_ratio)

        # Reduce
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # Process
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # Expand
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Channel attention mechanism that learns to weight channels
    based on global information.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize SE block.

        Args:
            channels: Number of channels
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Attention-weighted tensor of shape [B, C, H, W]
        """
        b, c, _, _ = x.size()

        # Squeeze
        y = self.avg_pool(x).view(b, c)

        # Excite
        y = self.fc(y).view(b, c, 1, 1)

        # Scale
        return x * y.expand_as(x)


class SEResidualBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation attention.

    Combines residual learning with channel attention for
    improved feature representation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        se_reduction: int = 16,
        dropout: float = 0.0
    ):
        """
        Initialize SE residual block.

        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            se_reduction: SE reduction ratio
            dropout: Dropout probability
        """
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.se = SEBlock(channels, se_reduction)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE attention
        out = self.se(out)

        out = out + residual
        out = self.relu(out)

        return out


class ResidualTower(nn.Module):
    """
    Stack of residual blocks (tower).

    Used as the main feature extractor in AlphaZero-style networks.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        block_type: str = "basic",
        dropout: float = 0.0
    ):
        """
        Initialize residual tower.

        Args:
            channels: Number of channels
            num_blocks: Number of residual blocks
            block_type: "basic", "bottleneck", or "se"
            dropout: Dropout probability
        """
        super().__init__()

        blocks = []
        for _ in range(num_blocks):
            if block_type == "basic":
                blocks.append(ResidualBlock(channels, dropout=dropout))
            elif block_type == "bottleneck":
                blocks.append(BottleneckBlock(channels, dropout=dropout))
            elif block_type == "se":
                blocks.append(SEResidualBlock(channels, dropout=dropout))
            else:
                raise ValueError(f"Unknown block type: {block_type}")

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all blocks.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        return self.blocks(x)
