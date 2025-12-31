"""
Attention mechanisms for chess neural networks.

Provides:
    - SelfAttention: Basic self-attention for spatial features
    - MultiHeadAttention: Multi-head attention with projections
    - TransformerBlock: Full transformer encoder block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for 2D spatial features.

    Applies attention over the spatial dimensions (H, W) of the
    feature map, allowing the model to learn long-range dependencies.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        """
        Initialize self-attention.

        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # Query, Key, Value projections
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W  # Number of spatial positions

        # Get Q, K, V
        qkv = self.qkv(x)  # [B, 3C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(2, 3).reshape(B, C, H, W)

        # Output projection
        out = self.proj(out)

        return x + out  # Residual connection


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for sequence data.

    Standard transformer-style attention that can be used
    with flattened spatial features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention.

        Args:
            query: Query tensor [B, seq_len, embed_dim]
            key: Key tensor (defaults to query for self-attention)
            value: Value tensor (defaults to query for self-attention)
            mask: Optional attention mask

        Returns:
            Output tensor [B, seq_len, embed_dim]
        """
        if key is None:
            key = query
        if value is None:
            value = query

        B, T, E = query.shape

        # Project Q, K, V
        q = self.q_proj(query)  # [B, T, E]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = attn @ v  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        # Output projection
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.

    Consists of:
    1. Multi-head self-attention with residual
    2. Feed-forward network with residual
    3. Layer normalization
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [B, seq_len, embed_dim]
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), mask=mask)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """
    Stack of transformer blocks.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        """
        Initialize transformer encoder.

        Args:
            embed_dim: Embedding dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
        """
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor [B, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [B, seq_len, embed_dim]
        """
        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.norm(x)


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Computes attention weights for spatial positions
    based on channel statistics.
    """

    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention.

        Args:
            kernel_size: Convolution kernel size
        """
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, H, W]
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(combined)
        attn = self.sigmoid(attn)

        return x * attn
