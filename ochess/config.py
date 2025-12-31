"""
Configuration management for Torch o'Chess.

Provides centralized configuration for all modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    """Configuration for data processing."""

    sequence_length: int = 5  # Number of past positions to consider
    cache_dir: str = "cache"


@dataclass
class DataGenConfig:
    """Configuration for data generation."""

    stockfish_path: str = "/usr/local/bin/stockfish"
    stockfish_threads: int = 4
    stockfish_hash_mb: int = 256
    analysis_depth: int = 20
    num_positions: int = 10000
    output_dir: str = "datasets"
    include_bad_moves: bool = True
    multipv: int = 3  # Number of top moves to analyze


@dataclass
class ModelConfig:
    """Configuration for neural network models."""

    # Embedding dimensions
    piece_embed_dim: int = 64
    position_embed_dim: int = 64
    hidden_dim: int = 256

    # Architecture
    num_residual_blocks: int = 8
    num_transformer_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.3

    # Output
    num_moves: int = 4096  # 64 * 64 possible from-to combinations

    # Input
    sequence_length: int = 5
    flip_board_for_black: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100

    # Loss weights
    move_loss_weight: float = 3.0
    score_loss_weight: float = 0.5
    capture_loss_weight: float = 1.0
    outcome_loss_weight: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5

    # Early stopping
    early_stopping_patience: int = 10

    # Hardware
    device: str = "cuda"  # or "cpu" or "mps"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    stockfish_path: str = "/usr/local/bin/stockfish"
    num_games: int = 100
    stockfish_levels: List[int] = field(default_factory=lambda: [1, 5, 10, 15, 20])
    time_limit: float = 0.1  # seconds per move for Stockfish


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    datagen: DataGenConfig = field(default_factory=DataGenConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            data=DataConfig(**data.get("data", {})),
            datagen=DataGenConfig(**data.get("datagen", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        from dataclasses import asdict

        data = {
            "data": asdict(self.data),
            "datagen": asdict(self.datagen),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
