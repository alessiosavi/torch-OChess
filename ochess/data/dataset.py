"""
PyTorch Dataset for chess training data.

Key features:
    - Pre-computed tensor caching (no FEN parsing during training)
    - Efficient data loading
    - Support for sequences of positions
    - Automatic train/val splitting
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

from ochess.data.fen_parser import FenParser
from ochess.data.move_encoder import MoveEncoder
from ochess.data.score_encoder import ScoreEncoder

logger = logging.getLogger(__name__)


class ChessDataset(Dataset):
    """
    Chess training dataset with pre-computed tensors.

    This dataset pre-processes all FEN strings to tensors during
    initialization, avoiding the CPU bottleneck during training.

    Features:
        - Pre-computed board tensors
        - Caching to disk for fast loading
        - Support for temporal sequences
        - Multiple target outputs
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 5,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize chess dataset.

        Args:
            data_path: Path to parquet data file
            sequence_length: Number of positions per sequence
            cache_dir: Directory for cached tensors
            use_cache: Whether to use disk caching
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.use_cache = use_cache

        # Initialize encoders
        self.fen_parser = FenParser()
        self.move_encoder = MoveEncoder()
        self.score_encoder = ScoreEncoder(normalize=True)

        # Set cache path
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.data_path.parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_name = f"{self.data_path.stem}_seq{sequence_length}.pt"
        self.cache_path = self.cache_dir / cache_name

        # Load or preprocess data
        if use_cache and self.cache_path.exists():
            logger.info(f"Loading cached data from {self.cache_path}")
            self._load_cache()
        else:
            logger.info(f"Preprocessing data from {self.data_path}")
            self._preprocess_data()
            if use_cache:
                self._save_cache()

    def _preprocess_data(self):
        """Convert raw parquet data to tensors."""
        # Load parquet
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(df)} positions from parquet")

        # Group by game_id if available
        if 'game_id' in df.columns:
            self._preprocess_grouped(df)
        else:
            self._preprocess_ungrouped(df)

    def _preprocess_grouped(self, df: pd.DataFrame):
        """Preprocess data grouped by game."""
        sequences = []

        for game_id, game_df in df.groupby('game_id'):
            # Sort by move number
            game_df = game_df.sort_values('move_number')

            # Create sliding windows
            for i in range(len(game_df) - self.sequence_length + 1):
                window = game_df.iloc[i:i + self.sequence_length]
                seq = self._process_window(window)
                if seq is not None:
                    sequences.append(seq)

        self._finalize_sequences(sequences)

    def _preprocess_ungrouped(self, df: pd.DataFrame):
        """Preprocess data without game grouping (single positions)."""
        sequences = []

        for idx, row in df.iterrows():
            seq = self._process_single(row)
            if seq is not None:
                sequences.append(seq)

        self._finalize_sequences(sequences)

    def _process_window(self, window: pd.DataFrame) -> Optional[Dict]:
        """Process a window of positions."""
        try:
            # Convert FENs to tensors
            boards = []
            colors = []

            for _, row in window.iterrows():
                board = self.fen_parser.to_tensor(row['fen'])
                boards.append(board)
                colors.append(0 if self.fen_parser.is_white_to_move(row['fen']) else 1)

            boards = torch.stack(boards)  # [seq_len, 8, 8]
            colors = torch.tensor(colors, dtype=torch.long)

            # Get target from last position
            last_row = window.iloc[-1]

            # Target move
            target_move = self.move_encoder.encode(last_row['best_move_uci'])

            # Score (already normalized in data generation)
            score = last_row['score_normalized']

            # Capture flag
            is_capture = int(last_row['is_capture'])

            # Outcome (if available)
            outcome = None
            if 'outcome' in last_row and pd.notna(last_row.get('outcome')):
                outcome = int(last_row['outcome'])

            return {
                'boards': boards,
                'colors': colors,
                'target_move': target_move,
                'score': score,
                'is_capture': is_capture,
                'outcome': outcome
            }

        except Exception as e:
            logger.warning(f"Failed to process window: {e}")
            return None

    def _process_single(self, row: pd.Series) -> Optional[Dict]:
        """Process a single position (replicated for sequence)."""
        try:
            board = self.fen_parser.to_tensor(row['fen'])
            color = 0 if self.fen_parser.is_white_to_move(row['fen']) else 1

            # Replicate for sequence
            boards = board.unsqueeze(0).expand(self.sequence_length, -1, -1).clone()
            colors = torch.tensor([color] * self.sequence_length, dtype=torch.long)

            target_move = self.move_encoder.encode(row['best_move_uci'])
            score = row['score_normalized']
            is_capture = int(row['is_capture'])

            outcome = None
            if 'outcome' in row and pd.notna(row.get('outcome')):
                outcome = int(row['outcome'])

            return {
                'boards': boards,
                'colors': colors,
                'target_move': target_move,
                'score': score,
                'is_capture': is_capture,
                'outcome': outcome
            }

        except Exception as e:
            logger.warning(f"Failed to process row: {e}")
            return None

    def _finalize_sequences(self, sequences: List[Dict]):
        """Stack all sequences into tensors."""
        if not sequences:
            raise ValueError("No valid sequences found in data")

        self.boards = torch.stack([s['boards'] for s in sequences])
        self.colors = torch.stack([s['colors'] for s in sequences])
        self.target_moves = torch.tensor([s['target_move'] for s in sequences], dtype=torch.long)
        self.scores = torch.tensor([s['score'] for s in sequences], dtype=torch.float32)
        self.is_captures = torch.tensor([s['is_capture'] for s in sequences], dtype=torch.long)

        # Handle optional outcomes
        has_outcomes = all(s['outcome'] is not None for s in sequences)
        if has_outcomes:
            self.outcomes = torch.tensor([s['outcome'] for s in sequences], dtype=torch.long)
        else:
            self.outcomes = None

        self.length = len(sequences)
        logger.info(f"Preprocessed {self.length} sequences")

    def _save_cache(self):
        """Save preprocessed data to disk."""
        data = {
            'boards': self.boards,
            'colors': self.colors,
            'target_moves': self.target_moves,
            'scores': self.scores,
            'is_captures': self.is_captures,
            'outcomes': self.outcomes,
            'length': self.length,
            'sequence_length': self.sequence_length
        }
        torch.save(data, self.cache_path)
        logger.info(f"Saved cache to {self.cache_path}")

    def _load_cache(self):
        """Load preprocessed data from disk."""
        data = torch.load(self.cache_path)
        self.boards = data['boards']
        self.colors = data['colors']
        self.target_moves = data['target_moves']
        self.scores = data['scores']
        self.is_captures = data['is_captures']
        self.outcomes = data['outcomes']
        self.length = data['length']

        cached_seq_len = data.get('sequence_length', self.sequence_length)
        if cached_seq_len != self.sequence_length:
            logger.warning(
                f"Cache sequence length ({cached_seq_len}) differs from "
                f"requested ({self.sequence_length}). Using cached value."
            )
            self.sequence_length = cached_seq_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - 'boards': [seq_len, 8, 8]
            - 'colors': [seq_len]
            - 'target_move': scalar
            - 'score': scalar
            - 'is_capture': scalar
            - 'outcome': scalar (optional)
        """
        item = {
            'boards': self.boards[idx],
            'colors': self.colors[idx],
            'target_move': self.target_moves[idx],
            'score': self.scores[idx],
            'is_capture': self.is_captures[idx]
        }

        if self.outcomes is not None:
            item['outcome'] = self.outcomes[idx]

        return item


def create_dataloader(
    dataset: ChessDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create DataLoader for chess dataset.

    Args:
        dataset: ChessDataset instance
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
        drop_last: Drop incomplete last batch

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )


def train_val_split(
    dataset: ChessDataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Full dataset
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from torch.utils.data import Subset

    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
