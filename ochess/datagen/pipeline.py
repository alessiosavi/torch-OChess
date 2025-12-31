"""
End-to-end data generation pipeline.

Combines position generation and Stockfish analysis to create
fully labeled training data for the chess neural network.

Pipeline steps:
1. Generate diverse chess positions
2. Analyze each position with Stockfish (best move, score)
3. Optionally generate bad move examples
4. Save to efficient parquet format
"""

import chess
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Generator, Dict, Any
from tqdm import tqdm
import random
import logging
import json

from ochess.datagen.stockfish_wrapper import StockfishWrapper, AnalysisResult
from ochess.datagen.position_generator import (
    PositionGenerator,
    GeneratedPosition,
    PositionType
)
from ochess.data.score_encoder import ScoreEncoder

logger = logging.getLogger(__name__)


@dataclass
class LabeledPosition:
    """A chess position with all training labels."""
    # Position
    fen: str
    white_to_move: bool
    move_number: int
    position_type: str

    # Best move (Stockfish)
    best_move_uci: str
    score_cp: int              # Centipawns from White's perspective
    score_normalized: float    # Normalized for training (side-to-move perspective)

    # Move properties
    is_capture: bool
    is_check: bool
    is_mate: bool

    # Alternative good moves (optional)
    alternative_moves: Optional[List[str]] = None
    alternative_scores: Optional[List[int]] = None

    # Bad move for contrastive learning (optional)
    bad_move_uci: Optional[str] = None
    bad_move_score_cp: Optional[int] = None

    # Metadata
    source: str = ""
    game_id: Optional[str] = None


class DataGenerationPipeline:
    """
    End-to-end pipeline for generating labeled training data.

    Features:
    - Multiple position generation strategies (random, openings, endgames)
    - Stockfish analysis for labeling
    - Good and bad move examples
    - Efficient parquet output
    - Progress tracking and resumability

    Usage:
        pipeline = DataGenerationPipeline(
            stockfish_path="/usr/local/bin/stockfish",
            output_dir="datasets/"
        )
        path = pipeline.generate_dataset(
            num_positions=10000,
            output_name="train_data"
        )
    """

    def __init__(
        self,
        stockfish_path: str,
        output_dir: str = "datasets",
        analysis_depth: int = 20,
        stockfish_threads: int = 4,
        stockfish_hash_mb: int = 256,
        include_bad_moves: bool = True,
        multipv: int = 3,
        seed: Optional[int] = None
    ):
        """
        Initialize the data generation pipeline.

        Args:
            stockfish_path: Path to Stockfish binary
            output_dir: Directory for output files
            analysis_depth: Stockfish search depth
            stockfish_threads: Threads for Stockfish
            stockfish_hash_mb: Hash table size
            include_bad_moves: Generate bad move examples
            multipv: Number of top moves to analyze
            seed: Random seed for reproducibility
        """
        self.stockfish_path = stockfish_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analysis_depth = analysis_depth
        self.stockfish_threads = stockfish_threads
        self.stockfish_hash_mb = stockfish_hash_mb
        self.include_bad_moves = include_bad_moves
        self.multipv = multipv
        self.seed = seed

        self.score_encoder = ScoreEncoder(normalize=True, scale=1000.0)
        self.position_generator = PositionGenerator(seed=seed)

    def generate_dataset(
        self,
        num_positions: int,
        output_name: str,
        generation_method: str = "mixed",
        batch_size: int = 100,
        save_intermediate: bool = True
    ) -> Path:
        """
        Generate a complete labeled dataset.

        Args:
            num_positions: Total positions to generate
            output_name: Name for output file (without extension)
            generation_method: "random", "openings", "mixed", or "endgame"
            batch_size: Positions per batch for analysis
            save_intermediate: Save progress every batch

        Returns:
            Path to generated parquet file
        """
        logger.info(f"Generating {num_positions} positions using {generation_method}")

        # Generate positions
        if generation_method == "random":
            positions = list(self.position_generator.generate_random_game_positions(
                num_positions
            ))
        elif generation_method == "openings":
            positions = list(self.position_generator.generate_from_openings(
                num_positions
            ))
        elif generation_method == "endgame":
            positions = list(self.position_generator.generate_endgame_positions(
                num_positions
            ))
        else:  # mixed
            positions = list(self.position_generator.generate_mixed(num_positions))

        logger.info(f"Generated {len(positions)} unique positions")

        # Analyze with Stockfish
        all_labeled = []
        intermediate_path = self.output_dir / f"{output_name}_intermediate.parquet"

        with StockfishWrapper(
            self.stockfish_path,
            threads=self.stockfish_threads,
            hash_mb=self.stockfish_hash_mb
        ) as engine:

            # Process in batches with progress bar
            for i in tqdm(range(0, len(positions), batch_size), desc="Analyzing"):
                batch = positions[i:i + batch_size]
                labeled_batch = self._analyze_batch(engine, batch)
                all_labeled.extend(labeled_batch)

                # Save intermediate results
                if save_intermediate and len(all_labeled) % (batch_size * 10) == 0:
                    self._save_to_parquet(all_labeled, intermediate_path)

        # Final save
        output_path = self.output_dir / f"{output_name}.parquet"
        self._save_to_parquet(all_labeled, output_path)

        # Remove intermediate file
        if intermediate_path.exists():
            intermediate_path.unlink()

        # Save metadata
        self._save_metadata(output_name, len(all_labeled), generation_method)

        logger.info(f"Generated {len(all_labeled)} labeled positions -> {output_path}")
        return output_path

    def _analyze_batch(
        self,
        engine: StockfishWrapper,
        positions: List[GeneratedPosition]
    ) -> List[LabeledPosition]:
        """
        Analyze a batch of positions with Stockfish.

        Args:
            engine: StockfishWrapper instance
            positions: List of positions to analyze

        Returns:
            List of LabeledPosition objects
        """
        labeled = []

        for pos in positions:
            try:
                labeled_pos = self._analyze_single(engine, pos)
                if labeled_pos is not None:
                    labeled.append(labeled_pos)
            except Exception as e:
                logger.warning(f"Failed to analyze position: {e}")
                continue

        return labeled

    def _analyze_single(
        self,
        engine: StockfishWrapper,
        pos: GeneratedPosition
    ) -> Optional[LabeledPosition]:
        """
        Analyze a single position.

        Args:
            engine: StockfishWrapper instance
            pos: Position to analyze

        Returns:
            LabeledPosition or None if analysis fails
        """
        board = chess.Board(pos.fen)

        # Skip terminal positions
        if board.is_game_over():
            return None

        # Analyze position
        results = engine.analyze(
            board,
            depth=self.analysis_depth,
            multipv=self.multipv if self.include_bad_moves else 1
        )

        if not results or results[0].best_move is None:
            return None

        best = results[0]
        best_move = best.best_move

        # Get score normalized for side-to-move
        score_normalized = self.score_encoder.encode_from_centipawns(
            best.score_cp,
            pos.white_to_move
        )

        # Check move properties
        is_capture = board.is_capture(best_move)
        board.push(best_move)
        is_check = board.is_check()
        is_mate = board.is_checkmate()
        board.pop()

        # Get alternative good moves
        alt_moves = None
        alt_scores = None
        if len(results) > 1:
            alt_moves = [r.best_move.uci() for r in results[1:] if r.best_move]
            alt_scores = [r.score_cp for r in results[1:]]

        # Generate bad move example
        bad_move = None
        bad_score = None
        if self.include_bad_moves:
            bad_move, bad_score = self._find_bad_move(engine, board, results)

        return LabeledPosition(
            fen=pos.fen,
            white_to_move=pos.white_to_move,
            move_number=pos.move_number,
            position_type=pos.position_type.value,
            best_move_uci=best_move.uci(),
            score_cp=best.score_cp,
            score_normalized=score_normalized,
            is_capture=is_capture,
            is_check=is_check,
            is_mate=is_mate,
            alternative_moves=alt_moves,
            alternative_scores=alt_scores,
            bad_move_uci=bad_move,
            bad_move_score_cp=bad_score,
            source=pos.source,
            game_id=pos.game_id
        )

    def _find_bad_move(
        self,
        engine: StockfishWrapper,
        board: chess.Board,
        good_results: List[AnalysisResult]
    ) -> tuple:
        """
        Find a bad move for contrastive learning.

        Args:
            engine: StockfishWrapper instance
            board: Current position
            good_results: Analysis results for good moves

        Returns:
            Tuple of (bad_move_uci, bad_move_score_cp) or (None, None)
        """
        # Get set of good moves
        good_moves = {r.best_move for r in good_results if r.best_move}

        # Find moves not in top N
        legal_moves = list(board.legal_moves)
        bad_candidates = [m for m in legal_moves if m not in good_moves]

        if not bad_candidates:
            return None, None

        # Pick a random bad move
        bad_move = random.choice(bad_candidates)

        # Evaluate the bad move
        bad_score = engine.evaluate_move(board, bad_move, depth=10)

        return bad_move.uci(), bad_score

    def _save_to_parquet(
        self,
        labeled: List[LabeledPosition],
        path: Path
    ) -> None:
        """Save labeled positions to parquet file."""
        # Convert to dict format (handle None values and lists)
        records = []
        for pos in labeled:
            record = {
                "fen": pos.fen,
                "white_to_move": pos.white_to_move,
                "move_number": pos.move_number,
                "position_type": pos.position_type,
                "best_move_uci": pos.best_move_uci,
                "score_cp": pos.score_cp,
                "score_normalized": pos.score_normalized,
                "is_capture": pos.is_capture,
                "is_check": pos.is_check,
                "is_mate": pos.is_mate,
                "alternative_moves": json.dumps(pos.alternative_moves) if pos.alternative_moves else None,
                "alternative_scores": json.dumps(pos.alternative_scores) if pos.alternative_scores else None,
                "bad_move_uci": pos.bad_move_uci,
                "bad_move_score_cp": pos.bad_move_score_cp,
                "source": pos.source,
                "game_id": pos.game_id
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.to_parquet(path, compression="zstd", index=False)

    def _save_metadata(
        self,
        output_name: str,
        num_positions: int,
        generation_method: str
    ) -> None:
        """Save dataset metadata."""
        metadata = {
            "name": output_name,
            "num_positions": num_positions,
            "generation_method": generation_method,
            "analysis_depth": self.analysis_depth,
            "include_bad_moves": self.include_bad_moves,
            "multipv": self.multipv,
            "seed": self.seed
        }

        metadata_path = self.output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def generate_training_data(
    stockfish_path: str = "/usr/local/bin/stockfish",
    output_dir: str = "datasets",
    num_positions: int = 10000,
    output_name: str = "train_data",
    seed: Optional[int] = 42
) -> Path:
    """
    Convenience function to generate training data.

    Args:
        stockfish_path: Path to Stockfish binary
        output_dir: Output directory
        num_positions: Number of positions
        output_name: Output file name
        seed: Random seed

    Returns:
        Path to generated parquet file
    """
    pipeline = DataGenerationPipeline(
        stockfish_path=stockfish_path,
        output_dir=output_dir,
        seed=seed
    )

    return pipeline.generate_dataset(
        num_positions=num_positions,
        output_name=output_name
    )
