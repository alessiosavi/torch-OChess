#!/usr/bin/env python3
"""
Generate training data using Stockfish.

Usage:
    python scripts/generate_data.py --num-positions 10000 --output train_data
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ochess.datagen.pipeline import DataGenerationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate chess training data using Stockfish"
    )

    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="/usr/local/bin/stockfish",
        help="Path to Stockfish binary"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Output directory for data files"
    )
    parser.add_argument(
        "--num-positions",
        type=int,
        default=10000,
        help="Number of positions to generate"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="train_data",
        help="Name for output file"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "openings", "mixed", "endgame"],
        default="mixed",
        help="Position generation method"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Stockfish analysis depth"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of Stockfish threads"
    )
    parser.add_argument(
        "--no-bad-moves",
        action="store_true",
        help="Don't include bad move examples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create pipeline
    pipeline = DataGenerationPipeline(
        stockfish_path=args.stockfish_path,
        output_dir=args.output_dir,
        analysis_depth=args.depth,
        stockfish_threads=args.threads,
        include_bad_moves=not args.no_bad_moves,
        seed=args.seed
    )

    # Generate data
    output_path = pipeline.generate_dataset(
        num_positions=args.num_positions,
        output_name=args.output_name,
        generation_method=args.method
    )

    print(f"\nData generated successfully: {output_path}")


if __name__ == "__main__":
    main()
