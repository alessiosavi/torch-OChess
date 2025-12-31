#!/usr/bin/env python3
"""
Evaluate a trained model against Stockfish.

Usage:
    python scripts/evaluate.py --model checkpoints/best_model.pt --model-type hybrid --num-games 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ochess.evaluation.stockfish_eval import StockfishEvaluator
from ochess.utils import get_available_device, load_model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chess model against Stockfish"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["resnet", "transformer", "hybrid"],
        required=True,
        help="Model architecture type (resnet, transformer, hybrid)",
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="/usr/local/bin/stockfish",
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--num-games", type=int, default=10, help="Number of games per level"
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="1,2,3",
        help="Stockfish skill levels (comma-separated)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for inference"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results (JSON)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Check device
    device = get_available_device(args.device, logger)

    # Load model
    model = load_model(args.model, args.model_type, device)

    # Parse levels
    levels = [int(x) for x in args.levels.split(",")]

    # Create evaluator
    evaluator = StockfishEvaluator(
        model=model, stockfish_path=args.stockfish_path, device=device
    )

    # Run evaluation
    print(f"\nEvaluating against Stockfish levels: {levels}")
    print(f"Games per level: {args.num_games}")
    print("=" * 50)

    results = evaluator.play_match(num_games=args.num_games, stockfish_levels=levels)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    all_records = []
    for level, records in results.items():
        wins = sum(1 for r in records if (r.result.value == "1-0") == r.model_color)
        draws = sum(1 for r in records if r.result.value == "1/2-1/2")
        losses = len(records) - wins - draws

        win_rate = wins / len(records) if records else 0

        print(
            f"Level {level:2d}: W={wins:3d} D={draws:3d} L={losses:3d} "
            f"({win_rate*100:5.1f}% win rate)"
        )

        all_records.extend(records)

    # Overall stats
    total_wins = sum(
        1 for r in all_records if (r.result.value == "1-0") == r.model_color
    )
    total_draws = sum(1 for r in all_records if r.result.value == "1/2-1/2")
    total_losses = len(all_records) - total_wins - total_draws
    total_illegal = sum(r.model_illegal_moves for r in all_records)

    print("-" * 50)
    print(f"TOTAL:    W={total_wins:3d} D={total_draws:3d} L={total_losses:3d}")
    print(f"Illegal move predictions: {total_illegal}")
    print("=" * 50)

    # Save results
    if args.output:
        output_data = {
            level: [
                {
                    "result": r.result.value,
                    "model_color": "white" if r.model_color else "black",
                    "moves": len(r.moves),
                    "illegal_moves": r.model_illegal_moves,
                }
                for r in records
            ]
            for level, records in results.items()
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
