#!/usr/bin/env python3
"""
Play against a trained chess model.

Usage:
    python scripts/play.py --model checkpoints/best_model.pt --color white
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ochess.play.cli_interface import CLIInterface


def main():
    parser = argparse.ArgumentParser(
        description="Play chess against trained model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference"
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Your color (white or black)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    print(f"Loading model from {args.model}...")
    model = torch.load(args.model, map_location=args.device)
    if hasattr(model, 'eval'):
        model.eval()

    # Start game
    human_white = args.color == "white"
    interface = CLIInterface(model, args.device)
    interface.start_game(human_white)


if __name__ == "__main__":
    main()
