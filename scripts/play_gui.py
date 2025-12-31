#!/usr/bin/env python3
"""
Play against a trained chess model using a graphical interface.

Usage:
    python scripts/play_gui.py --model checkpoints/best_model.pt --model-type hybrid --color white
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ochess.play.gui_interface import GUIInterface
from ochess.utils import get_available_device, load_model


def main():
    parser = argparse.ArgumentParser(
        description="Play chess against trained model (GUI)"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["resnet", "transformer", "hybrid"],
        required=True,
        help="Model architecture type (resnet, transformer, hybrid)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Your color (white or black)",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=80,
        help="Size of each square in pixels (default: 80)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Check device
    device = get_available_device(args.device)

    # Load model
    model = load_model(args.model, args.model_type, device)

    # Start game
    human_white = args.color == "white"
    interface = GUIInterface(model, device, square_size=args.square_size)
    interface.start_game(human_white)


if __name__ == "__main__":
    main()
