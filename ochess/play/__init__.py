"""
Play interface module for Torch o'Chess.

Provides:
    - ChessEngine: Wrapper for model inference
    - CLIInterface: Command-line interface for human play
    - GUIInterface: Graphical interface for human play
    - BoardDisplay: Board visualization in terminal
"""

from ochess.play.engine import ChessEngine
from ochess.play.cli_interface import CLIInterface
from ochess.play.gui_interface import GUIInterface
from ochess.play.board_display import display_board, format_board

__all__ = [
    "ChessEngine",
    "CLIInterface",
    "GUIInterface",
    "display_board",
    "format_board",
]
