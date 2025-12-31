"""
Command-line interface for playing against the neural network.
"""

from typing import Optional

import chess
import torch.nn as nn

from ochess.play.board_display import display_board, format_game_state
from ochess.play.engine import ChessEngine


class CLIInterface:
    """
    Command-line interface for playing chess against the model.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize CLI interface.

        Args:
            model: Trained neural network
            device: Device for inference
        """
        self.engine = ChessEngine(model, device)
        self.board = chess.Board()
        self.human_is_white = True

    def start_game(self, human_plays_white: bool = True):
        """
        Start a new game.

        Args:
            human_plays_white: True if human plays White
        """
        self.board = chess.Board()
        self.engine.new_game()
        self.human_is_white = human_plays_white

        print("\n" + "=" * 50)
        print("     TORCH O'CHESS")
        print("=" * 50)
        print(f"You are playing as {'White' if human_plays_white else 'Black'}")
        print("\nCommands:")
        print("  - Enter moves in UCI format (e.g., 'e2e4')")
        print("  - 'quit' to exit")
        print("  - 'undo' to take back a move")
        print("  - 'legal' to see legal moves")
        print("  - 'eval' to see model's evaluation")
        print("=" * 50 + "\n")

        self._game_loop()

    def _game_loop(self):
        """Main game loop."""
        while not self.board.is_game_over():
            # Display board
            print(display_board(self.board, flip=not self.human_is_white))
            print()
            print(format_game_state(self.board))
            print()

            is_human_turn = (self.board.turn == chess.WHITE) == self.human_is_white

            if is_human_turn:
                move = self._get_human_move()
                if move is None:  # User quit
                    print("\nThanks for playing!")
                    return
            else:
                print("Computer is thinking...")
                move, was_illegal = self.engine.get_move(self.board)

                if was_illegal:
                    print("(Model predicted illegal move, using fallback)")

                print(f"Computer plays: {self.board.san(move)} ({move.uci()})")

            self.board.push(move)
            self.engine.history.append(self.board.fen())
            print()

        # Game over
        print(display_board(self.board))
        print()
        self._print_result()

    def _get_human_move(self) -> Optional[chess.Move]:
        """Get a move from the human player."""
        while True:
            try:
                user_input = input("Your move: ").strip().lower()

                if user_input == "quit":
                    return None

                if user_input == "undo":
                    if len(self.board.move_stack) >= 2:
                        self.board.pop()
                        self.board.pop()
                        print("Moves undone.")
                        print(display_board(self.board))
                    else:
                        print("Nothing to undo.")
                    continue

                if user_input == "legal":
                    moves = [self.board.san(m) for m in self.board.legal_moves]
                    print("Legal moves:", ", ".join(moves[:20]))
                    if len(moves) > 20:
                        print(f"  ... and {len(moves) - 20} more")
                    continue

                if user_input == "eval":
                    score = self.engine.get_evaluation(self.board)
                    side = "White" if self.board.turn else "Black"
                    print(f"Model evaluation: {score:.2f} ({side}'s perspective)")
                    continue

                # Try UCI format
                try:
                    move = chess.Move.from_uci(user_input)
                except ValueError:
                    # Try SAN format
                    try:
                        move = self.board.parse_san(user_input)
                    except ValueError:
                        print("Invalid move format. Use UCI (e2e4) or SAN (e4).")
                        continue

                if move in self.board.legal_moves:
                    return move
                else:
                    print("Illegal move. Try again.")

            except KeyboardInterrupt:
                print("\nGame interrupted.")
                return None

    def _print_result(self):
        """Print game result."""
        outcome = self.board.outcome()
        print("\n" + "=" * 30)

        if outcome is None:
            print("Game incomplete")
        elif outcome.winner is None:
            print("Game drawn!")
        elif (outcome.winner == chess.WHITE) == self.human_is_white:
            print("Congratulations! You win!")
        else:
            print("Computer wins!")

        if outcome:
            print(f"Termination: {outcome.termination.name}")

        print("=" * 30)


def play_game(model_path: str, device: str = "cuda", human_white: bool = True):
    """
    Convenience function to start a game.

    Args:
        model_path: Path to trained model
        device: Device for inference
        human_white: True if human plays White
    """
    import torch

    # Load model
    model = torch.load(model_path, map_location=device)
    if hasattr(model, "eval"):
        model.eval()

    # Start game
    interface = CLIInterface(model, device)
    interface.start_game(human_white)
