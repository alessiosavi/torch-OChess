"""
GUI interface for playing against the neural network.

Uses tkinter for cross-platform GUI support.
"""

import chess
import torch.nn as nn
from typing import Optional, Tuple
import tkinter as tk
from tkinter import messagebox

from ochess.play.engine import ChessEngine


# Unicode chess pieces for display
PIECE_UNICODE = {
    (chess.PAWN, chess.WHITE): "\u2659",
    (chess.KNIGHT, chess.WHITE): "\u2658",
    (chess.BISHOP, chess.WHITE): "\u2657",
    (chess.ROOK, chess.WHITE): "\u2656",
    (chess.QUEEN, chess.WHITE): "\u2655",
    (chess.KING, chess.WHITE): "\u2654",
    (chess.PAWN, chess.BLACK): "\u265F",
    (chess.KNIGHT, chess.BLACK): "\u265E",
    (chess.BISHOP, chess.BLACK): "\u265D",
    (chess.ROOK, chess.BLACK): "\u265C",
    (chess.QUEEN, chess.BLACK): "\u265B",
    (chess.KING, chess.BLACK): "\u265A",
}

# Colors
LIGHT_SQUARE = "#F0D9B5"
DARK_SQUARE = "#B58863"
HIGHLIGHT_COLOR = "#7CFC00"
SELECTED_COLOR = "#FFFF00"
LAST_MOVE_COLOR = "#AAD4FF"


class GUIInterface:
    """
    Graphical user interface for playing chess against the model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        square_size: int = 80
    ):
        """
        Initialize GUI interface.

        Args:
            model: Trained neural network
            device: Device for inference
            square_size: Size of each square in pixels
        """
        self.engine = ChessEngine(model, device)
        self.board = chess.Board()
        self.human_is_white = True
        self.square_size = square_size

        # GUI state
        self.selected_square: Optional[int] = None
        self.legal_moves_from_selected: list = []
        self.last_move: Optional[chess.Move] = None

        # Create window
        self.root = tk.Tk()
        self.root.title("Torch o'Chess")
        self.root.resizable(False, False)

        self._create_widgets()
        self._bind_events()

    def _create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root, bg="#333333")
        main_frame.pack(padx=10, pady=10)

        # Board canvas
        board_size = self.square_size * 8
        self.canvas = tk.Canvas(
            main_frame,
            width=board_size,
            height=board_size,
            bg="white",
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT)

        # Info panel
        info_frame = tk.Frame(main_frame, bg="#333333", width=200)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        info_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(
            info_frame,
            text="Torch o'Chess",
            font=("Helvetica", 16, "bold"),
            fg="white",
            bg="#333333"
        )
        title_label.pack(pady=(0, 10))

        # Status label
        self.status_label = tk.Label(
            info_frame,
            text="White to move",
            font=("Helvetica", 12),
            fg="white",
            bg="#333333"
        )
        self.status_label.pack(pady=5)

        # Move count label
        self.move_label = tk.Label(
            info_frame,
            text="Move: 1",
            font=("Helvetica", 11),
            fg="white",
            bg="#333333"
        )
        self.move_label.pack(pady=5)

        # Evaluation label
        self.eval_label = tk.Label(
            info_frame,
            text="Eval: 0.00",
            font=("Helvetica", 11),
            fg="white",
            bg="#333333"
        )
        self.eval_label.pack(pady=5)

        # Last move label
        self.last_move_label = tk.Label(
            info_frame,
            text="Last: -",
            font=("Helvetica", 11),
            fg="white",
            bg="#333333"
        )
        self.last_move_label.pack(pady=5)

        # Separator
        sep = tk.Frame(info_frame, height=2, bg="#555555")
        sep.pack(fill=tk.X, pady=10)

        # Buttons
        btn_frame = tk.Frame(info_frame, bg="#333333")
        btn_frame.pack(fill=tk.X, pady=5)

        self.new_game_btn = tk.Button(
            btn_frame,
            text="New Game",
            command=self._new_game_dialog,
            width=15
        )
        self.new_game_btn.pack(pady=3)

        self.undo_btn = tk.Button(
            btn_frame,
            text="Undo Move",
            command=self._undo_move,
            width=15
        )
        self.undo_btn.pack(pady=3)

        self.flip_btn = tk.Button(
            btn_frame,
            text="Flip Board",
            command=self._flip_board,
            width=15
        )
        self.flip_btn.pack(pady=3)

        self.quit_btn = tk.Button(
            btn_frame,
            text="Quit",
            command=self.root.quit,
            width=15
        )
        self.quit_btn.pack(pady=3)

        # Instructions
        instructions = tk.Label(
            info_frame,
            text="Click a piece to select,\nthen click destination\nto move.",
            font=("Helvetica", 9),
            fg="#AAAAAA",
            bg="#333333",
            justify=tk.LEFT
        )
        instructions.pack(pady=10)

    def _bind_events(self):
        """Bind mouse events."""
        self.canvas.bind("<Button-1>", self._on_click)

    def _draw_board(self):
        """Draw the chess board."""
        self.canvas.delete("all")
        flip = not self.human_is_white

        # Draw squares
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = (7 - rank) * self.square_size if not flip else rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                # Determine square color
                square = chess.square(file, rank)
                is_light = (file + rank) % 2 == 1

                if self.selected_square == square:
                    color = SELECTED_COLOR
                elif self.last_move and (square == self.last_move.from_square or
                                          square == self.last_move.to_square):
                    color = LAST_MOVE_COLOR
                elif square in [m.to_square for m in self.legal_moves_from_selected]:
                    color = HIGHLIGHT_COLOR
                else:
                    color = LIGHT_SQUARE if is_light else DARK_SQUARE

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)

                x = file * self.square_size + self.square_size // 2
                if flip:
                    y = rank * self.square_size + self.square_size // 2
                else:
                    y = (7 - rank) * self.square_size + self.square_size // 2

                symbol = PIECE_UNICODE.get((piece.piece_type, piece.color), "?")
                self.canvas.create_text(
                    x, y,
                    text=symbol,
                    font=("Helvetica", int(self.square_size * 0.7)),
                    fill="black" if piece.color == chess.WHITE else "#1a1a1a"
                )

        # Draw file labels
        for file in range(8):
            label = chr(ord('a') + file) if not flip else chr(ord('h') - file)
            x = file * self.square_size + self.square_size // 2
            self.canvas.create_text(
                x, 8 * self.square_size - 8,
                text=label,
                font=("Helvetica", 10),
                fill="#666666"
            )

        # Draw rank labels
        for rank in range(8):
            label = str(rank + 1) if not flip else str(8 - rank)
            y = (7 - rank) * self.square_size + self.square_size // 2 if not flip else rank * self.square_size + self.square_size // 2
            self.canvas.create_text(
                8, y,
                text=label,
                font=("Helvetica", 10),
                fill="#666666"
            )

    def _update_status(self):
        """Update status labels."""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_label.config(text=f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            self.status_label.config(text="Stalemate! Draw.")
        elif self.board.is_insufficient_material():
            self.status_label.config(text="Draw - Insufficient material")
        elif self.board.is_check():
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.config(text=f"{turn} in CHECK!")
        else:
            turn = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_label.config(text=f"{turn} to move")

        self.move_label.config(text=f"Move: {self.board.fullmove_number}")

        if self.last_move:
            # Get SAN for last move
            self.board.pop()
            san = self.board.san(self.last_move)
            self.board.push(self.last_move)
            self.last_move_label.config(text=f"Last: {san}")
        else:
            self.last_move_label.config(text="Last: -")

        # Update evaluation
        try:
            score = self.engine.get_evaluation(self.board)
            self.eval_label.config(text=f"Eval: {score:.2f}")
        except Exception:
            self.eval_label.config(text="Eval: N/A")

    def _on_click(self, event):
        """Handle click on board."""
        if self.board.is_game_over():
            return

        # Check if it's human's turn
        is_human_turn = (self.board.turn == chess.WHITE) == self.human_is_white
        if not is_human_turn:
            return

        # Convert click to square
        file = event.x // self.square_size
        rank = 7 - (event.y // self.square_size)

        if not self.human_is_white:
            file = 7 - file
            rank = 7 - rank

        if not (0 <= file < 8 and 0 <= rank < 8):
            return

        clicked_square = chess.square(file, rank)

        if self.selected_square is None:
            # Select piece
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.legal_moves_from_selected = [
                    m for m in self.board.legal_moves
                    if m.from_square == clicked_square
                ]
        else:
            # Try to make move
            move = None

            # Check if clicked on a legal destination
            for m in self.legal_moves_from_selected:
                if m.to_square == clicked_square:
                    # Handle promotion
                    if (self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                        chess.square_rank(clicked_square) in [0, 7]):
                        move = chess.Move(
                            self.selected_square, clicked_square,
                            promotion=chess.QUEEN
                        )
                    else:
                        move = m
                    break

            if move:
                self._make_move(move)
            else:
                # Check if clicking on another own piece
                piece = self.board.piece_at(clicked_square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = clicked_square
                    self.legal_moves_from_selected = [
                        m for m in self.board.legal_moves
                        if m.from_square == clicked_square
                    ]
                else:
                    # Deselect
                    self.selected_square = None
                    self.legal_moves_from_selected = []

        self._draw_board()

    def _make_move(self, move: chess.Move):
        """Make a move and let computer respond."""
        # Human move
        self.board.push(move)
        self.engine.history.append(self.board.fen())
        self.last_move = move
        self.selected_square = None
        self.legal_moves_from_selected = []

        self._draw_board()
        self._update_status()
        self.root.update()

        # Check for game over
        if self.board.is_game_over():
            self._show_game_over()
            return

        # Computer's turn
        self.status_label.config(text="Computer thinking...")
        self.root.update()

        comp_move, was_illegal = self.engine.get_move(self.board)

        if comp_move:
            self.board.push(comp_move)
            self.engine.history.append(self.board.fen())
            self.last_move = comp_move

        self._draw_board()
        self._update_status()

        if self.board.is_game_over():
            self._show_game_over()

    def _show_game_over(self):
        """Show game over dialog."""
        outcome = self.board.outcome()

        if outcome.winner is None:
            msg = "Game drawn!"
        elif (outcome.winner == chess.WHITE) == self.human_is_white:
            msg = "Congratulations! You win!"
        else:
            msg = "Computer wins!"

        messagebox.showinfo("Game Over", f"{msg}\n\n{outcome.termination.name}")

    def _new_game_dialog(self):
        """Show new game dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("New Game")
        dialog.geometry("250x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        label = tk.Label(dialog, text="Choose your color:", font=("Helvetica", 12))
        label.pack(pady=20)

        btn_frame = tk.Frame(dialog)
        btn_frame.pack()

        def start_as_white():
            dialog.destroy()
            self._start_new_game(True)

        def start_as_black():
            dialog.destroy()
            self._start_new_game(False)

        white_btn = tk.Button(
            btn_frame, text="Play as White", command=start_as_white, width=12
        )
        white_btn.pack(side=tk.LEFT, padx=5)

        black_btn = tk.Button(
            btn_frame, text="Play as Black", command=start_as_black, width=12
        )
        black_btn.pack(side=tk.LEFT, padx=5)

    def _start_new_game(self, human_white: bool):
        """Start a new game."""
        self.board = chess.Board()
        self.engine.new_game()
        self.human_is_white = human_white
        self.selected_square = None
        self.legal_moves_from_selected = []
        self.last_move = None

        self._draw_board()
        self._update_status()

        # If human is black, let computer move first
        if not human_white:
            self.status_label.config(text="Computer thinking...")
            self.root.update()

            comp_move, _ = self.engine.get_move(self.board)
            if comp_move:
                self.board.push(comp_move)
                self.engine.history.append(self.board.fen())
                self.last_move = comp_move

            self._draw_board()
            self._update_status()

    def _undo_move(self):
        """Undo last two moves (human + computer)."""
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            if self.engine.history:
                self.engine.history = self.engine.history[:-2]
            self.last_move = self.board.move_stack[-1] if self.board.move_stack else None
            self.selected_square = None
            self.legal_moves_from_selected = []
            self._draw_board()
            self._update_status()
        elif len(self.board.move_stack) == 1:
            self.board.pop()
            if self.engine.history:
                self.engine.history.pop()
            self.last_move = None
            self.selected_square = None
            self.legal_moves_from_selected = []
            self._draw_board()
            self._update_status()

    def _flip_board(self):
        """Flip the board view."""
        self.human_is_white = not self.human_is_white
        self._draw_board()

    def start_game(self, human_plays_white: bool = True):
        """
        Start a new game and run the GUI.

        Args:
            human_plays_white: True if human plays White
        """
        self.human_is_white = human_plays_white
        self.board = chess.Board()
        self.engine.new_game()

        self._draw_board()
        self._update_status()

        # If human is black, computer moves first
        if not human_plays_white:
            self.root.after(100, self._computer_first_move)

        self.root.mainloop()

    def _computer_first_move(self):
        """Make computer's first move when human is Black."""
        self.status_label.config(text="Computer thinking...")
        self.root.update()

        comp_move, _ = self.engine.get_move(self.board)
        if comp_move:
            self.board.push(comp_move)
            self.engine.history.append(self.board.fen())
            self.last_move = comp_move

        self._draw_board()
        self._update_status()


def play_gui(model_path: str, device: str = "cuda", human_white: bool = True):
    """
    Convenience function to start a GUI game.

    Args:
        model_path: Path to trained model
        device: Device for inference
        human_white: True if human plays White
    """
    import torch

    # Load model
    model = torch.load(model_path, map_location=device)
    if hasattr(model, 'eval'):
        model.eval()

    # Start game
    interface = GUIInterface(model, device)
    interface.start_game(human_white)
