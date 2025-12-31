"""
Board display utilities for terminal.
"""

import chess
from typing import Optional

# ASCII piece symbols
PIECE_SYMBOLS = {
    chess.PAWN: 'P', chess.KNIGHT: 'N', chess.BISHOP: 'B',
    chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K'
}

# Unicode piece symbols
UNICODE_PIECES = {
    (chess.PAWN, chess.WHITE): '\u2659',
    (chess.KNIGHT, chess.WHITE): '\u2658',
    (chess.BISHOP, chess.WHITE): '\u2657',
    (chess.ROOK, chess.WHITE): '\u2656',
    (chess.QUEEN, chess.WHITE): '\u2655',
    (chess.KING, chess.WHITE): '\u2654',
    (chess.PAWN, chess.BLACK): '\u265F',
    (chess.KNIGHT, chess.BLACK): '\u265E',
    (chess.BISHOP, chess.BLACK): '\u265D',
    (chess.ROOK, chess.BLACK): '\u265C',
    (chess.QUEEN, chess.BLACK): '\u265B',
    (chess.KING, chess.BLACK): '\u265A',
}


def display_board(
    board: chess.Board,
    flip: bool = False,
    use_unicode: bool = False,
    highlight_squares: Optional[list] = None
) -> str:
    """
    Create a string representation of the board.

    Args:
        board: Chess board
        flip: Show from Black's perspective
        use_unicode: Use Unicode chess symbols
        highlight_squares: Squares to highlight

    Returns:
        Multi-line string representation
    """
    lines = []

    # Header
    files = "   a b c d e f g h" if not flip else "   h g f e d c b a"
    lines.append(files)
    lines.append("  +-+-+-+-+-+-+-+-+")

    # Board
    ranks = range(7, -1, -1) if not flip else range(8)

    for rank in ranks:
        row = f"{rank + 1} |"
        file_range = range(8) if not flip else range(7, -1, -1)

        for file in file_range:
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece is None:
                symbol = '.'
            elif use_unicode:
                symbol = UNICODE_PIECES.get(
                    (piece.piece_type, piece.color), '?'
                )
            else:
                symbol = PIECE_SYMBOLS.get(piece.piece_type, '?')
                if piece.color == chess.BLACK:
                    symbol = symbol.lower()

            row += f"{symbol}|"

        lines.append(row)
        lines.append("  +-+-+-+-+-+-+-+-+")

    lines.append(files)

    return "\n".join(lines)


def format_board(board: chess.Board) -> str:
    """Simple board format for display."""
    return display_board(board, use_unicode=True)


def print_board(board: chess.Board, flip: bool = False):
    """Print board to console."""
    print(display_board(board, flip=flip))


def format_move(board: chess.Board, move: chess.Move) -> str:
    """Format move in SAN notation."""
    return board.san(move)


def format_game_state(board: chess.Board) -> str:
    """Format current game state."""
    lines = []

    turn = "White" if board.turn else "Black"
    lines.append(f"Turn: {turn}")

    if board.is_check():
        lines.append("CHECK!")

    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        lines.append(f"CHECKMATE! {winner} wins!")
    elif board.is_stalemate():
        lines.append("STALEMATE! Draw.")
    elif board.is_insufficient_material():
        lines.append("INSUFFICIENT MATERIAL! Draw.")
    elif board.can_claim_draw():
        lines.append("Draw can be claimed.")

    lines.append(f"Move: {board.fullmove_number}")

    return "\n".join(lines)
