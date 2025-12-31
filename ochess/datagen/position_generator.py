"""
Chess position generator for training data creation.

Generates diverse chess positions using various methods:
    - Random legal moves (high diversity, may include unrealistic positions)
    - From standard openings (realistic positions)
    - Tactical positions (positions with sharp moves)
"""

import chess
import random
from dataclasses import dataclass
from typing import List, Optional, Generator, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Classification of chess position types."""
    OPENING = "opening"       # First ~10 moves
    MIDDLEGAME = "middlegame"  # Moves 10-40
    ENDGAME = "endgame"       # Few pieces remaining
    TACTICAL = "tactical"     # Sharp positions with tactics


@dataclass
class GeneratedPosition:
    """A generated chess position with metadata."""
    fen: str
    position_type: PositionType
    move_number: int
    white_to_move: bool
    source: str  # How this position was generated
    game_id: Optional[str] = None


# Common chess openings for generating realistic positions
COMMON_OPENINGS = [
    # Italian Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    # Sicilian Defense
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
    # French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    # Caro-Kann Defense
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    # Queen's Gambit
    ["d2d4", "d7d5", "c2c4"],
    # King's Indian Defense
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
    # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    # Scotch Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],
    # London System
    ["d2d4", "d7d5", "c1f4"],
    # English Opening
    ["c2c4", "e7e5"],
    # Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6"],
    # Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    # Dutch Defense
    ["d2d4", "f7f5"],
    # Pirc Defense
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"],
]


class PositionGenerator:
    """
    Generate diverse chess positions for training data.

    Methods:
    1. Random games: Play random legal moves
    2. From openings: Start from known openings, then randomize
    3. Mixed: Combine various generation methods
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the position generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self._generated_fens: Set[str] = set()

    def generate_random_game_positions(
        self,
        num_positions: int,
        max_game_length: int = 100,
        skip_first_n: int = 4,
        deduplicate: bool = True
    ) -> Generator[GeneratedPosition, None, None]:
        """
        Generate positions by playing random legal moves.

        This creates diverse positions but may include unrealistic ones.

        Args:
            num_positions: Number of positions to generate
            max_game_length: Maximum moves per game
            skip_first_n: Skip very early positions (opening book territory)
            deduplicate: Skip duplicate FENs

        Yields:
            GeneratedPosition objects
        """
        positions_generated = 0
        game_counter = 0

        while positions_generated < num_positions:
            board = chess.Board()
            move_number = 0
            game_counter += 1
            game_id = f"random_{game_counter}"

            while not board.is_game_over() and move_number < max_game_length:
                if move_number >= skip_first_n:
                    fen = board.fen()

                    # Deduplicate
                    if deduplicate:
                        board_fen = fen.split()[0]  # Just the piece placement
                        if board_fen in self._generated_fens:
                            # Skip duplicate
                            legal_moves = list(board.legal_moves)
                            if not legal_moves:
                                break
                            move = random.choice(legal_moves)
                            board.push(move)
                            move_number += 1
                            continue
                        self._generated_fens.add(board_fen)

                    positions_generated += 1

                    yield GeneratedPosition(
                        fen=fen,
                        position_type=self._classify_position(board, move_number),
                        move_number=move_number,
                        white_to_move=board.turn == chess.WHITE,
                        source="random_game",
                        game_id=game_id
                    )

                    if positions_generated >= num_positions:
                        return

                # Play random move
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                # Slightly prefer captures and checks (more interesting positions)
                captures = [m for m in legal_moves if board.is_capture(m)]
                checks = [m for m in legal_moves if board.gives_check(m)]

                if random.random() < 0.3 and captures:
                    move = random.choice(captures)
                elif random.random() < 0.2 and checks:
                    move = random.choice(checks)
                else:
                    move = random.choice(legal_moves)

                board.push(move)
                move_number += 1

    def generate_from_openings(
        self,
        num_positions: int,
        openings: Optional[List[List[str]]] = None,
        continuation_depth: int = 30,
        deduplicate: bool = True
    ) -> Generator[GeneratedPosition, None, None]:
        """
        Generate positions starting from known openings.

        Args:
            num_positions: Number of positions to generate
            openings: List of opening move sequences (UCI format)
            continuation_depth: Random moves after opening
            deduplicate: Skip duplicate FENs

        Yields:
            GeneratedPosition objects
        """
        if openings is None:
            openings = COMMON_OPENINGS

        positions_generated = 0
        game_counter = 0

        while positions_generated < num_positions:
            # Pick a random opening
            opening = random.choice(openings)
            game_counter += 1
            game_id = f"opening_{game_counter}"

            board = chess.Board()

            # Play opening moves
            for uci_move in opening:
                try:
                    move = chess.Move.from_uci(uci_move)
                    if move in board.legal_moves:
                        board.push(move)
                except (ValueError, chess.IllegalMoveError):
                    break

            opening_length = len(board.move_stack)

            # Continue with semi-random moves
            for i in range(continuation_depth):
                if board.is_game_over():
                    break

                fen = board.fen()

                # Deduplicate
                if deduplicate:
                    board_fen = fen.split()[0]
                    if board_fen not in self._generated_fens:
                        self._generated_fens.add(board_fen)

                        positions_generated += 1
                        yield GeneratedPosition(
                            fen=fen,
                            position_type=self._classify_position(
                                board, opening_length + i
                            ),
                            move_number=opening_length + i,
                            white_to_move=board.turn == chess.WHITE,
                            source="from_opening",
                            game_id=game_id
                        )

                        if positions_generated >= num_positions:
                            return

                # Play move
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                move = random.choice(legal_moves)
                board.push(move)

    def generate_mixed(
        self,
        num_positions: int,
        random_ratio: float = 0.5,
        opening_ratio: float = 0.5
    ) -> Generator[GeneratedPosition, None, None]:
        """
        Generate positions using a mix of methods.

        Args:
            num_positions: Total positions to generate
            random_ratio: Fraction from random games
            opening_ratio: Fraction from openings

        Yields:
            GeneratedPosition objects
        """
        # Normalize ratios
        total = random_ratio + opening_ratio
        random_ratio /= total
        opening_ratio /= total

        num_random = int(num_positions * random_ratio)
        num_opening = num_positions - num_random

        # Generate from random games
        logger.info(f"Generating {num_random} positions from random games")
        for pos in self.generate_random_game_positions(num_random):
            yield pos

        # Generate from openings
        logger.info(f"Generating {num_opening} positions from openings")
        for pos in self.generate_from_openings(num_opening):
            yield pos

    def generate_endgame_positions(
        self,
        num_positions: int,
        max_pieces: int = 8
    ) -> Generator[GeneratedPosition, None, None]:
        """
        Generate endgame positions with few pieces.

        Args:
            num_positions: Number of positions to generate
            max_pieces: Maximum pieces on the board

        Yields:
            GeneratedPosition objects
        """
        positions_generated = 0
        game_counter = 0

        while positions_generated < num_positions:
            # Start from random and play until endgame
            board = chess.Board()
            game_counter += 1
            move_number = 0

            while not board.is_game_over() and move_number < 200:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break

                # Prefer captures to reach endgame faster
                captures = [m for m in legal_moves if board.is_capture(m)]
                if captures and random.random() < 0.6:
                    move = random.choice(captures)
                else:
                    move = random.choice(legal_moves)

                board.push(move)
                move_number += 1

                # Check if we're in endgame
                piece_count = len(board.piece_map())
                if piece_count <= max_pieces:
                    fen = board.fen()
                    board_fen = fen.split()[0]

                    if board_fen not in self._generated_fens:
                        self._generated_fens.add(board_fen)
                        positions_generated += 1

                        yield GeneratedPosition(
                            fen=fen,
                            position_type=PositionType.ENDGAME,
                            move_number=move_number,
                            white_to_move=board.turn == chess.WHITE,
                            source="endgame",
                            game_id=f"endgame_{game_counter}"
                        )

                        if positions_generated >= num_positions:
                            return

    def _classify_position(
        self,
        board: chess.Board,
        move_number: int
    ) -> PositionType:
        """
        Classify position as opening, middlegame, or endgame.

        Args:
            board: Current board state
            move_number: Move number in the game

        Returns:
            PositionType classification
        """
        piece_count = len(board.piece_map())

        # Check for tactical positions (checks, captures available)
        if board.is_check():
            return PositionType.TACTICAL

        if move_number < 10:
            return PositionType.OPENING
        elif piece_count <= 10 or move_number > 50:
            return PositionType.ENDGAME
        else:
            return PositionType.MIDDLEGAME

    def clear_dedup_cache(self) -> None:
        """Clear the deduplication cache."""
        self._generated_fens.clear()

    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            "unique_positions": len(self._generated_fens)
        }
