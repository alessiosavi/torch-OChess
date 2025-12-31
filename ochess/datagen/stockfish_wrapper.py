"""
Stockfish engine wrapper for position analysis and data generation.

Provides a clean interface to Stockfish for:
    - Position analysis
    - Best move extraction
    - Multi-PV (multiple principal variations) analysis
    - Configurable skill levels
"""

import chess
import chess.engine
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from analyzing a position."""
    best_move: chess.Move
    score_cp: int           # Centipawns from White's perspective
    score_str: str          # String representation (e.g., "+150", "#5")
    pv: List[chess.Move]    # Principal variation (sequence of best moves)
    depth: int              # Search depth reached
    is_mate: bool           # True if this is a mate score
    mate_in: Optional[int]  # Moves until mate (positive=White, negative=Black)

    def __str__(self) -> str:
        """String representation of analysis result."""
        move_str = self.best_move.uci() if self.best_move else "none"
        if self.is_mate:
            return f"Move: {move_str}, Score: #{self.mate_in}, Depth: {self.depth}"
        return f"Move: {move_str}, Score: {self.score_cp}cp, Depth: {self.depth}"


class StockfishWrapper:
    """
    Wrapper around Stockfish chess engine.

    Provides analysis capabilities for generating training data:
    - Single best move extraction
    - Multi-PV analysis for top N moves
    - Configurable search depth and time
    - Skill level adjustment for generating varied data

    Usage:
        with StockfishWrapper("/path/to/stockfish") as engine:
            board = chess.Board()
            result = engine.analyze(board, depth=20)
            print(result.best_move, result.score_cp)
    """

    def __init__(
        self,
        stockfish_path: str,
        threads: int = 4,
        hash_mb: int = 256,
        skill_level: int = 20
    ):
        """
        Initialize Stockfish wrapper.

        Args:
            stockfish_path: Path to Stockfish binary
            threads: Number of threads for analysis
            hash_mb: Hash table size in MB
            skill_level: Stockfish skill level (0-20, 20 is strongest)
        """
        self.path = Path(stockfish_path)
        self.threads = threads
        self.hash_mb = hash_mb
        self.skill_level = skill_level
        self.engine: Optional[chess.engine.SimpleEngine] = None

        if not self.path.exists():
            raise FileNotFoundError(f"Stockfish not found at: {self.path}")

    def __enter__(self) -> "StockfishWrapper":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def start(self) -> None:
        """Start the Stockfish engine."""
        if self.engine is not None:
            return

        logger.info(f"Starting Stockfish from {self.path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        self.engine.configure({
            "Threads": self.threads,
            "Hash": self.hash_mb,
            "Skill Level": self.skill_level
        })
        logger.info(f"Stockfish started (threads={self.threads}, hash={self.hash_mb}MB)")

    def stop(self) -> None:
        """Stop the Stockfish engine."""
        if self.engine is not None:
            logger.info("Stopping Stockfish")
            self.engine.quit()
            self.engine = None

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self.engine is not None

    def analyze(
        self,
        board: chess.Board,
        depth: int = 20,
        time_limit: Optional[float] = None,
        multipv: int = 1
    ) -> List[AnalysisResult]:
        """
        Analyze a position.

        Args:
            board: Chess position to analyze
            depth: Search depth (ignored if time_limit is set)
            time_limit: Time limit in seconds (overrides depth)
            multipv: Number of principal variations to return

        Returns:
            List of AnalysisResult objects (one per PV)

        Raises:
            RuntimeError: If engine is not started
        """
        if self.engine is None:
            raise RuntimeError("Engine not started. Call start() or use context manager.")

        # Set limit
        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=depth)

        # Analyze
        info_list = self.engine.analyse(board, limit, multipv=multipv)

        # Handle single PV vs multi PV
        if not isinstance(info_list, list):
            info_list = [info_list]

        # Parse results
        results = []
        for info in info_list:
            result = self._parse_info(info, board)
            if result is not None:
                results.append(result)

        return results

    def get_best_move(
        self,
        board: chess.Board,
        depth: int = 20,
        time_limit: Optional[float] = None
    ) -> Tuple[chess.Move, int]:
        """
        Get best move and score for a position.

        Args:
            board: Chess position
            depth: Search depth
            time_limit: Time limit in seconds

        Returns:
            Tuple of (best_move, score_in_centipawns)
        """
        results = self.analyze(board, depth=depth, time_limit=time_limit, multipv=1)
        if not results:
            # Fallback: return first legal move
            move = list(board.legal_moves)[0]
            return move, 0

        return results[0].best_move, results[0].score_cp

    def get_top_moves(
        self,
        board: chess.Board,
        n: int = 3,
        depth: int = 15
    ) -> List[Tuple[chess.Move, int]]:
        """
        Get top N moves with their scores.

        Args:
            board: Chess position
            n: Number of moves to return
            depth: Search depth

        Returns:
            List of (move, score) tuples
        """
        results = self.analyze(board, depth=depth, multipv=n)
        return [(r.best_move, r.score_cp) for r in results]

    def evaluate_move(
        self,
        board: chess.Board,
        move: chess.Move,
        depth: int = 15
    ) -> int:
        """
        Evaluate a specific move.

        Args:
            board: Current position
            move: Move to evaluate
            depth: Search depth

        Returns:
            Score in centipawns after the move (from original side's perspective)
        """
        # Make the move
        board.push(move)

        # Analyze resulting position
        results = self.analyze(board, depth=depth, multipv=1)

        # Undo the move
        board.pop()

        if not results:
            return 0

        # Score is from opponent's perspective after the move
        # Flip sign to get from original side's perspective
        return -results[0].score_cp

    def _parse_info(
        self,
        info: chess.engine.InfoDict,
        board: chess.Board
    ) -> Optional[AnalysisResult]:
        """
        Parse Stockfish analysis info dict.

        Args:
            info: Analysis info from python-chess
            board: The analyzed position

        Returns:
            AnalysisResult or None if parsing fails
        """
        # Get score
        if "score" not in info:
            return None

        score = info["score"].white()  # Always from White's perspective

        # Parse score
        is_mate = score.is_mate()
        mate_in = None
        score_cp = 0
        score_str = ""

        if is_mate:
            mate_in = score.mate()
            # Convert mate to high centipawn value
            if mate_in is not None:
                if mate_in > 0:
                    score_cp = 10000 - mate_in * 100
                else:
                    score_cp = -10000 - mate_in * 100
                score_str = f"#{mate_in}"
        else:
            score_cp = score.score() or 0
            score_str = f"{score_cp:+d}"

        # Get best move
        best_move = None
        pv = []
        if "pv" in info and info["pv"]:
            best_move = info["pv"][0]
            pv = info["pv"]

        return AnalysisResult(
            best_move=best_move,
            score_cp=score_cp,
            score_str=score_str,
            pv=pv,
            depth=info.get("depth", 0),
            is_mate=is_mate,
            mate_in=mate_in
        )

    def set_skill_level(self, level: int) -> None:
        """
        Set Stockfish skill level.

        Args:
            level: Skill level 0-20 (20 is strongest)
        """
        if self.engine is not None:
            self.engine.configure({"Skill Level": level})
        self.skill_level = level

    def play_move(
        self,
        board: chess.Board,
        depth: int = 20,
        time_limit: Optional[float] = None
    ) -> chess.Move:
        """
        Get the move Stockfish would play.

        This is similar to get_best_move but uses the play() method
        which is slightly more efficient for just getting the move.

        Args:
            board: Current position
            depth: Search depth
            time_limit: Time limit in seconds

        Returns:
            The move Stockfish would play
        """
        if self.engine is None:
            raise RuntimeError("Engine not started")

        if time_limit is not None:
            limit = chess.engine.Limit(time=time_limit)
        else:
            limit = chess.engine.Limit(depth=depth)

        result = self.engine.play(board, limit)
        return result.move
