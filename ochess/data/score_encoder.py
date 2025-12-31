"""
Chess score encoding with proper perspective handling.

THIS IS THE CRITICAL FIX for the color problem in the original code.

The Problem:
    The original code used `abs(score)` which lost perspective information.
    This caused the network to only play effectively as Black.

The Solution:
    Always encode scores from the SIDE-TO-MOVE perspective:
    - Positive score = Good for the player whose turn it is
    - Negative score = Bad for the player whose turn it is
    - Zero = Equal position

Stockfish Convention:
    Stockfish always reports scores from White's perspective:
    - +150 means White is better by ~1.5 pawns
    - -150 means Black is better by ~1.5 pawns

Conversion:
    For White to move: keep the sign (Stockfish perspective = my perspective)
    For Black to move: flip the sign (I'm Black, so negative White = positive me)

Mate Scores:
    Mate-in-N is converted to a high centipawn value:
    - White mates in 5: +10000 - 500 = +9500 cp
    - Black mates in 3: -10000 + 300 = -9700 cp
    Closer mates have higher absolute values.
"""

import re
from typing import Optional, Tuple
import torch


class ScoreEncoder:
    """
    Encode and decode chess evaluation scores.

    Key features:
    1. Converts from Stockfish perspective (always White) to side-to-move perspective
    2. Handles mate scores properly
    3. Normalizes to a reasonable range for neural network training

    Attributes:
        MATE_SCORE: Base centipawn value for mate (10000 cp)
        MAX_SCORE: Maximum score before clipping (15000 cp)
        normalize: Whether to normalize scores
        scale: Division factor for normalization (default 1000.0)
    """

    MATE_SCORE = 10000  # Base centipawns for mate
    MAX_SCORE = 15000   # Clip scores above this

    def __init__(self, normalize: bool = True, scale: float = 1000.0):
        """
        Initialize the score encoder.

        Args:
            normalize: If True, divide scores by scale
            scale: Division factor for normalization (default 1000.0)
                   With scale=1000, +1000cp becomes +1.0
        """
        self.normalize = normalize
        self.scale = scale

    def encode(
        self,
        score_str: str,
        white_to_move: bool
    ) -> float:
        """
        Convert Stockfish score string to side-to-move perspective.

        This is the main encoding function. It:
        1. Parses the Stockfish score string
        2. Converts to centipawns
        3. Flips sign if Black to move
        4. Optionally normalizes

        Args:
            score_str: Stockfish score string
                - Regular scores: "+150", "-50", "150", "-100"
                - Mate scores: "#5", "#+5" (White mates in 5)
                              "#-5" (Black mates in 5)
            white_to_move: True if White to move in the position

        Returns:
            Score from side-to-move perspective (positive = good for me)

        Examples:
            >>> enc = ScoreEncoder(normalize=False)
            >>> enc.encode("+150", white_to_move=True)   # White winning, White's turn
            150.0  # Good for me (White)
            >>> enc.encode("+150", white_to_move=False)  # White winning, Black's turn
            -150.0  # Bad for me (Black)
            >>> enc.encode("#5", white_to_move=True)     # White mates in 5
            9500.0  # Very good for White
        """
        # Parse score string to centipawns (White's perspective)
        centipawns = self._parse_score_string(score_str)

        # Convert to side-to-move perspective
        # Stockfish gives score from White's view
        # If Black to move, flip the sign
        if not white_to_move:
            centipawns = -centipawns

        # Clip extreme values
        centipawns = max(-self.MAX_SCORE, min(self.MAX_SCORE, centipawns))

        # Optionally normalize
        if self.normalize:
            return centipawns / self.scale

        return float(centipawns)

    def encode_from_centipawns(
        self,
        centipawns: int,
        white_to_move: bool
    ) -> float:
        """
        Encode score from centipawns value (White's perspective).

        Args:
            centipawns: Score in centipawns from White's perspective
            white_to_move: True if White to move

        Returns:
            Score from side-to-move perspective
        """
        if not white_to_move:
            centipawns = -centipawns

        centipawns = max(-self.MAX_SCORE, min(self.MAX_SCORE, centipawns))

        if self.normalize:
            return centipawns / self.scale

        return float(centipawns)

    def decode(
        self,
        encoded_score: float,
        white_to_move: bool
    ) -> int:
        """
        Convert encoded score back to centipawns (White's perspective).

        Args:
            encoded_score: Score from side-to-move perspective
            white_to_move: True if White to move in original position

        Returns:
            Centipawns from White's perspective
        """
        # De-normalize
        if self.normalize:
            centipawns = encoded_score * self.scale
        else:
            centipawns = encoded_score

        # Convert back to White's perspective
        if not white_to_move:
            centipawns = -centipawns

        return int(centipawns)

    def _parse_score_string(self, score_str: str) -> int:
        """
        Parse Stockfish score string to centipawns (White's perspective).

        Handles formats:
        - Regular: "+150", "-50", "150", "-100"
        - Mate: "#5", "#+5", "#-5", "M5", "M-5"

        Args:
            score_str: Score string to parse

        Returns:
            Centipawns from White's perspective
        """
        score_str = score_str.strip()

        # Handle mate scores
        if '#' in score_str or score_str.upper().startswith('M'):
            return self._parse_mate_score(score_str)

        # Handle "cp" suffix (e.g., "+150cp")
        score_str = score_str.lower().replace("cp", "")

        # Parse as integer
        try:
            # Handle with or without + sign
            return int(score_str)
        except ValueError:
            # Try parsing as float and convert
            try:
                return int(float(score_str))
            except ValueError:
                # Default to 0 for unparseable scores
                return 0

    def _parse_mate_score(self, score_str: str) -> int:
        """
        Convert mate-in-N to centipawn equivalent.

        Closer mates are worth more:
        - #1 (mate in 1) -> ±(MATE_SCORE - 100) = ±9900
        - #5 (mate in 5) -> ±(MATE_SCORE - 500) = ±9500
        - #50 (mate in 50) -> ±(MATE_SCORE - 5000) = ±5000

        Formats:
        - "#5" or "#+5" -> White mates in 5 -> positive
        - "#-5" -> Black mates in 5 -> negative
        - "M5" or "M+5" -> White mates in 5
        - "M-5" -> Black mates in 5

        Args:
            score_str: Mate score string

        Returns:
            Centipawns from White's perspective
        """
        score_str = score_str.strip().upper()

        # Remove M or # prefix
        score_str = score_str.replace("M", "").replace("#", "")

        # Remove + (explicit positive)
        score_str = score_str.replace("+", "")

        # Parse the number
        try:
            moves = int(score_str)
        except ValueError:
            return 0  # Default for unparseable

        if moves >= 0:
            # White is mating
            return self.MATE_SCORE - abs(moves) * 100
        else:
            # Black is mating
            return -(self.MATE_SCORE - abs(moves) * 100)

    def is_mate_score(self, score_str: str) -> bool:
        """Check if score string represents a mate."""
        score_str = score_str.strip().upper()
        return '#' in score_str or score_str.startswith('M')

    def get_mate_distance(self, score_str: str) -> Optional[int]:
        """
        Get mate distance from score string.

        Args:
            score_str: Score string

        Returns:
            Positive int if White mating, negative if Black mating,
            None if not a mate score
        """
        if not self.is_mate_score(score_str):
            return None

        score_str = score_str.strip().upper()
        score_str = score_str.replace("M", "").replace("#", "").replace("+", "")

        try:
            return int(score_str)
        except ValueError:
            return None

    def encode_batch(
        self,
        scores: list,
        white_to_move: list
    ) -> torch.Tensor:
        """
        Encode a batch of scores.

        Args:
            scores: List of score strings or centipawn values
            white_to_move: List of booleans

        Returns:
            torch.FloatTensor of encoded scores
        """
        encoded = []
        for score, wtm in zip(scores, white_to_move):
            if isinstance(score, str):
                encoded.append(self.encode(score, wtm))
            else:
                encoded.append(self.encode_from_centipawns(int(score), wtm))

        return torch.tensor(encoded, dtype=torch.float32)

    def score_to_win_probability(self, encoded_score: float) -> float:
        """
        Convert encoded score to win probability estimate.

        Uses a sigmoid-like function to map scores to [0, 1].
        This is useful for understanding what scores mean.

        Args:
            encoded_score: Score from side-to-move perspective

        Returns:
            Estimated win probability for the side to move
        """
        import math

        # De-normalize if needed
        if self.normalize:
            cp = encoded_score * self.scale
        else:
            cp = encoded_score

        # Sigmoid transformation
        # At cp=0: 50% win probability
        # At cp=300: ~75% win probability
        # At cp=1000: ~95% win probability
        return 1.0 / (1.0 + math.exp(-cp / 300.0))

    def win_probability_to_score(self, win_prob: float) -> float:
        """
        Convert win probability back to score.

        Args:
            win_prob: Win probability in [0, 1]

        Returns:
            Encoded score (normalized if self.normalize)
        """
        import math

        # Avoid log(0) or log(inf)
        win_prob = max(0.001, min(0.999, win_prob))

        # Inverse sigmoid
        cp = -300.0 * math.log(1.0 / win_prob - 1.0)

        if self.normalize:
            return cp / self.scale
        return cp

    @staticmethod
    def explain_score(centipawns: int) -> str:
        """
        Provide human-readable interpretation of a score.

        Args:
            centipawns: Score in centipawns from White's perspective

        Returns:
            Human-readable description
        """
        abs_cp = abs(centipawns)
        side = "White" if centipawns > 0 else "Black"

        if abs_cp >= 9000:
            return f"{side} has forced mate"
        elif abs_cp >= 500:
            return f"{side} is winning (+{abs_cp/100:.1f} pawns)"
        elif abs_cp >= 200:
            return f"{side} has clear advantage (+{abs_cp/100:.1f} pawns)"
        elif abs_cp >= 50:
            return f"{side} has slight advantage (+{abs_cp/100:.1f} pawns)"
        else:
            return "Equal position"
