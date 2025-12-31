"""Tests for score encoder - THE CRITICAL FIX."""

import pytest
from ochess.data.score_encoder import ScoreEncoder


class TestScoreEncoder:
    """Tests for score encoding with proper perspective handling."""

    @pytest.fixture
    def encoder(self):
        return ScoreEncoder(normalize=True, scale=1000.0)

    @pytest.fixture
    def encoder_unnorm(self):
        return ScoreEncoder(normalize=False)

    # ==================== BASIC ENCODING ====================

    def test_positive_score_white_to_move(self, encoder):
        """White winning, White's turn = positive (good for me)."""
        score = encoder.encode("+150", white_to_move=True)
        assert score > 0
        assert abs(score - 0.15) < 0.01

    def test_positive_score_black_to_move(self, encoder):
        """White winning, Black's turn = negative (bad for me)."""
        score = encoder.encode("+150", white_to_move=False)
        assert score < 0
        assert abs(score + 0.15) < 0.01

    def test_negative_score_white_to_move(self, encoder):
        """Black winning, White's turn = negative (bad for me)."""
        score = encoder.encode("-100", white_to_move=True)
        assert score < 0
        assert abs(score + 0.1) < 0.01

    def test_negative_score_black_to_move(self, encoder):
        """Black winning, Black's turn = positive (good for me)."""
        score = encoder.encode("-100", white_to_move=False)
        assert score > 0
        assert abs(score - 0.1) < 0.01

    def test_zero_score(self, encoder):
        """Equal position = zero."""
        score_w = encoder.encode("0", white_to_move=True)
        score_b = encoder.encode("0", white_to_move=False)

        assert abs(score_w) < 0.01
        assert abs(score_b) < 0.01

    # ==================== PERSPECTIVE SYMMETRY ====================

    def test_perspective_symmetry(self, encoder):
        """Flipping perspective should flip sign."""
        white_score = encoder.encode("+200", white_to_move=True)
        black_score = encoder.encode("+200", white_to_move=False)

        # Should be opposite signs
        assert white_score * black_score < 0
        # Should have same magnitude
        assert abs(abs(white_score) - abs(black_score)) < 0.001

    def test_this_is_the_critical_fix(self, encoder_unnorm):
        """
        THIS IS THE CRITICAL FIX!

        Old code: return abs(score)  # WRONG - loses perspective
        New code: flip sign for Black's turn

        This test verifies the fix works correctly.
        """
        # Position: White is better by 300cp

        # White to move: I'm winning, score = +300
        white_view = encoder_unnorm.encode("+300", white_to_move=True)
        assert white_view == 300

        # Black to move: I'm losing, score = -300
        black_view = encoder_unnorm.encode("+300", white_to_move=False)
        assert black_view == -300

        # The OLD broken code would give:
        # abs(300) = 300 for both colors (WRONG!)

        # The NEW fixed code gives:
        # +300 for White (good for me)
        # -300 for Black (bad for me)

    # ==================== MATE SCORES ====================

    def test_mate_score_white_mating(self, encoder):
        """White mates in 5 = very positive for White."""
        score = encoder.encode("#+5", white_to_move=True)
        assert score > 9  # Very high positive

    def test_mate_score_black_mating(self, encoder):
        """Black mates in 3 = very negative for White."""
        score_w = encoder.encode("#-3", white_to_move=True)
        assert score_w < -9  # Very high negative

        # For Black, this is good
        score_b = encoder.encode("#-3", white_to_move=False)
        assert score_b > 9  # Very high positive

    def test_closer_mate_higher_score(self, encoder_unnorm):
        """Mate in 1 should score higher than mate in 10."""
        mate_1 = encoder_unnorm.encode("#+1", white_to_move=True)
        mate_10 = encoder_unnorm.encode("#+10", white_to_move=True)

        assert mate_1 > mate_10  # Closer mate is better

    def test_mate_score_formats(self, encoder):
        """Test various mate score formats."""
        formats = ["#5", "#+5", "M5", "M+5"]

        for fmt in formats:
            score = encoder.encode(fmt, white_to_move=True)
            assert score > 9  # All should be high positive

    # ==================== DECODING ====================

    def test_decode_roundtrip(self, encoder):
        """Test encode -> decode roundtrip."""
        original_cp = 150
        encoded = encoder.encode_from_centipawns(original_cp, white_to_move=True)
        decoded = encoder.decode(encoded, white_to_move=True)

        assert abs(decoded - original_cp) < 1

    def test_decode_with_perspective_flip(self, encoder):
        """Test decode preserves perspective correctly."""
        # White's view: +150
        encoded = encoder.encode_from_centipawns(150, white_to_move=True)
        assert abs(encoder.decode(encoded, white_to_move=True) - 150) < 1

        # Black's view of same position: -150
        encoded_b = encoder.encode_from_centipawns(150, white_to_move=False)
        assert abs(encoder.decode(encoded_b, white_to_move=False) - 150) < 1

    # ==================== EDGE CASES ====================

    def test_score_clipping(self, encoder_unnorm):
        """Test extreme scores are clipped."""
        extreme = encoder_unnorm.encode("+20000", white_to_move=True)
        assert extreme <= 15000  # Should be clipped

    def test_various_score_formats(self, encoder):
        """Test various score string formats."""
        formats = ["+150", "150", "-50", "+0", "0"]

        for fmt in formats:
            score = encoder.encode(fmt, white_to_move=True)
            assert isinstance(score, float)

    # ==================== UTILITY FUNCTIONS ====================

    def test_is_mate_score(self, encoder):
        """Test mate score detection."""
        assert encoder.is_mate_score("#5")
        assert encoder.is_mate_score("#+5")
        assert encoder.is_mate_score("#-5")
        assert encoder.is_mate_score("M5")
        assert not encoder.is_mate_score("+150")
        assert not encoder.is_mate_score("-50")

    def test_get_mate_distance(self, encoder):
        """Test extracting mate distance."""
        assert encoder.get_mate_distance("#5") == 5
        assert encoder.get_mate_distance("#+5") == 5
        assert encoder.get_mate_distance("#-5") == -5
        assert encoder.get_mate_distance("+150") is None

    def test_score_to_win_probability(self, encoder):
        """Test converting score to win probability."""
        # Equal position = 50%
        prob_equal = encoder.score_to_win_probability(0.0)
        assert abs(prob_equal - 0.5) < 0.05

        # Winning position = high probability
        prob_winning = encoder.score_to_win_probability(1.0)  # +1000cp
        assert prob_winning > 0.9

        # Losing position = low probability
        prob_losing = encoder.score_to_win_probability(-1.0)  # -1000cp
        assert prob_losing < 0.1

    def test_explain_score(self):
        """Test score explanation."""
        assert "winning" in ScoreEncoder.explain_score(500).lower()
        assert "losing" not in ScoreEncoder.explain_score(500).lower()
        assert "equal" in ScoreEncoder.explain_score(10).lower()
        assert "mate" in ScoreEncoder.explain_score(9500).lower()
