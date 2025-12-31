# The Critical Fix: Score Encoding

This document explains the most important fix in the Torch o'Chess rewrite - proper handling of evaluation scores for both colors.

## The Problem

### Original Code (Broken)

In the original `OChess.py`, line 141 contained this code:

```python
def encode_score(score_str):
    # ... parsing logic ...
    return abs(score)  # BUG: Loses perspective information!
```

### Why This Was Wrong

Using `abs(score)` means:

- White winning by 150 centipawns → score = 150
- Black winning by 150 centipawns → score = 150 (SAME!)

The network couldn't distinguish between winning and losing positions because **both looked positive**.

### The Symptom

The network only played effectively as Black. Why? Because:

1. Training data mixed White and Black positions
2. All scores were positive (due to `abs()`)
3. Network learned that "high score = good move"
4. But this only worked when randomly correct (50% of the time)
5. Black positions happened to align better with this broken logic

## The Solution

### Side-to-Move Perspective

The fix is simple but crucial: **encode scores from the current player's perspective**.

```python
class ScoreEncoder:
    def encode(self, stockfish_score: str, white_to_move: bool) -> float:
        """
        Encode score from side-to-move perspective.

        Convention:
        - Positive = good for the player whose turn it is
        - Negative = bad for the player whose turn it is
        - Zero = equal position
        """
        # Parse the centipawn value (Stockfish reports from White's view)
        centipawns = self._parse_score_string(stockfish_score)

        # THE CRITICAL FIX: Flip sign for Black's turn
        if not white_to_move:
            centipawns = -centipawns

        # Clip extreme values
        centipawns = max(-15000, min(15000, centipawns))

        # Normalize to roughly [-15, +15] range
        return centipawns / 1000.0
```

### Example Walkthrough

**Position**: White has a material advantage of 150 centipawns (+1.5 pawns)

| Scenario | Stockfish Says | Old Code | New Code |
|----------|---------------|----------|----------|
| White to move | +150 | +150 | +150 "I'm winning" |
| Black to move | +150 | +150 (WRONG) | -150 "I'm losing" |

**Position**: Black has a material advantage of 200 centipawns

| Scenario | Stockfish Says | Old Code | New Code |
|----------|---------------|----------|----------|
| White to move | -200 | +200 (WRONG) | -200 "I'm losing" |
| Black to move | -200 | +200 (WRONG) | +200 "I'm winning" |

## The Complete Picture: Board Flipping

Score encoding is only half the fix. We also need **board flipping**:

### Why Flip the Board?

Without flipping, the network sees:

- White pieces always at rows 6-7 (ranks 1-2)
- Black pieces always at rows 0-1 (ranks 7-8)

This creates an asymmetry - the network learns different patterns for each color.

### The Solution: Normalize Perspective

When it's Black's turn, we:

1. Flip the board 180 degrees
2. Swap piece colors (White ↔ Black)
3. Negate the score

Now the network **always** sees:

- "My" pieces at the bottom (rows 6-7)
- "Opponent" pieces at the top (rows 0-1)
- Positive score = I'm winning

```python
def normalize_for_player(board_tensor, white_to_move):
    """Normalize board to current player's perspective."""
    if white_to_move:
        return board_tensor  # No change needed

    # Flip board 180 degrees
    flipped = board_tensor.flip(0).flip(1)

    # Swap colors: White (1-6) ↔ Black (7-12)
    swapped = swap_piece_colors(flipped)

    return swapped
```

### Visual Example

**Original position (Black to move):**

```
   a  b  c  d  e  f  g  h
8  r  .  .  .  k  .  .  r    ← Black's pieces
7  p  p  p  .  .  p  p  p
6  .  .  .  .  .  .  .  .
5  .  .  .  p  p  .  .  .
4  .  .  .  P  P  .  .  .
3  .  .  .  .  .  .  .  .
2  P  P  P  .  .  P  P  P
1  R  .  .  .  K  .  .  R    ← White's pieces

Score: +100 (White is slightly better)
```

**After normalization:**

```
   a  b  c  d  e  f  g  h
8  R  .  .  .  K  .  .  R    ← "Opponent" (was White)
7  P  P  P  .  .  P  P  P
6  .  .  .  .  .  .  .  .
5  .  .  .  P  P  .  .  .    ← Colors swapped
4  .  .  .  p  p  .  .  .
3  .  .  .  .  .  .  .  .
2  p  p  p  .  .  p  p  p
1  r  .  .  .  k  .  .  r    ← "Me" (was Black)

Score: -100 (I'm slightly worse)
```

The network now sees:

- My pieces (lowercase, but network sees them as "friendly") at bottom
- Score is -100 meaning "I'm at a disadvantage"
- This is consistent regardless of which color is actually playing

## Implementation Details

### Score Encoding Class

```python
@dataclass
class ScoreEncoderConfig:
    normalize: bool = True      # Normalize to ~[-15, 15]
    scale: float = 1000.0       # Divide by this for normalization
    max_score: int = 15000      # Clip scores beyond this
    mate_score: int = 10000     # Base value for mate scores

class ScoreEncoder:
    def __init__(self, config: ScoreEncoderConfig = None):
        self.config = config or ScoreEncoderConfig()

    def encode(self, score_str: str, white_to_move: bool) -> float:
        """Main encoding function."""
        cp = self._parse_score_string(score_str)

        if not white_to_move:
            cp = -cp  # THE FIX

        cp = self._clip(cp)

        if self.config.normalize:
            return cp / self.config.scale
        return float(cp)

    def _parse_score_string(self, score_str: str) -> int:
        """Parse various score formats."""
        score_str = score_str.strip()

        # Mate scores: #+5, #-3, M5, M-3
        if '#' in score_str or 'M' in score_str.upper():
            return self._parse_mate_score(score_str)

        # Regular centipawn scores: +150, -50, 100
        return int(score_str.replace('+', ''))

    def _parse_mate_score(self, score_str: str) -> int:
        """Convert mate-in-N to centipawns."""
        # Extract the number
        import re
        match = re.search(r'[#M]([+-]?\d+)', score_str, re.IGNORECASE)
        if not match:
            return 0

        moves = int(match.group(1))

        # Closer mate = higher score
        # Mate in 1 > Mate in 10
        base = self.config.mate_score
        adjustment = abs(moves) * 100

        if moves > 0:  # I'm mating
            return base - adjustment
        else:  # I'm getting mated
            return -(base - adjustment)
```

### Mate Score Handling

Mate scores need special handling:

| Score String | Meaning | Encoded Value |
|--------------|---------|---------------|
| #+1 | White mates in 1 | +9900 |
| #+5 | White mates in 5 | +9500 |
| #-1 | Black mates in 1 | -9900 |
| #-5 | Black mates in 5 | -9500 |

Closer mates have higher magnitude because they're more valuable.

## Testing the Fix

The test suite includes explicit verification of this fix:

```python
def test_this_is_the_critical_fix(self):
    """
    Verify the fix works correctly.

    Old code: return abs(score)  # WRONG
    New code: flip sign for Black's turn
    """
    encoder = ScoreEncoder(normalize=False)

    # Position: White is better by 300cp

    # White to move: I'm winning
    white_view = encoder.encode("+300", white_to_move=True)
    assert white_view == 300

    # Black to move: I'm losing
    black_view = encoder.encode("+300", white_to_move=False)
    assert black_view == -300  # NOT +300!

def test_perspective_symmetry(self):
    """Flipping perspective should flip sign."""
    encoder = ScoreEncoder(normalize=True)

    white_score = encoder.encode("+200", white_to_move=True)
    black_score = encoder.encode("+200", white_to_move=False)

    # Should be opposite signs
    assert white_score * black_score < 0

    # Should have same magnitude
    assert abs(abs(white_score) - abs(black_score)) < 0.001
```

## Summary

| Aspect | Old (Broken) | New (Fixed) |
|--------|--------------|-------------|
| Score sign | Always positive | Relative to side-to-move |
| Board orientation | Fixed | Flipped for Black |
| Network sees | Asymmetric patterns | Consistent perspective |
| Result | Only plays Black well | Plays both colors |

The key insight: **The network should always think it's the same player at the bottom of the board, trying to maximize score.** This is achieved through:

1. **Score negation** when Black to move
2. **Board flipping** when Black to move
3. **Color swapping** when Black to move

Together, these transformations ensure the network learns symmetric, color-agnostic chess knowledge.
