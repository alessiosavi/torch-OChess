"""
Data generation module for Torch o'Chess.

Provides tools to generate labeled training data using Stockfish:
    - Position generation (random, from openings, tactical)
    - Position labeling (best moves, scores, captures)
    - Good vs bad move examples
"""

from ochess.datagen.stockfish_wrapper import StockfishWrapper, AnalysisResult
from ochess.datagen.position_generator import PositionGenerator, PositionType
from ochess.datagen.pipeline import DataGenerationPipeline, LabeledPosition

__all__ = [
    "StockfishWrapper",
    "AnalysisResult",
    "PositionGenerator",
    "PositionType",
    "DataGenerationPipeline",
    "LabeledPosition",
]
