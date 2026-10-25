"""Reusable field regression, trend analysis, and statistical inference.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
"""
from .alignment import align_years
from .inference import benjamini_hochberg, lag1_bartlett
from .regression import linear_regression
from .trends import circular_block_bootstrap, ols_trend, sen_trend

__version__ = "0.1.0"
__all__ = ["align_years", "linear_regression", "lag1_bartlett", "benjamini_hochberg",
           "ols_trend", "sen_trend", "circular_block_bootstrap"]
