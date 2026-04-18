"""Controlled adaptive loop (paper/shadow): observation → diagnosis → apply → evaluate."""

from .bot_hook import prepare_ml_buy_decision
from .engine import AdaptiveEngine
from .integration import effective_ml_threshold_for_entry
from .trading_integration import (
    BLOCK_ADAPTIVE,
    BLOCK_ADAPTIVE_BUCKET,
    build_adaptive_strategy_decision_details,
    evaluate_ml_buy_with_adaptive,
)

__all__ = [
    "AdaptiveEngine",
    "BLOCK_ADAPTIVE",
    "BLOCK_ADAPTIVE_BUCKET",
    "build_adaptive_strategy_decision_details",
    "effective_ml_threshold_for_entry",
    "evaluate_ml_buy_with_adaptive",
    "prepare_ml_buy_decision",
]
