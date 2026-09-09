"""
Módulo de evaluación y backtest.
"""

from .metrics import (
    hit_rate,
    expected_return,
    coverage,
    average_hits,
    hit_distribution,
    metrics_summary
)
from .backtest import BacktestEngine, run_quick_backtest

__all__ = [
    # Metrics
    "hit_rate",
    "expected_return",
    "coverage",
    "average_hits",
    "hit_distribution",
    "metrics_summary",
    
    # Backtest
    "BacktestEngine",
    "run_quick_backtest"
]
