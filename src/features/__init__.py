"""
Módulo de feature engineering.
"""

from .frequencies import (
    weighted_frequency,
    rolling_frequency,
    multi_window_frequencies,
    exponential_moving_average,
    frequency_features
)
from .gaps import calculate_gaps, gap_features
from .momentum import momentum, acceleration, volatility, momentum_features
from .statistical import (
    draw_statistics,
    pair_matrix,
    pair_strength,
    tercile_features,
    number_features
)
from .builder import (
    build_features_v1,
    build_features_v3,
    build_latest_features_v3
)

__all__ = [
    # Frequencies
    "weighted_frequency",
    "rolling_frequency",
    "multi_window_frequencies",
    "exponential_moving_average",
    "frequency_features",
    
    # Gaps
    "calculate_gaps",
    "gap_features",
    
    # Momentum
    "momentum",
    "acceleration",
    "volatility",
    "momentum_features",
    
    # Statistical
    "draw_statistics",
    "pair_matrix",
    "pair_strength",
    "tercile_features",
    "number_features",
    
    # Builder
    "build_features_v1",
    "build_features_v3",
    "build_latest_features_v3"
]
