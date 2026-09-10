"""
Módulo de entrenamiento y validación.
"""

from .trainer import (
    train_model,
    train_ensemble,
    train_with_validation,
    get_model_info
)
from .validator import (
    walk_forward_validation,
    expanding_window_validation,
    calculate_optimal_weights,
    cross_validate_models
)
from .hyperparams import (
    optimize_lgbm,
    optimize_xgb,
    optimize_rf,
    optimize_all_models
)

__all__ = [
    # Trainer
    "train_model",
    "train_ensemble",
    "train_with_validation",
    "get_model_info",
    
    # Validator
    "walk_forward_validation",
    "expanding_window_validation",
    "calculate_optimal_weights",
    "cross_validate_models",
    
    # Hyperparams
    "optimize_lgbm",
    "optimize_xgb",
    "optimize_rf",
    "optimize_all_models"
]
