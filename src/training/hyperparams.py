"""
Optimización de hiperparámetros con Optuna.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def optimize_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    cv: int = 3
) -> Dict[str, float]:
    """
    Optimiza hiperparámetros de LightGBM con Optuna.
    
    Args:
        X: Features
        y: Target
        n_trials: Número de intentos
        cv: Número de folds
    
    Returns:
        Mejores hiperparámetros
    """
    try:
        import optuna
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.warning("Optuna no instalado. Usando hiperparámetros por defecto.")
        return {
            "n_estimators": 200,
            "learning_rate": 0.02,
            "max_depth": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85
        }
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": 4,
            "verbosity": -1
        }
        
        model = LGBMRegressor(**params)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        
        return -scores.mean()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Mejores hiperparámetros LGBM: {study.best_params}")
    
    return study.best_params


def optimize_xgb(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    cv: int = 3
) -> Dict[str, float]:
    """
    Optimiza hiperparámetros de XGBoost con Optuna.
    
    Args:
        X: Features
        y: Target
        n_trials: Número de intentos
        cv: Número de folds
    
    Returns:
        Mejores hiperparámetros
    """
    try:
        import optuna
        from xgboost import XGBRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.warning("Optuna no instalado. Usando hiperparámetros por defecto.")
        return {
            "n_estimators": 180,
            "learning_rate": 0.025,
            "max_depth": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85
        }
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": 4,
            "tree_method": "hist",
            "verbosity": 0
        }
        
        model = XGBRegressor(**params)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        
        return -scores.mean()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Mejores hiperparámetros XGB: {study.best_params}")
    
    return study.best_params


def optimize_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    cv: int = 3
) -> Dict[str, float]:
    """
    Optimiza hiperparámetros de Random Forest con Optuna.
    
    Args:
        X: Features
        y: Target
        n_trials: Número de intentos
        cv: Número de folds
    
    Returns:
        Mejores hiperparámetros
    """
    try:
        import optuna
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.warning("Optuna no instalado. Usando hiperparámetros por defecto.")
        return {
            "n_estimators": 180,
            "max_depth": 12
        }
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "random_state": 42,
            "n_jobs": 4
        }
        
        model = RandomForestRegressor(**params)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        
        return -scores.mean()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Mejores hiperparámetros RF: {study.best_params}")
    
    return study.best_params


def optimize_all_models(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    cv: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Optimiza hiperparámetros de todos los modelos.
    
    Args:
        X: Features
        y: Target
        n_trials: Número de intentos por modelo
        cv: Número de folds
    
    Returns:
        Diccionario con mejores hiperparámetros por modelo
    """
    logger.info("Optimizando hiperparámetros de todos los modelos...")
    
    results = {
        "lightgbm": optimize_lgbm(X, y, n_trials, cv),
        "xgboost": optimize_xgb(X, y, n_trials, cv),
        "randomforest": optimize_rf(X, y, n_trials, cv)
    }
    
    return results
