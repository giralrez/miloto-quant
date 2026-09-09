"""
Validación de modelos con walk-forward y expanding window.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, BaseEstimator],
    n_splits: int = 5,
    model_type: str = "regressor"
) -> Dict[str, float]:
    """
    Validación walk-forward con TimeSeriesSplit.
    
    Args:
        X: Features
        y: Target
        models: Diccionario {nombre: modelo}
        n_splits: Número de splits
        model_type: Tipo de modelo ("classifier" o "regressor")
    
    Returns:
        Diccionario {nombre_modelo: score}
    """
    logger.info(f"Walk-Forward Validation con {n_splits} splits")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {}
    
    for name, model in models.items():
        logger.info(f"Evaluando {name}...")
        
        fold_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Entrenar
            model.fit(X_train, y_train)
            
            # Predecir
            y_pred = model.predict(X_test)
            
            # Calcular score
            if model_type == "regressor":
                mae = mean_absolute_error(y_test, y_pred)
                score = 1 / (mae + 1e-6)
            else:
                score = model.score(X_test, y_test)
            
            fold_scores.append(score)
        
        scores[name] = np.mean(fold_scores)
        logger.info(f"  {name}: {scores[name]:.6f}")
    
    return scores


def expanding_window_validation(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, BaseEstimator],
    min_train_size: int = 100,
    step: int = 50,
    model_type: str = "regressor"
) -> Dict[str, List[float]]:
    """
    Validación con ventana expansiva.
    
    Args:
        X: Features
        y: Target
        models: Diccionario {nombre: modelo}
        min_train_size: Tamaño mínimo de entrenamiento
        step: Tamaño del paso
        model_type: Tipo de modelo
    
    Returns:
        Diccionario {nombre_modelo: [scores_por_fold]}
    """
    logger.info(f"Expanding Window Validation (min_train={min_train_size}, step={step})")
    
    results = {name: [] for name in models}
    
    for start in range(min_train_size, len(X) - step, step):
        X_train = X[:start]
        y_train = y[:start]
        X_test = X[start:start + step]
        y_test = y[start:start + step]
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if model_type == "regressor":
                mae = mean_absolute_error(y_test, y_pred)
                score = 1 / (mae + 1e-6)
            else:
                score = model.score(X_test, y_test)
            
            results[name].append(score)
    
    for name in results:
        scores = results[name]
        logger.info(f"  {name}: mean={np.mean(scores):.6f}, std={np.std(scores):.6f}")
    
    return results


def calculate_optimal_weights(
    scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Calcula pesos óptimos basados en scores de validación.
    
    Args:
        scores: {nombre_modelo: score}
    
    Returns:
        Pesos óptimos normalizados
    """
    total = sum(scores.values())
    
    if total > 0:
        return {k: v / total for k, v in scores.items()}
    
    n = len(scores)
    return {k: 1.0 / n for k in scores}


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, BaseEstimator],
    cv: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Validación cruzada para múltiples modelos.
    
    Args:
        X: Features
        y: Target
        models: Diccionario {nombre: modelo}
        cv: Número de folds
    
    Returns:
        Diccionario {nombre_modelo: {mean_score, std_score, scores}}
    """
    from sklearn.model_selection import cross_val_score
    
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        scores = -scores  # Convertir a positivo
        
        results[name] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "scores": scores.tolist()
        }
        
        logger.info(f"{name}: MAE={np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    return results
