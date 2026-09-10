"""
Predicción de scores para números de lotería.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def predict_scores(
    models: Dict[str, BaseEstimator],
    weights: Dict[str, float],
    X_latest: np.ndarray,
    numbers: np.ndarray
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """
    Predice scores para cada número con cada modelo.
    
    Args:
        models: Diccionario {nombre: modelo}
        weights: Diccionario {nombre: peso}
        X_latest: Features para predicción (39 filas)
        numbers: Array de números [1, 2, ..., 39]
    
    Returns:
        Tupla de (scores_finales, scores_por_modelo)
    """
    final_scores = np.zeros(len(numbers))
    per_model = {}
    
    for idx, number in enumerate(numbers):
        feats = X_latest[idx].reshape(1, -1)
        
        model_scores = {}
        score = 0
        
        for name, model in models.items():
            pred = model.predict(feats)[0]
            pred = max(pred, 0)  # Asegurar positivo
            
            model_scores[name] = pred
            score += pred * weights.get(name, 1.0)
        
        per_model[int(number)] = model_scores
        final_scores[idx] = score
    
    return final_scores, per_model


def predict_top_numbers(
    scores: np.ndarray,
    numbers: np.ndarray,
    top_k: int = 10
) -> np.ndarray:
    """
    Obtiene los top-K números con mayor score.
    
    Args:
        scores: Array de scores
        numbers: Array de números
        top_k: Cantidad de top números a retornar
    
    Returns:
        Array con los top-K números ordenados por score
    """
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return numbers[top_indices]


def rank_numbers(
    scores: np.ndarray,
    numbers: np.ndarray
) -> np.ndarray:
    """
    Ordena números por score de mayor a menor.
    
    Args:
        scores: Array de scores
        numbers: Array de números
    
    Returns:
        Array de números ordenados por score
    """
    sorted_indices = np.argsort(scores)[::-1]
    return numbers[sorted_indices]
