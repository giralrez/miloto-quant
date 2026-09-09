"""
Métricas de evaluación para predicciones de lotería.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def hit_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 5
) -> float:
    """
    Calcula la tasa de aciertos en los top-K predichos.
    
    Args:
        y_true: Valores reales (conjunto de números)
        y_pred: Números predichos (top-K)
        k: Cantidad de números predichos
    
    Returns:
        Proporción de aciertos
    """
    true_set = set(y_true)
    pred_set = set(y_pred[:k])
    
    hits = len(true_set.intersection(pred_set))
    
    return hits / k


def expected_return(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 5,
    prizes: Dict[int, float] = None
) -> float:
    """
    Calcula el retorno esperado basado en aciertos.
    
    Args:
        y_true: Valores reales
        y_pred: Números predichos
        k: Cantidad de números predichos
        prizes: Diccionario {aciertos: premio}
    
    Returns:
        Retorno esperado
    """
    if prizes is None:
        prizes = {
            5: 1000000,
            4: 10000,
            3: 100,
            2: 10,
            1: 0,
            0: 0
        }
    
    true_set = set(y_true)
    pred_set = set(y_pred[:k])
    
    hits = len(true_set.intersection(pred_set))
    
    return prizes.get(hits, 0)


def coverage(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    k: int = 5
) -> float:
    """
    Calcula la cobertura de predicciones (cuántos sorteos tienen al menos 1 acierto).
    
    Args:
        y_true_list: Lista de valores reales por sorteo
        y_pred_list: Lista de predicciones por sorteo
        k: Cantidad de números predichos
    
    Returns:
        Proporción de sorteos con al menos 1 acierto
    """
    total = len(y_true_list)
    at_least_one = 0
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        true_set = set(y_true)
        pred_set = set(y_pred[:k])
        
        if len(true_set.intersection(pred_set)) > 0:
            at_least_one += 1
    
    return at_least_one / total if total > 0 else 0


def average_hits(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    k: int = 5
) -> float:
    """
    Calcula el promedio de aciertos por sorteo.
    
    Args:
        y_true_list: Lista de valores reales por sorteo
        y_pred_list: Lista de predicciones por sorteo
        k: Cantidad de números predichos
    
    Returns:
        Promedio de aciertos
    """
    total_hits = 0
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        true_set = set(y_true)
        pred_set = set(y_pred[:k])
        
        total_hits += len(true_set.intersection(pred_set))
    
    return total_hits / len(y_true_list) if y_true_list else 0


def hit_distribution(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    k: int = 5
) -> Dict[int, int]:
    """
    Calcula la distribución de aciertos.
    
    Args:
        y_true_list: Lista de valores reales por sorteo
        y_pred_list: Lista de predicciones por sorteo
        k: Cantidad de números predichos
    
    Returns:
        Diccionario {n_aciertos: frecuencia}
    """
    distribution = {i: 0 for i in range(k + 1)}
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        true_set = set(y_true)
        pred_set = set(y_pred[:k])
        
        hits = len(true_set.intersection(pred_set))
        distribution[hits] = distribution.get(hits, 0) + 1
    
    return distribution


def metrics_summary(
    hits: List[int],
    k: int = 5
) -> Dict[str, float]:
    """
    Genera un resumen de métricas.
    
    Args:
        hits: Lista de aciertos por sorteo
        k: Cantidad de números predichos
    
    Returns:
        Diccionario con métricas resumen
    """
    hits_array = np.array(hits)
    
    return {
        "total_draws": len(hits),
        "total_hits": int(hits_array.sum()),
        "average_hits": float(hits_array.mean()),
        "max_hits": int(hits_array.max()),
        "min_hits": int(hits_array.min()),
        "std_hits": float(hits_array.std()),
        "hits_per_draw": float(hits_array.mean() / k),
        "coverage": float(np.mean(hits_array > 0)),
    }
