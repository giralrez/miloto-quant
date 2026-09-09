"""
Motor de probabilidad final.
Combina scores ML con prior Bayesiano.
"""

import logging
from typing import Dict

import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


def bayesian_prior(
    df: np.ndarray,
    max_number: int = 39
) -> np.ndarray:
    """
    Calcula prior Bayesiano basado en frecuencias históricas.
    
    Args:
        df: DataFrame con sorteos históricos
        max_number: Número máximo
    
    Returns:
        Array de probabilidades prior
    """
    total_draws = len(df)
    prior = np.zeros(max_number)
    
    for n in range(1, max_number + 1):
        count = sum(n in row for row in df.values)
        prior[n - 1] = (count + 1) / (total_draws + 2)  # Suavizado de Laplace
    
    return prior


def final_probability(
    ml_scores: np.ndarray,
    bayes_prior: np.ndarray,
    ml_weight: float = 0.82,
    bayes_weight: float = 0.18,
    temperature: float = 0.78,
    clip_min: float = 0.001,
    clip_max: float = 0.35
) -> np.ndarray:
    """
    Calcula la probabilidad final combinando ML y Bayes.
    
    Args:
        ml_scores: Scores del modelo ML
        bayes_prior: Prior Bayesiano
        ml_weight: Peso del componente ML
        bayes_weight: Peso del componente Bayesiano
        temperature: Temperatura del softmax
        clip_min: Mínimo de clipping
        clip_max: Máximo de clipping
    
    Returns:
        Array de probabilidades finales
    """
    # Combinar ML y Bayes
    combined = ml_weight * ml_scores + bayes_weight * bayes_prior
    
    # Aplicar softmax con temperatura
    probs = softmax(combined / temperature)
    
    # Clipping
    probs = np.clip(probs, clip_min, clip_max)
    
    # Renormalizar
    probs = probs / probs.sum()
    
    return probs


def probability_features(
    df: np.ndarray,
    max_number: int = 39
) -> Dict[str, np.ndarray]:
    """
    Genera features de probabilidad para cada número.
    
    Args:
        df: DataFrame con sorteos históricos
        max_number: Número máximo
    
    Returns:
        Diccionario con diferentes medidas de probabilidad
    """
    total_draws = len(df)
    
    # Frecuencia absoluta
    freq_abs = np.zeros(max_number)
    for n in range(1, max_number + 1):
        freq_abs[n - 1] = sum(n in row for row in df.values)
    
    # Frecuencia relativa
    freq_rel = freq_abs / total_draws
    
    # Prior Bayesiano
    bayes = bayesian_prior(df, max_number)
    
    return {
        "frequency_absolute": freq_abs,
        "frequency_relative": freq_rel,
        "bayesian_prior": bayes
    }
