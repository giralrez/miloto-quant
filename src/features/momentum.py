"""
Cálculo de momentum, aceleración y volatilidad.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def momentum(
    freq_short: float,
    freq_long: float
) -> float:
    """
    Calcula el momentum (tendencia) de un número.
    
    Args:
        freq_short: Frecuencia a corto plazo
        freq_long: Frecuencia a largo plazo
    
    Returns:
        Valor de momentum (positivo = tendencia alcista)
    """
    return freq_short - freq_long


def acceleration(
    freq_very_short: float,
    freq_short: float
) -> float:
    """
    Calcula la aceleración de un número.
    
    Args:
        freq_very_short: Frecuencia muy corto plazo
        freq_short: Frecuencia corto plazo
    
    Returns:
        Valor de aceleración
    """
    return freq_very_short - freq_short


def volatility(
    freq_values: List[float]
) -> float:
    """
    Calcula la volatilidad de las frecuencias.
    
    Args:
        freq_values: Lista de frecuencias en diferentes ventanas
    
    Returns:
        Desviación estándar de las frecuencias
    """
    if not freq_values:
        return 0.0
    
    return float(np.std(freq_values))


def momentum_features(
    freq5: float,
    freq10: float,
    freq20: float,
    freq50: float
) -> Dict[str, float]:
    """
    Genera todas las features de momentum para un número.
    
    Args:
        freq5: Frecuencia últimos 5 sorteos
        freq10: Frecuencia últimos 10 sorteos
        freq20: Frecuencia últimos 20 sorteos
        freq50: Frecuencia últimos 50 sorteos
    
    Returns:
        Diccionario con features de momentum
    """
    # Momentum: comparación corto vs largo plazo
    mom_10_50 = momentum(freq10, freq50)
    mom_5_20 = momentum(freq5, freq20)
    
    # Aceleración: cambio en el momentum
    accel = acceleration(freq5, freq10)
    
    # Volatilidad
    vol = volatility([freq5, freq10, freq20, freq50])
    
    features = {
        "momentum_10_50": mom_10_50,
        "momentum_5_20": mom_5_20,
        "acceleration": accel,
        "volatility": vol,
    }
    
    return features
