"""
Cálculo de frecuencias de números en lotería.
Incluye frecuencias globales ponderadas y ventanas móviles.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def weighted_frequency(
    df: pd.DataFrame,
    alpha: float = 0.997
) -> Dict[int, float]:
    """
    Calcula frecuencia ponderada exponencial de cada número.
    
    Los sorteos más recientes tienen mayor peso mediante
    decaimiento exponencial con factor alpha.
    
    Args:
        df: DataFrame con sorteos históricos
        alpha: Factor de decaimiento exponencial (0 < alpha < 1)
    
    Returns:
        Diccionario {numero: frecuencia_ponderada}
    """
    n_draws = len(df)
    weights = np.array([alpha ** i for i in range(n_draws)][::-1])
    
    freq = {}
    for idx, row in enumerate(df.values):
        for num in row:
            num = int(num)
            freq[num] = freq.get(num, 0) + weights[idx]
    
    # Normalizar
    total = sum(freq.values())
    if total > 0:
        freq = {k: v / total for k, v in freq.items()}
    
    return freq


def rolling_frequency(
    df: pd.DataFrame,
    window: int = 20,
    max_number: int = 39
) -> Dict[int, float]:
    """
    Calcula frecuencia en una ventana móvil de los últimos N sorteos.
    
    Args:
        df: DataFrame con sorteos históricos
        window: Tamaño de la ventana
        max_number: Número máximo (para incluir todos los números)
    
    Returns:
        Diccionario {numero: frecuencia_en_ventana}
    """
    recent = df.tail(window)
    freq = {}
    
    for row in recent.values:
        for num in row:
            num = int(num)
            freq[num] = freq.get(num, 0) + 1
    
    # Normalizar por tamaño de ventana
    freq = {
        n: freq.get(n, 0) / max(window, 1)
        for n in range(1, max_number + 1)
    }
    
    return freq


def multi_window_frequencies(
    df: pd.DataFrame,
    windows: List[int] = None,
    max_number: int = 39
) -> Dict[int, Dict[int, float]]:
    """
    Calcula frecuencias para múltiples ventanas de tiempo.
    
    Args:
        df: DataFrame con sorteos históricos
        windows: Lista de tamaños de ventana
        max_number: Número máximo
    
    Returns:
        Diccionario {ventana: {numero: frecuencia}}
    """
    if windows is None:
        windows = [5, 10, 20, 50]
    
    result = {}
    for window in windows:
        result[window] = rolling_frequency(df, window, max_number)
    
    return result


def exponential_moving_average(
    series: List[float],
    alpha: float = 0.3
) -> float:
    """
    Calcula la media móvil exponencial de una serie.
    
    Args:
        series: Lista de valores
        alpha: Factor de suavizado (0 < alpha < 1)
    
    Returns:
        Último valor de la EMA
    """
    if not series:
        return 0.0
    
    ema = series[0]
    for val in series[1:]:
        ema = alpha * val + (1 - alpha) * ema
    
    return ema


def frequency_features(
    df: pd.DataFrame,
    number: int,
    windows: List[int] = None,
    max_number: int = 39
) -> Dict[str, float]:
    """
    Genera todas las features de frecuencia para un número específico.
    
    Args:
        df: DataFrame con sorteos históricos
        number: Número del cual calcular features
        windows: Lista de ventanas a usar
        max_number: Número máximo
    
    Returns:
        Diccionario con features de frecuencia
    """
    if windows is None:
        windows = [5, 10, 20, 50]
    
    # Frecuencia global ponderada
    global_freq = weighted_frequency(df)
    
    # Frecuencias por ventana
    rolling_freqs = {}
    for window in windows:
        freq = rolling_frequency(df, window, max_number)
        rolling_freqs[window] = freq.get(number, 0)
    
    # EMA de las frecuencias rolling
    ema_values = list(rolling_freqs.values())
    ema_signal = exponential_moving_average(ema_values)
    
    # Entropía de las frecuencias
    from scipy.stats import entropy as calc_entropy
    freq_values = [v + 1e-9 for v in rolling_freqs.values()]
    entropy_signal = calc_entropy(freq_values)
    
    features = {
        "global_freq": global_freq.get(number, 0),
        "ema_signal": ema_signal,
        "entropy": entropy_signal,
    }
    
    # Agregar frecuencias por ventana
    for window in windows:
        features[f"freq{window}"] = rolling_freqs.get(window, 0)
    
    return features
