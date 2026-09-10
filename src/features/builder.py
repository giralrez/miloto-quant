"""
Orquestador de feature engineering.
Combina todos los módulos para generar datasets de features.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .frequencies import (
    weighted_frequency,
    rolling_frequency,
    multi_window_frequencies,
    exponential_moving_average,
    frequency_features
)
from .gaps import calculate_gaps, gap_features
from .momentum import momentum_features
from .statistical import (
    pair_matrix,
    pair_strength,
    number_features
)

logger = logging.getLogger(__name__)


def build_features_v1(
    df: pd.DataFrame,
    min_history: int = 30,
    max_number: int = 39,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Genera features estilo V1 (1 fila por sorteo).
    
    Args:
        df: DataFrame con sorteos históricos
        min_history: Mínimo de sorteos históricos requeridos
        max_number: Número máximo
        windows: Ventanas de frecuencia a usar
    
    Returns:
        DataFrame con features
    """
    if windows is None:
        windows = [5, 10, 20, 50]
    
    rows = []
    
    for i in range(min_history, len(df)):
        historical = df.iloc[:i]
        
        # Frecuencias globales y rolling
        global_freq = weighted_frequency(historical)
        rolling_freqs = multi_window_frequencies(historical, windows, max_number)
        
        # Gaps
        gaps = calculate_gaps(historical, max_number)
        
        # Para cada número, calcular features
        for number in range(1, max_number + 1):
            feat = {}
            
            # Features del número
            feat.update(number_features(number, max_number))
            
            # Features de frecuencia
            freq_feats = frequency_features(
                historical, number, windows, max_number
            )
            feat.update(freq_feats)
            
            # Features de gap
            gap_feats = gap_features(historical, number, max_number)
            feat.update(gap_feats)
            
            # Guardar con el sorteo de referencia
            feat["draw_idx"] = i
            feat["number"] = number
            
            rows.append(feat)
    
    return pd.DataFrame(rows)


def build_features_v3(
    df: pd.DataFrame,
    min_history: int = 60,
    max_number: int = 39,
    windows: List[int] = None,
    pair_matrix_window: int = 80
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera features estilo V3 (motor de ranking).
    
    Args:
        df: DataFrame con sorteos históricos
        min_history: Mínimo de sorteos históricos requeridos
        max_number: Número máximo
        windows: Ventanas de frecuencia a usar
        pair_matrix_window: Ventana para matriz de co-ocurrencia
    
    Returns:
        Tupla de (X, y) donde X son features y y es el target binario
    """
    if windows is None:
        windows = [5, 10, 20, 50]
    
    X = []
    y = []
    
    for i in range(min_history, len(df) - 1):
        historical = df.iloc[:i]
        next_draw = set(df.iloc[i + 1].values)
        
        # Frecuencias
        global_freq = weighted_frequency(historical)
        rolling_freqs = multi_window_frequencies(historical, windows, max_number)
        
        # Gaps
        gaps = calculate_gaps(historical, max_number)
        
        # Matriz de co-ocurrencia
        pair_mat = pair_matrix(historical, pair_matrix_window, max_number)
        
        # Para cada número, generar vector de features
        for number in range(1, max_number + 1):
            freq_vals = [rolling_freqs[w].get(number, 0) for w in windows]
            
            feat = [
                # Frecuencias
                global_freq.get(number, 0),
                *[rolling_freqs[w].get(number, 0) for w in windows],
                
                # Momentum y aceleración
                freq_vals[1] - freq_vals[3] if len(freq_vals) > 3 else 0,  # momentum_10_50
                freq_vals[0] - freq_vals[2] if len(freq_vals) > 2 else 0,  # momentum_5_20
                freq_vals[0] - freq_vals[1] if len(freq_vals) > 1 else 0,  # acceleration
                
                # Gap
                gaps.get(number, len(df)),
                
                # Pair strength
                pair_strength(pair_mat, number),
                
                # Volatilidad
                float(np.std(freq_vals)),
                
                # EMA
                exponential_moving_average(freq_vals),
                
                # Features del número
                number / max_number,
                number % 2,
                int(number <= 13),
                int(14 <= number <= 26),
                int(number >= 27),
            ]
            
            X.append(feat)
            y.append(1 if number in next_draw else 0)
    
    return np.array(X), np.array(y)


def build_latest_features_v3(
    df: pd.DataFrame,
    max_number: int = 39,
    windows: List[int] = None,
    pair_matrix_window: int = 80
) -> np.ndarray:
    """
    Genera features para predicción del próximo sorteo (V3).
    
    Args:
        df: DataFrame con todos los sorteos históricos
        max_number: Número máximo
        windows: Ventanas de frecuencia a usar
        pair_matrix_window: Ventana para matriz de co-ocurrencia
    
    Returns:
        Array de features (39 filas, una por número)
    """
    if windows is None:
        windows = [5, 10, 20, 50]
    
    historical = df.copy()
    
    # Frecuencias
    global_freq = weighted_frequency(historical)
    rolling_freqs = multi_window_frequencies(historical, windows, max_number)
    
    # Gaps
    gaps = calculate_gaps(historical, max_number)
    
    # Matriz de co-ocurrencia
    pair_mat = pair_matrix(historical, pair_matrix_window, max_number)
    
    rows = []
    
    for number in range(1, max_number + 1):
        freq_vals = [rolling_freqs[w].get(number, 0) for w in windows]
        
        feat = [
            # Frecuencias
            global_freq.get(number, 0),
            *[rolling_freqs[w].get(number, 0) for w in windows],
            
            # Momentum y aceleración
            freq_vals[1] - freq_vals[3] if len(freq_vals) > 3 else 0,
            freq_vals[0] - freq_vals[2] if len(freq_vals) > 2 else 0,
            freq_vals[0] - freq_vals[1] if len(freq_vals) > 1 else 0,
            
            # Gap
            gaps.get(number, len(df)),
            
            # Pair strength
            pair_strength(pair_mat, number),
            
            # Volatilidad
            float(np.std(freq_vals)),
            
            # EMA
            exponential_moving_average(freq_vals),
            
            # Features del número
            number / max_number,
            number % 2,
            int(number <= 13),
            int(14 <= number <= 26),
            int(number >= 27),
        ]
        
        rows.append(feat)
    
    return np.array(rows)
