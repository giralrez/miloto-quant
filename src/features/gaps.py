"""
Cálculo de gaps (sorteos desde la última aparición de un número).
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_gaps(
    df: pd.DataFrame,
    max_number: int = 39
) -> Dict[int, int]:
    """
    Calcula cuántos sorteos han pasado desde la última aparición de cada número.
    
    Args:
        df: DataFrame con sorteos históricos (orden cronológico ascendente)
        max_number: Número máximo a considerar
    
    Returns:
        Diccionario {numero: gap}
    """
    gaps = {}
    
    for number in range(1, max_number + 1):
        last_seen = None
        
        # Buscar desde el sorteo más reciente hacia atrás
        for idx in reversed(range(len(df))):
            if number in df.iloc[idx].values:
                last_seen = len(df) - idx
                break
        
        # Si nunca apareció, el gap es el total de sorteos
        gaps[number] = last_seen if last_seen is not None else len(df)
    
    return gaps


def gap_features(
    df: pd.DataFrame,
    number: int,
    max_number: int = 39
) -> Dict[str, float]:
    """
    Genera features de gap para un número específico.
    
    Args:
        df: DataFrame con sorteos históricos
        number: Número del cual calcular gaps
        max_number: Número máximo
    
    Returns:
        Diccionario con features de gap
    """
    gaps = calculate_gaps(df, max_number)
    
    gap = gaps.get(number, len(df))
    
    # Normalizar gap (0 a 1, donde 1 = nunca ha salido)
    normalized_gap = gap / max(len(df), 1)
    
    features = {
        "gap": gap,
        "normalized_gap": normalized_gap,
    }
    
    return features
