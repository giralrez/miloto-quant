"""
Features estadísticos y de co-ocurrencia.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from itertools import combinations

logger = logging.getLogger(__name__)


def draw_statistics(row: pd.Series) -> Dict[str, float]:
    """
    Calcula estadísticos de un sorteo individual.
    
    IMPORTANTE: Estas features solo deben usarse en backtest,
    ya que describen el sorteo que se intenta predecir.
    
    Args:
        row: Fila del DataFrame con los 5 números
    
    Returns:
        Diccionario con estadísticos del sorteo
    """
    nums = row.values.astype(int)
    
    stats = {
        "sum": nums.sum(),
        "mean": nums.mean(),
        "std": nums.std(),
        "range": nums.max() - nums.min(),
        "even_count": sum(1 for n in nums if n % 2 == 0),
        "odd_count": sum(1 for n in nums if n % 2 != 0),
        "consecutive_pairs": sum(
            1 for i in range(len(nums) - 1)
            if nums[i + 1] - nums[i] == 1
        ),
    }
    
    # Diferencias entre posiciones adyacentes
    for i in range(len(nums) - 1):
        stats[f"diff_{i}"] = nums[i + 1] - nums[i]
    
    return stats


def pair_matrix(
    df: pd.DataFrame,
    window: int = 80,
    max_number: int = 39
) -> np.ndarray:
    """
    Calcula la matriz de co-ocurrencia de pares.
    
    Args:
        df: DataFrame con sorteos históricos
        window: Ventana de sorteos a considerar
        max_number: Número máximo
    
    Returns:
        Matriz de co-ocurrencia normalizada
    """
    matrix = np.zeros((max_number, max_number))
    
    # Usar solo los últimos N sorteos
    recent = df.tail(window)
    
    for row in recent.values:
        for a, b in combinations(row, 2):
            a, b = int(a), int(b)
            matrix[a - 1][b - 1] += 1
            matrix[b - 1][a - 1] += 1
    
    # Normalizar
    if matrix.max() > 0:
        matrix /= matrix.max()
    
    return matrix


def pair_strength(
    matrix: np.ndarray,
    number: int
) -> float:
    """
    Calcula la fuerza de co-ocurrencia de un número con todos los demás.
    
    Args:
        matrix: Matriz de co-ocurrencia
        number: Número a evaluar
    
    Returns:
        Promedio de co-ocurrencia con todos los demás números
    """
    return float(np.mean(matrix[number - 1]))


def tercile_features(number: int, max_number: int = 39) -> Dict[str, int]:
    """
    Codifica el número por terciles.
    
    Args:
        number: Número a codificar
        max_number: Número máximo
    
    Returns:
        Diccionario con features de terciles
    """
    tercil_size = max_number / 3
    
    return {
        "tercil_bajo": 1 if number <= tercil_size else 0,
        "tercil_medio": 1 if tercil_size < number <= 2 * tercil_size else 0,
        "tercil_alto": 1 if number > 2 * tercil_size else 0,
    }


def number_features(
    number: int,
    max_number: int = 39
) -> Dict[str, float]:
    """
    Genera features básicas del número.
    
    Args:
        number: Número a evaluar
        max_number: Número máximo
    
    Returns:
        Diccionario con features del número
    """
    features = {
        "number_norm": number / max_number,
        "is_odd": number % 2,
    }
    
    # Agregar terciles
    features.update(tercile_features(number, max_number))
    
    return features
