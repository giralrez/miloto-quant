"""Tests para el módulo de features."""

import sys
import os

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.frequencies import (
    weighted_frequency,
    rolling_frequency,
    multi_window_frequencies,
    exponential_moving_average,
    frequency_features
)
from src.features.gaps import calculate_gaps, gap_features
from src.features.momentum import momentum, acceleration, volatility, momentum_features
from src.features.statistical import (
    draw_statistics,
    pair_matrix,
    pair_strength,
    tercile_features,
    number_features
)
from src.features.builder import (
    build_features_v1,
    build_features_v3,
    build_latest_features_v3
)


# Datos de prueba
SAMPLE_DATA = [
    [1, 5, 12, 25, 38],
    [3, 8, 15, 22, 33],
    [7, 14, 21, 28, 35],
    [10, 18, 25, 30, 38],
    [2, 9, 16, 23, 31],
    [5, 12, 19, 26, 34],
    [8, 15, 22, 29, 36],
    [11, 18, 25, 32, 39],
    [4, 11, 18, 25, 32],
    [7, 14, 21, 28, 35],
    [1, 8, 15, 22, 29],
    [3, 10, 17, 24, 31],
    [6, 13, 20, 27, 34],
    [9, 16, 23, 30, 37],
    [2, 9, 16, 23, 30],
    [5, 12, 19, 26, 33],
    [8, 15, 22, 29, 36],
    [11, 18, 25, 32, 39],
    [4, 11, 18, 25, 32],
    [7, 14, 21, 28, 35],
    [1, 8, 15, 22, 29],
    [3, 10, 17, 24, 31],
    [6, 13, 20, 27, 34],
    [9, 16, 23, 30, 37],
    [2, 9, 16, 23, 30],
]


def test_weighted_frequency():
    """Test frecuencia ponderada."""
    df = pd.DataFrame(SAMPLE_DATA[:10])
    
    freq = weighted_frequency(df, alpha=0.997)
    
    assert isinstance(freq, dict)
    assert len(freq) > 0
    # Todas las frecuencias deben ser positivas
    assert all(v >= 0 for v in freq.values())


def test_rolling_frequency():
    """Test frecuencia rolling."""
    df = pd.DataFrame(SAMPLE_DATA[:10])
    
    freq = rolling_frequency(df, window=5, max_number=39)
    
    assert isinstance(freq, dict)
    assert len(freq) == 39
    # Debe retornar frecuencias para todos los números
    assert all(0 <= v <= 1 for v in freq.values())


def test_multi_window_frequencies():
    """Test múltiples ventanas de frecuencia."""
    df = pd.DataFrame(SAMPLE_DATA[:20])
    
    result = multi_window_frequencies(df, windows=[5, 10, 20], max_number=39)
    
    assert isinstance(result, dict)
    assert 5 in result
    assert 10 in result
    assert 20 in result
    assert len(result[5]) == 39


def test_exponential_moving_average():
    """Test media móvil exponencial."""
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    ema = exponential_moving_average(series, alpha=0.3)
    
    assert isinstance(ema, float)
    # EMA debe estar entre el mínimo y máximo de la serie
    assert min(series) <= ema <= max(series)


def test_frequency_features():
    """Test features de frecuencia para un número."""
    df = pd.DataFrame(SAMPLE_DATA[:20])
    
    features = frequency_features(df, number=5, windows=[5, 10, 20], max_number=39)
    
    assert isinstance(features, dict)
    assert "global_freq" in features
    assert "ema_signal" in features
    assert "entropy" in features
    assert "freq5" in features
    assert "freq10" in features
    assert "freq20" in features


def test_calculate_gaps():
    """Test cálculo de gaps."""
    df = pd.DataFrame(SAMPLE_DATA[:10])
    
    gaps = calculate_gaps(df, max_number=39)
    
    assert isinstance(gaps, dict)
    assert len(gaps) == 39
    # Todos los gaps deben ser positivos
    assert all(v > 0 for v in gaps.values())


def test_gap_features():
    """Test features de gap para un número."""
    df = pd.DataFrame(SAMPLE_DATA[:10])
    
    features = gap_features(df, number=5, max_number=39)
    
    assert isinstance(features, dict)
    assert "gap" in features
    assert "normalized_gap" in features
    assert features["gap"] > 0
    assert 0 <= features["normalized_gap"] <= 1


def test_momentum():
    """Test cálculo de momentum."""
    mom = momentum(0.5, 0.3)
    
    assert mom == 0.2


def test_acceleration():
    """Test cálculo de aceleración."""
    accel = acceleration(0.6, 0.4)
    
    assert abs(accel - 0.2) < 1e-10


def test_volatility():
    """Test cálculo de volatilidad."""
    vol = volatility([0.1, 0.2, 0.3, 0.4])
    
    assert isinstance(vol, float)
    assert vol > 0


def test_momentum_features():
    """Test features de momentum."""
    features = momentum_features(0.5, 0.4, 0.3, 0.2)
    
    assert isinstance(features, dict)
    assert "momentum_10_50" in features
    assert "momentum_5_20" in features
    assert "acceleration" in features
    assert "volatility" in features


def test_draw_statistics():
    """Test estadísticos de un sorteo."""
    row = pd.Series([1, 5, 12, 25, 38])
    
    stats = draw_statistics(row)
    
    assert isinstance(stats, dict)
    assert stats["sum"] == 81
    assert stats["mean"] == 16.2
    assert stats["even_count"] == 2
    assert stats["odd_count"] == 3


def test_pair_matrix():
    """Test matriz de co-ocurrencia."""
    df = pd.DataFrame(SAMPLE_DATA[:10])
    
    matrix = pair_matrix(df, window=10, max_number=39)
    
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (39, 39)
    # Debe ser simétrica
    np.testing.assert_array_equal(matrix, matrix.T)


def test_pair_strength():
    """Test fuerza de co-ocurrencia."""
    df = pd.DataFrame(SAMPLE_DATA[:10])
    matrix = pair_matrix(df, window=10, max_number=39)
    
    strength = pair_strength(matrix, number=5)
    
    assert isinstance(strength, float)
    assert strength >= 0


def test_tercile_features():
    """Test features de terciles."""
    features = tercile_features(5, max_number=39)
    
    assert isinstance(features, dict)
    assert "tercil_bajo" in features
    assert "tercil_medio" in features
    assert "tercil_alto" in features
    # Solo uno debe ser 1
    assert sum(features.values()) == 1


def test_number_features():
    """Test features básicas del número."""
    features = number_features(5, max_number=39)
    
    assert isinstance(features, dict)
    assert "number_norm" in features
    assert "is_odd" in features
    assert features["number_norm"] == 5 / 39
    assert features["is_odd"] == 1


def test_build_features_v1():
    """Test builder V1."""
    # Usar min_history bajo para tener datos suficientes
    df = pd.DataFrame(SAMPLE_DATA[:25])
    
    X = build_features_v1(df, min_history=10, max_number=39, windows=[5, 10])
    
    assert isinstance(X, pd.DataFrame)
    assert len(X) > 0
    assert "draw_idx" in X.columns
    assert "number" in X.columns


def test_build_features_v3():
    """Test builder V3."""
    # Usar min_history bajo para tener datos suficientes
    df = pd.DataFrame(SAMPLE_DATA[:25])
    
    X, y = build_features_v3(df, min_history=10, max_number=39, windows=[5, 10])
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    # Con 2 ventanas, se generan 15 features
    assert X.shape[1] == 15


def test_build_latest_features_v3():
    """Test features para predicción V3."""
    df = pd.DataFrame(SAMPLE_DATA[:25])
    
    X = build_latest_features_v3(df, max_number=39, windows=[5, 10])
    
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 39  # Un fila por número
    # Con 2 ventanas, se generan 15 features
    assert X.shape[1] == 15


if __name__ == "__main__":
    test_weighted_frequency()
    test_rolling_frequency()
    test_multi_window_frequencies()
    test_exponential_moving_average()
    test_frequency_features()
    test_calculate_gaps()
    test_gap_features()
    test_momentum()
    test_acceleration()
    test_volatility()
    test_momentum_features()
    test_draw_statistics()
    test_pair_matrix()
    test_pair_strength()
    test_tercile_features()
    test_number_features()
    test_build_features_v1()
    test_build_features_v3()
    test_build_latest_features_v3()
    print("OK: Todos los tests del modulo de features pasaron")
