"""Tests para el módulo de evaluación."""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    hit_rate,
    expected_return,
    coverage,
    average_hits,
    hit_distribution,
    metrics_summary
)
from src.evaluation.backtest import BacktestEngine


def test_hit_rate():
    """Test tasa de aciertos."""
    y_true = np.array([1, 5, 12, 25, 38])
    y_pred = np.array([1, 5, 10, 20, 30])
    
    rate = hit_rate(y_true, y_pred, k=5)
    
    assert rate == 0.4  # 2 aciertos de 5


def test_expected_return():
    """Test retorno esperado."""
    y_true = np.array([1, 5, 12, 25, 38])
    y_pred = np.array([1, 5, 10, 20, 30])
    
    ret = expected_return(y_true, y_pred, k=5)
    
    assert ret == 10  # 2 aciertos = 10


def test_coverage():
    """Test cobertura."""
    y_true_list = [
        np.array([1, 5, 12, 25, 38]),
        np.array([3, 8, 15, 22, 33]),
        np.array([7, 14, 21, 28, 35])
    ]
    y_pred_list = [
        np.array([1, 5, 10, 20, 30]),  # 2 aciertos
        np.array([1, 2, 3, 4, 5]),      # 1 acierto
        np.array([10, 20, 30, 40, 50])  # 0 aciertos
    ]
    
    cov = coverage(y_true_list, y_pred_list, k=5)
    
    assert abs(cov - 2/3) < 1e-10  # 2 de 3 con al menos 1 acierto


def test_average_hits():
    """Test promedio de aciertos."""
    y_true_list = [
        np.array([1, 5, 12, 25, 38]),
        np.array([3, 8, 15, 22, 33]),
        np.array([7, 14, 21, 28, 35])
    ]
    y_pred_list = [
        np.array([1, 5, 10, 20, 30]),  # 2 aciertos
        np.array([1, 2, 3, 4, 5]),      # 1 acierto
        np.array([10, 20, 30, 40, 50])  # 0 aciertos
    ]
    
    avg = average_hits(y_true_list, y_pred_list, k=5)
    
    assert abs(avg - 1.0) < 1e-10  # Promedio = (2+1+0)/3 = 1.0


def test_hit_distribution():
    """Test distribución de aciertos."""
    y_true_list = [
        np.array([1, 5, 12, 25, 38]),
        np.array([3, 8, 15, 22, 33]),
        np.array([7, 14, 21, 28, 35])
    ]
    y_pred_list = [
        np.array([1, 5, 10, 20, 30]),  # 2 aciertos
        np.array([1, 2, 3, 4, 5]),      # 1 acierto
        np.array([10, 20, 30, 40, 50])  # 0 aciertos
    ]
    
    dist = hit_distribution(y_true_list, y_pred_list, k=5)
    
    assert dist[0] == 1
    assert dist[1] == 1
    assert dist[2] == 1


def test_metrics_summary():
    """Test resumen de métricas."""
    hits = [2, 1, 0, 3, 2, 1, 0, 2, 1, 3]
    
    summary = metrics_summary(hits, k=5)
    
    assert summary["total_draws"] == 10
    assert summary["total_hits"] == 15
    assert summary["average_hits"] == 1.5
    assert summary["max_hits"] == 3
    assert summary["min_hits"] == 0


def test_backtest_engine_initialization():
    """Test inicialización del motor de backtest."""
    engine = BacktestEngine(
        min_history=30,
        n_steps=10,
        n_estimators=10,
        verbose=False
    )
    
    assert engine.min_history == 30
    assert engine.n_steps == 10
    assert engine.n_estimators == 10


if __name__ == "__main__":
    test_hit_rate()
    test_expected_return()
    test_coverage()
    test_average_hits()
    test_hit_distribution()
    test_metrics_summary()
    test_backtest_engine_initialization()
    print("OK: Todos los tests del modulo de evaluacion pasaron")
