"""Tests para el módulo de optimización."""

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.optimization import (
    parallel_train_models,
    parallel_predict,
    optimize_dataframe_dtypes,
    vectorized_gap_calculation
)
from src.models.lightgbm_model import create_lgbm_regressor
from src.models.random_forest import create_rf_regressor


def test_parallel_train_models():
    """Test entrenamiento paralelo."""
    X = np.random.rand(100, 17)
    y = np.random.rand(100)
    
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    trained = parallel_train_models(models, X, y, n_jobs=1)
    
    assert len(trained) == 2
    assert all(hasattr(m, "predict") for m in trained.values())


def test_parallel_predict():
    """Test predicción paralela."""
    X_train = np.random.rand(100, 17)
    y_train = np.random.rand(100)
    X_test = np.random.rand(10, 17)
    
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    for model in models.values():
        model.fit(X_train, y_train)
    
    weights = {"lgbm": 0.6, "rf": 0.4}
    
    predictions = parallel_predict(models, weights, X_test, n_jobs=1)
    
    assert predictions.shape[0] == 10


def test_optimize_dataframe_dtypes():
    """Test optimización de tipos de datos."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        "c": [100, 200, 300, 400, 500]
    })
    
    optimized = optimize_dataframe_dtypes(df)
    
    # Verificar que se optimizó
    assert optimized["a"].dtype == np.uint8
    assert optimized["b"].dtype == np.float32


def test_vectorized_gap_calculation():
    """Test cálculo vectorizado de gaps."""
    data = [
        [1, 5, 12, 25, 38],
        [3, 8, 15, 22, 33],
        [7, 14, 21, 28, 35],
    ]
    df = pd.DataFrame(data)
    
    gaps = vectorized_gap_calculation(df, max_number=39)
    
    assert gaps.shape[0] == 39
    # El número 1 apareció en el sorteo 0, gap = 3 sorteos
    assert gaps[0] == 3
    # El número 2 no apareció nunca, gap = 3 (total sorteos)
    assert gaps[1] == 3


if __name__ == "__main__":
    test_parallel_train_models()
    test_parallel_predict()
    test_optimize_dataframe_dtypes()
    test_vectorized_gap_calculation()
    print("OK: Todos los tests del modulo de optimizacion pasaron")
