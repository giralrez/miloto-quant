"""Tests para el módulo de entrenamiento."""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import (
    train_model,
    train_ensemble,
    train_with_validation,
    get_model_info
)
from src.training.validator import (
    walk_forward_validation,
    expanding_window_validation,
    calculate_optimal_weights,
    cross_validate_models
)
from src.models.lightgbm_model import create_lgbm_regressor
from src.models.random_forest import create_rf_regressor


# Datos de prueba
np.random.seed(42)
X_TEST = np.random.rand(200, 17)
y_TEST = np.random.rand(200)
X_VAL = np.random.rand(50, 17)
y_VAL = np.random.rand(50)


def test_train_model():
    """Test entrenamiento de un modelo individual."""
    model = create_lgbm_regressor(n_estimators=10, random_state=42)
    
    trained_model = train_model(model, X_TEST, y_TEST, "lgbm_test")
    
    assert trained_model is not None
    assert hasattr(trained_model, "predict")


def test_train_ensemble():
    """Test entrenamiento de ensamble."""
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    trained_models = train_ensemble(models, X_TEST, y_TEST)
    
    assert len(trained_models) == 2
    assert "lgbm" in trained_models
    assert "rf" in trained_models


def test_train_with_validation():
    """Test entrenamiento con validación."""
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    trained_models = train_with_validation(
        models, X_TEST, y_TEST, X_VAL, y_VAL
    )
    
    assert len(trained_models) == 2


def test_get_model_info():
    """Test información del modelo."""
    model = create_lgbm_regressor(n_estimators=10, random_state=42)
    model.fit(X_TEST, y_TEST)
    
    info = get_model_info(model)
    
    assert "type" in info
    assert "params" in info
    assert info["type"] == "LGBMRegressor"


def test_walk_forward_validation():
    """Test validación walk-forward."""
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    scores = walk_forward_validation(
        X_TEST, y_TEST, models, n_splits=3, model_type="regressor"
    )
    
    assert isinstance(scores, dict)
    assert len(scores) == 2
    assert all(isinstance(v, float) for v in scores.values())


def test_expanding_window_validation():
    """Test validación con ventana expansiva."""
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42)
    }
    
    results = expanding_window_validation(
        X_TEST, y_TEST, models, min_train_size=100, step=50
    )
    
    assert isinstance(results, dict)
    assert "lgbm" in results
    assert len(results["lgbm"]) > 0


def test_calculate_optimal_weights():
    """Test cálculo de pesos óptimos."""
    scores = {
        "model_a": 0.8,
        "model_b": 0.6,
        "model_c": 0.4
    }
    
    weights = calculate_optimal_weights(scores)
    
    assert abs(sum(weights.values()) - 1.0) < 1e-10
    assert weights["model_a"] > weights["model_b"]


def test_cross_validate_models():
    """Test validación cruzada."""
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42)
    }
    
    results = cross_validate_models(X_TEST, y_TEST, models, cv=3)
    
    assert isinstance(results, dict)
    assert "lgbm" in results
    assert "mean" in results["lgbm"]
    assert "std" in results["lgbm"]


if __name__ == "__main__":
    test_train_model()
    test_train_ensemble()
    test_train_with_validation()
    test_get_model_info()
    test_walk_forward_validation()
    test_expanding_window_validation()
    test_calculate_optimal_weights()
    test_cross_validate_models()
    print("OK: Todos los tests del modulo de entrenamiento pasaron")
