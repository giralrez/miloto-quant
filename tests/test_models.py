"""Tests para el módulo de modelos."""

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lightgbm_model import create_lgbm_classifier, create_lgbm_regressor
from src.models.xgboost_model import create_xgb_classifier, create_xgb_regressor
from src.models.random_forest import create_rf_classifier, create_rf_regressor
from src.models.extra_trees import create_et_classifier, create_et_regressor
from src.models.ensemble import DynamicEnsemble, normalize_weights


def test_create_lgbm_classifier():
    """Test creación de clasificador LightGBM."""
    model = create_lgbm_classifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    assert model is not None
    assert model.n_estimators == 50
    assert model.learning_rate == 0.05


def test_create_lgbm_regressor():
    """Test creación de regresor LightGBM."""
    model = create_lgbm_regressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    assert model is not None
    assert model.n_estimators == 50


def test_create_xgb_classifier():
    """Test creación de clasificador XGBoost."""
    model = create_xgb_classifier(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    assert model is not None
    assert model.n_estimators == 50


def test_create_xgb_regressor():
    """Test creación de regresor XGBoost."""
    model = create_xgb_regressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    assert model is not None
    assert model.n_estimators == 50


def test_create_rf_classifier():
    """Test creación de clasificador Random Forest."""
    model = create_rf_classifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    assert model is not None
    assert model.n_estimators == 50


def test_create_rf_regressor():
    """Test creación de regresor Random Forest."""
    model = create_rf_regressor(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    assert model is not None
    assert model.n_estimators == 50


def test_create_et_classifier():
    """Test creación de clasificador Extra Trees."""
    model = create_et_classifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    assert model is not None
    assert model.n_estimators == 50


def test_create_et_regressor():
    """Test creación de regresor Extra Trees."""
    model = create_et_regressor(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    assert model is not None
    assert model.n_estimators == 50


def test_normalize_weights():
    """Test normalización de pesos."""
    scores = {
        "model_a": 0.8,
        "model_b": 0.6,
        "model_c": 0.4
    }
    
    weights = normalize_weights(scores)
    
    assert abs(sum(weights.values()) - 1.0) < 1e-10
    assert weights["model_a"] > weights["model_b"]
    assert weights["model_b"] > weights["model_c"]


def test_dynamic_ensemble_initialization():
    """Test inicialización del ensamble dinámico."""
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "xgb": create_xgb_regressor(n_estimators=10, random_state=42)
    }
    
    ensemble = DynamicEnsemble(models, use_dynamic_weights=True)
    
    assert ensemble is not None
    assert len(ensemble.models) == 2
    assert ensemble.use_dynamic_weights is True


def test_dynamic_ensemble_calculate_weights():
    """Test cálculo de pesos en ensamble."""
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "xgb": create_xgb_regressor(n_estimators=10, random_state=42)
    }
    
    ensemble = DynamicEnsemble(models, use_dynamic_weights=True)
    
    validation_scores = {
        "lgbm": 0.8,
        "xgb": 0.6
    }
    
    weights = ensemble.calculate_weights(validation_scores)
    
    assert abs(sum(weights.values()) - 1.0) < 1e-10
    assert weights["lgbm"] > weights["xgb"]


def test_dynamic_ensemble_predict():
    """Test predicción del ensamble."""
    # Crear datos de prueba
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    
    # Crear y entrenar modelos
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    for model in models.values():
        model.fit(X_train, y_train)
    
    # Crear ensamble
    ensemble = DynamicEnsemble(models, use_dynamic_weights=True)
    ensemble.calculate_weights({"lgbm": 0.7, "rf": 0.5})
    
    # Predecir
    X_test = np.random.rand(5, 10)
    predictions = ensemble.predict_proba(X_test, model_type="regressor")
    
    assert predictions.shape[0] == 5
    assert all(p >= 0 for p in predictions)


def test_dynamic_ensemble_predict_scores():
    """Test predicción de scores por número."""
    # Crear datos de prueba
    np.random.seed(42)
    X_train = np.random.rand(100, 17)
    y_train = np.random.rand(100)
    
    # Crear y entrenar modelos
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    for model in models.values():
        model.fit(X_train, y_train)
    
    # Crear ensamble
    ensemble = DynamicEnsemble(models, use_dynamic_weights=True)
    ensemble.calculate_weights({"lgbm": 0.7, "rf": 0.5})
    
    # Predecir scores
    numbers = np.arange(1, 40)
    X_latest = np.random.rand(39, 17)
    
    final_scores, per_model = ensemble.predict_scores(X_latest, numbers)
    
    assert final_scores.shape[0] == 39
    assert len(per_model) == 39
    assert all(s >= 0 for s in final_scores)


if __name__ == "__main__":
    test_create_lgbm_classifier()
    test_create_lgbm_regressor()
    test_create_xgb_classifier()
    test_create_xgb_regressor()
    test_create_rf_classifier()
    test_create_rf_regressor()
    test_create_et_classifier()
    test_create_et_regressor()
    test_normalize_weights()
    test_dynamic_ensemble_initialization()
    test_dynamic_ensemble_calculate_weights()
    test_dynamic_ensemble_predict()
    test_dynamic_ensemble_predict_scores()
    print("OK: Todos los tests del modulo de modelos pasaron")
