"""Tests de integración para el pipeline completo."""

import sys
import os
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_full_pipeline():
    """Test del pipeline completo con datos sintéticos."""
    from src.data.loader import load_data, get_data_stats
    from src.features.builder import build_features_v3, build_latest_features_v3
    from src.features.statistical import pair_matrix
    from src.models.lightgbm_model import create_lgbm_regressor
    from src.models.xgboost_model import create_xgb_regressor
    from src.models.random_forest import create_rf_regressor
    from src.models.ensemble import DynamicEnsemble
    from src.training.trainer import train_ensemble
    from src.training.validator import walk_forward_validation, calculate_optimal_weights
    from src.prediction.predictor import predict_scores, predict_top_numbers
    from src.prediction.probability import bayesian_prior, final_probability
    from src.prediction.monte_carlo import optimize_combinations
    
    # Crear datos sintéticos
    np.random.seed(42)
    n_draws = 100
    data = []
    for _ in range(n_draws):
        draw = sorted(np.random.choice(range(1, 40), size=5, replace=False))
        data.append(draw)
    
    df = pd.DataFrame(data)
    
    # 1. Cargar datos
    stats = get_data_stats(df)
    assert stats["n_draws"] == n_draws
    
    # 2. Construir features
    X, y = build_features_v3(df, min_history=60, max_number=39, windows=[5, 10])
    assert len(X) > 0
    assert len(y) > 0
    
    # 3. Crear modelos
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "xgb": create_xgb_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    # 4. Entrenar
    trained_models = train_ensemble(models, X, y)
    assert len(trained_models) == 3
    
    # 5. Predecir
    X_latest = build_latest_features_v3(df, max_number=39, windows=[5, 10])
    numbers = np.arange(1, 40)
    weights = {"lgbm": 0.4, "xgb": 0.35, "rf": 0.25}
    
    final_scores, per_model = predict_scores(trained_models, weights, X_latest, numbers)
    assert final_scores.shape[0] == 39
    
    # 6. Probabilidad final
    bayes = bayesian_prior(df, max_number=39)
    probs = final_probability(final_scores, bayes)
    assert probs.shape[0] == 39
    assert abs(probs.sum() - 1.0) < 1e-6
    
    # 7. Top números
    top_numbers = predict_top_numbers(final_scores, numbers, top_k=5)
    assert len(top_numbers) == 5
    
    # 8. Monte Carlo
    pair_mat = pair_matrix(df, window=80, max_number=39)
    combinations = optimize_combinations(probs, pair_mat, n_samples=1000, top_k=5)
    assert len(combinations) == 5


def test_data_to_prediction_flow():
    """Test flujo de datos a predicción."""
    from src.data.loader import load_data
    from src.features.builder import build_features_v3, build_latest_features_v3
    from src.models.lightgbm_model import create_lgbm_regressor
    from src.models.ensemble import DynamicEnsemble
    from src.prediction.predictor import predict_scores
    from src.prediction.probability import bayesian_prior, final_probability
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        for i in range(80):
            draw = sorted(np.random.choice(range(1, 40), size=5, replace=False).tolist())
            f.write(",".join(map(str, draw)) + "\n")
        temp_path = f.name
    
    try:
        # Cargar
        df = load_data(temp_path)
        assert len(df) > 0
        
        # Features
        X, y = build_features_v3(df, min_history=60, max_number=39, windows=[5, 10])
        
        # Modelo
        model = create_lgbm_regressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Predicción
        X_latest = build_latest_features_v3(df, max_number=39, windows=[5, 10])
        numbers = np.arange(1, 40)
        
        final_scores, _ = predict_scores(
            {"lgbm": model}, {"lgbm": 1.0}, X_latest, numbers
        )
        
        # Probabilidad
        bayes = bayesian_prior(df, max_number=39)
        probs = final_probability(final_scores, bayes)
        
        assert probs.shape[0] == 39
        assert all(p >= 0 for p in probs)
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_full_pipeline()
    test_data_to_prediction_flow()
    print("OK: Todos los tests de integracion pasaron")
