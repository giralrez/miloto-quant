"""Tests para el módulo de predicción."""

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction.predictor import predict_scores, predict_top_numbers, rank_numbers
from src.prediction.probability import bayesian_prior, final_probability, probability_features
from src.prediction.monte_carlo import (
    optimize_combinations,
    beam_search_combinations,
    greedy_selection
)
from src.models.lightgbm_model import create_lgbm_regressor
from src.models.random_forest import create_rf_regressor


# Datos de prueba
np.random.seed(42)
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
]
df_test = pd.DataFrame(SAMPLE_DATA)


def test_predict_scores():
    """Test predicción de scores."""
    # Crear modelos entrenados
    X_train = np.random.rand(100, 17)
    y_train = np.random.rand(100)
    
    models = {
        "lgbm": create_lgbm_regressor(n_estimators=10, random_state=42),
        "rf": create_rf_regressor(n_estimators=10, random_state=42)
    }
    
    for model in models.values():
        model.fit(X_train, y_train)
    
    weights = {"lgbm": 0.6, "rf": 0.4}
    X_latest = np.random.rand(39, 17)
    numbers = np.arange(1, 40)
    
    final_scores, per_model = predict_scores(models, weights, X_latest, numbers)
    
    assert final_scores.shape[0] == 39
    assert len(per_model) == 39
    assert all(s >= 0 for s in final_scores)


def test_predict_top_numbers():
    """Test predicción de top números."""
    scores = np.random.rand(39)
    numbers = np.arange(1, 40)
    
    top_numbers = predict_top_numbers(scores, numbers, top_k=5)
    
    assert len(top_numbers) == 5
    assert all(1 <= n <= 39 for n in top_numbers)


def test_rank_numbers():
    """Test ranking de números."""
    scores = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    numbers = np.array([1, 2, 3, 4, 5])
    
    ranked = rank_numbers(scores, numbers)
    
    assert ranked[0] == 4  # Mayor score
    assert ranked[-1] == 1  # Menor score


def test_bayesian_prior():
    """Test prior Bayesiano."""
    prior = bayesian_prior(df_test, max_number=39)
    
    assert prior.shape[0] == 39
    assert all(0 <= p <= 1 for p in prior)
    # El prior no necesita sumar 1, solo debe ser válido
    assert all(p >= 0 for p in prior)


def test_final_probability():
    """Test probabilidad final."""
    ml_scores = np.random.rand(39)
    bayes_prior = np.random.rand(39)
    bayes_prior = bayes_prior / bayes_prior.sum()
    
    probs = final_probability(ml_scores, bayes_prior)
    
    assert probs.shape[0] == 39
    assert all(0 <= p <= 1 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6


def test_probability_features():
    """Test features de probabilidad."""
    features = probability_features(df_test, max_number=39)
    
    assert "frequency_absolute" in features
    assert "frequency_relative" in features
    assert "bayesian_prior" in features
    
    assert features["frequency_absolute"].shape[0] == 39
    assert features["frequency_relative"].shape[0] == 39
    assert features["bayesian_prior"].shape[0] == 39


def test_optimize_combinations():
    """Test optimización con Monte Carlo."""
    probs = np.random.rand(39)
    probs = probs / probs.sum()
    pair_mat = np.random.rand(39, 39)
    pair_mat = (pair_mat + pair_mat.T) / 2  # Simetrizar
    
    combinations = optimize_combinations(
        probs, pair_mat, n_samples=1000, top_k=5
    )
    
    assert len(combinations) == 5
    assert all(len(combo) == 5 for combo, _ in combinations)
    assert all(all(1 <= n <= 39 for n in combo) for combo, _ in combinations)


def test_beam_search_combinations():
    """Test beam search."""
    probs = np.random.rand(39)
    probs = probs / probs.sum()
    pair_mat = np.random.rand(39, 39)
    pair_mat = (pair_mat + pair_mat.T) / 2
    
    combinations = beam_search_combinations(
        probs, pair_mat, beam_width=50
    )
    
    # beam_search retorna todas las combinaciones del beam
    assert len(combinations) > 0
    assert all(len(combo) == 5 for combo, _ in combinations)


def test_greedy_selection():
    """Test selección greedy."""
    probs = np.random.rand(39)
    probs = probs / probs.sum()
    pair_mat = np.random.rand(39, 39)
    pair_mat = (pair_mat + pair_mat.T) / 2
    
    combo = greedy_selection(probs, pair_mat)
    
    assert len(combo) == 5
    assert all(1 <= n <= 39 for n in combo)
    assert len(set(combo)) == 5  # Sin duplicados


if __name__ == "__main__":
    test_predict_scores()
    test_predict_top_numbers()
    test_rank_numbers()
    test_bayesian_prior()
    test_final_probability()
    test_probability_features()
    test_optimize_combinations()
    test_beam_search_combinations()
    test_greedy_selection()
    print("OK: Todos los tests del modulo de prediccion pasaron")
