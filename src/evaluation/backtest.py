"""
Backtest walk-forward para evaluar estrategias de predicción.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..data.loader import load_data
from ..features.builder import build_features_v3, build_latest_features_v3
from ..features.statistical import pair_matrix
from ..models.ensemble import DynamicEnsemble
from ..models.lightgbm_model import create_lgbm_regressor
from ..models.xgboost_model import create_xgb_regressor
from ..models.random_forest import create_rf_regressor
from ..prediction.predictor import predict_scores
from ..prediction.probability import bayesian_prior, final_probability
from ..prediction.monte_carlo import optimize_combinations
from .metrics import metrics_summary

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Motor de backtest walk-forward.
    
    Re-entrena los modelos en cada paso para simular predicción real.
    """
    
    def __init__(
        self,
        min_history: int = 60,
        n_steps: int = 50,
        start_ratio: float = 0.80,
        windows: List[int] = None,
        max_number: int = 39,
        n_estimators: int = 50,
        monte_carlo_samples: int = 10000,
        verbose: bool = True
    ):
        """
        Inicializa el motor de backtest.
        
        Args:
            min_history: Mínimo de sorteos históricos
            n_steps: Número de pasos de backtest
            start_ratio: Ratio de inicio del backtest
            windows: Ventanas de frecuencia
            max_number: Número máximo
            n_estimators: Número de estimadores por modelo
            monte_carlo_samples: Muestras Monte Carlo
            verbose: Si True, imprime progreso
        """
        self.min_history = min_history
        self.n_steps = n_steps
        self.start_ratio = start_ratio
        self.windows = windows or [5, 10, 20, 50]
        self.max_number = max_number
        self.n_estimators = n_estimators
        self.monte_carlo_samples = monte_carlo_samples
        self.verbose = verbose
    
    def run(
        self,
        df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Ejecuta el backtest completo.
        
        Args:
            df: DataFrame con sorteos históricos
        
        Returns:
            Diccionario con resultados del backtest
        """
        start = int(len(df) * self.start_ratio)
        total_steps = min(self.n_steps, len(df) - start - 2)
        
        if self.verbose:
            print(f"\nBACKTEST: {total_steps} pasos desde sorteo {start}")
        
        hits_list = []
        all_predictions = []
        all_actuals = []
        
        for step, i in enumerate(range(start, start + total_steps)):
            if self.verbose and step % 10 == 0:
                print(f"  Paso {step + 1}/{total_steps}...")
            
            # Datos hasta el sorteo i
            partial_df = df.iloc[:i]
            
            # Construir features
            X_train, y_train = build_features_v3(
                partial_df,
                min_history=self.min_history,
                max_number=self.max_number,
                windows=self.windows
            )
            
            if len(X_train) == 0:
                continue
            
            # Crear y entrenar modelos
            models = {
                "lightgbm": create_lgbm_regressor(
                    n_estimators=self.n_estimators, random_state=42
                ),
                "xgboost": create_xgb_regressor(
                    n_estimators=self.n_estimators, random_state=42
                ),
                "randomforest": create_rf_regressor(
                    n_estimators=self.n_estimators, random_state=42
                )
            }
            
            for model in models.values():
                model.fit(X_train, y_train)
            
            # Crear ensamble con pesos iguales (simplificación para backtest)
            weights = {
                "lightgbm": 0.4,
                "xgboost": 0.35,
                "randomforest": 0.25
            }
            
            # Predecir
            X_latest = build_latest_features_v3(
                partial_df,
                max_number=self.max_number,
                windows=self.windows
            )
            
            numbers = np.arange(1, self.max_number + 1)
            final_scores, _ = predict_scores(models, weights, X_latest, numbers)
            
            # Probabilidad final
            bayes = bayesian_prior(partial_df, max_number=self.max_number)
            probs = final_probability(final_scores, bayes)
            
            # Monte Carlo para combinaciones
            pair_mat = pair_matrix(partial_df, window=80, max_number=self.max_number)
            combinations = optimize_combinations(
                probs, pair_mat,
                n_samples=self.monte_carlo_samples,
                top_k=1,
                max_number=self.max_number
            )
            
            # Predicción: top-5 por score
            pred_indices = np.argsort(final_scores)[-5:]
            pred = numbers[pred_indices]
            
            # Valor real
            actual = df.iloc[i + 1].values.astype(int)
            
            # Calcular aciertos
            hit_count = len(set(pred).intersection(set(actual)))
            hits_list.append(hit_count)
            
            all_predictions.append(pred)
            all_actuals.append(actual)
        
        # Métricas
        summary = metrics_summary(hits_list, k=5)
        
        if self.verbose:
            print(f"\nRESULTADOS BACKTEST:")
            print(f"  Promedio aciertos: {summary['average_hits']:.4f}")
            print(f"  Maximo aciertos: {summary['max_hits']}")
            print(f"  Cobertura (>=1 acierto): {summary['coverage']:.2%}")
        
        return {
            "hits": hits_list,
            "predictions": all_predictions,
            "actuals": all_actuals,
            "summary": summary
        }


def run_quick_backtest(
    filepath: str,
    n_steps: int = 20,
    n_estimators: int = 50
) -> Dict[str, any]:
    """
    Ejecuta un backtest rápido.
    
    Args:
        filepath: Ruta al CSV de histórico
        n_steps: Número de pasos
        n_estimators: Número de estimadores
    
    Returns:
        Resultados del backtest
    """
    df = load_data(filepath)
    
    engine = BacktestEngine(
        n_steps=n_steps,
        n_estimators=n_estimators,
        verbose=True
    )
    
    return engine.run(df)
