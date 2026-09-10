"""
Ensamble de modelos con pesos dinámicos.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class DynamicEnsemble:
    """
    Ensamble de modelos con pesos dinámicos.
    
    Los pesos se calculan normalizando los scores de validación
    de cada modelo.
    """
    
    def __init__(
        self,
        models: Dict[str, BaseEstimator],
        use_dynamic_weights: bool = True
    ):
        """
        Inicializa el ensamble.
        
        Args:
            models: Diccionario {nombre: modelo}
            use_dynamic_weights: Si True, calcula pesos dinámicos
        """
        self.models = models
        self.use_dynamic_weights = use_dynamic_weights
        self.weights = {name: 1.0 / len(models) for name in models}
        self.scores = {}
    
    def calculate_weights(
        self,
        validation_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcula pesos normalizando los scores de validación.
        
        Args:
            validation_scores: {nombre_modelo: score}
        
        Returns:
            Diccionario de pesos normalizados
        """
        self.scores = validation_scores
        
        if not self.use_dynamic_weights:
            return self.weights
        
        # Normalizar scores (mayor score = mayor peso)
        total = sum(validation_scores.values())
        
        if total > 0:
            self.weights = {
                name: score / total
                for name, score in validation_scores.items()
            }
        
        logger.info(f"Pesos calculados: {self.weights}")
        
        return self.weights
    
    def predict_proba(
        self,
        X: np.ndarray,
        model_type: str = "classifier"
    ) -> np.ndarray:
        """
        Predice probabilidades combinando todos los modelos.
        
        Args:
            X: Features de entrada
            model_type: Tipo de modelo ("classifier" o "regressor")
        
        Returns:
            Probabilidades/scores combinados
        """
        all_predictions = []
        
        for name, model in self.models.items():
            if model_type == "classifier":
                # Para clasificadores, usar predict_proba
                proba = model.predict_proba(X)
                # Si es binario, tomar la columna positiva
                if proba.shape[1] == 2:
                    proba = proba[:, 1]
                else:
                    proba = proba.T[1] if proba.shape[1] > 1 else proba
            else:
                # Para regresores, usar predict directamente
                proba = model.predict(X)
                # Asegurar que sea positivo
                proba = np.maximum(proba, 0)
            
            all_predictions.append(proba * self.weights.get(name, 1.0))
        
        # Combinar predicciones
        combined = np.sum(all_predictions, axis=0)
        
        return combined
    
    def predict_scores(
        self,
        X: np.ndarray,
        numbers: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
        """
        Predice scores para cada número con cada modelo.
        
        Args:
            X: Features (39 filas, una por número)
            numbers: Array de números [1, 2, ..., 39]
        
        Returns:
            Tupla de (scores_finales, scores_por_modelo)
        """
        final_scores = np.zeros(len(numbers))
        per_model = {}
        
        for idx, number in enumerate(numbers):
            feats = X[idx].reshape(1, -1)
            
            model_scores = {}
            score = 0
            
            for name, model in self.models.items():
                pred = model.predict(feats)[0]
                pred = max(pred, 0)  # Asegurar positivo
                
                model_scores[name] = pred
                score += pred * self.weights.get(name, 1.0)
            
            per_model[int(number)] = model_scores
            final_scores[idx] = score
        
        return final_scores, per_model
    
    def get_weights(self) -> Dict[str, float]:
        """Retorna los pesos actuales."""
        return self.weights.copy()
    
    def get_scores(self) -> Dict[str, float]:
        """Retorna los scores de validación."""
        return self.scores.copy()


def normalize_weights(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normaliza scores para usar como pesos.
    
    Args:
        scores: {nombre: score}
    
    Returns:
        Pesos normalizados
    """
    total = sum(scores.values())
    
    if total > 0:
        return {k: v / total for k, v in scores.items()}
    
    # Si total es 0, pesos iguales
    n = len(scores)
    return {k: 1.0 / n for k in scores}
