"""
Entrenamiento de modelos ML.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def train_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "model"
) -> BaseEstimator:
    """
    Entrena un modelo individual.
    
    Args:
        model: Modelo a entrenar
        X: Features de entrenamiento
        y: Target de entrenamiento
        model_name: Nombre del modelo (para logging)
    
    Returns:
        Modelo entrenado
    """
    logger.info(f"Entrenando {model_name}...")
    
    model.fit(X, y)
    
    logger.info(f"{model_name} entrenado exitosamente")
    
    return model


def train_ensemble(
    models: Dict[str, BaseEstimator],
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, BaseEstimator]:
    """
    Entrena un ensamble completo de modelos.
    
    Args:
        models: Diccionario {nombre: modelo}
        X: Features de entrenamiento
        y: Target de entrenamiento
    
    Returns:
        Diccionario con modelos entrenados
    """
    trained_models = {}
    
    for name, model in models.items():
        trained_models[name] = train_model(model, X, y, name)
    
    return trained_models


def train_with_validation(
    models: Dict[str, BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> Dict[str, BaseEstimator]:
    """
    Entrena modelos con opción de validación.
    
    Args:
        models: Diccionario {nombre: modelo}
        X: Features de entrenamiento
        y: Target de entrenamiento
        X_val: Features de validación (opcional)
        y_val: Target de validación (opcional)
    
    Returns:
        Diccionario con modelos entrenados
    """
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Entrenando {name}...")
        
        # Entrenar
        model.fit(X, y)
        
        # Evaluar en validación si está disponible
        if X_val is not None and y_val is not None:
            train_score = model.score(X, y)
            val_score = model.score(X_val, y_val)
            logger.info(f"  {name} - Train: {train_score:.4f}, Val: {val_score:.4f}")
        
        trained_models[name] = model
    
    return trained_models


def get_model_info(model: BaseEstimator) -> dict:
    """
    Obtiene información de un modelo entrenado.
    
    Args:
        model: Modelo entrenado
    
    Returns:
        Diccionario con información del modelo
    """
    info = {
        "type": type(model).__name__,
        "params": model.get_params(),
    }
    
    # Para modelos con feature_importances_
    if hasattr(model, "feature_importances_"):
        info["feature_importances"] = model.feature_importances_
    
    # Para modelos con n_estimators
    if hasattr(model, "n_estimators"):
        info["n_estimators"] = model.n_estimators
    
    return info
