"""
Wrapper para LightGBM.
"""

import logging
from typing import Optional

from lightgbm import LGBMClassifier, LGBMRegressor

logger = logging.getLogger(__name__)


def create_lgbm_classifier(
    n_estimators: int = 200,
    learning_rate: float = 0.02,
    max_depth: int = 6,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    random_state: int = 42,
    n_jobs: int = 4,
    **kwargs
) -> LGBMClassifier:
    """
    Crea un clasificador LightGBM.
    
    Args:
        n_estimators: Número de árboles
        learning_rate: Tasa de aprendizaje
        max_depth: Profundidad máxima
        subsample: Submuestreo de filas
        colsample_bytree: Submuestreo de columnas
        random_state: Semilla aleatoria
        n_jobs: Número de hilos
        **kwargs: Parámetros adicionales
    
    Returns:
        Clasificador LGBM configurado
    """
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
        **kwargs
    )
    
    return model


def create_lgbm_regressor(
    n_estimators: int = 200,
    learning_rate: float = 0.02,
    max_depth: int = 6,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    random_state: int = 42,
    n_jobs: int = 4,
    **kwargs
) -> LGBMRegressor:
    """
    Crea un regresor LightGBM.
    
    Args:
        n_estimators: Número de árboles
        learning_rate: Tasa de aprendizaje
        max_depth: Profundidad máxima
        subsample: Submuestreo de filas
        colsample_bytree: Submuestreo de columnas
        random_state: Semilla aleatoria
        n_jobs: Número de hilos
        **kwargs: Parámetros adicionales
    
    Returns:
        Regresor LGBM configurado
    """
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
        **kwargs
    )
    
    return model
