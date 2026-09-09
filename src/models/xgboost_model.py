"""
Wrapper para XGBoost.
"""

import logging

from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


def create_xgb_classifier(
    n_estimators: int = 180,
    learning_rate: float = 0.025,
    max_depth: int = 5,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    random_state: int = 42,
    n_jobs: int = 4,
    **kwargs
) -> XGBClassifier:
    """
    Crea un clasificador XGBoost.
    
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
        Clasificador XGB configurado
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        **kwargs
    )
    
    return model


def create_xgb_regressor(
    n_estimators: int = 180,
    learning_rate: float = 0.025,
    max_depth: int = 5,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    random_state: int = 42,
    n_jobs: int = 4,
    **kwargs
) -> XGBRegressor:
    """
    Crea un regresor XGBoost.
    
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
        Regresor XGB configurado
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
        **kwargs
    )
    
    return model
