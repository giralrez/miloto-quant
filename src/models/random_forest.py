"""
Wrapper para Random Forest.
"""

import logging

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


def create_rf_classifier(
    n_estimators: int = 180,
    max_depth: int = 12,
    random_state: int = 42,
    n_jobs: int = 4,
    **kwargs
) -> RandomForestClassifier:
    """
    Crea un clasificador Random Forest.
    
    Args:
        n_estimators: Número de árboles
        max_depth: Profundidad máxima
        random_state: Semilla aleatoria
        n_jobs: Número de hilos
        **kwargs: Parámetros adicionales
    
    Returns:
        Clasificador RF configurado
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    
    return model


def create_rf_regressor(
    n_estimators: int = 180,
    max_depth: int = 12,
    random_state: int = 42,
    n_jobs: int = 4,
    **kwargs
) -> RandomForestRegressor:
    """
    Crea un regresor Random Forest.
    
    Args:
        n_estimators: Número de árboles
        max_depth: Profundidad máxima
        random_state: Semilla aleatoria
        n_jobs: Número de hilos
        **kwargs: Parámetros adicionales
    
    Returns:
        Regresor RF configurado
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    
    return model
