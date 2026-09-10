"""
Wrapper para Extra Trees.
"""

import logging

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

logger = logging.getLogger(__name__)


def create_et_classifier(
    n_estimators: int = 180,
    max_depth: int = 12,
    random_state: int = 42,
    n_jobs: int = 4,
    **kwargs
) -> ExtraTreesClassifier:
    """
    Crea un clasificador Extra Trees.
    
    Args:
        n_estimators: Número de árboles
        max_depth: Profundidad máxima
        random_state: Semilla aleatoria
        n_jobs: Número de hilos
        **kwargs: Parámetros adicionales
    
    Returns:
        Clasificador ET configurado
    """
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    
    return model


def create_et_regressor(
    n_estimators: int = 180,
    max_depth: int = 12,
    random_state: int = 42,
    n_jobs: int = 4,
    **kwargs
) -> ExtraTreesRegressor:
    """
    Crea un regresor Extra Trees.
    
    Args:
        n_estimators: Número de árboles
        max_depth: Profundidad máxima
        random_state: Semilla aleatoria
        n_jobs: Número de hilos
        **kwargs: Parámetros adicionales
    
    Returns:
        Regresor ET configurado
    """
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    
    return model
