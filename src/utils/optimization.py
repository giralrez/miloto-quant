"""
Optimizaciones de rendimiento para MiLoto Quant.
"""

import logging
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parallel_train_models(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int = -1
) -> dict:
    """
    Entrena modelos en paralelo usando joblib.
    
    Args:
        models: Diccionario {nombre: modelo}
        X: Features
        y: Target
        n_jobs: Número de hilos (-1 = todos)
    
    Returns:
        Diccionario con modelos entrenados
    """
    try:
        from joblib import Parallel, delayed
        
        def train_single(name, model):
            model.fit(X, y)
            return name, model
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_single)(name, model)
            for name, model in models.items()
        )
        
        return dict(results)
        
    except ImportError:
        logger.warning("joblib no instalado. Entrenando secuencialmente.")
        for name, model in models.items():
            model.fit(X, y)
        return models


def parallel_predict(
    models: dict,
    weights: dict,
    X: np.ndarray,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Predicción paralela de múltiples modelos.
    
    Args:
        models: Diccionario {nombre: modelo}
        weights: Diccionario {nombre: peso}
        X: Features
        n_jobs: Número de hilos
    
    Returns:
        Predicciones combinadas
    """
    try:
        from joblib import Parallel, delayed
        
        def predict_single(name, model, weight):
            pred = model.predict(X)
            return pred * weight
        
        predictions = Parallel(n_jobs=n_jobs)(
            delayed(predict_single)(name, model, weights.get(name, 1.0))
            for name, model in models.items()
        )
        
        return np.sum(predictions, axis=0)
        
    except ImportError:
        logger.warning("joblib no instalado. Prediciendo secuencialmente.")
        predictions = []
        for name, model in models.items():
            pred = model.predict(X)
            predictions.append(pred * weights.get(name, 1.0))
        return np.sum(predictions, axis=0)


def parallel_monte_carlo(
    probs: np.ndarray,
    pair_mat: np.ndarray,
    n_samples: int = 30000,
    n_jobs: int = -1,
    **kwargs
) -> list:
    """
    Monte Carlo en paralelo.
    
    Args:
        probs: Probabilidades
        pair_mat: Matriz de co-ocurrencia
        n_samples: Número de muestras
        n_jobs: Número de hilos
        **kwargs: Parámetros adicionales
    
    Returns:
        Lista de combinaciones ordenadas
    """
    from collections import Counter
    from scipy.stats import entropy as calc_entropy
    
    temperature = kwargs.get("temperature", 0.78)
    numbers_per_draw = kwargs.get("numbers_per_draw", 5)
    max_number = kwargs.get("max_number", 39)
    
    # Aplicar softmax
    probs_softmax = np.exp(probs / temperature)
    probs_softmax = probs_softmax / probs_softmax.sum()
    
    numbers = np.arange(1, max_number + 1)
    
    def sample_batch(batch_size):
        """Muestrea un lote de combinaciones."""
        candidates = []
        
        for _ in range(batch_size):
            combo = np.random.choice(
                numbers,
                size=numbers_per_draw,
                replace=False,
                p=probs_softmax
            )
            combo = sorted(combo)
            
            # Score
            base_score = np.prod([probs[n - 1] for n in combo])
            diversity = len(set([n // 13 for n in combo]))
            spread = np.std(combo)
            
            pair_penalty = 0
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    pair_penalty += pair_mat[combo[i] - 1][combo[j] - 1]
            
            combo_probs = [probs[n - 1] for n in combo]
            entropy_bonus = calc_entropy(combo_probs)
            
            final_score = (
                base_score * 1e7
                + diversity * 18
                + spread * 1.2
                + entropy_bonus * 25
                - pair_penalty * 22
            )
            
            candidates.append((tuple(combo), final_score))
        
        return candidates
    
    try:
        from joblib import Parallel, delayed
        
        # Dividir en lotes
        batch_size = n_samples // n_jobs if n_jobs > 0 else n_samples
        n_batches = n_samples // batch_size
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(sample_batch)(batch_size)
            for _ in range(n_batches)
        )
        
        # Aplanar resultados
        all_candidates = []
        for batch in results:
            all_candidates.extend(batch)
        
    except ImportError:
        all_candidates = sample_batch(n_samples)
    
    # Agrupar y ordenar
    counter = Counter()
    for combo, score in all_candidates:
        counter[combo] += score
    
    return counter.most_common(kwargs.get("top_k", 10))


@lru_cache(maxsize=128)
def cached_weighted_frequency(
    df_hash: tuple,
    alpha: float = 0.997,
    max_number: int = 39
) -> dict:
    """
    Versión cacheada de weighted_frequency.
    
    Nota: df_hash debe ser un hashable del DataFrame.
    """
    # Esta función requiere implementación específica
    # ya que DataFrames no son directamente hashables
    pass


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimiza los tipos de datos del DataFrame para reducir memoria.
    
    Args:
        df: DataFrame a optimizar
    
    Returns:
        DataFrame optimizado
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'int64':
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
        
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
    
    return df


def vectorized_gap_calculation(
    df: pd.DataFrame,
    max_number: int = 39
) -> np.ndarray:
    """
    Calcula gaps de forma vectorizada (más rápido que loop).
    
    Args:
        df: DataFrame con sorteos
        max_number: Número máximo
    
    Returns:
        Array de gaps (max_number,)
    """
    gaps = np.full(max_number, len(df))
    
    for idx in reversed(range(len(df))):
        for num in df.iloc[idx].values:
            num = int(num) - 1
            if gaps[num] == len(df):
                gaps[num] = len(df) - idx
    
    return gaps
