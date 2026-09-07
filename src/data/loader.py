"""
Carga y preprocesamiento de datos de lotería.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .validator import validate_dataframe, ValidationResult

logger = logging.getLogger(__name__)


def load_data(
    filepath: str,
    max_number: int = 39,
    limit: Optional[int] = None,
    reverse_chronology: bool = True
) -> pd.DataFrame:
    """
    Carga y preprocesa datos de lotería desde un CSV.
    
    Args:
        filepath: Ruta al archivo CSV
        max_number: Número máximo permitido (default: 39)
        limit: Límite máximo de filas a cargar (default: None)
        reverse_chronology: Si True, invierte el orden cronológico
                           (primera fila = más reciente)
    
    Returns:
        DataFrame preprocesado
    
    Raises:
        ValueError: Si los datos no son válidos
    """
    logger.info(f"Cargando datos desde: {filepath}")
    
    # Cargar CSV
    df = pd.read_csv(filepath, header=None)
    logger.info(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas")
    
    # Limpiar datos
    df = _clean_data(df, max_number)
    
    # Validar
    validation = validate_dataframe(
        df,
        max_number=max_number,
        numbers_per_draw=5,
        require_no_duplicates=True
    )
    
    if not validation.is_valid:
        raise ValueError(
            f"Datos inválidos:\n" + "\n".join(validation.errors)
        )
    
    # Aplicar límite
    if limit is not None and limit > 0:
        df = df.head(limit)
        logger.info(f"Límite aplicado: {len(df)} filas")
    
    # Invertir cronología (primera fila = más reciente → última = más reciente)
    if reverse_chronology:
        df = df[::-1].reset_index(drop=True)
        logger.info("Cronología invertida para entrenamiento")
    
    logger.info(f"Datos finales: {len(df)} sorteos")
    
    return df


def _clean_data(df: pd.DataFrame, max_number: int) -> pd.DataFrame:
    """
    Limpia y normaliza los datos del CSV.
    
    Args:
        df: DataFrame crudo
        max_number: Número máximo permitido
    
    Returns:
        DataFrame limpio
    """
    # Eliminar filas completamente vacías
    df = df.dropna(how="all")
    
    # Convertir a numérico
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Eliminar filas con valores no numéricos
    df = df.dropna()
    
    # Convertir a entero
    df = df.astype(int)
    
    # Filtrar valores fuera de rango
    df = df[
        (df >= 1).all(axis=1) &
        (df <= max_number).all(axis=1)
    ]
    
    # Eliminar duplicados por fila
    def has_duplicates(row):
        return len(row) != len(set(row))
    
    df = df[~df.apply(has_duplicates, axis=1)]
    
    # Ordenar cada fila (números en orden ascendente)
    df = df.apply(sorted, axis=1, result_type="expand")
    
    return df


def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Calcula estadísticas básicas del dataset.
    
    Args:
        df: DataFrame de lotería
    
    Returns:
        Diccionario con estadísticas
    """
    stats = {
        "n_draws": len(df),
        "n_columns": len(df.columns),
        "min_number": int(df.min().min()),
        "max_number": int(df.max().max()),
        "mean_number": float(df.mean().mean()),
        "std_number": float(df.std().mean()),
    }
    
    # Frecuencia de cada número
    all_numbers = df.values.flatten()
    freq = pd.Series(all_numbers).value_counts().sort_index()
    stats["number_frequencies"] = freq.to_dict()
    
    # Números más y menos comunes
    stats["most_common"] = freq.idxmax()
    stats["least_common"] = freq.idxmin()
    
    return stats


def print_data_stats(stats: dict) -> None:
    """Imprime estadísticas del dataset."""
    print("\n=== Estadísticas del Dataset ===")
    print(f"Sorteos: {stats['n_draws']}")
    print(f"Columnas: {stats['n_columns']}")
    print(f"Rango de números: [{stats['min_number']}, {stats['max_number']}]")
    print(f"Promedio: {stats['mean_number']:.2f}")
    print(f"Desviación estándar: {stats['std_number']:.2f}")
    print(f"Número más común: {stats['most_common']}")
    print(f"Número menos común: {stats['least_common']}")
