"""
Validación de datos de entrada para MiLoto Quant.
Verifica formato, rangos y consistencia del CSV.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de la validación de datos."""
    is_valid: bool
    n_rows: int
    n_cols: int
    n_duplicates: int
    n_out_of_range: int
    errors: List[str]
    warnings: List[str]


def validate_dataframe(
    df: pd.DataFrame,
    max_number: int = 39,
    numbers_per_draw: int = 5,
    require_no_duplicates: bool = True
) -> ValidationResult:
    """
    Valida un DataFrame con datos de lotería.
    
    Args:
        df: DataFrame a validar
        max_number: Número máximo permitido (default: 39)
        numbers_per_draw: Números por sorteo (default: 5)
        require_no_duplicates: Si True, falla si hay duplicados por fila
    
    Returns:
        ValidationResult con el resultado de la validación
    """
    errors = []
    warnings = []
    n_duplicates = 0
    n_out_of_range = 0
    
    # Verificar número de columnas
    if len(df.columns) != numbers_per_draw:
        errors.append(
            f"Se esperaban {numbers_per_draw} columnas, "
            f"se encontraron {len(df.columns)}"
        )
    
    # Verificar que no haya valores nulos
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        errors.append(f"Se encontraron {null_count} valores nulos")
    
    # Verificar que todos sean enteros
    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]):
            try:
                df[col] = df[col].astype(int)
            except (ValueError, TypeError):
                errors.append(
                    f"Columna {col} contiene valores no enteros"
                )
    
    # Verificar rango [1, max_number]
    for col in df.columns:
        out_of_range = ((df[col] < 1) | (df[col] > max_number)).sum()
        n_out_of_range += out_of_range
        if out_of_range > 0:
            warnings.append(
                f"Columna {col}: {out_of_range} valores fuera de rango [1, {max_number}]"
            )
    
    # Verificar duplicados por fila
    if require_no_duplicates:
        for idx, row in df.iterrows():
            if len(row) != len(set(row)):
                n_duplicates += 1
        
        if n_duplicates > 0:
            errors.append(
                f"{n_duplicates} filas tienen números duplicados"
            )
    
    # Verificar orden (advertencia, no error)
    if len(df) > 1:
        # La primera fila debería ser la más reciente
        logger.info("Asumiendo: primera fila = sorteo más reciente")
    
    is_valid = len(errors) == 0
    
    result = ValidationResult(
        is_valid=is_valid,
        n_rows=len(df),
        n_cols=len(df.columns),
        n_duplicates=n_duplicates,
        n_out_of_range=n_out_of_range,
        errors=errors,
        warnings=warnings
    )
    
    return result


def validate_csv_file(
    filepath: str,
    max_number: int = 39,
    numbers_per_draw: int = 5,
    require_no_duplicates: bool = True
) -> Tuple[Optional[pd.DataFrame], ValidationResult]:
    """
    Valida un archivo CSV de lotería.
    
    Args:
        filepath: Ruta al archivo CSV
        max_number: Número máximo permitido
        numbers_per_draw: Números por sorteo
        require_no_duplicates: Si True, falla si hay duplicados
    
    Returns:
        Tupla de (DataFrame, ValidationResult)
    """
    try:
        df = pd.read_csv(filepath, header=None)
        logger.info(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas")
    except Exception as e:
        result = ValidationResult(
            is_valid=False,
            n_rows=0,
            n_cols=0,
            n_duplicates=0,
            n_out_of_range=0,
            errors=[f"Error al leer CSV: {str(e)}"],
            warnings=[]
        )
        return None, result
    
    validation = validate_dataframe(
        df,
        max_number=max_number,
        numbers_per_draw=numbers_per_draw,
        require_no_duplicates=require_no_duplicates
    )
    
    return df, validation


def print_validation_report(result: ValidationResult) -> None:
    """Imprime un reporte de validación legible."""
    print("\n=== Reporte de Validación ===")
    print(f"Filas: {result.n_rows}")
    print(f"Columnas: {result.n_cols}")
    print(f"Duplicados: {result.n_duplicates}")
    print(f"Fuera de rango: {result.n_out_of_range}")
    
    if result.errors:
        print("\nERRORES:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("\nADVERTENCIAS:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    print(f"\nEstado: {'VALIDO' if result.is_valid else 'INVALIDO'}")
