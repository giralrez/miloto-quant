"""
Módulo de carga y validación de datos.
"""

from .loader import load_data, get_data_stats, print_data_stats
from .validator import (
    validate_dataframe,
    validate_csv_file,
    print_validation_report,
    ValidationResult
)

__all__ = [
    "load_data",
    "get_data_stats",
    "print_data_stats",
    "validate_dataframe",
    "validate_csv_file",
    "print_validation_report",
    "ValidationResult"
]
