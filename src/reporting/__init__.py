"""
Módulo de generación de reportes.
"""

from .pdf_report import (
    generate_report,
    generate_metrics_json,
    generate_predictions_csv
)

__all__ = [
    "generate_report",
    "generate_metrics_json",
    "generate_predictions_csv"
]
