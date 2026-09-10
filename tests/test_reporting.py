"""Tests para el módulo de reporting."""

import sys
import os
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reporting.pdf_report import (
    generate_report,
    generate_metrics_json,
    generate_predictions_csv
)


def test_generate_report():
    """Test generación de reporte PDF."""
    # Datos de prueba
    probabilities = np.random.rand(39)
    probabilities = probabilities / probabilities.sum()
    
    combinations = [
        ((1, 5, 12, 25, 38), 100.0),
        ((3, 8, 15, 22, 33), 90.0),
        ((7, 14, 21, 28, 35), 80.0),
    ]
    
    weights = {"lgbm": 0.4, "xgb": 0.35, "rf": 0.25}
    hits = [2, 1, 0, 3, 2]
    metrics = {"average_hits": 1.6, "max_hits": 3}
    
    # Generar en directorio temporal
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_report.pdf")
        
        generate_report(
            probabilities=probabilities,
            combinations=combinations,
            weights=weights,
            hits=hits,
            metrics=metrics,
            output_path=output_path
        )
        
        # Verificar que se creó el archivo
        assert os.path.exists(output_path)


def test_generate_metrics_json():
    """Test generación de métricas JSON."""
    metrics = {
        "average_hits": 1.6,
        "max_hits": 3,
        "total_draws": 50
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "metrics.json")
        
        generate_metrics_json(metrics, output_path)
        
        assert os.path.exists(output_path)
        
        # Verificar contenido
        import json
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded["average_hits"] == 1.6
        assert loaded["max_hits"] == 3


def test_generate_predictions_csv():
    """Test generación de CSV de predicciones."""
    predictions = [
        np.array([1, 5, 12, 25, 38]),
        np.array([3, 8, 15, 22, 33])
    ]
    actuals = [
        np.array([1, 5, 10, 20, 30]),
        np.array([3, 8, 15, 22, 33])
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "predictions.csv")
        
        generate_predictions_csv(predictions, actuals, output_path)
        
        assert os.path.exists(output_path)
        
        # Verificar contenido
        import pandas as pd
        df = pd.read_csv(output_path)
        
        assert len(df) == 2
        assert "hits" in df.columns
        assert df.iloc[0]["hits"] == 2  # 2 aciertos
        assert df.iloc[1]["hits"] == 5  # 5 aciertos


if __name__ == "__main__":
    test_generate_report()
    test_generate_metrics_json()
    test_generate_predictions_csv()
    print("OK: Todos los tests del modulo de reporting pasaron")
