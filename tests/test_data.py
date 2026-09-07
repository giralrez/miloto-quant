"""Tests para el módulo de datos."""

import sys
import os
import tempfile

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.validator import validate_dataframe, validate_csv_file
from src.data.loader import load_data, get_data_stats


def test_validate_dataframe_valid():
    """Test validación de DataFrame válido."""
    data = [
        [1, 5, 10, 15, 20],
        [2, 8, 12, 18, 25],
        [3, 10, 15, 22, 30],
        [4, 12, 20, 28, 35],
        [5, 15, 25, 30, 38]
    ]
    df = pd.DataFrame(data)
    
    result = validate_dataframe(df)
    
    assert result.is_valid is True
    assert result.n_rows == 5
    assert result.n_cols == 5
    assert result.n_duplicates == 0


def test_validate_dataframe_duplicates():
    """Test validación con duplicados."""
    # Crear DataFrame con duplicados por fila (usar listas de listas)
    data = [
        [1, 1, 10, 15, 20],  # Duplicado: 1 aparece dos veces
        [2, 8, 12, 18, 25],
        [3, 3, 15, 22, 30],  # Duplicado: 3 aparece dos veces
        [4, 12, 20, 28, 35],
        [5, 15, 25, 30, 38]
    ]
    df = pd.DataFrame(data)
    
    result = validate_dataframe(df, require_no_duplicates=True)
    
    assert result.is_valid is False
    assert result.n_duplicates == 2


def test_validate_dataframe_out_of_range():
    """Test validación con valores fuera de rango."""
    data = [
        [1, 0, 10, 15, 20],   # 0 fuera de rango
        [2, 8, 12, 18, 25],
        [3, 10, 15, 22, 30],
        [4, 12, 20, 28, 35],
        [5, 40, 25, 30, 38]   # 40 fuera de rango
    ]
    df = pd.DataFrame(data)
    
    result = validate_dataframe(df, max_number=39)
    
    assert result.n_out_of_range == 2
    assert len(result.warnings) > 0


def test_validate_dataframe_wrong_columns():
    """Test validación con número incorrecto de columnas."""
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    df = pd.DataFrame(data)
    
    result = validate_dataframe(df, numbers_per_draw=5)
    
    assert result.is_valid is False
    assert len(result.errors) > 0


def test_validate_csv_file():
    """Test validación de archivo CSV."""
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1,5,12,25,38\n")
        f.write("3,8,15,22,33\n")
        f.write("7,14,21,28,35\n")
        temp_path = f.name
    
    try:
        df, result = validate_csv_file(temp_path)
        
        assert df is not None
        assert result.is_valid is True
        assert result.n_rows == 3
    finally:
        os.unlink(temp_path)


def test_load_data():
    """Test carga de datos."""
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1,5,12,25,38\n")
        f.write("3,8,15,22,33\n")
        f.write("7,14,21,28,35\n")
        f.write("10,18,25,30,38\n")
        temp_path = f.name
    
    try:
        df = load_data(temp_path, reverse_chronology=True)
        
        assert len(df) == 4
        assert len(df.columns) == 5
        # Después de invertir, la última fila del CSV debería ser la primera
        assert df.iloc[0][0] == 10
    finally:
        os.unlink(temp_path)


def test_load_data_with_limit():
    """Test carga de datos con límite."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        for i in range(10):
            f.write(f"{i+1},{i+6},{i+11},{i+16},{i+21}\n")
        temp_path = f.name
    
    try:
        df = load_data(temp_path, limit=5)
        
        assert len(df) == 5
    finally:
        os.unlink(temp_path)


def test_get_data_stats():
    """Test estadísticas de datos."""
    data = [
        [1, 5, 10, 15, 20],
        [2, 8, 12, 18, 25],
        [3, 10, 15, 22, 30]
    ]
    df = pd.DataFrame(data)
    
    stats = get_data_stats(df)
    
    assert stats["n_draws"] == 3
    assert stats["n_columns"] == 5
    assert stats["min_number"] == 1
    assert stats["max_number"] == 30
    assert "number_frequencies" in stats


if __name__ == "__main__":
    test_validate_dataframe_valid()
    test_validate_dataframe_duplicates()
    test_validate_dataframe_out_of_range()
    test_validate_dataframe_wrong_columns()
    test_validate_csv_file()
    test_load_data()
    test_load_data_with_limit()
    test_get_data_stats()
    print("OK: Todos los tests del módulo de datos pasaron")
