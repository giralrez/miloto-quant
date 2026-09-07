"""Tests para el módulo de configuración."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config, PipelineConfig


def test_default_config():
    """Test que la configuración por defecto se crea correctamente."""
    config = get_config()
    
    assert isinstance(config, PipelineConfig)
    assert config.data.max_number == 39
    assert config.data.numbers_per_draw == 5
    assert config.model.random_state == 42
    assert config.ensemble.use_dynamic_weights is True


def test_config_to_dict():
    """Test que la configuración se convierte a diccionario."""
    config = get_config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert "data" in config_dict
    assert "features" in config_dict
    assert "model" in config_dict
    assert "ensemble" in config_dict
    assert "monte_carlo" in config_dict


def test_config_values():
    """Test que los valores de configuración son correctos."""
    config = get_config()
    
    assert config.features.alpha_decay == 0.997
    assert config.features.rolling_windows == [5, 10, 20, 50]
    assert config.monte_carlo.n_samples == 30000
    assert config.backtest.n_steps == 50


if __name__ == "__main__":
    test_default_config()
    test_config_to_dict()
    test_config_values()
    print("OK: Todos los tests de configuracion pasaron")
