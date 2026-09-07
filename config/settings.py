"""
Configuración centralizada de MiLoto Quant.
Todos los parámetros del sistema están definidos aquí.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Configuración de carga y validación de datos."""
    max_number: int = 39
    numbers_per_draw: int = 5
    min_history: int = 30
    max_history: int = 10000
    require_no_duplicates: bool = True
    require_sorted: bool = True


@dataclass
class FeatureConfig:
    """Configuración de feature engineering."""
    alpha_decay: float = 0.997
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_alpha: float = 0.3
    pair_matrix_window: int = 80


@dataclass
class ModelConfig:
    """Configuración de modelos ML."""
    random_state: int = 42
    n_jobs: int = 4
    
    # LightGBM
    lgbm_n_estimators: int = 200
    lgbm_learning_rate: float = 0.02
    lgbm_max_depth: int = 6
    lgbm_subsample: float = 0.85
    lgbm_colsample_bytree: float = 0.85
    
    # XGBoost
    xgb_n_estimators: int = 180
    xgb_learning_rate: float = 0.025
    xgb_max_depth: int = 5
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    
    # RandomForest
    rf_n_estimators: int = 180
    rf_max_depth: int = 12
    
    # ExtraTrees
    et_n_estimators: int = 180
    et_max_depth: int = 12


@dataclass
class EnsembleConfig:
    """Configuración del ensamble de modelos."""
    use_dynamic_weights: bool = True
    fixed_weights: dict = field(default_factory=lambda: {
        "lightgbm": 0.40,
        "xgboost": 0.35,
        "randomforest": 0.25
    })
    ml_weight: float = 0.82
    bayes_weight: float = 0.18
    temperature: float = 0.78
    clip_min: float = 0.001
    clip_max: float = 0.35


@dataclass
class MonteCarloConfig:
    """Configuración de Monte Carlo y optimización combinatoria."""
    n_samples: int = 30000
    temperature: float = 0.78
    top_k: int = 10
    
    # Pesos del scoring compuesto
    base_score_weight: float = 1e7
    diversity_weight: float = 18
    spread_weight: float = 1.2
    entropy_weight: float = 25
    pair_penalty_weight: float = 22
    recency_weight: float = 10


@dataclass
class BacktestConfig:
    """Configuración del backtest."""
    start_ratio: float = 0.80
    n_steps: int = 50
    retrain_every: int = 1


@dataclass
class ValidationConfig:
    """Configuración de validación cruzada."""
    n_splits: int = 5
    test_size: Optional[int] = None


@dataclass
class ReportConfig:
    """Configuración de generación de reportes."""
    output_dir: str = "output"
    pdf_filename: str = "reporte_quant_miloto.pdf"
    save_png: bool = True
    save_csv: bool = True
    save_json: bool = True


@dataclass
class PipelineConfig:
    """Configuración completa del pipeline."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    
    def to_dict(self) -> dict:
        """Convierte la configuración a diccionario."""
        return {
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "model": self.model.__dict__,
            "ensemble": self.ensemble.__dict__,
            "monte_carlo": self.monte_carlo.__dict__,
            "backtest": self.backtest.__dict__,
            "validation": self.validation.__dict__,
            "report": self.report.__dict__,
        }


def get_config() -> PipelineConfig:
    """Retorna la configuración por defecto."""
    return PipelineConfig()


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """Carga configuración desde un archivo YAML."""
    import yaml
    
    config = PipelineConfig()
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'data' in data:
            for k, v in data['data'].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)
        
        if 'features' in data:
            for k, v in data['features'].items():
                if hasattr(config.features, k):
                    setattr(config.features, k, v)
        
        if 'model' in data:
            for k, v in data['model'].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        
        if 'ensemble' in data:
            for k, v in data['ensemble'].items():
                if hasattr(config.ensemble, k):
                    setattr(config.ensemble, k, v)
        
        if 'monte_carlo' in data:
            for k, v in data['monte_carlo'].items():
                if hasattr(config.monte_carlo, k):
                    setattr(config.monte_carlo, k, v)
        
        if 'backtest' in data:
            for k, v in data['backtest'].items():
                if hasattr(config.backtest, k):
                    setattr(config.backtest, k, v)
        
        if 'validation' in data:
            for k, v in data['validation'].items():
                if hasattr(config.validation, k):
                    setattr(config.validation, k, v)
        
        if 'report' in data:
            for k, v in data['report'].items():
                if hasattr(config.report, k):
                    setattr(config.report, k, v)
    
    return config
