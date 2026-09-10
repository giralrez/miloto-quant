"""
MiLoto Quant - Entry Point Principal
Pipeline de predicción de lotería con ML ensemble.
"""

import argparse
import sys
import logging
import os
from pathlib import Path

import numpy as np

from config.settings import get_config, load_config_from_yaml

# Imports del pipeline
from src.data.loader import load_data, get_data_stats, print_data_stats
from src.features.builder import build_features_v3, build_latest_features_v3
from src.features.statistical import pair_matrix
from src.models.lightgbm_model import create_lgbm_regressor
from src.models.xgboost_model import create_xgb_regressor
from src.models.random_forest import create_rf_regressor
from src.models.ensemble import DynamicEnsemble
from src.training.trainer import train_ensemble
from src.training.validator import walk_forward_validation, calculate_optimal_weights
from src.prediction.predictor import predict_scores, predict_top_numbers
from src.prediction.probability import bayesian_prior, final_probability
from src.prediction.monte_carlo import optimize_combinations
from src.evaluation.backtest import BacktestEngine
from src.reporting.pdf_report import generate_report, generate_metrics_json, generate_predictions_csv


def setup_logging(verbose: bool = False):
    """Configura el logging del sistema."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_pipeline(args):
    """
    Ejecuta el pipeline completo de predicción.
    
    Args:
        args: Argumentos de línea de comandos
    """
    logger = logging.getLogger(__name__)
    
    # Cargar configuración
    if args.config:
        config = load_config_from_yaml(args.config)
        logger.info(f"Configuración cargada desde: {args.config}")
    else:
        config = get_config()
        logger.info("Usando configuración por defecto")
    
    # Crear directorio de salida
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Cargar datos
    logger.info("PASO 1: Cargando datos...")
    df = load_data(
        args.file,
        max_number=config.data.max_number,
        limit=config.data.max_history
    )
    stats = get_data_stats(df)
    print_data_stats(stats)
    
    # 2. Construir features
    logger.info("PASO 2: Construyendo features...")
    X, y = build_features_v3(
        df,
        min_history=config.data.min_history,
        max_number=config.data.max_number,
        windows=config.features.rolling_windows
    )
    logger.info(f"Features: X={X.shape}, y={y.shape}")
    
    # 3. Validación walk-forward
    logger.info("PASO 3: Validación walk-forward...")
    models = {
        "lightgbm": create_lgbm_regressor(
            n_estimators=config.model.lgbm_n_estimators,
            learning_rate=config.model.lgbm_learning_rate,
            max_depth=config.model.lgbm_max_depth,
            subsample=config.model.lgbm_subsample,
            colsample_bytree=config.model.lgbm_colsample_bytree,
            random_state=config.model.random_state,
            n_jobs=config.model.n_jobs
        ),
        "xgboost": create_xgb_regressor(
            n_estimators=config.model.xgb_n_estimators,
            learning_rate=config.model.xgb_learning_rate,
            max_depth=config.model.xgb_max_depth,
            subsample=config.model.xgb_subsample,
            colsample_bytree=config.model.xgb_colsample_bytree,
            random_state=config.model.random_state,
            n_jobs=config.model.n_jobs
        ),
        "randomforest": create_rf_regressor(
            n_estimators=config.model.rf_n_estimators,
            max_depth=config.model.rf_max_depth,
            random_state=config.model.random_state,
            n_jobs=config.model.n_jobs
        )
    }
    
    scores = walk_forward_validation(
        X, y, models,
        n_splits=config.validation.n_splits,
        model_type="regressor"
    )
    
    # Calcular pesos
    weights = calculate_optimal_weights(scores)
    logger.info(f"Pesos óptimos: {weights}")
    
    # 4. Entrenar ensamble final
    logger.info("PASO 4: Entrenando ensamble final...")
    trained_models = train_ensemble(models, X, y)
    
    ensemble = DynamicEnsemble(trained_models, use_dynamic_weights=True)
    ensemble.calculate_weights(weights)
    
    # 5. Predecir
    logger.info("PASO 5: Prediciendo...")
    X_latest = build_latest_features_v3(
        df,
        max_number=config.data.max_number,
        windows=config.features.rolling_windows
    )
    
    numbers = np.arange(1, config.data.max_number + 1)
    final_scores, per_model = predict_scores(
        trained_models, weights, X_latest, numbers
    )
    
    # Probabilidad final
    bayes = bayesian_prior(df, max_number=config.data.max_number)
    probs = final_probability(
        final_scores, bayes,
        ml_weight=config.ensemble.ml_weight,
        bayes_weight=config.ensemble.bayes_weight,
        temperature=config.ensemble.temperature,
        clip_min=config.ensemble.clip_min,
        clip_max=config.ensemble.clip_max
    )
    
    # Top números
    top_numbers = predict_top_numbers(final_scores, numbers, top_k=10)
    logger.info(f"Top 10 números: {top_numbers}")
    
    # Monte Carlo
    logger.info("PASO 6: Optimización combinatoria...")
    pair_mat = pair_matrix(
        df,
        window=config.features.pair_matrix_window,
        max_number=config.data.max_number
    )
    
    combinations = optimize_combinations(
        probs, pair_mat,
        n_samples=config.monte_carlo.n_samples,
        temperature=config.monte_carlo.temperature,
        top_k=config.monte_carlo.top_k,
        max_number=config.data.max_number
    )
    
    # 7. Backtest
    logger.info("PASO 7: Backtest...")
    if args.backtest:
        backtest_engine = BacktestEngine(
            min_history=config.data.min_history,
            n_steps=config.backtest.n_steps,
            start_ratio=config.backtest.start_ratio,
            windows=config.features.rolling_windows,
            max_number=config.data.max_number,
            n_estimators=config.model.lgbm_n_estimators,
            monte_carlo_samples=config.monte_carlo.n_samples,
            verbose=True
        )
        backtest_results = backtest_engine.run(df)
        hits = backtest_results["hits"]
        metrics = backtest_results["summary"]
    else:
        hits = []
        metrics = {"average_hits": 0, "max_hits": 0}
    
    # 8. Generar reportes
    logger.info("PASO 8: Generando reportes...")
    pdf_path = os.path.join(args.output, config.report.pdf_filename)
    generate_report(
        probabilities=probs,
        combinations=combinations,
        weights=weights,
        hits=hits,
        metrics=metrics,
        output_path=pdf_path
    )
    
    # Métricas JSON
    metrics_path = os.path.join(args.output, "metrics.json")
    generate_metrics_json(metrics, metrics_path)
    
    # Resumen final
    print("\n" + "="*50)
    print("PIPELINE COMPLETADO")
    print("="*50)
    print(f"\nTop 10 números sugeridos: {list(top_numbers)}")
    print(f"\nTop 5 combinaciones:")
    for i, (combo, score) in enumerate(combinations[:5], 1):
        print(f"  {i}. {combo} (score: {score:.2f})")
    
    if hits:
        print(f"\nBacktest:")
        print(f"  Promedio aciertos: {metrics['average_hits']:.4f}")
        print(f"  Máximo aciertos: {metrics['max_hits']}")
    
    print(f"\nReportes generados en: {args.output}/")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="MiLoto Quant - Sistema de predicción de lotería",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py --file historico.csv
  python main.py --file historico.csv --config config/local.yaml
  python main.py --file historico.csv --output output/
  python main.py --file historico.csv --backtest
  python main.py --file historico.csv --verbose
        """
    )
    
    parser.add_argument(
        "--file",
        required=True,
        help="Ruta al CSV con histórico de sorteos"
    )
    
    parser.add_argument(
        "--config",
        default=None,
        help="Ruta a archivo YAML de configuración (opcional)"
    )
    
    parser.add_argument(
        "--output",
        default="output",
        help="Directorio de salida para reportes (default: output)"
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Ejecutar backtest completo"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activar logging detallado"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Verificar que el archivo existe
    if not Path(args.file).exists():
        print(f"ERROR: Archivo no encontrado: {args.file}")
        sys.exit(1)
    
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
