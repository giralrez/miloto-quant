"""
MiLoto Quant - Entry Point Principal
Pipeline de predicción de lotería con ML ensemble.
"""

import argparse
import sys
import logging
from pathlib import Path

from config.settings import get_config, load_config_from_yaml


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


def main():
    parser = argparse.ArgumentParser(
        description="MiLoto Quant - Sistema de predicción de lotería",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py --file historico.csv
  python main.py --file historico.csv --config config/local.yaml
  python main.py --file historico.csv --output output/
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
        "--verbose",
        action="store_true",
        help="Activar logging detallado"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Iniciando MiLoto Quant")
    
    # Cargar configuración
    if args.config:
        config = load_config_from_yaml(args.config)
        logger.info(f"Configuración cargada desde: {args.config}")
    else:
        config = get_config()
        logger.info("Usando configuración por defecto")
    
    # Actualizar directorio de salida
    config.report.output_dir = args.output
    
    # Verificar que el archivo existe
    if not Path(args.file).exists():
        logger.error(f"Archivo no encontrado: {args.file}")
        sys.exit(1)
    
    logger.info(f"Archivo de datos: {args.file}")
    logger.info(f"Directorio de salida: {args.output}")
    
    # TODO: Implementar pipeline completo en siguientes fases
    logger.info("Pipeline en desarrollo - Fase 1 completada")
    logger.info("Próximas fases: Feature Engineering, Modelos, Predicción, etc.")
    
    print("\nOK: MiLoto Quant inicializado correctamente")
    print(f"Configuracion: {'Personalizada' if args.config else 'Por defecto'}")
    print(f"Datos: {args.file}")
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()
