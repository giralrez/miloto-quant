# MiLoto Quant

Sistema de predicción de números de lotería basado en Machine Learning ensemble.

## Descripción

MiLoto Quant utiliza un ensamble de modelos de ML (LightGBM, XGBoost, RandomForest) para predecir combinaciones de 5 números (rango 1-39) en sorteos de lotería. El sistema analiza patrones históricos de frecuencia, gaps, momentum y co-ocurrencia para generar probabilidades y sugerir las 10 mejores combinaciones.

## Pipeline Principal

El sistema ejecuta un pipeline completo de 8 pasos:

1. **Ingesta de datos** → Carga y validación de CSV con sorteos históricos
2. **Feature Engineering** → Frecuencias multi-ventana, gaps, momentum, EMA, volatilidad, pair matrix
3. **Validación** → Walk-forward con TimeSeriesSplit para calcular pesos dinámicos
4. **Entrenamiento** → Ensamble de 3 modelos (LGBM, XGB, RF) con pesos optimizados
5. **Predicción** → Scores por número + probabilidad final (ML + Bayesiano)
6. **Optimización combinatoria** → Monte Carlo (30,000 muestras) con scoring compuesto
7. **Backtest** → Evaluación walk-forward con re-entrenamiento (opcional)
8. **Reporting** → PDF con tablas, gráficos y combinaciones sugeridas

## Instalación

```bash
pip install -r requirements.txt
```

## Uso del Pipeline

### Ejecución básica

```bash
python main.py --file historico.csv
```

### Ejecutar con backtest

```bash
python main.py --file historico.csv --backtest
```

### Configuración personalizada

```bash
python main.py --file historico.csv --config config/local.yaml
```

### Directorio de salida personalizado

```bash
python main.py --file historico.csv --output mis_resultados/
```

### Modo verbose (logging detallado)

```bash
python main.py --file historico.csv --verbose
```

### Combinación de opciones

```bash
python main.py --file historico.csv --config config/local.yaml --output output/ --backtest --verbose
```

### Argumentos disponibles

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--file` | Ruta al CSV con histórico de sorteos | (requerido) |
| `--config` | Ruta a archivo YAML de configuración | None |
| `--output` | Directorio de salida para reportes | `output/` |
| `--backtest` | Ejecutar backtest completo | False |
| `--verbose` | Activar logging detallado | False |

## Formato de Datos

CSV sin encabezado, 5 enteros separados por coma (1-39, sin duplicados). **Primera fila = sorteo más reciente** (se invierte internamente).

Ejemplo:
```
1,5,12,25,38
3,8,15,22,33
```

## Salida del Pipeline

- `output/reporte_quant_miloto.pdf` — Reporte completo con tablas y gráficos
- `output/metrics.json` — Métricas de evaluación en formato JSON
- `output/probs_probs.png` — Gráfico de probabilidades
- `output/probs_backtest.png` — Histograma de aciertos (si se ejecuta backtest)
- **Consola**: Top-10 combinaciones sugeridas con scores

## Configuración

El pipeline es configurable mediante archivos YAML o editando `config/settings.py`:

```yaml
data:
  max_number: 39
  numbers_per_draw: 5
  min_history: 30

features:
  alpha_decay: 0.997
  rolling_windows: [5, 10, 20, 50]

model:
  lgbm_n_estimators: 200
  lgbm_learning_rate: 0.02
  xgb_n_estimators: 180

ensemble:
  ml_weight: 0.82
  bayes_weight: 0.18
  temperature: 0.78

monte_carlo:
  n_samples: 30000
  top_k: 10
```

## Estructura del Proyecto

```
miloto_quant/
├── main.py                          # Entry point principal
├── config/
│   ├── settings.py                  # Configuración centralizada
│   └── default.yaml                 # Configuración por defecto
├── src/
│   ├── data/                        # Carga y validación de datos
│   │   ├── loader.py
│   │   └── validator.py
│   ├── features/                    # Feature engineering
│   │   ├── frequencies.py
│   │   ├── gaps.py
│   │   ├── momentum.py
│   │   ├── statistical.py
│   │   └── builder.py
│   ├── models/                      # Modelos ML
│   │   ├── lightgbm_model.py
│   │   ├── xgboost_model.py
│   │   ├── random_forest.py
│   │   ├── extra_trees.py
│   │   └── ensemble.py
│   ├── training/                    # Entrenamiento y validación
│   │   ├── trainer.py
│   │   ├── validator.py
│   │   └── hyperparams.py
│   ├── prediction/                  # Predicción y Monte Carlo
│   │   ├── predictor.py
│   │   ├── probability.py
│   │   └── monte_carlo.py
│   ├── evaluation/                  # Evaluación y backtest
│   │   ├── metrics.py
│   │   └── backtest.py
│   ├── reporting/                   # Generación de reportes
│   │   └── pdf_report.py
│   └── utils/                       # Optimizaciones
│       └── optimization.py
├── tests/                           # Tests unitarios e integración
├── data/                            # Datos de entrada
├── output/                          # Reportes generados
├── miloto_quant.py                  # V1 original (legacy)
├── backup.py                        # V3 original (legacy)
├── requirements.txt
├── .gitignore
└── README.md
```

## Modelos Utilizados

| Modelo | Tipo | Hiperparámetros por defecto |
|--------|------|-----------------------------|
| LightGBM | Regresor | n_estimators=200, lr=0.02, max_depth=6 |
| XGBoost | Regresor | n_estimators=180, lr=0.025, max_depth=5 |
| RandomForest | Regresor | n_estimators=180, max_depth=12 |

## Métricas de Evaluación

- **Walk-Forward Validation**: MAE invertido como score
- **Hit Rate**: Proporción de aciertos en top-5
- **Coverage**: Sorteos con al menos 1 acierto
- **Average Hits**: Promedio de aciertos por sorteo

## Notas

- `RANDOM_STATE=42` para reproducibilidad
- Los scripts legacy (`miloto_quant.py`, `backup.py`) se mantienen por compatibilidad
- El pipeline principal está en `main.py`
- Los tests se ejecutan con: `python -m pytest tests/`
