# 🎰 MiLoto Quant

**Sistema de predicción de números de lotería basado en ensamble de Machine Learning.**

MiLoto Quant combina LightGBM, XGBoost y Random Forest en un ensamble con pesos dinámicos, entrenado sobre patrones históricos de frecuencia, gaps, momentum y co-ocurrencia, para generar probabilidades y sugerir las combinaciones con mayor score esperado.

---

## ⚡ En una mirada

| | |
|---|---|
| **Qué predice** | Combinaciones de 5 números (rango 1-39) |
| **Modelos** | LightGBM + XGBoost + Random Forest (ensamble con pesos dinámicos) |
| **Features** | Frecuencias multi-ventana, gaps, momentum, EMA, volatilidad, pair matrix |
| **Validación** | Walk-forward con `TimeSeriesSplit` — sin fuga de información temporal |
| **Optimización** | Monte Carlo (30,000 muestras) con scoring compuesto |
| **Salida** | Top-10 combinaciones + reporte PDF con tablas y gráficos |

## 📑 Contenido

- [Pipeline principal](#-pipeline-principal)
- [Instalación](#-instalación)
- [Uso del pipeline](#-uso-del-pipeline)
- [Formato de datos](#-formato-de-datos)
- [Salida del pipeline](#-salida-del-pipeline)
- [Configuración](#-configuración)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Modelos utilizados](#-modelos-utilizados)
- [Métricas de evaluación](#-métricas-de-evaluación)
- [Notas](#-notas)

## 🔄 Pipeline principal

El sistema ejecuta un pipeline de 8 pasos, de punta a punta:

| # | Paso | Qué hace |
|---|---|---|
| 1 | **Ingesta de datos** | Carga y validación de CSV con sorteos históricos |
| 2 | **Feature Engineering** | Frecuencias multi-ventana, gaps, momentum, EMA, volatilidad, pair matrix |
| 3 | **Validación** | Walk-forward con `TimeSeriesSplit` para calcular pesos dinámicos |
| 4 | **Entrenamiento** | Ensamble de 3 modelos (LGBM, XGB, RF) con pesos optimizados |
| 5 | **Predicción** | Scores por número + probabilidad final (ML + Bayesiano) |
| 6 | **Optimización combinatoria** | Monte Carlo (30,000 muestras) con scoring compuesto |
| 7 | **Backtest** *(opcional)* | Evaluación walk-forward con re-entrenamiento |
| 8 | **Reporting** | PDF con tablas, gráficos y combinaciones sugeridas |

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

## ▶️ Uso del pipeline

**Ejecución básica**

```bash
python main.py --file historico.csv
```

**Con backtest**

```bash
python main.py --file historico.csv --backtest
```

**Configuración personalizada**

```bash
python main.py --file historico.csv --config config/local.yaml
```

**Directorio de salida personalizado**

```bash
python main.py --file historico.csv --output mis_resultados/
```

**Modo verbose (logging detallado)**

```bash
python main.py --file historico.csv --verbose
```

**Combinación de opciones**

```bash
python main.py --file historico.csv --config config/local.yaml --output output/ --backtest --verbose
```

### Argumentos disponibles

| Argumento | Descripción | Default |
|---|---|---|
| `--file` | Ruta al CSV con histórico de sorteos | *(requerido)* |
| `--config` | Ruta a archivo YAML de configuración | `None` |
| `--output` | Directorio de salida para reportes | `output/` |
| `--backtest` | Ejecutar backtest completo | `False` |
| `--verbose` | Activar logging detallado | `False` |

## 📄 Formato de datos

CSV sin encabezado, 5 enteros separados por coma (1-39, sin duplicados). **La primera fila es el sorteo más reciente** (se invierte internamente).

```
1,5,12,25,38
3,8,15,22,33
```

## 📤 Salida del pipeline

| Archivo | Contenido |
|---|---|
| `output/reporte_quant_miloto.pdf` | Reporte completo con tablas y gráficos |
| `output/metrics.json` | Métricas de evaluación en formato JSON |
| `output/probs_probs.png` | Gráfico de probabilidades |
| `output/probs_backtest.png` | Histograma de aciertos (si se ejecuta backtest) |
| **Consola** | Top-10 combinaciones sugeridas con scores |

## ⚙️ Configuración

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

## 🗂️ Estructura del proyecto

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

## 🤖 Modelos utilizados

| Modelo | Tipo | Hiperparámetros por defecto |
|---|---|---|
| LightGBM | Regresor | `n_estimators=200`, `lr=0.02`, `max_depth=6` |
| XGBoost | Regresor | `n_estimators=180`, `lr=0.025`, `max_depth=5` |
| RandomForest | Regresor | `n_estimators=180`, `max_depth=12` |

## 📈 Métricas de evaluación

- **Walk-Forward Validation** — MAE invertido como score.
- **Hit Rate** — proporción de aciertos en el top-5.
- **Coverage** — sorteos con al menos 1 acierto.
- **Average Hits** — promedio de aciertos por sorteo.

## 📝 Notas

- `RANDOM_STATE=42` para reproducibilidad.
- Los scripts legacy (`miloto_quant.py`, `backup.py`) se mantienen por compatibilidad.
- El pipeline principal está en `main.py`.
- Los tests se ejecutan con: `python -m pytest tests/`.

## 👤 Autor

**Andrés Giraldo Ramírez**
Software Engineer | Data Analytics / ML / Data Engineering
GitHub: [@giralrez](https://github.com/giralrez)
