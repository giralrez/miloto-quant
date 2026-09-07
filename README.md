# MiLoto Quant

Sistema de predicción de números de lotería basado en Machine Learning ensemble.

## Descripción

MiLoto Quant utiliza un ensamble de modelos de ML (LightGBM, XGBoost, RandomForest, ExtraTrees) para predecir combinaciones de 5 números (rango 1-39) en sorteos de lotería. El sistema analiza patrones históricos de frecuencia, gaps, momentum y co-ocurrencia para generar probabilidades y sugerir las 10 mejores combinaciones.

## Funcionalidad

### Pipeline Principal

1. **Ingesta de datos** → CSV con sorteos históricos (5 números por fila)
2. **Feature Engineering** → Cálculo de frecuencias, gaps, momentum, EMA, volatilidad
3. **Entrenamiento** → Ensamble de 4 modelos con pesos dinámicos
4. **Predicción** → Motor de ranking + Monte Carlo (30,000 muestras)
5. **Evaluación** → Backtest walk-forward + métricas Hamming/F1/Jaccard
6. **Reporte** → PDF con tablas, gráficos y combinaciones sugeridas

### Versiones

| Versión | Archivo | Enfoque | Modelos |
|---------|---------|---------|---------|
| V1 | `miloto_quant.py` | Clasificador multi-output | 3 (LGBM, XGB, RF) |
| V3 | `backup.py` | Motor de ranking + regresores | 4 (LGBM, XGB, RF, ET) |

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Versión original
python miloto_quant.py --file historico.csv

# Versión V3 (ranking engine)
python backup.py --file historico.csv
```

## Formato de Datos

CSV sin encabezado, 5 enteros separados por coma (1-39, sin duplicados). **Primera fila = sorteo más reciente** (se invierte internamente).

Ejemplo:
```
1,5,12,25,38
3,8,15,22,33
```

## Salida

- `reporte_quant_miloto.pdf` (V1) o `reporte_quant_miloto_v3.pdf` (V3)
- Consola: Top-10 combinaciones sugeridas con scores
- Backtest: Promedio y máximo de aciertos

## Configuración

| Parámetro | V1 | V3 | Descripción |
|-----------|----|----|-------------|
| `MIN_HISTORY` | 30 | 60 | Sorteos mínimos para iniciar |
| `MONTE_CARLO_SAMPLES` | 15,000 | 30,000 | Muestras para optimización |
| `TEMPERATURE` | 1.15 | 0.78 | Temperatura del softmax |
| `ALPHA` | 0.997 | 0.996 | Decaimiento exponencial |

## Estructura del Proyecto

```
miloto_quant/
├── miloto_quant.py          # V1 - Clasificador
├── backup.py                # V3 - Ranking engine
├── feature_builder.py       # (placeholder)
├── lottery4d_predictor/     # (placeholder)
├── historico.csv            # Datos de entrada
├── requirements.txt         # Dependencias
├── .gitignore               # Archivos ignorados por git
└── README.md                # Este archivo
```
