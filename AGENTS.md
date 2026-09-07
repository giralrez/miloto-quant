## instrucciones generales
-  empieza siempre tu respuesta con el emoji 🤖
-  responde siempre en español

# MiLoto Quant

Predicción de números de lotería usando ensamble ML (LightGBM, XGBoost, RandomForest).

## Estructura del proyecto

- `miloto_quant.py` — Versión original (clasificador, multi-output)
- `backup.py` — Reescritura V3 (regresor, motor de ranking con pesos dinámicos)
- `feature_builder.py` — Placeholder vacío
- `lottery4d_predictor/` — Placeholder vacío
- `.venv/`, `.venv-1/` — Entornos virtuales (dependencias no fijadas)

## Ejecución

Ambos scripts requieren un CSV con sorteos históricos (5 números por fila, rango 1–39):

```bash
python miloto_quant.py --file historico.csv
python backup.py --file historico.csv
```

Salida: reporte PDF (`reporte_quant_miloto.pdf` o `reporte_quant_miloto_v3.pdf`) + predicciones impresas.

## Dependencias (no están en requirements.txt)

pandas, numpy, scikit-learn, lightgbm, xgboost, scipy, reportlab, matplotlib

## Diferencias clave entre versiones

- `miloto_quant.py`: Pesos fijos del ensamble (LGBM 40%, XGB 35%, RF 25%), usa `MultiOutputClassifier`
- `backup.py`: Pesos dinámicos de validación, usa regresores, agrega señales EMA y features de momentum

## Formato de datos

CSV sin encabezado, 5 enteros separados por coma por fila (1–39, sin duplicados). **Primera fila = sorteo más reciente** (ambos scripts invierten la cronología internamente).

## Notas

- Sin tests, sin linting, sin CI
- `RANDOM_STATE=42` para reproducibilidad
- Ambos scripts suprimen todos los warnings
