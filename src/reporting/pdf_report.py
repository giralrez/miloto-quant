"""
Generación de reportes PDF para MiLoto Quant.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    probabilities: np.ndarray,
    combinations: List[Tuple[tuple, float]],
    weights: Dict[str, float],
    hits: List[int],
    metrics: Dict[str, float],
    output_path: str = "reporte_quant_miloto.pdf",
    model_probs: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Genera el reporte PDF completo.
    
    Args:
        probabilities: Probabilidades finales de cada número
        combinations: Lista de combinaciones (combo, score)
        weights: Pesos de los modelos
        hits: Lista de aciertos del backtest
        metrics: Métricas de evaluación
        output_path: Ruta de salida del PDF
        model_probs: Probabilidades por modelo (opcional)
    """
    try:
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image
        )
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("reportlab o matplotlib no instalado. Saltando generación de PDF.")
        return
    
    # Crear documento
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Título
    elements.append(Paragraph("MiLoto Quant - Reporte", styles["Title"]))
    elements.append(Spacer(1, 20))
    
    # Fecha
    elements.append(Paragraph(
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))
    
    # Tabla de probabilidades
    elements.append(Paragraph("Top 15 Probabilidades", styles["Heading2"]))
    prob_table = [["Número", "Probabilidad"]]
    
    sorted_probs = sorted(
        enumerate(probabilities, start=1),
        key=lambda x: x[1],
        reverse=True
    )
    
    for num, prob in sorted_probs[:15]:
        prob_table.append([str(num), f"{prob:.6f}"])
    
    table = Table(prob_table)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Tabla de pesos
    if weights:
        elements.append(Paragraph("Pesos del Ensamble", styles["Heading2"]))
        weight_table = [["Modelo", "Peso"]]
        
        for name, weight in weights.items():
            weight_table.append([name, f"{weight:.4f}"])
        
        table2 = Table(weight_table)
        table2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.green),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table2)
        elements.append(Spacer(1, 20))
    
    # Tabla de combinaciones
    elements.append(Paragraph("Top 10 Combinaciones", styles["Heading2"]))
    combo_table = [["Combinación", "Score"]]
    
    for combo, score in combinations[:10]:
        combo_table.append([str(combo), f"{score:.2f}"])
    
    table3 = Table(combo_table)
    table3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.red),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table3)
    elements.append(Spacer(1, 20))
    
    # Gráfico de probabilidades
    plt.figure(figsize=(10, 5))
    nums = [x[0] for x in sorted_probs[:15]]
    probs = [x[1] for x in sorted_probs[:15]]
    plt.bar(nums, probs, color='steelblue')
    plt.title("Top 15 Probabilidades")
    plt.xlabel("Número")
    plt.ylabel("Probabilidad")
    plt.tight_layout()
    
    plot_path = output_path.replace('.pdf', '_probs.png')
    plt.savefig(plot_path)
    plt.close()
    
    elements.append(Image(plot_path, width=450, height=250))
    elements.append(Spacer(1, 20))
    
    # Histograma de backtest
    if hits:
        plt.figure(figsize=(10, 5))
        plt.hist(hits, bins=range(max(hits) + 2), color='steelblue', edgecolor='black')
        plt.title("Distribución de Aciertos - Backtest")
        plt.xlabel("Número de Aciertos")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        
        hist_path = output_path.replace('.pdf', '_backtest.png')
        plt.savefig(hist_path)
        plt.close()
        
        elements.append(Image(hist_path, width=450, height=250))
    
    # Construir PDF
    doc.build(elements)
    logger.info(f"Reporte generado: {output_path}")


def generate_metrics_json(
    metrics: Dict[str, float],
    output_path: str = "output/metrics.json"
) -> None:
    """
    Genera archivo JSON con métricas.
    
    Args:
        metrics: Diccionario de métricas
        output_path: Ruta de salida
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Métricas guardadas en: {output_path}")


def generate_predictions_csv(
    predictions: List[np.ndarray],
    actuals: List[np.ndarray],
    output_path: str = "output/predictions.csv"
) -> None:
    """
    Genera CSV con predicciones y valores reales.
    
    Args:
        predictions: Lista de predicciones por sorteo
        actuals: Lista de valores reales por sorteo
        output_path: Ruta de salida
    """
    import os
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        rows.append({
            "draw_index": i,
            "predicted": "-".join(map(str, sorted(pred))),
            "actual": "-".join(map(str, sorted(actual))),
            "hits": len(set(pred).intersection(set(actual)))
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Predicciones guardadas en: {output_path}")
