"""
Optimización combinatoria con Monte Carlo.
"""

import logging
from typing import List, Tuple
from collections import Counter

import numpy as np
from scipy.stats import entropy as calc_entropy

logger = logging.getLogger(__name__)


def optimize_combinations(
    probs: np.ndarray,
    pair_mat: np.ndarray,
    n_samples: int = 30000,
    temperature: float = 0.78,
    top_k: int = 10,
    numbers_per_draw: int = 5,
    max_number: int = 39,
    weights: dict = None
) -> List[Tuple[tuple, float]]:
    """
    Optimiza combinaciones usando Monte Carlo.
    
    Args:
        probs: Probabilidades de cada número
        pair_mat: Matriz de co-ocurrencia
        n_samples: Número de muestras
        temperature: Temperatura del softmax
        top_k: Número de mejores combinaciones a retornar
        numbers_per_draw: Números por sorteo
        max_number: Número máximo
        weights: Pesos del scoring compuesto
    
    Returns:
        Lista de tuplas (combinacion, score)
    """
    if weights is None:
        weights = {
            "base_score": 1e7,
            "diversity": 18,
            "spread": 1.2,
            "entropy": 25,
            "pair_penalty": 22
        }
    
    # Aplicar softmax con temperatura
    probs_softmax = np.exp(probs / temperature)
    probs_softmax = probs_softmax / probs_softmax.sum()
    
    numbers = np.arange(1, max_number + 1)
    candidates = []
    
    for _ in range(n_samples):
        # Muestrear combinación
        combo = np.random.choice(
            numbers,
            size=numbers_per_draw,
            replace=False,
            p=probs_softmax
        )
        combo = sorted(combo)
        
        # Score base: producto de probabilidades
        base_score = np.prod([probs[n - 1] for n in combo])
        
        # Diversidad: cuántos terciles diferentes
        diversity = len(set([n // 13 for n in combo]))
        
        # Spread: desviación estándar de los números
        spread = np.std(combo)
        
        # Pair penalty: penalización por co-ocurrencia
        pair_penalty = 0
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                pair_penalty += pair_mat[combo[i] - 1][combo[j] - 1]
        
        # Entropy bonus
        combo_probs = [probs[n - 1] for n in combo]
        entropy_bonus = calc_entropy(combo_probs)
        
        # Score compuesto
        final_score = (
            base_score * weights["base_score"]
            + diversity * weights["diversity"]
            + spread * weights["spread"]
            + entropy_bonus * weights["entropy"]
            - pair_penalty * weights["pair_penalty"]
        )
        
        candidates.append((tuple(combo), final_score))
    
    # Agrupar y promediar scores
    counter = Counter()
    for combo, score in candidates:
        counter[combo] += score
    
    # Retornar top-K
    return counter.most_common(top_k)


def beam_search_combinations(
    probs: np.ndarray,
    pair_mat: np.ndarray,
    beam_width: int = 100,
    numbers_per_draw: int = 5,
    max_number: int = 39
) -> List[Tuple[tuple, float]]:
    """
    Beam search para encontrar mejores combinaciones.
    
    Args:
        probs: Probabilidades de cada número
        pair_mat: Matriz de co-ocurrencia
        beam_width: Ancho del beam
        numbers_per_draw: Números por sorteo
        max_number: Número máximo
    
    Returns:
        Lista de tuplas (combinacion, score)
    """
    numbers = np.arange(1, max_number + 1)
    
    # Inicializar beam con combinaciones vacías
    beam = [((), 0.0)]
    
    for pos in range(numbers_per_draw):
        candidates = []
        
        for combo, score in beam:
            for num in numbers:
                if num not in combo:
                    new_combo = combo + (num,)
                    
                    # Calcular score incremental
                    new_score = score + np.log(probs[num - 1] + 1e-10)
                    
                    # Penalización por co-ocurrencia
                    for existing in combo:
                        new_score -= pair_mat[existing - 1][num - 1] * 0.1
                    
                    candidates.append((new_combo, new_score))
        
        # Mantener top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]
    
    # Ordenar por score y retornar
    beam.sort(key=lambda x: x[1], reverse=True)
    
    # Convertir scores a escala positiva
    result = []
    for combo, score in beam:
        # Normalizar score
        normalized_score = np.exp(score) * 1e6
        result.append((combo, normalized_score))
    
    return result


def greedy_selection(
    probs: np.ndarray,
    pair_mat: np.ndarray,
    numbers_per_draw: int = 5,
    max_number: int = 39
) -> tuple:
    """
    Selección greedy de la mejor combinación.
    
    Args:
        probs: Probabilidades de cada número
        pair_mat: Matriz de co-ocurrencia
        numbers_per_draw: Números por sorteo
        max_number: Número máximo
    
    Returns:
        Tupla con la mejor combinación
    """
    numbers = np.arange(1, max_number + 1)
    selected = []
    
    for _ in range(numbers_per_draw):
        best_num = None
        best_score = -np.inf
        
        for num in numbers:
            if num not in selected:
                # Score: probabilidad - penalización por co-ocurrencia
                score = probs[num - 1]
                
                for existing in selected:
                    score -= pair_mat[existing - 1][num - 1] * 0.1
                
                if score > best_score:
                    best_score = score
                    best_num = num
        
        if best_num is not None:
            selected.append(best_num)
    
    return tuple(sorted(selected))
