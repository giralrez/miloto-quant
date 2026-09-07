import argparse
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from itertools import combinations

from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from scipy.special import softmax
from scipy.stats import entropy

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


# ==========================================================
# CONFIG
# ==========================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

MAX_NUMBER = 39
DRAW_SIZE = 5

MIN_HISTORY = 60

ALPHA = 0.996

MONTE_CARLO_SAMPLES = 30000

BACKTEST_STEPS = 20

TEMPERATURE = 0.78


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data(path):

    df = pd.read_csv(path, header=None)

    df = df.dropna()

    for c in df.columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna()

    df = df.astype(int)

    df = df[
        (df >= 1).all(axis=1)
        &
        (df <= MAX_NUMBER).all(axis=1)
    ]

    def duplicated(row):
        return len(set(row)) != len(row)

    df = df[
        ~df.apply(duplicated, axis=1)
    ]

    df = df.apply(
        sorted,
        axis=1,
        result_type="expand"
    )

    # PRIMERA FILA = MÁS RECIENTE
    # invertir cronología
    df = df[::-1].reset_index(drop=True)

    return df


# ==========================================================
# WEIGHTS
# ==========================================================

def exp_weights(n):

    return np.array([
        ALPHA ** i
        for i in range(n)
    ][::-1])


# ==========================================================
# FREQUENCIES
# ==========================================================

def weighted_frequency(df):

    weights = exp_weights(len(df))

    freq = Counter()

    for idx, row in enumerate(df.values):

        for n in row:
            freq[n] += weights[idx]

    total = sum(freq.values())

    return {

        k: v / total
        for k, v in freq.items()
    }


def rolling_frequency(df, window):

    recent = df.tail(window)

    freq = Counter()

    for row in recent.values:

        for n in row:
            freq[n] += 1

    return {

        n: freq.get(n, 0) / max(window, 1)
        for n in range(1, MAX_NUMBER + 1)
    }


# ==========================================================
# GAPS
# ==========================================================

def calculate_gaps(df):

    gaps = {}

    for n in range(1, MAX_NUMBER + 1):

        last_seen = None

        for idx in reversed(range(len(df))):

            if n in df.iloc[idx].values:

                last_seen = len(df) - idx
                break

        gaps[n] = (
            last_seen
            if last_seen
            else len(df)
        )

    return gaps


# ==========================================================
# PAIR MATRIX
# ==========================================================

def pair_matrix(df):

    matrix = np.zeros(
        (MAX_NUMBER, MAX_NUMBER)
    )

    for row in df.values:

        for a, b in combinations(row, 2):

            matrix[a-1][b-1] += 1
            matrix[b-1][a-1] += 1

    if matrix.max() > 0:
        matrix /= matrix.max()

    return matrix


# ==========================================================
# EMA
# ==========================================================

def exponential_moving_average(
    series,
    alpha=0.3
):

    ema = []

    current = series[0]

    for val in series:

        current = (
            alpha * val
            +
            (1-alpha) * current
        )

        ema.append(current)

    return ema[-1]


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def build_dataset(df):

    X = []
    y = []

    for i in range(
        MIN_HISTORY,
        len(df)-1
    ):

        historical = df.iloc[:i]

        next_draw = set(
            df.iloc[i+1].values
        )

        global_freq = weighted_frequency(
            historical
        )

        freq5 = rolling_frequency(
            historical,
            5
        )

        freq10 = rolling_frequency(
            historical,
            10
        )

        freq20 = rolling_frequency(
            historical,
            20
        )

        freq50 = rolling_frequency(
            historical,
            50
        )

        gaps = calculate_gaps(
            historical
        )

        pair_mat = pair_matrix(
            historical.tail(80)
        )

        for number in range(1, MAX_NUMBER + 1):

            momentum = (
                freq10[number]
                -
                freq50[number]
            )

            acceleration = (
                freq5[number]
                -
                freq20[number]
            )

            pair_strength = np.mean(
                pair_mat[number-1]
            )

            volatility = np.std([
                freq5[number],
                freq10[number],
                freq20[number],
                freq50[number]
            ])

            ema_signal = exponential_moving_average([
                freq5[number],
                freq10[number],
                freq20[number],
                freq50[number]
            ])

            feature_vector = [

                global_freq.get(number, 0),

                freq5[number],
                freq10[number],
                freq20[number],
                freq50[number],

                momentum,
                acceleration,

                gaps[number],

                pair_strength,

                volatility,

                ema_signal,

                entropy([
                    freq5[number] + 1e-9,
                    freq10[number] + 1e-9,
                    freq20[number] + 1e-9,
                    freq50[number] + 1e-9
                ]),

                number / MAX_NUMBER,

                number % 2,

                int(number <= 13),
                int(14 <= number <= 26),
                int(number >= 27)

            ]

            X.append(feature_vector)

            y.append(
                1 if number in next_draw else 0
            )

    return np.array(X), np.array(y)


# ==========================================================
# MODELS
# ==========================================================

def build_models():

    return {

        "lightgbm":
        LGBMRegressor(
            n_estimators=200,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbosity=-1
        ),

        "xgboost":
        XGBRegressor(
            n_estimators=180,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist"
        ),

        "randomforest":
        RandomForestRegressor(
            n_estimators=180,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=4
        ),

        "extratrees":
        ExtraTreesRegressor(
            n_estimators=180,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=4
        )
    }


# ==========================================================
# WALK FORWARD VALIDATION
# ==========================================================

def evaluate_models(X, y):

    print("\n📊 WALK FORWARD VALIDATION")

    tscv = TimeSeriesSplit(
        n_splits=3
    )

    scores = {}

    for name, model in build_models().items():

        maes = []

        print(f"\n🔍 Evaluando {name}")

        for train_idx, test_idx in tscv.split(X):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(
                X_test
            )

            maes.append(
                mean_absolute_error(
                    y_test,
                    pred
                )
            )

        scores[name] = 1 / (
            np.mean(maes) + 1e-6
        )

        print(
            f"score={scores[name]:.6f}"
        )

    return scores


# ==========================================================
# TRAIN
# ==========================================================

def train_models(X, y):

    models = build_models()

    print("\n🚀 ENTRENAMIENTO FINAL")

    for name, model in models.items():

        print(f"✅ Entrenando {name}")

        model.fit(X, y)

    return models


# ==========================================================
# DYNAMIC WEIGHTS
# ==========================================================

def normalize_weights(scores):

    total = sum(scores.values())

    return {

        k: v / total
        for k, v in scores.items()
    }


# ==========================================================
# PREDICT NUMBER SCORES
# ==========================================================

def predict_scores(
    models,
    weights,
    latest_features
):

    final_scores = np.zeros(MAX_NUMBER)

    per_model = {}

    for idx, number in enumerate(
        range(1, MAX_NUMBER + 1)
    ):

        feats = latest_features[idx].reshape(1,-1)

        score = 0

        model_scores = {}

        for name, model in models.items():

            pred = model.predict(feats)[0]

            pred = max(pred, 0)

            model_scores[name] = pred

            score += (
                pred * weights[name]
            )

        per_model[number] = model_scores

        final_scores[idx] = score

    return final_scores, per_model


# ==========================================================
# BAYESIAN PRIOR
# ==========================================================

def bayesian_prior(df):

    prior = {}

    total = len(df)

    for n in range(1, MAX_NUMBER + 1):

        count = sum(
            n in row.values
            for _, row in df.iterrows()
        )

        prior[n] = (
            count + 1
        ) / (
            total + 2
        )

    return np.array([
        prior[n]
        for n in range(1, MAX_NUMBER + 1)
    ])


# ==========================================================
# FINAL PROBABILITIES
# ==========================================================

def final_probability_engine(
    ml_scores,
    bayes
):

    probs = (

        0.82 * ml_scores

        +

        0.18 * bayes

    )

    probs = softmax(
        probs / TEMPERATURE
    )

    probs = np.clip(
        probs,
        0.001,
        0.35
    )

    probs /= probs.sum()

    return probs


# ==========================================================
# COMBINATORIAL OPTIMIZER
# ==========================================================

def optimize_combinations(
    probs,
    pair_mat
):

    nums = np.arange(
        1,
        MAX_NUMBER + 1
    )

    candidates = []

    for _ in range(
        MONTE_CARLO_SAMPLES
    ):

        combo = np.random.choice(
            nums,
            size=5,
            replace=False,
            p=probs
        )

        combo = sorted(combo)

        base_score = np.prod([
            probs[n-1]
            for n in combo
        ])

        diversity = len(set([
            n // 10
            for n in combo
        ]))

        spread = np.std(combo)

        pair_penalty = 0

        for a, b in combinations(combo, 2):

            pair_penalty += (
                pair_mat[a-1][b-1]
            )

        entropy_bonus = entropy([
            probs[n-1]
            for n in combo
        ])

        final_score = (

            base_score * 10000000

            +

            diversity * 18

            +

            spread * 1.2

            +

            entropy_bonus * 25

            -

            pair_penalty * 22
        )

        candidates.append(
            (
                tuple(combo),
                final_score
            )
        )

    counter = Counter()

    for combo, score in candidates:

        counter[combo] += score

    return counter.most_common(10)


# ==========================================================
# BUILD LATEST FEATURES
# ==========================================================

def latest_feature_vectors(df):

    historical = df.copy()

    global_freq = weighted_frequency(
        historical
    )

    freq5 = rolling_frequency(
        historical,
        5
    )

    freq10 = rolling_frequency(
        historical,
        10
    )

    freq20 = rolling_frequency(
        historical,
        20
    )

    freq50 = rolling_frequency(
        historical,
        50
    )

    gaps = calculate_gaps(
        historical
    )

    pair_mat = pair_matrix(
        historical.tail(80)
    )

    rows = []

    for number in range(1, MAX_NUMBER + 1):

        momentum = (
            freq10[number]
            -
            freq50[number]
        )

        acceleration = (
            freq5[number]
            -
            freq20[number]
        )

        pair_strength = np.mean(
            pair_mat[number-1]
        )

        volatility = np.std([
            freq5[number],
            freq10[number],
            freq20[number],
            freq50[number]
        ])

        ema_signal = exponential_moving_average([
            freq5[number],
            freq10[number],
            freq20[number],
            freq50[number]
        ])

        rows.append([

            global_freq.get(number, 0),

            freq5[number],
            freq10[number],
            freq20[number],
            freq50[number],

            momentum,
            acceleration,

            gaps[number],

            pair_strength,

            volatility,

            ema_signal,

            entropy([
                freq5[number] + 1e-9,
                freq10[number] + 1e-9,
                freq20[number] + 1e-9,
                freq50[number] + 1e-9
            ]),

            number / MAX_NUMBER,

            number % 2,

            int(number <= 13),
            int(14 <= number <= 26),
            int(number >= 27)

        ])

    return np.array(rows)


# ==========================================================
# BACKTEST
# ==========================================================

def backtest(df):

    print("\n🧪 BACKTEST")

    hits = []

    start = int(len(df) * 0.80)

    total = min(
        start + BACKTEST_STEPS,
        len(df)-2
    ) - start

    for idx, i in enumerate(

        range(
            start,
            min(
                start + BACKTEST_STEPS,
                len(df)-2
            )
        )
    ):

        print(
            f"\n📈 Iteración {idx+1}/{total}"
        )

        partial_df = df.iloc[:i]

        X_train, y_train = build_dataset(
            partial_df
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        scores = evaluate_models(
            X_train,
            y_train
        )

        weights = normalize_weights(
            scores
        )

        models = train_models(
            X_train,
            y_train
        )

        latest_feats = latest_feature_vectors(
            partial_df
        )

        latest_feats = scaler.transform(
            latest_feats
        )

        ml_scores, _ = predict_scores(
            models,
            weights,
            latest_feats
        )

        bayes = bayesian_prior(
            partial_df
        )

        probs = final_probability_engine(
            ml_scores,
            bayes
        )

        pred = np.argsort(
            probs
        )[-5:] + 1

        actual = set(
            df.iloc[i+1].values
        )

        hit = len(
            set(pred).intersection(actual)
        )

        hits.append(hit)

    return hits


# ==========================================================
# REPORT
# ==========================================================

def generate_report(
    probs,
    combinations,
    weights,
    hits
):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        "reporte_quant_miloto_v3.pdf",
        pagesize=A4
    )

    elements = []

    elements.append(
        Paragraph(
            "MiLoto Quant Ranking Engine",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1,20)
    )

    # ======================================================
    # WEIGHTS
    # ======================================================

    weight_table = [
        ["Modelo", "Peso"]
    ]

    for k,v in weights.items():

        weight_table.append([
            k,
            f"{v:.4f}"
        ])

    tbl = Table(weight_table)

    tbl.setStyle(TableStyle([

        ('BACKGROUND',(0,0),(-1,0),colors.darkblue),

        ('TEXTCOLOR',(0,0),(-1,0),colors.white),

        ('GRID',(0,0),(-1,-1),1,colors.black)

    ]))

    elements.append(tbl)

    elements.append(
        Spacer(1,20)
    )

    # ======================================================
    # TOP PROBS
    # ======================================================

    top = sorted(
        enumerate(probs, start=1),
        key=lambda x: x[1],
        reverse=True
    )

    prob_table = [
        ["Número", "Probabilidad"]
    ]

    for n,p in top[:15]:

        prob_table.append([
            str(n),
            f"{p:.6f}"
        ])

    tbl2 = Table(prob_table)

    tbl2.setStyle(TableStyle([

        ('BACKGROUND',(0,0),(-1,0),colors.green),

        ('TEXTCOLOR',(0,0),(-1,0),colors.white),

        ('GRID',(0,0),(-1,-1),1,colors.black)

    ]))

    elements.append(tbl2)

    elements.append(
        Spacer(1,20)
    )

    # ======================================================
    # COMBINATIONS
    # ======================================================

    combo_table = [
        ["Combinación", "Score"]
    ]

    for combo,score in combinations:

        combo_table.append([
            str(combo),
            f"{score:.2f}"
        ])

    tbl3 = Table(combo_table)

    tbl3.setStyle(TableStyle([

        ('BACKGROUND',(0,0),(-1,0),colors.red),

        ('TEXTCOLOR',(0,0),(-1,0),colors.white),

        ('GRID',(0,0),(-1,-1),1,colors.black)

    ]))

    elements.append(tbl3)

    elements.append(
        Spacer(1,20)
    )

    # ======================================================
    # PROB PLOT
    # ======================================================

    plt.figure(figsize=(10,5))

    nums = [x[0] for x in top[:15]]

    vals = [x[1] for x in top[:15]]

    plt.bar(nums, vals)

    plt.title(
        "Top Probabilidades"
    )

    plt.tight_layout()

    plt.savefig(
        "top_probs_v3.png"
    )

    plt.close()

    elements.append(
        Image(
            "top_probs_v3.png",
            width=450,
            height=250
        )
    )

    # ======================================================
    # BACKTEST
    # ======================================================

    plt.figure(figsize=(10,5))

    plt.hist(hits, bins=6)

    plt.title(
        "Distribución Hits Backtest"
    )

    plt.tight_layout()

    plt.savefig(
        "backtest_v3.png"
    )

    plt.close()

    elements.append(
        Image(
            "backtest_v3.png",
            width=450,
            height=250
        )
    )

    doc.build(elements)

    print(
        "\n📄 reporte_quant_miloto_v3.pdf generado"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        required=True,
        help="historico.csv"
    )

    args = parser.parse_args()

    print("\n📥 Cargando histórico...")

    df = load_data(
        args.file
    )

    print(
        f"✅ Sorteos: {len(df)}"
    )

    print(
        "\n⚙️ Construyendo dataset ranking..."
    )

    X, y = build_dataset(df)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print(
        f"X={X_scaled.shape}"
    )

    print(
        f"y={y.shape}"
    )

    scores = evaluate_models(
        X_scaled,
        y
    )

    weights = normalize_weights(
        scores
    )

    print("\n🧠 PESOS DINÁMICOS")

    print(weights)

    models = train_models(
        X_scaled,
        y
    )

    latest = latest_feature_vectors(
        df
    )

    latest = scaler.transform(
        latest
    )

    ml_scores, per_model = predict_scores(
        models,
        weights,
        latest
    )

    bayes = bayesian_prior(df)

    final_probs = final_probability_engine(
        ml_scores,
        bayes
    )

    pair_mat = pair_matrix(df)

    combinations = optimize_combinations(
        final_probs,
        pair_mat
    )

    print("\n🏆 TOP COMBINACIONES")

    for idx, (combo, score) in enumerate(
        combinations,
        start=1
    ):

        print(
            f"{idx}. {combo} -> {score:.2f}"
        )

    print(
        "\n🎯 COMBINACIÓN SUGERIDA"
    )

    print(combinations[0][0])

    hits = backtest(df)

    print("\n📈 RESULTADOS BACKTEST")

    print(
        f"Promedio hits: {np.mean(hits):.4f}"
    )

    print(
        f"Máximo hits: {np.max(hits)}"
    )

    generate_report(
        final_probs,
        combinations,
        weights,
        hits
    )

    print("\n✅ PROCESO FINALIZADO")


if __name__ == "__main__":
    main()
