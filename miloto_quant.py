import argparse
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    hamming_loss,
    f1_score,
    jaccard_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.multioutput import MultiOutputClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

MAX_NUMBER = 39
NUMBERS_PER_DRAW = 5

ALPHA = 0.997

MONTE_CARLO_SAMPLES = 15000

MIN_HISTORY = 30

TEMPERATURE = 1.15


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path, limit=10000):

    df = pd.read_csv(path, header=None)

    df = df.dropna(how="all")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df = df[
        (df >= 1).all(axis=1)
        & (df <= MAX_NUMBER).all(axis=1)
    ]

    def has_duplicates(row):
        return len(row) != len(set(row))

    df = df[~df.apply(has_duplicates, axis=1)]

    df = df.apply(sorted, axis=1, result_type="expand")

    # CSV:
    # primera fila = MÁS RECIENTE
    # invertimos para entrenar cronológicamente
    df = df[::-1].reset_index(drop=True)

    return df.head(limit)


# ============================================================
# WEIGHTS
# ============================================================

def exponential_weights(n):
    return np.array([ALPHA ** i for i in range(n)][::-1])


# ============================================================
# BASIC STATS
# ============================================================

def calculate_number_frequencies(df):

    freqs = Counter()

    weights = exponential_weights(len(df))

    for idx, row in enumerate(df.values):

        for num in row:
            freqs[int(num)] += weights[idx]

    total = sum(freqs.values())

    return {
        k: v / total
        for k, v in freqs.items()
    }


def rolling_frequency(df, window=20):

    recent = df.tail(window)

    freq = Counter()

    for row in recent.values:
        for n in row:
            freq[int(n)] += 1

    return {
        n: freq.get(n, 0) / window
        for n in range(1, MAX_NUMBER + 1)
    }


def calculate_gaps(df):

    gaps = {}

    for number in range(1, MAX_NUMBER + 1):

        last_seen = None

        for idx in reversed(range(len(df))):

            if number in df.iloc[idx].values:

                last_seen = len(df) - idx
                break

        gaps[number] = last_seen if last_seen else len(df)

    return gaps


# ============================================================
# FEATURE ENGINEERING (CAUSAL)
# ============================================================

def build_causal_features(df):

    rows = []

    for i in range(MIN_HISTORY, len(df)-1):

        historical = df.iloc[:i]

        current = df.iloc[i]

        global_freq = calculate_number_frequencies(historical)

        rolling_freqs = rolling_frequency(historical)

        gaps = calculate_gaps(historical)

        nums = current.values

        feat = {}

        feat["sum"] = nums.sum()
        feat["mean"] = nums.mean()
        feat["std"] = nums.std()
        feat["range"] = nums.max() - nums.min()

        feat["even_count"] = sum(
            n % 2 == 0 for n in nums
        )

        feat["odd_count"] = 5 - feat["even_count"]

        feat["consecutive_pairs"] = sum(
            nums[i+1] - nums[i] == 1
            for i in range(len(nums)-1)
        )

        for j in range(NUMBERS_PER_DRAW - 1):

            feat[f"diff_{j}"] = nums[j+1] - nums[j]

        for n in range(1, MAX_NUMBER + 1):

            feat[f"global_freq_{n}"] = global_freq.get(n, 0)

            feat[f"rolling_freq_{n}"] = rolling_freqs.get(n, 0)

            feat[f"gap_{n}"] = gaps.get(n, 0)

        rows.append(feat)

    return pd.DataFrame(rows)


# ============================================================
# TARGETS
# ============================================================

def build_targets(df):

    y = []

    for i in range(MIN_HISTORY + 1, len(df)):

        draw = df.iloc[i].values

        target = np.zeros(MAX_NUMBER)

        for n in draw:
            target[int(n)-1] = 1

        y.append(target)

    return np.array(y)


# ============================================================
# BAYESIAN
# ============================================================

def bayesian_probability(df):

    probs = {}

    alpha_prior = 1
    beta_prior = 1

    total_draws = len(df)

    for n in range(1, MAX_NUMBER + 1):

        appearances = sum(
            n in row.values
            for _, row in df.iterrows()
        )

        posterior = (
            appearances + alpha_prior
        ) / (
            total_draws + alpha_prior + beta_prior
        )

        probs[n] = posterior

    return probs


# ============================================================
# CORRELATION PENALTY
# ============================================================

def build_co_occurrence_matrix(df):

    matrix = np.zeros((MAX_NUMBER, MAX_NUMBER))

    for row in df.values:

        for a in row:

            for b in row:

                if a != b:
                    matrix[a-1][b-1] += 1

    matrix /= matrix.max()

    return matrix


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {

        "lightgbm": MultiOutputClassifier(

            LGBMClassifier(
                n_estimators=80,
                learning_rate=0.03,
                max_depth=6,
                random_state=RANDOM_STATE,
                n_jobs=4,
                verbosity=-1
            ),

            n_jobs=4
        ),

        "xgboost": MultiOutputClassifier(

            XGBClassifier(
                n_estimators=80,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=4,
                tree_method="hist"
            ),

            n_jobs=4
        ),

        "randomforest": MultiOutputClassifier(

            RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=4
            )
        )
    }


# ============================================================
# VALIDATION
# ============================================================

def walk_forward_validation(X, y, models):

    print("\n📊 Walk Forward Validation")

    tscv = TimeSeriesSplit(n_splits=3)

    scores = {}

    for name, model in models.items():

        print(f"\n🔍 Evaluando {name}")

        h_scores = []
        f_scores = []
        j_scores = []

        for train_idx, test_idx in tscv.split(X):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            h_scores.append(
                1 - hamming_loss(y_test, y_pred)
            )

            f_scores.append(
                f1_score(
                    y_test,
                    y_pred,
                    average="micro"
                )
            )

            j_scores.append(
                jaccard_score(
                    y_test,
                    y_pred,
                    average="samples"
                )
            )

        scores[name] = {

            "hamming": np.mean(h_scores),

            "f1": np.mean(f_scores),

            "jaccard": np.mean(j_scores)
        }

        print(scores[name])

    return scores


# ============================================================
# TRAIN
# ============================================================

def train_models(X, y):

    models = build_models()

    print("\n🚀 Entrenando modelos")

    for name, model in models.items():

        print(f"✅ Entrenando {name}")

        model.fit(X, y)

    return models


# ============================================================
# ENSEMBLE
# ============================================================

def ensemble_probabilities(models, X_latest):

    weights = {

        "lightgbm": 0.40,
        "xgboost": 0.35,
        "randomforest": 0.25
    }

    final_probs = np.zeros(MAX_NUMBER)

    per_model_probs = {}

    for name, model in models.items():

        probs = []

        for estimator in model.estimators_:

            proba = estimator.predict_proba(X_latest)[0]

            if len(proba) > 1:
                p = proba[1]
            else:
                p = 0.0

            probs.append(p)

        probs = np.array(probs)

        per_model_probs[name] = probs

        final_probs += probs * weights[name]

    return final_probs, per_model_probs


# ============================================================
# STABILITY
# ============================================================

def stabilize_probabilities(
    current_probs,
    historical_probs
):

    final_probs = (
        0.65 * current_probs +
        0.20 * historical_probs +
        0.15 * np.mean(current_probs)
    )

    final_probs = np.clip(
        final_probs,
        0.015,
        0.12
    )

    final_probs = final_probs / final_probs.sum()

    return final_probs


# ============================================================
# MONTE CARLO
# ============================================================

def monte_carlo_sampling(
    probabilities,
    co_matrix
):

    probabilities = softmax(
        probabilities / TEMPERATURE
    )

    combos = []

    numbers = np.arange(1, MAX_NUMBER + 1)

    for _ in range(MONTE_CARLO_SAMPLES):

        sample = np.random.choice(
            numbers,
            size=NUMBERS_PER_DRAW,
            replace=False,
            p=probabilities
        )

        sample = sorted(sample)

        penalty = 0

        for i in range(len(sample)):
            for j in range(i+1, len(sample)):

                penalty += co_matrix[
                    sample[i]-1
                ][
                    sample[j]-1
                ]

        score = 1 / (1 + penalty)

        combos.append(
            (
                tuple(sample),
                score
            )
        )

    counter = Counter()

    for combo, score in combos:

        counter[combo] += score

    return counter.most_common(10)


# ============================================================
# BACKTEST
# ============================================================

def backtest(df, X, y):

    print("\n🧪 Ejecutando backtest")

    hits = []

    start = int(len(X) * 0.7)

    for i in range(
        start,
        min(start + 50, len(X)-1)
    ):

        X_train = X.iloc[:i]
        y_train = y[:i]

        X_test = X.iloc[i:i+1]

        models = train_models(
            X_train,
            y_train
        )

        probs, _ = ensemble_probabilities(
            models,
            X_test
        )

        pred = np.argsort(probs)[-5:] + 1

        actual = set(
            df.iloc[
                i + MIN_HISTORY + 1
            ].values
        )

        hit_count = len(
            set(pred).intersection(actual)
        )

        hits.append(hit_count)

    print(
        f"\n📈 Promedio aciertos: {np.mean(hits):.4f}"
    )

    print(
        f"🏆 Máximo aciertos: {np.max(hits)}"
    )

    return hits


# ============================================================
# REPORT
# ============================================================

def generate_report(
    probabilities,
    combinations,
    metrics,
    model_probs,
    hits
):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        "reporte_quant_miloto.pdf",
        pagesize=A4
    )

    elements = []

    elements.append(
        Paragraph(
            "MiLoto Predictor Quant",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    # =======================================================
    # PROBABILITIES
    # =======================================================

    prob_table = [
        ["Número", "Probabilidad"]
    ]

    sorted_probs = sorted(
        enumerate(probabilities, start=1),
        key=lambda x: x[1],
        reverse=True
    )

    for num, prob in sorted_probs[:15]:

        prob_table.append([
            str(num),
            f"{prob:.4f}"
        ])

    table = Table(prob_table)

    table.setStyle(TableStyle([

        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),

        ('TEXTCOLOR', (0,0), (-1,0), colors.white),

        ('GRID', (0,0), (-1,-1), 1, colors.black),

        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')

    ]))

    elements.append(table)

    elements.append(Spacer(1, 20))

    # =======================================================
    # ENSEMBLE TABLE
    # =======================================================

    ensemble_table = [[
        "Número",
        "LGBM",
        "XGB",
        "RF",
        "Final"
    ]]

    for i in range(MAX_NUMBER):

        ensemble_table.append([

            str(i+1),

            f"{model_probs['lightgbm'][i]:.4f}",

            f"{model_probs['xgboost'][i]:.4f}",

            f"{model_probs['randomforest'][i]:.4f}",

            f"{probabilities[i]:.4f}"
        ])

    ensemble_tbl = Table(ensemble_table)

    ensemble_tbl.setStyle(TableStyle([

        ('BACKGROUND', (0,0), (-1,0), colors.green),

        ('TEXTCOLOR', (0,0), (-1,0), colors.white),

        ('GRID', (0,0), (-1,-1), 1, colors.black),

        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')

    ]))

    elements.append(ensemble_tbl)

    elements.append(Spacer(1, 20))

    # =======================================================
    # COMBINATIONS
    # =======================================================

    combo_table = [
        ["Combinación", "Score"]
    ]

    for combo, score in combinations:

        clean_combo = tuple(
            int(x) for x in combo
        )

        combo_table.append([
            str(clean_combo),
            f"{score:.2f}"
        ])

    combo_tbl = Table(combo_table)

    combo_tbl.setStyle(TableStyle([

        ('BACKGROUND', (0,0), (-1,0), colors.red),

        ('TEXTCOLOR', (0,0), (-1,0), colors.white),

        ('GRID', (0,0), (-1,-1), 1, colors.black),

        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')

    ]))

    elements.append(combo_tbl)

    elements.append(Spacer(1, 20))

    # =======================================================
    # PLOT
    # =======================================================

    plt.figure(figsize=(12,6))

    nums = [x[0] for x in sorted_probs[:15]]

    probs = [x[1] for x in sorted_probs[:15]]

    plt.bar(nums, probs)

    plt.title("Top 15 Probabilidades")

    plt.xlabel("Número")

    plt.ylabel("Probabilidad")

    plt.tight_layout()

    plt.savefig("probabilidades.png")

    plt.close()

    elements.append(
        Image(
            "probabilidades.png",
            width=450,
            height=250
        )
    )

    # =======================================================
    # BACKTEST HIST
    # =======================================================

    plt.figure(figsize=(10,5))

    plt.hist(hits, bins=6)

    plt.title("Distribución Hits Backtest")

    plt.xlabel("Aciertos")

    plt.ylabel("Frecuencia")

    plt.tight_layout()

    plt.savefig("backtest_hits.png")

    plt.close()

    elements.append(
        Image(
            "backtest_hits.png",
            width=450,
            height=250
        )
    )

    doc.build(elements)

    print(
        "\n📄 Reporte generado: reporte_quant_miloto.pdf"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        required=True,
        help="CSV histórico MiLoto"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10000
    )

    args = parser.parse_args()

    print("\n📥 Cargando datos...")

    df = load_data(
        args.file,
        args.limit
    )

    print(
        f"✅ Sorteos cargados: {len(df)}"
    )

    print("\n⚙️ Construyendo features causales...")

    X = build_causal_features(df)

    y = build_targets(df)

    print(X.shape)
    print(y.shape)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    metrics = walk_forward_validation(
        X_scaled,
        y,
        build_models()
    )

    models = train_models(
        X_scaled,
        y
    )

    latest_features = X_scaled.iloc[-1:]

    ensemble_probs, model_probs = (
        ensemble_probabilities(
            models,
            latest_features
        )
    )

    bayes_probs = bayesian_probability(df)

    historical_probs = np.array([
        bayes_probs[i]
        for i in range(1, MAX_NUMBER + 1)
    ])

    final_probs = stabilize_probabilities(
        ensemble_probs,
        historical_probs
    )

    co_matrix = build_co_occurrence_matrix(df)

    combinations = monte_carlo_sampling(
        final_probs,
        co_matrix
    )

    print("\n🏆 COMBINACIÓN SUGERIDA")

    best_combo = tuple(
        int(x)
        for x in combinations[0][0]
    )

    print(best_combo)

    hits = backtest(
        df,
        X_scaled,
        y
    )

    generate_report(
        final_probs,
        combinations,
        metrics,
        model_probs,
        hits
    )

    print("\n✅ PROCESO FINALIZADO")


if __name__ == "__main__":
    main()