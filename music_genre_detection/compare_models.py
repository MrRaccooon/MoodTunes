"""
compare_models.py
-----------------
Compares 7 classifiers on the GTZAN feature set using proper 10-fold
stratified cross-validation (scaler inside the CV loop via Pipeline).

Usage
-----
  python compare_models.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from train import load_data, FEATURES_CSV

OUTPUT_IMAGE = os.path.join("outputs", "model_comparison.png")


def compare() -> None:
    os.makedirs("outputs", exist_ok=True)

    X, y = load_data(FEATURES_CSV)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "SVM (RBF, C=10)":     make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", probability=True, random_state=42)),
        "MLP (256,128)":       make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)),
        "Random Forest":       make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1, random_state=42)),
        "Gradient Boosting":   make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        "KNN (k=7)":           make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    }

    results = []

    print(f"\n{'Model':^25s} | {'10-Fold CV':^20s}")
    print("-" * 50)

    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        results.append((name, scores.mean(), scores.std()))
        print(f"  {name:23s} | {scores.mean()*100:5.1f}% +/- {scores.std()*100:.1f}%")

    results.sort(key=lambda r: r[1], reverse=True)
    print("-" * 50)
    print(f"\n  Best: {results[0][0]}  ({results[0][1]*100:.1f}%)")

    # ── bar chart ──
    names  = [r[0] for r in results]
    means  = [r[1] for r in results]
    stds   = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison - 10-Fold Stratified CV")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0.4, 0.9)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m*100:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close()
    print(f"\nChart saved -> {OUTPUT_IMAGE}")


if __name__ == "__main__":
    compare()
