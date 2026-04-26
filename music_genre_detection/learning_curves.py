"""
learning_curves.py
------------------
Plots training-set-size vs accuracy (train & CV) to visualise
overfitting / underfitting behaviour of Random Forest.

Usage
-----
  python learning_curves.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler

from train import load_data, FEATURES_CSV

OUTPUT_IMAGE = os.path.join("outputs", "learning_curve.png")


def plot_learning_curve() -> None:
    os.makedirs("outputs", exist_ok=True)

    X, y_raw = load_data(FEATURES_CSV)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    train_sizes, train_scores, cv_scores = learning_curve(
        clf, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    cv_mean = cv_scores.mean(axis=1)
    cv_std = cv_scores.std(axis=1)

    print("\nTraining size | Train Acc | CV Acc")
    print("-" * 42)
    for size, tm, cm in zip(train_sizes, train_mean, cv_mean):
        print(f"  {size:>5d}       | {tm*100:6.2f}%   | {cm*100:6.2f}%")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15)
    ax.plot(train_sizes, train_mean, "o-", label="Training score")
    ax.plot(train_sizes, cv_mean, "o-", label="Cross-validation score")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve — Random Forest (200 trees)")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close()

    print(f"\nPlot saved -> {OUTPUT_IMAGE}")


if __name__ == "__main__":
    plot_learning_curve()
