"""
compare_models.py
-----------------
Trains 5 classifiers on the same features.csv, compares them via
5-fold cross-validation, prints a results table, and saves a bar chart.

Usage
-----
  python compare_models.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from train import load_data, FEATURES_CSV

OUTPUT_IMAGE = os.path.join("outputs", "model_comparison.png")


def compare() -> None:
    os.makedirs("outputs", exist_ok=True)

    X, y_raw = load_data(FEATURES_CSV)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)":          SVC(kernel="rbf", probability=True, random_state=42),
        "KNN (k=7)":          KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    }

    names, cv_means, cv_stds, test_accs = [], [], [], []

    print(f"\n{'Model':^25s} | {'5-Fold CV':^18s} | {'Test Acc':^10s}")
    print("-" * 60)

    for name, clf in models.items():
        cv = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        names.append(name)
        cv_means.append(cv.mean())
        cv_stds.append(cv.std())
        test_accs.append(test_acc)

        print(f"  {name:23s} | {cv.mean():.4f} +/- {cv.std():.4f} | {test_acc*100:6.2f}%")

    print("-" * 60)

    # ── bar chart ──
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, cv_means, width, yerr=cv_stds, label="5-Fold CV", capsize=4)
    ax.bar(x + width / 2, test_accs, width, label="Test Set", capsize=4)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison — Genre Classification")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close()
    print(f"\nChart saved -> {OUTPUT_IMAGE}")


if __name__ == "__main__":
    compare()
