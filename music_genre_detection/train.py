"""
train.py
--------
Loads data/features.csv, trains a Random Forest classifier,
evaluates it, saves the model + scaler, and plots a confusion matrix.

Usage
-----
  python train.py
  python train.py --n_estimators 200 --test_size 0.2
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

FEATURES_CSV  = os.path.join("data",   "features.csv")
MODEL_PATH    = os.path.join("models", "rf_model.pkl")
SCALER_PATH   = os.path.join("models", "scaler.pkl")
ENCODER_PATH  = os.path.join("models", "label_encoder.pkl")
CM_IMAGE_PATH = os.path.join("outputs","confusion_matrix.png")


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X  = df.drop(columns=["label"]).values.astype(np.float32)
    y  = df["label"].values
    return X, y


def train(n_estimators: int, test_size: float, random_state: int) -> None:
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"Loading features from {FEATURES_CSV} ...")
    X, y_raw = load_data(FEATURES_CSV)
    print(f"  Dataset shape : {X.shape}  ({len(set(y_raw))} classes)")

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    print(f"  Classes       : {list(le.classes_)}")

    # ── 3. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 4. Scale features ────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 5. Train model ────────────────────────────────────────────────────────
    print(f"\nTraining Random Forest  (n_estimators={n_estimators}) ...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 6. Cross-validation (on training fold) ────────────────────────────────
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # ── 7. Test set evaluation ────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy      : {acc:.4f}  ({acc*100:.2f}%)\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── 8. Confusion matrix ───────────────────────────────────────────────────
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - Test Accuracy: {acc*100:.1f}%", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CM_IMAGE_PATH, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {CM_IMAGE_PATH}")

    # ── 9. Feature importance (top 10) ───────────────────────────────────────
    importances = clf.feature_importances_
    df_feat = pd.read_csv(FEATURES_CSV)
    feature_names = df_feat.drop(columns=["label"]).columns.tolist()
    top10_idx = np.argsort(importances)[::-1][:10]
    print("\nTop-10 most important features:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"  {rank:2d}. {feature_names[idx]:25s}  {importances[idx]:.4f}")

    # ── 10. Save artefacts ────────────────────────────────────────────────────
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le,     ENCODER_PATH)
    print(f"\nModel   saved -> {MODEL_PATH}")
    print(f"Scaler  saved -> {SCALER_PATH}")
    print(f"Encoder saved -> {ENCODER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int,   default=200)
    parser.add_argument("--test_size",    type=float, default=0.2)
    parser.add_argument("--random_state", type=int,   default=42)
    args = parser.parse_args()

    train(args.n_estimators, args.test_size, args.random_state)