"""
tune.py
-------
Runs RandomizedSearchCV on Random Forest hyperparameters, prints
before/after accuracy, and saves the best model to models/.

Usage
-----
  python tune.py
  python tune.py --n_iter 50
"""

import os
import argparse
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from train import load_data, FEATURES_CSV

MODEL_PATH   = os.path.join("models", "rf_model.pkl")
SCALER_PATH  = os.path.join("models", "scaler.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

PARAM_DIST = {
    "n_estimators":    [100, 200, 300, 400, 500],
    "max_depth":       [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":    ["sqrt", "log2", None],
}


def tune(n_iter: int, random_state: int) -> None:
    os.makedirs("models", exist_ok=True)

    X, y_raw = load_data(FEATURES_CSV)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── baseline (default RF) ──
    baseline = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"\nBaseline RF accuracy : {baseline_acc*100:.2f}%")

    # ── randomized search ──
    print(f"\nRunning RandomizedSearchCV  (n_iter={n_iter}, 5-fold) ...")
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        cv=5,
        scoring="accuracy",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_acc = accuracy_score(y_test, search.best_estimator_.predict(X_test))
    print(f"\nBest CV score        : {search.best_score_*100:.2f}%")
    print(f"Best test accuracy   : {best_acc*100:.2f}%")
    print(f"Improvement          : {(best_acc - baseline_acc)*100:+.2f}%")
    print(f"Best params          : {search.best_params_}")

    # ── save ──
    joblib.dump(search.best_estimator_, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\nTuned model saved -> {MODEL_PATH}")
    print(f"Scaler  saved     -> {SCALER_PATH}")
    print(f"Encoder saved     -> {ENCODER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=30,
                        help="Number of random param combos to try")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    tune(args.n_iter, args.random_state)
