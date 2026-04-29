import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
predict_ensemble.py
-------------------
Trains an ensemble of SVM + Random Forest + MLP on features.csv, saves the
ensemble, and predicts genre via soft voting (averaged probabilities).

First run trains and saves the ensemble (~30s). Subsequent runs load from disk.

Usage
-----
  python predict_ensemble.py train
  python predict_ensemble.py predict samples/disco_test.wav
  python predict_ensemble.py predict samples/metal_test.wav --top 5
"""

import os
import argparse
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

from train import load_data, FEATURES_CSV
from feature_extractor import extract_features

ENSEMBLE_PATH = os.path.join("models", "ensemble_model.pkl")
SCALER_PATH   = os.path.join("models", "ensemble_scaler.pkl")
ENCODER_PATH  = os.path.join("models", "ensemble_encoder.pkl")


def train_ensemble(random_state: int = 42) -> None:
    os.makedirs("models", exist_ok=True)

    X, y_raw = load_data(FEATURES_CSV)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ensemble = VotingClassifier(
        estimators=[
            ("svm", SVC(C=10, kernel="rbf", probability=True, random_state=random_state)),
            ("rf",  RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=random_state)),
        ],
        voting="soft",
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    print("Running 10-fold CV on ensemble ...")
    scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  Ensemble 10-fold CV : {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%")

    print("\nTraining final ensemble on full dataset ...")
    ensemble.fit(X_scaled, y)

    joblib.dump(ensemble, ENSEMBLE_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"  Ensemble saved -> {ENSEMBLE_PATH}")


def predict_file(file_path: str, top_n: int = 3) -> dict:
    if not os.path.exists(ENSEMBLE_PATH):
        raise FileNotFoundError(
            f"Ensemble not found at '{ENSEMBLE_PATH}'. Run: python predict_ensemble.py train"
        )

    ensemble = joblib.load(ENSEMBLE_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    features = extract_features(file_path).reshape(1, -1)
    features = scaler.transform(features)

    proba = ensemble.predict_proba(features)[0]
    top_idx = np.argsort(proba)[::-1][:top_n]

    predicted = encoder.classes_[top_idx[0]]
    confidence = proba[top_idx[0]]

    top_predictions = [(encoder.classes_[i], float(proba[i])) for i in top_idx]

    return {
        "predicted_genre": predicted,
        "confidence": float(confidence),
        "top_n": top_predictions,
    }


def print_result(file_path: str, result: dict) -> None:
    bar_width = 30
    print(f"\n{'='*55}")
    print(f"  File       : {os.path.basename(file_path)}")
    print(f"  Method     : Ensemble (SVM + RF + MLP)")
    print(f"  Genre      : {result['predicted_genre'].upper()}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"\n  Top predictions:")
    for genre, prob in result["top_n"]:
        filled = int(prob * bar_width)
        bar = "#" * filled + "." * (bar_width - filled)
        print(f"    {genre:12s} [{bar}] {prob*100:5.1f}%")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble genre predictor (SVM + RF + MLP soft voting)."
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Train and save the ensemble model")

    pred = sub.add_parser("predict", help="Predict genre of an audio file")
    pred.add_argument("file", help="Path to .wav audio file")
    pred.add_argument("--top", type=int, default=3,
                      help="Show top-N predictions (default: 3)")

    args = parser.parse_args()

    if args.command == "train":
        train_ensemble()
    elif args.command == "predict":
        try:
            result = predict_file(args.file, top_n=args.top)
            print_result(args.file, result)
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Prediction failed: {e}")
            sys.exit(1)
    else:
        parser.print_help()
