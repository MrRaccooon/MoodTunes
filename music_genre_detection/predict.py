"""
predict.py
----------
Loads a trained model and predicts the genre of a new audio file.

Usage
-----
  python predict.py samples/test_rock.wav
  python predict.py samples/test_jazz.wav --top 3
"""

import os
import sys
import argparse
import numpy as np
import joblib

from feature_extractor import extract_features

MODEL_PATH   = os.path.join("models", "rf_model.pkl")
SCALER_PATH  = os.path.join("models", "scaler.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run train.py first."
        )
    clf     = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return clf, scaler, encoder


def predict(file_path: str, top_n: int = 3) -> dict:
    """
    Predict the genre of an audio file.

    Returns
    -------
    dict with keys:
        predicted_genre : str
        confidence      : float  (0–1)
        top_n           : list of (genre, probability) tuples
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    clf, scaler, encoder = load_model()

    # Extract + scale
    features = extract_features(file_path).reshape(1, -1)
    features = scaler.transform(features)

    # Predict
    proba  = clf.predict_proba(features)[0]           # shape: (n_classes,)
    top_idx = np.argsort(proba)[::-1][:top_n]

    predicted_label = encoder.classes_[top_idx[0]]
    confidence      = proba[top_idx[0]]

    top_predictions = [
        (encoder.classes_[i], float(proba[i]))
        for i in top_idx
    ]

    return {
        "predicted_genre" : predicted_label,
        "confidence"      : float(confidence),
        "top_n"           : top_predictions,
    }


def print_result(file_path: str, result: dict) -> None:
    bar_width = 30
    print(f"\n{'─'*50}")
    print(f"  File   : {os.path.basename(file_path)}")
    print(f"  Genre  : {result['predicted_genre'].upper()}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"\n  Top predictions:")
    for genre, prob in result["top_n"]:
        filled = int(prob * bar_width)
        bar    = "█" * filled + "░" * (bar_width - filled)
        print(f"    {genre:12s} {bar} {prob*100:5.1f}%")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict music genre from an audio file."
    )
    parser.add_argument("file", help="Path to .wav audio file")
    parser.add_argument(
        "--top", type=int, default=3,
        help="Show top-N genre predictions (default: 3)"
    )
    args = parser.parse_args()

    try:
        result = predict(args.file, top_n=args.top)
        print_result(args.file, result)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        sys.exit(1)