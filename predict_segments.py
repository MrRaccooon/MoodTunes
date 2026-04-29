import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
predict_segments.py
-------------------
Splits an audio file into short segments, predicts each one independently,
and uses majority voting (with averaged probabilities as tie-breaker) to
produce a more robust final prediction.

Usage
-----
  python predict_segments.py samples/disco_test.wav
  python predict_segments.py samples/metal_test.wav --segment_sec 5 --top 5
"""

import os
import sys
import argparse
import numpy as np
import librosa

from feature_extractor import SAMPLE_RATE, N_MFCC
from predict import load_model


def _features_from_signal(y: np.ndarray, sr: int) -> np.ndarray:
    """Same feature pipeline as feature_extractor.extract_features,
    but operates on a pre-loaded signal array instead of a file path."""
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


def predict_segments(file_path: str, segment_sec: int = 3, top_n: int = 3) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    clf, scaler, encoder = load_model()
    n_classes = len(encoder.classes_)

    y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    seg_samples = sr * segment_sec
    n_segments = max(1, len(y_full) // seg_samples)

    all_proba = np.zeros((n_segments, n_classes))
    seg_labels = []

    for i in range(n_segments):
        start = i * seg_samples
        segment = y_full[start : start + seg_samples]

        if len(segment) < seg_samples:
            segment = np.pad(segment, (0, seg_samples - len(segment)))

        feat = _features_from_signal(segment, sr).reshape(1, -1)
        feat = scaler.transform(feat)

        proba = clf.predict_proba(feat)[0]
        all_proba[i] = proba
        seg_labels.append(encoder.classes_[np.argmax(proba)])

    # ── majority vote ──
    from collections import Counter
    vote_counts = Counter(seg_labels)
    majority_genre = vote_counts.most_common(1)[0][0]

    # ── averaged probabilities as secondary ranking ──
    avg_proba = all_proba.mean(axis=0)
    top_idx = np.argsort(avg_proba)[::-1][:top_n]

    top_predictions = [
        (encoder.classes_[i], float(avg_proba[i]))
        for i in top_idx
    ]

    return {
        "predicted_genre": majority_genre,
        "confidence": float(avg_proba[list(encoder.classes_).index(majority_genre)]),
        "top_n": top_predictions,
        "n_segments": n_segments,
        "segment_votes": dict(vote_counts),
    }


def print_result(file_path: str, result: dict) -> None:
    bar_width = 30
    print(f"\n{'─'*55}")
    print(f"  File     : {os.path.basename(file_path)}")
    print(f"  Mode     : segment voting ({result['n_segments']} segments)")
    print(f"  Votes    : {result['segment_votes']}")
    print(f"  Genre    : {result['predicted_genre'].upper()}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"\n  Top predictions (averaged probabilities):")
    for genre, prob in result["top_n"]:
        filled = int(prob * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"    {genre:12s} {bar} {prob*100:5.1f}%")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict genre using per-segment majority voting."
    )
    parser.add_argument("file", help="Path to .wav audio file")
    parser.add_argument("--segment_sec", type=int, default=3,
                        help="Segment length in seconds (default: 3)")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top-N genre predictions (default: 3)")
    args = parser.parse_args()

    try:
        result = predict_segments(args.file, segment_sec=args.segment_sec, top_n=args.top)
        print_result(args.file, result)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        sys.exit(1)
