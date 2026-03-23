"""
feature_extractor.py
--------------------
Extracts audio features from a .wav file using librosa.
Returns a flat numpy array ready for model input.

Features extracted:
  - MFCC (20 coefficients, mean + std)  → 40 values
  - Chroma STFT (12, mean + std)        → 24 values
  - Spectral Centroid (mean + std)      →  2 values
  - Spectral Rolloff  (mean + std)      →  2 values
  - Zero Crossing Rate (mean + std)     →  2 values
  - RMS Energy (mean + std)             →  2 values
                                Total   → 72 values
"""

import numpy as np
import librosa


SAMPLE_RATE = 22050
DURATION    = 30       # seconds — GTZAN clips are exactly 30s
N_MFCC      = 20


def extract_features(file_path: str) -> np.ndarray:
    """
    Load an audio file and extract features.

    Parameters
    ----------
    file_path : str
        Path to a .wav (or any librosa-supported) audio file.

    Returns
    -------
    np.ndarray
        1-D feature vector of shape (72,).

    Raises
    ------
    Exception
        Propagates any librosa loading errors to the caller.
    """
    # Load audio — mono, fixed sample rate, trim/pad to DURATION seconds
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    # Pad with silence if the clip is shorter than DURATION
    target_length = SAMPLE_RATE * DURATION
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    features = []

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # --- Chroma STFT ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # --- Spectral Centroid ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # --- Spectral Rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # --- Zero Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # --- RMS Energy ---
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    # Quick sanity check — run: python feature_extractor.py <path_to_wav>
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python feature_extractor.py <path_to_wav>")
    else:
        vec = extract_features(path)
        print(f"Feature vector shape : {vec.shape}")
        print(f"First 10 values      : {vec[:10]}")