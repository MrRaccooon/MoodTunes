import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
predict_youtube.py
------------------
Downloads audio from a YouTube URL via yt-dlp and predicts its genre
using the trained model.

Usage
-----
  python predict_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python predict_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --top 5
  python predict_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --keep
"""

import os
import sys
import argparse
import subprocess
import shutil
import glob as globmod

import numpy as np
import joblib
import librosa

from feature_extractor import SAMPLE_RATE, DURATION, N_MFCC

TEMP_AUDIO = os.path.join("samples", "yt_temp")
TEMP_WAV   = TEMP_AUDIO + ".wav"

MODEL_PATH   = os.path.join("models", "rf_model.pkl")
SCALER_PATH  = os.path.join("models", "scaler.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")
SEGMENT_LEN  = DURATION  # 30 seconds, matching training data


def _get_ffmpeg() -> str | None:
    """Find an ffmpeg executable."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def download_audio(url: str, output_base: str) -> str:
    """Download best audio track, convert to wav. Returns wav path."""
    if shutil.which("yt-dlp") is None:
        print("[ERROR] yt-dlp not found.  Install with:  pip install yt-dlp")
        sys.exit(1)

    ffmpeg = _get_ffmpeg()
    if ffmpeg is None:
        print("[ERROR] ffmpeg not found.  Install with:  pip install imageio-ffmpeg")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)

    for old in globmod.glob(output_base + ".*"):
        os.remove(old)

    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-o", output_base + ".%(ext)s",
        "--no-playlist",
        "--quiet",
        url,
    ]
    print("  Downloading audio from YouTube ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] yt-dlp failed:\n{result.stderr}")
        sys.exit(1)

    matches = globmod.glob(output_base + ".*")
    if not matches:
        print("[ERROR] No audio file downloaded.")
        sys.exit(1)

    downloaded = matches[0]
    wav_path = output_base + ".wav"

    if downloaded.endswith(".wav"):
        return downloaded

    print("  Converting to wav ...")
    conv = subprocess.run(
        [ffmpeg, "-y", "-i", downloaded, "-ar", "22050", "-ac", "1", wav_path],
        capture_output=True, text=True,
    )
    os.remove(downloaded)
    if conv.returncode != 0 or not os.path.exists(wav_path):
        print(f"[ERROR] ffmpeg conversion failed:\n{conv.stderr}")
        sys.exit(1)

    print(f"  Saved -> {wav_path}")
    return wav_path


def _extract_segment_features(y_seg: np.ndarray, sr: int) -> np.ndarray:
    """Extract the 72-dim feature vector from a signal array (same as feature_extractor)."""
    target_len = sr * SEGMENT_LEN
    if len(y_seg) < target_len:
        y_seg = np.pad(y_seg, (0, target_len - len(y_seg)))

    feats = []
    mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=N_MFCC)
    feats.extend(np.mean(mfcc, axis=1)); feats.extend(np.std(mfcc, axis=1))
    chroma = librosa.feature.chroma_stft(y=y_seg, sr=sr)
    feats.extend(np.mean(chroma, axis=1)); feats.extend(np.std(chroma, axis=1))
    for fn in (librosa.feature.spectral_centroid, librosa.feature.spectral_rolloff):
        v = fn(y=y_seg, sr=sr)
        feats.append(np.mean(v)); feats.append(np.std(v))
    zcr = librosa.feature.zero_crossing_rate(y_seg)
    feats.append(np.mean(zcr)); feats.append(np.std(zcr))
    rms = librosa.feature.rms(y=y_seg)
    feats.append(np.mean(rms)); feats.append(np.std(rms))
    return np.array(feats, dtype=np.float32)


def predict_full_song(wav_path: str, top_n: int = 3) -> dict:
    """
    Load the FULL audio, split into 30-sec segments, predict each,
    and aggregate probabilities across all segments.
    """
    clf     = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    total_duration = len(y) / sr
    seg_samples = sr * SEGMENT_LEN

    # Build segments -- at least 10 seconds to be useful
    min_useful = sr * 10
    segments = []
    for start in range(0, len(y), seg_samples):
        seg = y[start : start + seg_samples]
        if len(seg) >= min_useful:
            segments.append(seg)

    if not segments:
        segments = [y]

    n_segs = len(segments)
    print(f"  Analyzing {n_segs} segment(s)  ({total_duration:.0f}s total)")

    all_proba = []
    for i, seg in enumerate(segments):
        feats = _extract_segment_features(seg, sr).reshape(1, -1)
        feats = scaler.transform(feats)
        proba = clf.predict_proba(feats)[0]
        all_proba.append(proba)
        seg_top = encoder.classes_[np.argmax(proba)]
        print(f"    Segment {i+1}/{n_segs} -> {seg_top} ({np.max(proba)*100:.0f}%)")

    avg_proba = np.mean(all_proba, axis=0)
    top_idx = np.argsort(avg_proba)[::-1][:top_n]

    return {
        "predicted_genre": encoder.classes_[top_idx[0]],
        "confidence": float(avg_proba[top_idx[0]]),
        "top_n": [(encoder.classes_[i], float(avg_proba[i])) for i in top_idx],
        "n_segments": n_segs,
        "duration": total_duration,
    }


def print_yt_result(file_path: str, result: dict) -> None:
    bar_width = 30
    print(f"\n{'='*50}")
    print(f"  File      : {os.path.basename(file_path)}")
    print(f"  Duration  : {result['duration']:.0f}s  ({result['n_segments']} segments)")
    print(f"  Genre     : {result['predicted_genre'].upper()}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"\n  Top predictions (averaged across all segments):")
    for genre, prob in result["top_n"]:
        filled = int(prob * bar_width)
        bar    = "#" * filled + "." * (bar_width - filled)
        print(f"    {genre:12s} {bar} {prob*100:5.1f}%")
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict music genre from a YouTube URL (full-song analysis)."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--top", type=int, default=5,
                        help="Show top-N genre predictions (default: 5)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the downloaded audio file after prediction")
    args = parser.parse_args()

    audio_path = download_audio(args.url, TEMP_AUDIO)

    try:
        result = predict_full_song(audio_path, top_n=args.top)
        print_yt_result(audio_path, result)
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        sys.exit(1)
    finally:
        if not args.keep and os.path.exists(audio_path):
            os.remove(audio_path)
            print("  (temp audio file cleaned up)")


if __name__ == "__main__":
    main()
