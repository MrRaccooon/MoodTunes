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

from predict import predict, print_result

TEMP_AUDIO = os.path.join("samples", "yt_temp")
TEMP_WAV   = TEMP_AUDIO + ".wav"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict music genre from a YouTube URL."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top-N genre predictions (default: 3)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the downloaded audio file after prediction")
    args = parser.parse_args()

    audio_path = download_audio(args.url, TEMP_AUDIO)

    try:
        result = predict(audio_path, top_n=args.top)
        print_result(audio_path, result)
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        sys.exit(1)
    finally:
        if not args.keep and os.path.exists(audio_path):
            os.remove(audio_path)
            print("  (temp audio file cleaned up)")


if __name__ == "__main__":
    main()
