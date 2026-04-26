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

from predict import predict, print_result

TEMP_WAV = os.path.join("samples", "yt_temp.wav")


def download_audio(url: str, output_path: str) -> None:
    if shutil.which("yt-dlp") is None:
        print("[ERROR] yt-dlp not found.  Install with:  pip install yt-dlp")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # yt-dlp appends the extension itself, so strip .wav from the template
    template = output_path.rsplit(".", 1)[0]

    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "-o", template + ".%(ext)s",
        "--no-playlist",
        "--quiet",
        url,
    ]
    print(f"  Downloading audio from YouTube ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] yt-dlp failed:\n{result.stderr}")
        sys.exit(1)

    if not os.path.exists(output_path):
        print(f"[ERROR] Expected file not found at {output_path}")
        sys.exit(1)

    print(f"  Saved -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict music genre from a YouTube URL."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--top", type=int, default=3,
                        help="Show top-N genre predictions (default: 3)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the downloaded .wav file after prediction")
    args = parser.parse_args()

    download_audio(args.url, TEMP_WAV)

    try:
        result = predict(TEMP_WAV, top_n=args.top)
        print_result(TEMP_WAV, result)
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        sys.exit(1)
    finally:
        if not args.keep and os.path.exists(TEMP_WAV):
            os.remove(TEMP_WAV)
            print("  (temp audio file cleaned up)")


if __name__ == "__main__":
    main()
