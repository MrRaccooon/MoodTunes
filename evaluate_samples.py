"""
evaluate_samples.py
-------------------
Evaluates the trained model on a folder of labelled audio samples.
Useful for demonstrating prediction on new audio during the review.

Expected folder layout
-----------------------
samples/
  blues/
    some_blues_clip.wav
  rock/
    some_rock_clip.wav
  ...

OR a flat folder where filenames begin with the genre name:
  samples/
    jazz_test1.wav
    metal_test2.wav

Usage
-----
  python evaluate_samples.py
  python evaluate_samples.py --samples_dir samples --mode flat
"""

import os
import argparse
from predict import predict, print_result, load_model

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz",  "metal",     "pop",     "reggae", "rock"]


def evaluate_folder_mode(samples_dir: str) -> None:
    """Subfolder per genre — compares prediction vs actual label."""
    correct, total = 0, 0

    for genre in GENRES:
        genre_dir = os.path.join(samples_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for fn in os.listdir(genre_dir):
            if not fn.endswith(".wav"):
                continue
            path   = os.path.join(genre_dir, fn)
            result = predict(path, top_n=3)
            match  = (result["predicted_genre"] == genre)
            mark   = "✓" if match else "✗"
            print(f"  {mark} [{genre:10s}] → predicted: {result['predicted_genre']:10s}  "
                  f"({result['confidence']*100:.1f}%)")
            correct += int(match)
            total   += 1

    if total:
        print(f"\n  Accuracy on samples: {correct}/{total} = {correct/total*100:.1f}%\n")


def evaluate_flat_mode(samples_dir: str) -> None:
    """Flat folder — filename must start with genre name (e.g. jazz_clip.wav)."""
    correct, total = 0, 0

    for fn in sorted(os.listdir(samples_dir)):
        if not fn.endswith(".wav"):
            continue
        true_genre = None
        for g in GENRES:
            if fn.lower().startswith(g):
                true_genre = g
                break

        path   = os.path.join(samples_dir, fn)
        result = predict(path, top_n=3)
        print_result(path, result)

        if true_genre:
            match = (result["predicted_genre"] == true_genre)
            mark  = "✓ Correct" if match else f"✗ Expected: {true_genre}"
            print(f"  Label check: {mark}\n")
            correct += int(match)
            total   += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", default="samples")
    parser.add_argument(
        "--mode", choices=["folder", "flat"], default="flat",
        help="'folder' = subfolder per genre, 'flat' = genre prefix in filename"
    )
    args = parser.parse_args()

    print(f"\nEvaluating samples in: {args.samples_dir}  (mode={args.mode})\n")

    if args.mode == "folder":
        evaluate_folder_mode(args.samples_dir)
    else:
        evaluate_flat_mode(args.samples_dir)