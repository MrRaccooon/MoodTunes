"""
build_dataset.py
----------------
Walks the GTZAN dataset directory, extracts features for every audio file,
and saves a single CSV: data/features.csv

Expected GTZAN folder layout
-----------------------------
data/
  genres_original/
    blues/
      blues.00000.wav
      blues.00001.wav
      ...
    classical/
      ...
    country/
    disco/
    hiphop/
    jazz/
    metal/
    pop/
    reggae/
    rock/

Download GTZAN from:
  https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
  (free, requires Kaggle login)

Usage
-----
  python build_dataset.py
  python build_dataset.py --data_dir /path/to/genres_original
"""

import os
import argparse
import numpy as np
import csv
from feature_extractor import extract_features, N_MFCC

# ── column names ─────────────────────────────────────────────────────────────
FEATURE_COLS = (
    [f"mfcc_mean_{i}"     for i in range(N_MFCC)] +
    [f"mfcc_std_{i}"      for i in range(N_MFCC)] +
    [f"chroma_mean_{i}"   for i in range(12)]     +
    [f"chroma_std_{i}"    for i in range(12)]     +
    ["centroid_mean", "centroid_std",
     "rolloff_mean",  "rolloff_std",
     "zcr_mean",      "zcr_std",
     "rms_mean",      "rms_std"]
)
HEADER = FEATURE_COLS + ["label"]

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz",  "metal",     "pop",     "reggae", "rock"]


def build(data_dir: str, output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    total, skipped = 0, 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for genre in GENRES:
            genre_dir = os.path.join(data_dir, genre)
            if not os.path.isdir(genre_dir):
                print(f"  [WARN] Genre folder not found: {genre_dir}")
                continue

            files = [fn for fn in sorted(os.listdir(genre_dir))
                     if fn.endswith(".wav")]
            print(f"  {genre:12s}  {len(files)} files")

            for fn in files:
                path = os.path.join(genre_dir, fn)
                try:
                    vec = extract_features(path)
                    writer.writerow(list(vec) + [genre])
                    total += 1
                except Exception as e:
                    print(f"    [SKIP] {fn}: {e}")
                    skipped += 1

    print(f"\nDone. {total} samples saved → {output_csv}  ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join("data", "genres_original"),
        help="Path to GTZAN genres_original folder"
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join("data", "features.csv"),
    )
    args = parser.parse_args()

    print(f"Building dataset from: {args.data_dir}")
    build(args.data_dir, args.output_csv)