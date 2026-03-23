# Music Genre Detection

A machine-learning pipeline that classifies music into 10 genres using
audio features extracted with **librosa** and a **Random Forest** classifier.

**Genres:** blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock

---

## Project Structure

```
music_genre_detection/
│
├── feature_extractor.py    # Core: extracts audio features from a .wav file
├── build_dataset.py        # Walks GTZAN folder, saves data/features.csv
├── train.py                # Trains & evaluates the model, saves to models/
├── predict.py              # Predicts genre for a single audio file
├── evaluate_samples.py     # Batch-tests a folder of audio samples
│
├── data/                   # Put GTZAN's genres_original/ here
├── models/                 # Trained model artefacts saved here
├── outputs/                # Confusion matrix image saved here
├── samples/                # Put your test audio clips here
│
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+. Tested on 3.10 and 3.12.

### 2. Download the GTZAN dataset

1. Go to: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. Download and unzip.
3. Copy the `genres_original/` folder into `data/`:

```
data/
  genres_original/
    blues/
      blues.00000.wav
      blues.00001.wav
      ...
    classical/
    country/
    ...
```

---

## Running the Pipeline

### Step 1 — Build the feature dataset (takes ~5–10 min)

```bash
python build_dataset.py
```

Walks every `.wav` file in `data/genres_original/`, extracts 72 features per clip,
and writes them all to `data/features.csv`.

### Step 2 — Train the model

```bash
python train.py
```

What it does:
- Loads `data/features.csv`
- Splits 80% train / 20% test
- Scales features with StandardScaler
- Trains a Random Forest (200 trees)
- Runs 5-fold cross-validation
- Prints accuracy + classification report
- Saves confusion matrix → `outputs/confusion_matrix.png`
- Saves model → `models/rf_model.pkl`

**Expected accuracy: ~65–78%**

Optional flags:
```bash
python train.py --n_estimators 300 --test_size 0.2
```

### Step 3 — Predict a new audio file

```bash
python predict.py samples/my_song.wav
python predict.py samples/my_song.wav --top 5   # show top-5 genres
```

Output example:
```
──────────────────────────────────────────────────
  File   : my_song.wav
  Genre  : JAZZ
  Confidence : 74.3%

  Top predictions:
    jazz         ██████████████████████░░░░░░░░  74.3%
    blues        ████░░░░░░░░░░░░░░░░░░░░░░░░░░  14.1%
    classical    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7.2%
──────────────────────────────────────────────────
```

### Step 4 — Batch evaluate sample clips

Place some `.wav` clips in `samples/` with filenames starting with the genre:
```
samples/
  jazz_clip1.wav
  rock_test.wav
  blues_sample.wav
```

```bash
python evaluate_samples.py
```

---

## Features Extracted

| Feature               | Dimensions | What it captures                    |
|-----------------------|-----------|--------------------------------------|
| MFCC (mean + std)     | 40        | Timbre, tone quality                 |
| Chroma (mean + std)   | 24        | Harmonic/pitch content               |
| Spectral Centroid     |  2        | "Brightness" of sound                |
| Spectral Rolloff      |  2        | High-frequency energy distribution   |
| Zero Crossing Rate    |  2        | Percussiveness, noisiness            |
| RMS Energy            |  2        | Loudness / energy                    |
| **Total**             | **72**    |                                      |

---

## Review 2 Improvements (planned)

- [ ] CNN on Mel spectrograms (treat spectrogram as image → ~85-90% accuracy)
- [ ] Data augmentation (pitch shift, time stretch, noise injection)
- [ ] Streamlit or Flask web UI for live demo
- [ ] Per-segment voting (split 30s clip into 3s segments, majority vote)
- [ ] Confusion matrix analysis to fix genre pairs with high error rates

---

## Troubleshooting

**`librosa.load` fails on a file** — some GTZAN files (e.g. jazz.00054.wav) are
corrupt. The `build_dataset.py` script skips them automatically.

**Low accuracy** — ensure `build_dataset.py` completed without many skips.
Re-run `train.py` with `--n_estimators 300` for a small boost.

**ModuleNotFoundError** — make sure you ran `pip install -r requirements.txt`
in the same Python environment you're using to run the scripts.