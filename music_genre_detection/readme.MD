# MoodTunes - Music Genre Detection Using Machine Learning

**Author:** Prabhat Anil Bajpai
**Institution:** CMR Institute of Technology
**Branch:** ISE (Information Science and Engineering)

A machine learning pipeline that classifies music audio into **10 genres** by extracting audio features using `librosa` and training multiple classifiers (`SVM`, `Random Forest`, `MLP Neural Network`). The system can predict genres from local `.wav` files or directly from **YouTube URLs**.

**Supported Genres:** Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Running the Pipeline](#running-the-pipeline)
6. [All Available Commands](#all-available-commands)
7. [Features Extracted](#features-extracted)
8. [Model Comparison and Results](#model-comparison-and-results)
9. [Key Findings and Learnings](#key-findings-and-learnings)
10. [Troubleshooting](#troubleshooting)
11. [Future Scope](#future-scope)
12. [References](#references)

---

## Project Overview

The goal of this project is to understand how machine learning works end-to-end:
from raw audio data to a trained model that can predict the genre of any song.

The pipeline covers:
- **Data collection** -- using the GTZAN dataset (1000 audio clips, 10 genres)
- **Feature engineering** -- extracting 72 numerical features from each audio clip
- **Model training** -- training and comparing multiple classifiers
- **Evaluation** -- cross-validation, confusion matrix, learning curves
- **Prediction** -- predicting genre from local files or YouTube URLs

This project uses **classical machine learning** (not deep learning) to demonstrate
that hand-crafted audio features combined with the right classifier can achieve
competitive results (~76% accuracy across 10 genres).

---

## How It Works

The system works in three stages:

```
Stage 1: Feature Extraction
    Audio (.wav file)
        |
        v
    librosa loads audio at 22050 Hz, 30 seconds
        |
        v
    Extract 72 features (MFCCs, Chroma, Spectral, Energy)
        |
        v
    Feature vector: [0.23, -1.45, 0.87, ...]  (72 numbers)


Stage 2: Model Training
    features.csv (999 rows x 72 features + label)
        |
        v
    StandardScaler (normalize values)
        |
        v
    Train classifier (SVM / Random Forest / MLP)
        |
        v
    Saved model (.pkl file)


Stage 3: Prediction
    New audio file (or YouTube URL)
        |
        v
    Extract 72 features (same as Stage 1)
        |
        v
    Load saved model -> predict probabilities
        |
        v
    Output: "GENRE: ROCK  Confidence: 85.3%"
```

---

## Project Structure

```
music_genre_detection/
|
|-- feature_extractor.py     # Extracts 72 audio features from a .wav file
|-- build_dataset.py         # Walks GTZAN dataset, builds data/features.csv
|-- train.py                 # Trains SVM or Random Forest, saves model
|-- predict.py               # Predicts genre of a single audio file
|-- predict_segments.py      # Splits audio into 3s segments, majority vote
|-- predict_youtube.py       # Downloads audio from YouTube URL, predicts genre
|-- predict_ensemble.py      # Ensemble of SVM + RF + MLP (soft voting)
|-- compare_models.py        # Compares 6 classifiers with 10-fold CV
|-- tune.py                  # Hyperparameter tuning with RandomizedSearchCV
|-- learning_curves.py       # Plots learning curve (overfitting analysis)
|-- evaluate_samples.py      # Batch-tests a folder of sample clips
|
|-- data/
|   |-- genres_original/     # GTZAN dataset (not committed, ~1.2 GB)
|   |   |-- blues/
|   |   |-- classical/
|   |   |-- country/
|   |   |-- ...
|   |-- features.csv         # Extracted features (999 rows x 73 columns)
|
|-- models/
|   |-- rf_model.pkl          # Trained classifier (SVM or RF)
|   |-- scaler.pkl            # StandardScaler fitted on training data
|   |-- label_encoder.pkl     # Maps genre names to/from numbers
|   |-- ensemble_model.pkl    # Ensemble model (SVM + RF + MLP)
|   |-- ensemble_scaler.pkl
|   |-- ensemble_encoder.pkl
|
|-- outputs/
|   |-- confusion_matrix.png  # Confusion matrix heatmap
|   |-- model_comparison.png  # Bar chart comparing 6 classifiers
|   |-- learning_curve.png    # Train vs CV accuracy plot
|
|-- samples/                  # Test audio clips
|-- requirements.txt          # Python dependencies
|-- readme.MD                 # This file
```

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher (tested on 3.10, 3.12, 3.13)
- `pip` package manager
- Internet connection (for YouTube prediction and downloading the dataset)

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd MoodTunes/music_genre_detection
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `librosa` | Audio loading and feature extraction |
| `numpy` | Numerical operations |
| `scikit-learn` | ML classifiers, evaluation, preprocessing |
| `matplotlib` | Plotting confusion matrix, charts |
| `joblib` | Saving/loading trained models |
| `pandas` | Reading/writing CSV data |
| `yt-dlp` | Downloading audio from YouTube |

### Step 3: Download the GTZAN dataset

1. Go to: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. Download and unzip
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
    disco/
    hiphop/
    jazz/
    metal/
    pop/
    reggae/
    rock/
```

Each genre folder contains 100 audio clips, each 30 seconds long.

### Step 4 (optional): Install ffmpeg for YouTube support

YouTube prediction requires `ffmpeg` to convert audio. If not already installed:

```bash
pip install imageio-ffmpeg
```

Or install system-wide from https://ffmpeg.org/download.html.

---

## Running the Pipeline

Run these commands **in order** from the `music_genre_detection/` directory.

### Step 1 -- Build the feature dataset (5-10 minutes)

```bash
python build_dataset.py
```

This walks every `.wav` file in `data/genres_original/`, extracts 72 features
per clip, and writes them to `data/features.csv`.

Output:
```
Building dataset from: data/genres_original
  blues          100 files
  classical      100 files
  country        100 files
  ...
Done. 999 samples saved -> data/features.csv  (1 skipped)
```

### Step 2 -- Train the model

Train with **SVM** (recommended, ~76% accuracy):
```bash
python train.py --model svm
```

Or train with **Random Forest** (original, ~71% accuracy):
```bash
python train.py --model rf
```

Output:
```
Training SVM  (C=10, kernel=rbf) ...
  5-fold CV accuracy : 0.7459 +/- 0.0316
  Test accuracy      : 0.7550  (75.50%)

              precision    recall  f1-score   support
       blues       0.80      0.80      0.80        20
   classical       0.86      0.95      0.90        20
   ...
```

The model is saved to `models/rf_model.pkl`.

### Step 3 -- Predict genre of an audio file

```bash
python predict.py samples/disco_test.wav
python predict.py samples/metal_test.wav --top 5
```

Output:
```
  File   : disco_test.wav
  Genre  : DISCO
  Confidence : 85.4%

  Top predictions:
    disco        [#########################.....]  85.4%
    rock         [#.............................]   5.0%
    pop          [#.............................]   4.5%
```

### Step 4 -- Predict genre from a YouTube URL

```bash
python predict_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --top 5
```

This downloads the audio, converts it to wav, runs prediction, and cleans up.

Use `--keep` to save the downloaded audio file:
```bash
python predict_youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --keep
```

---

## All Available Commands

### Core Pipeline

| Command | What it does |
|---------|-------------|
| `python build_dataset.py` | Extract features from GTZAN dataset to CSV |
| `python train.py --model svm` | Train SVM classifier (best accuracy) |
| `python train.py --model rf` | Train Random Forest classifier |
| `python train.py --model rf --n_estimators 300` | Train RF with 300 trees |
| `python predict.py <file.wav>` | Predict genre of a local audio file |
| `python predict.py <file.wav> --top 5` | Show top-5 predictions |

### YouTube Prediction

| Command | What it does |
|---------|-------------|
| `python predict_youtube.py "<youtube_url>"` | Predict genre from YouTube video |
| `python predict_youtube.py "<url>" --top 5` | Show top-5 predictions |
| `python predict_youtube.py "<url>" --keep` | Keep downloaded audio after prediction |

### Segment-Based Prediction

| Command | What it does |
|---------|-------------|
| `python predict_segments.py <file.wav>` | Split into 3s segments, majority vote |
| `python predict_segments.py <file.wav> --segment_sec 5` | Use 5-second segments |
| `python predict_segments.py <file.wav> --top 5` | Show top-5 predictions |

### Ensemble Prediction

| Command | What it does |
|---------|-------------|
| `python predict_ensemble.py train` | Train the ensemble (SVM + RF + MLP) |
| `python predict_ensemble.py predict <file.wav>` | Predict using ensemble |
| `python predict_ensemble.py predict <file.wav> --top 5` | Show top-5 predictions |

### Analysis and Evaluation

| Command | What it does |
|---------|-------------|
| `python compare_models.py` | Compare 6 classifiers with 10-fold CV |
| `python tune.py` | Hyperparameter tuning for Random Forest |
| `python tune.py --n_iter 50` | Try 50 random parameter combinations |
| `python learning_curves.py` | Plot training size vs accuracy |
| `python evaluate_samples.py` | Batch-test clips in `samples/` folder |
| `python feature_extractor.py <file.wav>` | Print raw feature vector for a file |

---

## Features Extracted

Each 30-second audio clip is converted into a vector of **72 numbers**:

| Feature | Count | What It Captures |
|---------|-------|-----------------|
| MFCC (mean + std of 20 coefficients) | 40 | Timbre and tone quality -- the "texture" of sound. MFCCs mimic how the human ear perceives frequency. |
| Chroma STFT (mean + std of 12 bins) | 24 | Harmonic and pitch content -- which musical notes are present. Maps onto the 12 pitch classes (C, C#, D, ..., B). |
| Spectral Centroid (mean + std) | 2 | "Brightness" -- whether the sound has more high-frequency or low-frequency energy. Metal tends to have a high centroid, classical tends to be lower. |
| Spectral Rolloff (mean + std) | 2 | Where 85% of the spectral energy lies. Helps distinguish noisy (rock) from tonal (classical) sounds. |
| Zero Crossing Rate (mean + std) | 2 | How often the signal crosses zero. Higher for percussive and noisy sounds (metal, rock), lower for smooth sounds (jazz, classical). |
| RMS Energy (mean + std) | 2 | Loudness and energy level. Metal and rock have high RMS, classical and jazz have more dynamic range. |
| **Total** | **72** | |

We compute both the **mean** (average value across time) and **standard deviation**
(how much it varies) for each feature, because some genres are characterized by
consistency (classical) while others by variation (rock).

---

## Model Comparison and Results

### 10-Fold Stratified Cross-Validation Results

| Rank | Model | Accuracy | Std Dev |
|------|-------|----------|---------|
| 1 | **SVM (RBF, C=10)** | **75.8%** | 4.8% |
| 2 | MLP Neural Network (256, 128) | 74.4% | 4.8% |
| 3 | Random Forest (200 trees) | 71.0% | 3.3% |
| 4 | Logistic Regression | 70.7% | 4.8% |
| 5 | Gradient Boosting (200 trees) | 67.4% | 4.0% |
| 6 | KNN (k=7) | 66.6% | 5.2% |

**SVM (Support Vector Machine)** won the comparison and is the default model.

### Per-Genre Performance (SVM, test set)

| Genre | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Blues | 0.80 | 0.80 | 0.80 |
| Classical | 0.86 | 0.95 | 0.90 |
| Country | 0.67 | 0.80 | 0.73 |
| Disco | 0.67 | 0.50 | 0.57 |
| Hip-Hop | 0.71 | 0.85 | 0.77 |
| Jazz | 0.82 | 0.90 | 0.86 |
| Metal | 0.94 | 0.85 | 0.89 |
| Pop | 0.68 | 0.75 | 0.71 |
| Reggae | 0.65 | 0.55 | 0.59 |
| Rock | 0.75 | 0.60 | 0.67 |

**Best classified:** Classical (0.90 F1), Metal (0.89 F1), Jazz (0.86 F1)
**Most confused:** Disco/Reggae/Rock often overlap due to similar rhythm patterns.

### YouTube Prediction Tests

| Song | Artist | Predicted | Confidence | Correct |
|------|--------|-----------|------------|---------|
| Never Gonna Give You Up | Rick Astley | DISCO | 81.0% | Yes |
| Nothing Else Matters | Metallica | ROCK | 46.4% | Yes |
| Lose Yourself | Eminem | HIPHOP | 39.4% | Yes |
| Fur Elise | Beethoven | CLASSICAL | 72.8% | Yes |
| Bohemian Rhapsody | Queen | REGGAE | 22.9% | No |
| The Thrill Is Gone | BB King | POP | 52.4% | No |

4 out of 6 correct. Misclassifications happen because the model only analyzes
the first 30 seconds, and some songs are genre-defying (Bohemian Rhapsody starts
as a piano ballad, not rock).

---

## Key Findings and Learnings

### 1. Feature Engineering Matters More Than the Model

The 72 hand-crafted features (MFCCs, Chroma, Spectral) are the backbone of this
system. Without good features, even the best classifier would fail. MFCCs alone
carry most of the discriminative information -- the top-10 most important features
are almost all MFCC coefficients.

### 2. SVM Outperforms Random Forest on Small Datasets

With only ~100 samples per class, SVM's ability to find optimal decision boundaries
in high-dimensional space (72 features) gives it an edge. Random Forest tends to
overfit on small datasets -- the learning curve shows 100% train accuracy but only
~56% cross-validation accuracy, a classic overfitting signature.

### 3. The Learning Curve Reveals Overfitting

The learning curve plot (`outputs/learning_curve.png`) shows:
- Training accuracy = 100% (model memorizes the training data)
- CV accuracy = ~56% with full data, and climbs slowly

This tells us the model would benefit from **more training data**. The gap between
train and CV score is a textbook example of high variance (overfitting).

### 4. Cross-Validation Method Matters

Initially we used 5-fold CV on the training split only, which gave inflated numbers.
Switching to proper 10-fold stratified CV on the full dataset with the scaler inside
the pipeline gave honest, reproducible results.

### 5. Some Genres Are Inherently Hard to Separate

Rock, Country, and Blues share similar instrumentation (guitar, drums). Disco and
Pop share similar electronic production. The confusion matrix shows these pairs
are frequently mixed up. This is not a model failure -- even human listeners
disagree on genre boundaries.

### 6. Segment Voting Improves Robustness

Splitting a song into 10 segments of 3 seconds each and taking a majority vote
makes predictions more stable. Even if one segment is unusual (e.g., a quiet
intro), the other segments carry the prediction. For the disco test clip,
all 10 segments independently agreed on "disco".

### 7. Domain Shift Is Real

The model is trained on GTZAN clips (recorded in the 1990s-2000s, studio quality).
Real YouTube audio has different recording quality, mixing, and compression. Despite
this, the model performs surprisingly well, showing the features are robust.

### 8. Ensemble Does Not Always Beat the Best Single Model

Our ensemble (SVM + RF + MLP) achieved 75.5% vs SVM's 75.8%. This happens because
the weaker models (RF, MLP) can drag down the ensemble average when they disagree
with the correct SVM prediction.

---

## Troubleshooting

**`pip install` fails with compilation errors**
Your Python version may not have pre-built wheels. Use `>=` version pins (already
done in `requirements.txt`) to allow compatible newer versions.

**`librosa.load` fails on a file**
Some GTZAN files (e.g., `jazz.00054.wav`) are corrupt. `build_dataset.py` skips
them automatically. If your own file fails, ensure it is a valid audio file.

**Unicode errors on Windows (characters like arrows or blocks)**
The scripts handle this automatically with a UTF-8 stdout wrapper. If you still
see errors, set the environment variable before running:
```bash
set PYTHONIOENCODING=utf-8
```

**`predict_youtube.py` fails with ffmpeg error**
Install ffmpeg: `pip install imageio-ffmpeg`. The script auto-detects the ffmpeg
binary from this package.

**`predict_youtube.py` fails with "No supported JavaScript runtime"**
Install the yt-dlp JavaScript runtime: `pip install yt-dlp[default]`.

**Low accuracy on your own audio**
The model is trained on 30-second, single-instrument/genre clips. Real songs with
multiple genres, intros, or talking segments will have lower confidence. Try the
segment voting approach (`predict_segments.py`) for better results.

---

## Future Scope

- **CNN on Mel Spectrograms** -- Treat spectrograms as images and use a
  Convolutional Neural Network. Expected accuracy: 85-90%.
- **Data Augmentation** -- Pitch shifting, time stretching, and noise injection
  to artificially expand the training set.
- **Per-Segment Analysis with Visualization** -- Show a timeline of genre
  predictions across a full song (genre can change within a song).
- **Transfer Learning** -- Use pre-trained audio models (YAMNet, VGGish) as
  feature extractors for higher accuracy with less training data.
- **Web Interface** -- A Streamlit or Flask app for live demo.
- **Mood Detection** -- Extend beyond genre to detect mood/emotion from audio,
  which is a harder and more subjective classification task.

---

## References

1. Tzanetakis, G. and Cook, P. (2002). "Musical genre classification of audio signals." *IEEE Transactions on Speech and Audio Processing*, 10(5), 293-302.
2. GTZAN Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
3. librosa documentation: https://librosa.org/doc/latest/
4. scikit-learn documentation: https://scikit-learn.org/stable/
5. McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python." *Proceedings of the 14th Python in Science Conference*.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Audio Processing | librosa, soundfile |
| ML Framework | scikit-learn |
| Visualization | matplotlib |
| Data Handling | pandas, numpy |
| YouTube Audio | yt-dlp, ffmpeg |
| Model Persistence | joblib |

---

*Project developed as part of the Machine Learning coursework at CMR Institute of Technology.*
