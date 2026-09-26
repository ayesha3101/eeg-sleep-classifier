# EEG Sleep Stage Classifier

A machine learning pipeline for classifying sleep stages (Wake, N1, N2, N3, and REM) from EEG signals using the Sleep-EDF Expanded dataset from PhysioNet. The project uses spectral feature extraction and a Random Forest classifier.

## Pipeline

1. Download a subset of Sleep-EDF recordings.
2. Preprocess the EEG signals by filtering and dividing them into standard 30-second epochs.
3. Extract spectral band-power features for delta, theta, alpha, beta, and gamma frequency bands.
4. Train a Random Forest classifier using the extracted features and labeled epochs.
5. Evaluate the model using a confusion matrix, per-stage accuracy, and a hypnogram plot.

## Project Structure

```text
data/
├── raw/                # Original EDF recordings, not tracked in Git
└── processed/          # Extracted features and labels, not tracked in Git

src/                    # Data loading, preprocessing, feature extraction, and modeling
notebooks/              # Exploratory data analysis
results/                # Evaluation plots and trained model artifacts
```

## Dataset

This project uses the Sleep-EDF Expanded dataset provided by PhysioNet.
