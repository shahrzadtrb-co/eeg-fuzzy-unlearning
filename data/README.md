# Data

This project uses the public **Complete EEG dataset** on Kaggle:
Aman Anand, *Complete EEG dataset* — https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset

The raw files are not included in this repository.

## Setup

1. Download the dataset from the Kaggle page above.
2. Extract it.
3. Put the subject files (`s00.csv` … `s35.csv`) in `data/raw/`.

Each file is one subject: a 60-second recording with 31,000 rows and 19 EEG channels
(Fp1, Fp2, F3, F4, F7, F8, T3, T4, C3, C4, T5, T6, P3, P4, O1, O2, Fz, Cz, Pz).
Extra columns beyond the first 19 are ignored.

No dataset? `eeg-unlearn run --synthetic` runs the full pipeline on generated EEG-like
signals. That is useful for checking the setup, but its numbers mean nothing for real EEG.
