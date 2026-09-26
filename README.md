# Privacy-Preserving EEG Classification with Fuzzy Unlearning

Can a trained EEG classifier "forget" one person? EEG signals carry subject-specific patterns that can identify individuals, and under the GDPR a person can ask for their data to be deleted, including its influence on a model that is already in use.

This project builds an end-to-end subject-level unlearning pipeline and compares three settings:

- **Baseline:** Random Forest trained on all subjects
- **Exact unlearning:** retrained from scratch without the forgotten subject (the reference for true removal)
- **Fuzzy unlearning:** retrained with lower sample weights for the retained windows most similar to the forgotten subject

Written for the Real Life Security Seminar, University of Passau, December 2025. The full report is in [`paper/`](paper/Final_Report_EEG_Unlearning_Torabi.pdf).

## Results

Retained subjects (test set):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline | 0.8971 | 0.8712 |
| Exact unlearning | 0.9029 | 0.8998 |
| Fuzzy unlearning | 0.8800 | 0.8719 |

Forgotten subject (test windows). Lower confidence and higher entropy mean the model has forgotten more:

| Model | Mean max. confidence | Predictive entropy |
|---|---|---|
| Baseline | 0.3064 | 2.9003 |
| Exact unlearning | 0.1106 | 3.2650 |
| Fuzzy unlearning | 0.1208 | 3.2357 |

**What this shows:** exact retraining gives the best balance of forgetting and utility. Fuzzy unlearning moves the model strongly in the same direction, with a small loss in retained accuracy. Forgetting cannot be judged by accuracy alone, because the forgotten subject is no longer a class, so it is measured through confidence and entropy instead.

## Limitations

- **Small forget set.** The forgotten subject contributes 31 training windows and only 5 test windows, so the forgetting numbers are indicative, not conclusive.
- **Correlated samples.** Overlapping sliding windows make neighbouring training samples highly similar. Training accuracy is about 0.99 for all models, a sign of overfitting.
- **Fuzzy unlearning still retrains.** In this implementation it re-weights the retained data and retrains, so it does not yet save compute compared with exact retraining.
- **Scarce data.** Public EEG datasets suited to subject-level unlearning are rare, which limits how far the results generalise.

## Pipeline

1. Load 36 subject recordings (19 EEG channels, mental-arithmetic task) from CSV
2. Split each recording chronologically, 80% train and 20% test, so test windows are always later, unseen segments
3. Segment into sliding windows: overlapping for training, non-overlapping for testing
4. Extract five statistical features per channel, plus band power in the delta, theta, alpha, beta and gamma bands
5. Build the retain set and the forget set for the selected subject
6. Train the baseline, exact and fuzzy Random Forest models
7. Evaluate retained-subject accuracy and macro F1, and forgotten-subject confidence and entropy

**Tech:** Python · scikit-learn (Random Forest) · Jupyter

## Dataset

Complete EEG dataset by Aman Anand, hosted on Kaggle:
https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset

This repository does not include the raw files.

## How to run

1. Download the dataset from Kaggle and extract it
2. Create the folder `data/raw/` and place the EEG CSV files there
3. Open `notebooks/eeg_fuzzy_unlearning.ipynb` and run all cells

## Repository structure

```text
eeg-fuzzy-unlearning/
├── README.md
├── .gitignore
├── notebooks/
│   └── eeg_fuzzy_unlearning.ipynb
├── paper/
│   └── Final_Report_EEG_Unlearning_Torabi.pdf
└── data/
    └── README.md        (download instructions; put the raw CSVs in data/raw/)
```
