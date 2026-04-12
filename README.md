# Privacy-Preserving EEG Classification with Fuzzy Unlearning

This project presents an end-to-end EEG subject-level unlearning pipeline using a public EEG dataset from Kaggle. It compares three settings: baseline training, exact retraining after subject removal, and fuzzy unlearning through similarity-based sample weighting.

## Project goal

EEG signals can contain subject-specific patterns that make them privacy-sensitive. This project studies whether the influence of one selected subject can be reduced in a trained model while preserving classification utility for the remaining subjects.

## What this project does

The notebook performs the following steps:

1. Loads multi-subject EEG recordings from CSV files
2. Assigns the 19 standard EEG channel names
3. Splits each subject recording into train and test portions
4. Segments the signals into sliding windows
5. Extracts statistical and spectral features from each window
6. Builds retain and forget datasets
7. Trains a baseline Random Forest classifier
8. Trains an exact unlearning model by removing the forgotten subject
9. Trains a fuzzy unlearning model using similarity-based sample weighting
10. Compares the models in terms of retained-subject utility and forgotten-subject uncertainty

## Dataset

This project uses the public Complete EEG dataset hosted on Kaggle.

Source:  
Aman Anand, Complete EEG dataset, Kaggle  
https://www.kaggle.com/datasets/amananandrai/complete-eeg-dataset

The raw dataset files are not included in this repository. To run the notebook:

1. Download the dataset manually from the Kaggle page
2. Extract the files
3. Place the EEG CSV files inside `data/raw/`

## Repository structure

```text
eeg-fuzzy-unlearning/
├── .gitignore
├── README.md
├── notebook/
│   └── eeg_fuzzy_unlearning.ipynb
├── paper/
│   └── Final_Report_EEG_Unlearning_Torabi.pdf
├── data/
│   ├── README.md
│   └── raw/
└── figures/
