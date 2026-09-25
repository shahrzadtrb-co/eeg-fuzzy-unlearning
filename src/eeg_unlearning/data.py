"""Loading EEG recordings and building windowed train/test feature sets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_unlearning.config import ExperimentConfig
from eeg_unlearning.features import extract_windows

_SUBJECT_FILE = re.compile(r"^s(\d+)\.csv$")


@dataclass
class WindowedDataset:
    """Feature matrices for every subject, split in time into train and test."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    fs: float

    @property
    def subjects(self) -> np.ndarray:
        return np.unique(self.y_train)


def load_recordings(data_dir: str | Path, n_channels: int) -> dict[int, np.ndarray]:
    """Read ``sNN.csv`` files into {subject_id: array (samples, channels)}."""
    data_dir = Path(data_dir)
    recordings = {}
    for path in sorted(data_dir.glob("s*.csv")):
        match = _SUBJECT_FILE.match(path.name)
        if not match:
            continue
        df = pd.read_csv(path, header=None).iloc[:, :n_channels]
        recordings[int(match.group(1))] = df.to_numpy(dtype=float)
    if not recordings:
        raise FileNotFoundError(
            f"No subject files (s00.csv, s01.csv, ...) found in {data_dir.resolve()}. "
            "Download the dataset as described in data/README.md, or use --synthetic."
        )
    return recordings


def build_dataset(recordings: dict[int, np.ndarray], cfg: ExperimentConfig) -> WindowedDataset:
    """Split each recording in time, then window it and extract features.

    The first ``train_ratio`` of every recording is used for training with
    overlapping windows; the remainder is the test set with non-overlapping
    windows, so no test window shares samples with a training window.
    """
    first = next(iter(recordings.values()))
    fs = cfg.sampling_rate or round(first.shape[0] / cfg.recording_seconds)
    win_len = int(cfg.window_seconds * fs)
    train_step = int(cfg.step_seconds * fs)
    test_step = win_len

    parts: dict[str, list] = {"Xtr": [], "ytr": [], "Xte": [], "yte": []}
    for subject, eeg in sorted(recordings.items()):
        split = int(eeg.shape[0] * cfg.train_ratio)
        Xtr = extract_windows(eeg[:split], fs, cfg.bands, win_len, train_step)
        Xte = extract_windows(eeg[split:], fs, cfg.bands, win_len, test_step)
        parts["Xtr"].append(Xtr)
        parts["ytr"].append(np.full(len(Xtr), subject))
        parts["Xte"].append(Xte)
        parts["yte"].append(np.full(len(Xte), subject))

    return WindowedDataset(
        X_train=np.vstack(parts["Xtr"]),
        y_train=np.concatenate(parts["ytr"]),
        X_test=np.vstack(parts["Xte"]),
        y_test=np.concatenate(parts["yte"]),
        fs=fs,
    )


def synthetic_recordings(
    n_subjects: int = 36,
    n_channels: int = 19,
    fs: int = 128,
    seconds: float = 60.0,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Generate EEG-like signals with a subject-specific spectral fingerprint.

    Each subject gets its own mix of delta/theta/alpha/beta/gamma oscillations
    per channel plus pink-ish noise. Used for tests and for running the
    pipeline end to end without the Kaggle dataset; results on this data say
    nothing about real EEG.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    centre_freqs = np.array([2.0, 6.0, 10.0, 20.0, 38.0])
    recordings = {}
    for subject in range(n_subjects):
        amps = rng.uniform(0.2, 2.0, size=(n_channels, len(centre_freqs)))
        freqs = centre_freqs * rng.uniform(0.85, 1.15, size=len(centre_freqs))
        phases = rng.uniform(0, 2 * np.pi, size=(n_channels, len(centre_freqs)))
        signal = np.einsum(
            "cb,bct->tc",
            amps,
            np.sin(2 * np.pi * freqs[:, None, None] * t[None, None, :] + phases.T[:, :, None]),
        )
        noise = np.cumsum(rng.normal(size=(n, n_channels)), axis=0)
        noise -= noise.mean(0)
        noise /= noise.std(0)
        recordings[subject] = signal + 0.8 * noise + rng.normal(scale=0.5, size=(n, n_channels))
    return recordings
