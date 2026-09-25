"""Sliding-window feature extraction: per-channel statistics and band power."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


def band_powers(window: np.ndarray, fs: float, bands: dict[str, tuple[float, float]]) -> np.ndarray:
    """Band power per channel via Welch's PSD, integrated over each band.

    ``window`` has shape (samples, channels). Returns shape (channels, n_bands).
    The PSD is computed once per window and reused for every band.
    """
    nperseg = min(256, window.shape[0])
    freqs, psd = welch(window, fs=fs, nperseg=nperseg, axis=0)
    out = np.empty((window.shape[1], len(bands)))
    for b, (low, high) in enumerate(bands.values()):
        mask = (freqs >= low) & (freqs <= high)
        out[:, b] = trapezoid(psd[mask], freqs[mask], axis=0)
    return out


def feature_names(channel_names: list[str], bands: dict[str, tuple[float, float]]) -> list[str]:
    names = []
    for ch in channel_names:
        names += [f"{ch}_{stat}" for stat in ("mean", "std", "var", "min", "max")]
        names += [f"{ch}_bp_{band}" for band in bands]
    return names


def window_features(
    window: np.ndarray, fs: float, bands: dict[str, tuple[float, float]]
) -> np.ndarray:
    """Feature vector for one window, ordered as :func:`feature_names`."""
    stats = np.stack(
        [window.mean(0), window.std(0), window.var(0), window.min(0), window.max(0)], axis=1
    )
    return np.hstack([stats, band_powers(window, fs, bands)]).ravel()


def extract_windows(
    eeg: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    win_len: int,
    step: int,
) -> np.ndarray:
    """Slide a window over ``eeg`` (samples, channels) and stack feature vectors."""
    starts = range(0, eeg.shape[0] - win_len + 1, step)
    rows = [window_features(eeg[s : s + win_len], fs, bands) for s in starts]
    n_features = eeg.shape[1] * (5 + len(bands))
    return np.vstack(rows) if rows else np.empty((0, n_features))
