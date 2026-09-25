import numpy as np
from scipy.signal import welch

from eeg_unlearning.config import CHANNEL_NAMES, EEG_BANDS
from eeg_unlearning.features import band_powers, extract_windows, feature_names, window_features


def notebook_features(window, fs, bands):
    """Per-channel loop from the original notebook, kept as a reference."""
    out = []
    for ch in range(window.shape[1]):
        s = window[:, ch]
        out += [s.mean(), s.std(), s.var(), s.min(), s.max()]
        for low, high in bands.values():
            freqs, psd = welch(s, fs=fs, nperseg=min(256, len(s)))
            m = (freqs >= low) & (freqs <= high)
            out.append(np.sum((psd[m][1:] + psd[m][:-1]) / 2 * np.diff(freqs[m])))
    return np.array(out)


def test_vectorised_features_match_notebook_reference():
    rng = np.random.default_rng(0)
    window = rng.normal(size=(1034, 19))
    np.testing.assert_allclose(
        window_features(window, 517, EEG_BANDS), notebook_features(window, 517, EEG_BANDS)
    )


def test_band_power_peaks_in_the_right_band():
    fs = 256
    t = np.arange(4 * fs) / fs
    alpha = np.sin(2 * np.pi * 10 * t)[:, None]
    powers = band_powers(alpha, fs, EEG_BANDS)[0]
    assert list(EEG_BANDS)[int(np.argmax(powers))] == "alpha"


def test_feature_names_line_up_with_vector():
    names = feature_names(CHANNEL_NAMES, EEG_BANDS)
    vec = window_features(np.ones((256, 19)), 128, EEG_BANDS)
    assert len(names) == len(vec) == 19 * 10
    assert names[:6] == ["Fp1_mean", "Fp1_std", "Fp1_var", "Fp1_min", "Fp1_max", "Fp1_bp_delta"]


def test_window_count():
    eeg = np.zeros((1000, 3))
    assert extract_windows(eeg, 100, EEG_BANDS, win_len=200, step=150).shape == (6, 30)
    assert extract_windows(eeg[:100], 100, EEG_BANDS, win_len=200, step=150).shape == (0, 30)
