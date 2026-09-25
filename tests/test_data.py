import numpy as np
import pandas as pd
import pytest

from eeg_unlearning.data import build_dataset, load_recordings


def test_load_recordings_reads_subject_files_and_trims_channels(tmp_path):
    rng = np.random.default_rng(0)
    for sid in (0, 7):
        path = tmp_path / f"s{sid:02d}.csv"
        pd.DataFrame(rng.random((50, 21))).to_csv(path, header=False, index=False)
    (tmp_path / "notes.csv").write_text("ignored")
    recs = load_recordings(tmp_path, n_channels=19)
    assert sorted(recs) == [0, 7]
    assert recs[7].shape == (50, 19)


def test_load_recordings_explains_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_recordings(tmp_path, n_channels=19)


def test_split_is_temporal_and_balanced(dataset, recordings):
    n_subjects = len(recordings)
    assert dataset.fs == 128
    assert sorted(dataset.subjects) == list(range(n_subjects))
    # 48 s train at 2 s windows / 1.5 s step -> 31; 12 s test at 2 s non-overlapping -> 6
    assert np.all(np.bincount(dataset.y_train) == 31)
    assert np.all(np.bincount(dataset.y_test) == 6)


def test_paper_setup_window_counts(cfg):
    """36 recordings of 31000 rows x 19 channels, as in the notebook: 1116 train windows."""
    recs = {s: np.random.default_rng(s).normal(size=(31000, 19)) for s in range(36)}
    data = build_dataset(recs, cfg)
    assert data.fs == 517
    assert len(data.X_train) == 1116
    assert np.sum(data.y_train != 14) == 1085
    assert np.sum(data.y_test != 14) == 175
    assert np.sum(data.y_test == 14) == 5
