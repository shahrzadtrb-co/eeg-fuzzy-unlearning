import pytest

from eeg_unlearning.config import ExperimentConfig
from eeg_unlearning.data import build_dataset, synthetic_recordings


@pytest.fixture(scope="session")
def cfg():
    return ExperimentConfig(subject_to_forget=3, n_estimators=40)


@pytest.fixture(scope="session")
def recordings():
    return synthetic_recordings(n_subjects=8, fs=128, seconds=60.0, seed=1)


@pytest.fixture(scope="session")
def dataset(recordings, cfg):
    return build_dataset(recordings, cfg)
