"""Subject-level machine unlearning for EEG classification.

Compares three settings on a multi-subject EEG identification task:
baseline training, exact unlearning (retraining without the subject) and
fuzzy unlearning (similarity-based sample reweighting).
"""

from eeg_unlearning.config import ExperimentConfig
from eeg_unlearning.pipeline import run_experiment, run_sweep

__all__ = ["ExperimentConfig", "run_experiment", "run_sweep"]
__version__ = "1.0.0"
