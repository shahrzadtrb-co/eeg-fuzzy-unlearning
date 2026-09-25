"""The three model settings: baseline, exact unlearning and fuzzy unlearning."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from eeg_unlearning.config import ExperimentConfig


def make_forest(cfg: ExperimentConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=cfg.seed,
    )


def fuzzy_weights(X_retain: np.ndarray, X_forget: np.ndarray, k: int = 20) -> np.ndarray:
    """Down-weight retained samples that look like the forgotten subject.

    For each retained sample, ``d`` is its mean Euclidean distance to the ``k``
    nearest forget-set samples. The weight is a Gaussian membership complement

        w = 1 - exp(-d^2 / (2 * sigma^2)),  sigma = median(d)

    so samples close to the forgotten subject get weights near 0 and distant
    samples get weights near 1.
    """
    k = min(k, len(X_forget))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X_forget)
    dist, _ = nn.kneighbors(X_retain)
    d = dist.mean(axis=1)
    sigma = np.median(d) + 1e-12
    return np.clip(1.0 - np.exp(-(d**2) / (2.0 * sigma**2)), 0.0, 1.0)
