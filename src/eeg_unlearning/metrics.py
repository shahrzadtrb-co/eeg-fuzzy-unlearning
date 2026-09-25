"""Utility and forgetting metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def utility(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def forgetting(model, X_forget: np.ndarray, subject: int) -> dict[str, float]:
    """How strongly the model still recognises the forgotten subject.

    ``forget_hit_rate`` is the fraction of forget-subject windows still
    classified as that subject (0 is ideal after unlearning). Mean max
    confidence and mean entropy describe how certain the model is on them.
    """
    proba = model.predict_proba(X_forget)
    pred = model.classes_[proba.argmax(axis=1)]
    return {
        "forget_hit_rate": float(np.mean(pred == subject)),
        "forget_max_confidence": float(proba.max(axis=1).mean()),
        "forget_entropy": float(entropy(proba).mean()),
    }


def agreement(model_a, model_b, X: np.ndarray) -> dict[str, float]:
    """Prediction agreement and mean absolute probability gap between two models."""
    if not np.array_equal(model_a.classes_, model_b.classes_):
        raise ValueError("Models were trained on different class sets")
    pa, pb = model_a.predict_proba(X), model_b.predict_proba(X)
    return {
        "prediction_agreement": float(np.mean(pa.argmax(1) == pb.argmax(1))),
        "mean_abs_proba_diff": float(np.mean(np.abs(pa - pb))),
    }
