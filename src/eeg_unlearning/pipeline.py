"""End-to-end experiment: baseline vs. exact vs. fuzzy unlearning."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from eeg_unlearning.config import ExperimentConfig
from eeg_unlearning.data import WindowedDataset
from eeg_unlearning.metrics import agreement, forgetting, utility
from eeg_unlearning.unlearning import fuzzy_weights, make_forest


def _fit_timed(model, X, y, **fit_kwargs) -> float:
    start = time.perf_counter()
    model.fit(X, y, **fit_kwargs)
    return time.perf_counter() - start


def run_experiment(data: WindowedDataset, cfg: ExperimentConfig) -> dict[str, Any]:
    """Train all three models for one forgotten subject and one seed.

    Returns a nested dict of metrics per model plus fuzzy-vs-exact agreement.
    """
    subject = cfg.subject_to_forget
    if subject not in data.subjects:
        raise ValueError(f"Subject {subject} not in dataset (subjects: {data.subjects.tolist()})")

    tr_forget = data.y_train == subject
    te_forget = data.y_test == subject
    X_rtr, y_rtr = data.X_train[~tr_forget], data.y_train[~tr_forget]
    X_ftr = data.X_train[tr_forget]
    X_rte, y_rte = data.X_test[~te_forget], data.y_test[~te_forget]
    X_fte = data.X_test[te_forget]

    results: dict[str, Any] = {
        "subject_to_forget": subject,
        "seed": cfg.seed,
        "n_train_retain": int(len(X_rtr)),
        "n_train_forget": int(len(X_ftr)),
        "n_test_retain": int(len(X_rte)),
        "n_test_forget": int(len(X_fte)),
    }

    # Baseline: trained on everyone, including the subject to forget.
    X_btr = np.vstack([X_rtr, X_ftr])
    y_btr = np.concatenate([y_rtr, data.y_train[tr_forget]])
    base_scaler = StandardScaler().fit(X_btr)
    baseline = make_forest(cfg)
    fit_s = _fit_timed(baseline, base_scaler.transform(X_btr), y_btr)
    results["baseline"] = {
        **utility(y_rte, baseline.predict(base_scaler.transform(X_rte))),
        **forgetting(baseline, base_scaler.transform(X_fte), subject),
        "oob_score": float(baseline.oob_score_),
        "fit_seconds": fit_s,
    }

    # Unlearned models only ever see statistics of the retained subjects.
    scaler = StandardScaler().fit(X_rtr)
    Xs_rtr, Xs_rte = scaler.transform(X_rtr), scaler.transform(X_rte)
    Xs_ftr, Xs_fte = scaler.transform(X_ftr), scaler.transform(X_fte)

    exact = make_forest(cfg)
    fit_s = _fit_timed(exact, Xs_rtr, y_rtr)
    results["exact"] = {
        **utility(y_rte, exact.predict(Xs_rte)),
        **forgetting(exact, Xs_fte, subject),
        "oob_score": float(exact.oob_score_),
        "fit_seconds": fit_s,
    }

    start = time.perf_counter()
    w = fuzzy_weights(Xs_rtr, Xs_ftr, k=cfg.fuzzy_k)
    weight_s = time.perf_counter() - start
    fuzzy = make_forest(cfg)
    fit_s = _fit_timed(fuzzy, Xs_rtr, y_rtr, sample_weight=w)
    results["fuzzy"] = {
        **utility(y_rte, fuzzy.predict(Xs_rte)),
        **forgetting(fuzzy, Xs_fte, subject),
        "oob_score": float(fuzzy.oob_score_),
        "fit_seconds": fit_s + weight_s,
        "weight_min": float(w.min()),
        "weight_mean": float(w.mean()),
        "weight_max": float(w.max()),
    }

    results["fuzzy_vs_exact"] = {
        "retain": agreement(fuzzy, exact, Xs_rte),
        "forget": agreement(fuzzy, exact, Xs_fte),
    }
    return results


def flatten(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested result dict into one row, e.g. ``exact.accuracy``."""
    row: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            row[prefix] = value

    walk("", result)
    return row


def run_sweep(
    data: WindowedDataset,
    cfg: ExperimentConfig,
    subjects: list[int] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Repeat the experiment for every (forgotten subject, seed) pair.

    A single forgotten subject gives only a handful of test windows; sweeping
    over subjects and seeds turns the comparison into a distribution.
    """
    subjects = subjects if subjects is not None else data.subjects.tolist()
    seeds = seeds if seeds is not None else [cfg.seed]
    rows = [
        flatten(run_experiment(data, replace(cfg, subject_to_forget=s, seed=seed)))
        for s in subjects
        for seed in seeds
    ]
    return pd.DataFrame(rows)


SUMMARY_METRICS = [
    "accuracy",
    "macro_f1",
    "forget_hit_rate",
    "forget_max_confidence",
    "forget_entropy",
    "fit_seconds",
]


def summarize_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric per model."""
    table = {}
    for model in ("baseline", "exact", "fuzzy"):
        cols = {m: df[f"{model}.{m}"] for m in SUMMARY_METRICS}
        table[model] = {
            **{f"{m}_mean": s.mean() for m, s in cols.items()},
            **{f"{m}_std": s.std(ddof=0) for m, s in cols.items()},
        }
    return pd.DataFrame(table).T
