"""Experiment configuration.

Defaults reproduce the setup reported in the paper and the notebook.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# fmt: off
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8",
    "T3", "T4", "C3", "C4", "T5", "T6",
    "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]
# fmt: on

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class ExperimentConfig:
    data_dir: str = "data/raw"
    output_dir: str = "results"

    # Recordings are 60 s long; the sampling rate is estimated from the row count
    # unless set explicitly.
    recording_seconds: float = 60.0
    sampling_rate: float | None = None

    window_seconds: float = 2.0
    step_seconds: float = 1.5
    train_ratio: float = 0.8

    subject_to_forget: int = 14
    seed: int = 42

    # Random Forest hyperparameters shared by all three models.
    n_estimators: int = 120
    max_depth: int = 8
    min_samples_split: int = 20
    min_samples_leaf: int = 8

    # Fuzzy unlearning: number of forget-set neighbours used for the distance.
    fuzzy_k: int = 20

    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(EEG_BANDS))
    channel_names: list[str] = field(default_factory=lambda: list(CHANNEL_NAMES))

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> ExperimentConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        if "bands" in raw:
            raw["bands"] = {k: tuple(v) for k, v in raw["bands"].items()}
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
