"""Figures for a single run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ("baseline", "exact", "fuzzy")
LABELS = ("Baseline", "Exact retraining", "Fuzzy unlearning")


def plot_summary(result: dict[str, Any], path: str | Path) -> Path:
    """Two panels: utility on retained subjects, behaviour on the forgotten one."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(MODELS))
    width = 0.38

    acc = [result[m]["accuracy"] for m in MODELS]
    f1 = [result[m]["macro_f1"] for m in MODELS]
    left.bar([i - width / 2 for i in x], acc, width, label="Accuracy", color="#3b6ea5")
    left.bar([i + width / 2 for i in x], f1, width, label="Macro F1", color="#8fb3d9")
    left.set_ylim(0.0, 1.05)
    left.set_title("Utility on retained subjects (higher is better)")

    hit = [result[m]["forget_hit_rate"] for m in MODELS]
    conf = [result[m]["forget_max_confidence"] for m in MODELS]
    right.bar([i - width / 2 for i in x], hit, width, label="Still identified", color="#b5523b")
    right.bar([i + width / 2 for i in x], conf, width, label="Max confidence", color="#e0a488")
    right.set_ylim(0.0, 1.05)
    right.set_title(f"Forgotten subject {result['subject_to_forget']} (lower is better)")

    for ax in (left, right):
        ax.set_xticks(list(x), LABELS)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
