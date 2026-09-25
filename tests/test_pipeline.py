import json

import pytest

from eeg_unlearning.cli import main
from eeg_unlearning.pipeline import run_experiment, run_sweep, summarize_sweep


def test_run_experiment_shows_forgetting(dataset, cfg):
    r = run_experiment(dataset, cfg)
    assert r["baseline"]["forget_hit_rate"] > 0.5
    # A model that never saw the subject cannot predict it.
    assert r["exact"]["forget_hit_rate"] == 0.0
    assert r["fuzzy"]["forget_hit_rate"] == 0.0
    assert r["exact"]["forget_max_confidence"] < r["baseline"]["forget_max_confidence"]
    for model in ("baseline", "exact", "fuzzy"):
        assert 0.0 <= r[model]["accuracy"] <= 1.0
    assert 0.0 <= r["fuzzy_vs_exact"]["retain"]["prediction_agreement"] <= 1.0


def test_run_experiment_is_reproducible(dataset, cfg):
    a, b = run_experiment(dataset, cfg), run_experiment(dataset, cfg)
    for model in ("baseline", "exact", "fuzzy"):
        assert a[model]["accuracy"] == b[model]["accuracy"]
        assert a[model]["macro_f1"] == b[model]["macro_f1"]
    # Parallel trees sum probabilities in a varying order, so allow float noise.
    diff_a = a["fuzzy_vs_exact"]["retain"]["mean_abs_proba_diff"]
    assert diff_a == pytest.approx(b["fuzzy_vs_exact"]["retain"]["mean_abs_proba_diff"])


def test_unknown_subject_is_rejected(dataset, cfg):
    from dataclasses import replace

    with pytest.raises(ValueError, match="not in dataset"):
        run_experiment(dataset, replace(cfg, subject_to_forget=99))


def test_sweep_covers_every_subject_and_seed(dataset, cfg):
    df = run_sweep(dataset, cfg, subjects=[0, 1, 2], seeds=[0, 1])
    assert len(df) == 6
    summary = summarize_sweep(df)
    assert list(summary.index) == ["baseline", "exact", "fuzzy"]
    assert summary.loc["exact", "forget_hit_rate_mean"] == 0.0


def test_cli_run_writes_results(tmp_path):
    main(["run", "--synthetic", "--output-dir", str(tmp_path)])
    result = json.loads((tmp_path / "run.json").read_text())
    assert result["synthetic_data"] is True
    assert result["subject_to_forget"] == 14
    assert (tmp_path / "summary.png").stat().st_size > 0
