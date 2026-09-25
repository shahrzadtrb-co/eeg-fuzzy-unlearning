"""Command-line entry point: ``eeg-unlearn run`` and ``eeg-unlearn sweep``."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from eeg_unlearning.config import ExperimentConfig
from eeg_unlearning.data import build_dataset, load_recordings, synthetic_recordings
from eeg_unlearning.pipeline import run_experiment, run_sweep, summarize_sweep
from eeg_unlearning.plots import plot_summary

log = logging.getLogger("eeg_unlearning")


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    overrides = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "subject_to_forget": getattr(args, "subject", None),
        "seed": getattr(args, "seed", None),
    }
    if args.config:
        return ExperimentConfig.from_yaml(args.config, **overrides)
    return replace(ExperimentConfig(), **{k: v for k, v in overrides.items() if v is not None})


def _load_data(cfg: ExperimentConfig, synthetic: bool):
    if synthetic:
        log.info("Using synthetic EEG (for smoke tests only, not real results)")
        recordings = synthetic_recordings(n_channels=len(cfg.channel_names), seed=cfg.seed)
    else:
        recordings = load_recordings(cfg.data_dir, len(cfg.channel_names))
    log.info("Loaded %d subjects; extracting features", len(recordings))
    data = build_dataset(recordings, cfg)
    log.info(
        "fs=%.0f Hz, %d train / %d test windows, %d features",
        data.fs,
        len(data.X_train),
        len(data.X_test),
        data.X_train.shape[1],
    )
    return data


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    data = _load_data(cfg, args.synthetic)
    result = run_experiment(data, cfg)
    result["config"] = cfg.to_dict()
    result["synthetic_data"] = args.synthetic

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(result, indent=2))
    plot_summary(result, out / "summary.png")

    print(f"\nForgot subject {cfg.subject_to_forget} (seed {cfg.seed})")
    print(f"{'model':<10}{'acc':>8}{'macroF1':>9}{'hit':>7}{'conf':>7}{'entropy':>9}{'fit s':>8}")
    for m in ("baseline", "exact", "fuzzy"):
        r = result[m]
        print(
            f"{m:<10}{r['accuracy']:>8.4f}{r['macro_f1']:>9.4f}{r['forget_hit_rate']:>7.2f}"
            f"{r['forget_max_confidence']:>7.3f}{r['forget_entropy']:>9.3f}{r['fit_seconds']:>8.2f}"
        )
    print(f"\nWrote {out / 'run.json'} and {out / 'summary.png'}")


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    data = _load_data(cfg, args.synthetic)
    subjects = args.subjects or data.subjects.tolist()
    seeds = args.seeds or [cfg.seed]
    log.info("Sweeping %d subjects x %d seeds", len(subjects), len(seeds))
    df = run_sweep(data, cfg, subjects=subjects, seeds=seeds)
    summary = summarize_sweep(df)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep_runs.csv", index=False)
    summary.to_csv(out / "sweep_summary.csv")
    print(summary.round(4).to_string())
    print(f"\nWrote {out / 'sweep_runs.csv'} and {out / 'sweep_summary.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eeg-unlearn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML config (see configs/default.yaml)")
    common.add_argument("--data-dir", help="folder with s00.csv ... sNN.csv")
    common.add_argument("--output-dir", help="where to write results")
    common.add_argument(
        "--synthetic", action="store_true", help="use generated EEG instead of the dataset"
    )

    run = sub.add_parser("run", parents=[common], help="one forgotten subject, one seed")
    run.add_argument("--subject", type=int, help="subject id to forget")
    run.add_argument("--seed", type=int)
    run.set_defaults(func=cmd_run)

    sweep = sub.add_parser("sweep", parents=[common], help="repeat over subjects and seeds")
    sweep.add_argument("--subjects", type=int, nargs="+", help="default: every subject")
    sweep.add_argument("--seeds", type=int, nargs="+", help="default: the config seed")
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
