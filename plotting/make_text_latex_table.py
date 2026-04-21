"""Generate a LaTeX table from text PAL pooling results.

Columns: Dataset | Avg tokens | CLS token | Mean pool | PAL acc | PAL fit time

PAL = liw_tvf variant, evaluated at the iteration with the highest val_accuracy.
Fit time is cumulative fit_time_s up to (and including) that iteration.

When multiple result directories are supplied (--results-dirs), each dataset gets
one sub-row per directory (row grouping), with a thin rule between dataset groups.
Avg. tokens is only printed on the first sub-row since it does not change.

Average token length is read from preprocessed HDF5 attention masks (all splits combined).
Falls back to hardcoded values if HDF5 files are not found.

Usage
-----
    # single training set
    python plotting/make_text_latex_table.py

    # two training sets in the same directory (different variant suffixes)
    python plotting/make_text_latex_table.py \\
        --results-dirs results/text_pal_pooling results/text_pal_pooling \\
        --pal-variants liw_tvf liw_tvf_ds_ms \\
        --run-labels Full 4k

    # two training sets in different directories
    python plotting/make_text_latex_table.py \\
        --results-dirs results/text_pal_pooling results/text_pal_pooling_4k \\
        --run-labels Full 4k

    python plotting/make_text_latex_table.py --metric auroc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


DATASETS = ["imdb", "20news", "ag_news", "yelp"]
DATASET_LABELS: dict[str, str] = {
    "imdb":    "IMDB",
    "20news":  "20 Newsgroups",
    "ag_news": "AG News",
    "yelp":    "Yelp",
}

DEFAULT_PAL_VARIANT = "liw_tvf"

HDF5_BASE = Path("/scratch/hermanb/temp_datasets/extracted_features")

# Fallback values (mean token length, all splits combined, ELECTRA tokenizer)
_AVG_TOKENS_FALLBACK: dict[str, float] = {
    "imdb":    273.3,
    "20news":  174.5,
    "ag_news":  53.1,
    "yelp":    129.2,
}


def avg_token_length(dataset: str) -> Optional[float]:
    """Compute mean token length from HDF5 attention masks; fall back to hardcoded value."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        return _AVG_TOKENS_FALLBACK.get(dataset)

    all_lengths = []
    for split in ("train", "val", "test"):
        h5_path = HDF5_BASE / dataset / "electra/preprocessed" / f"{split}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as hf:
            masks = hf["attention_masks"][:]
        all_lengths.append(masks.sum(axis=1))

    if not all_lengths:
        return _AVG_TOKENS_FALLBACK.get(dataset)

    import numpy as np
    return float(np.concatenate(all_lengths).mean())


def _load(path: Path) -> dict:
    raw = path.read_text()
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw)


def _iter_stages(stages: list[dict]) -> list[dict]:
    return [s for s in stages if s.get("tag", "").startswith("iter_")]


def _best_val_info(stages: list[dict], metric: str) -> Optional[tuple[float, float]]:
    """Return (test_metric, cumulative_fit_time_s) at best-val iteration."""
    test_key = "test_accuracy" if metric == "acc" else "test_auroc"
    iters = [
        s for s in _iter_stages(stages)
        if s.get("val_accuracy") is not None
    ]
    if not iters:
        return None
    best_idx = max(range(len(iters)), key=lambda i: iters[i]["val_accuracy"])
    test_val = iters[best_idx].get(test_key)
    cumulative_fit = sum(s.get("fit_time_s", 0.0) or 0.0 for s in iters[: best_idx + 1])
    return test_val, cumulative_fit


def _load_single(results_json: Path, metric: str) -> Optional[dict]:
    """Extract CLS, mean-pool, and best-val PAL info from one results.json."""
    if not results_json.exists():
        return None
    data = _load(results_json)
    stages = data.get("stages", [])
    bsl = data.get("baselines", {})

    test_key_cls  = "cls_token"  if metric == "acc" else "cls_token_auroc"
    test_key_mean = "mean_pool"  if metric == "acc" else "mean_pool_auroc"

    pal_info = _best_val_info(stages, metric)
    return {
        "cls":       bsl.get(test_key_cls),
        "mean_pool": bsl.get(test_key_mean),
        "pal_test":  pal_info[0] if pal_info else None,
        "pal_fit_s": pal_info[1] if pal_info else None,
    }


def _mean(vals: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else None


def load_row(results_dir: Path, dataset: str, metric: str, pal_variant: str = DEFAULT_PAL_VARIANT) -> Optional[dict]:
    """Load CLS, mean-pool, and PAL values for one dataset.

    If the experiment directory contains seed_* subdirectories, values are averaged
    across seeds. Best-val iteration selection is applied per seed before averaging.
    """
    exp_dir = results_dir / f"{dataset}_{pal_variant}"
    if not exp_dir.exists():
        return None

    seed_dirs = sorted(exp_dir.glob("seed_*"))
    if seed_dirs:
        seed_rows = [_load_single(sd / "results.json", metric) for sd in seed_dirs]
        seed_rows = [r for r in seed_rows if r is not None]
        if not seed_rows:
            return None
        return {
            "cls":       _mean([r["cls"]       for r in seed_rows]),
            "mean_pool": _mean([r["mean_pool"]  for r in seed_rows]),
            "pal_test":  _mean([r["pal_test"]   for r in seed_rows]),
            "pal_fit_s": _mean([r["pal_fit_s"]  for r in seed_rows]),
        }

    return _load_single(exp_dir / "results.json", metric)


def _fmt_pct(v: Optional[float], bold: bool = False) -> str:
    if v is None:
        return "--"
    s = f"{v * 100:.2f}"
    return rf"\textbf{{{s}}}" if bold else s


def _fmt_time(s: Optional[float]) -> str:
    if s is None:
        return "--"
    if s < 60:
        return f"{s:.0f}s"
    return f"{s / 60:.1f}m"


def make_table(
    results_dirs: list[Path],
    run_labels: list[str],
    datasets: list[str],
    metric: str,
    pal_variants: Optional[list[str]] = None,
) -> str:
    metric_label = "Accuracy (\\%)" if metric == "acc" else "AUROC (\\%)"
    multi = len(results_dirs) > 1
    variants = pal_variants or [DEFAULT_PAL_VARIANT] * len(results_dirs)

    # Collect data: dataset -> list of (label, row|None)
    dataset_runs: list[tuple[str, list[tuple[str, Optional[dict]]]]] = []
    for ds in datasets:
        runs = []
        for label, rdir, variant in zip(run_labels, results_dirs, variants):
            row = load_row(rdir, ds, metric, pal_variant=variant)
            if row is None:
                print(f"[warn] no results for '{ds}' in {rdir}")
            runs.append((label, row))
        if any(r is not None for _, r in runs):
            dataset_runs.append((ds, runs))

    if not dataset_runs:
        raise SystemExit("No results found.")

    # Sort datasets by descending average token length
    dataset_runs.sort(key=lambda x: avg_token_length(x[0]) or 0, reverse=True)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Text classification results. "
        r"PAL accuracy is taken at the best-validation iteration; "
        r"fit time is cumulative up to that iteration.}"
    )
    lines.append(r"  \label{tab:text_results}")
    lines.append(r"  \begin{tabular}{lclccccc}" if multi else r"  \begin{tabular}{lcccccc}")
    lines.append(r"    \toprule")
    if multi:
        lines.append(
            rf"    Dataset & Avg.\ tokens & Train set & CLS token & Mean pool"
            rf" & PAL ({metric_label}) & PAL fit time \\"
        )
    else:
        lines.append(
            rf"    Dataset & Avg.\ tokens & CLS token & Mean pool"
            rf" & PAL ({metric_label}) & PAL fit time \\"
        )
    lines.append(r"    \midrule")

    n_cols = 7 if multi else 6

    for ds_idx, (ds, runs) in enumerate(dataset_runs):
        if ds_idx > 0:
            lines.append(rf"    \cmidrule{{1-{n_cols}}}")

        ds_label  = DATASET_LABELS.get(ds, ds)
        avg_tok   = avg_token_length(ds)
        avg_tok_s = f"{avg_tok:.0f}" if avg_tok is not None else "--"

        for sub_idx, (run_label, row) in enumerate(runs):
            # Determine which accuracy column is highest for bolding
            acc_vals = {
                "cls":  row["cls"]       if row else None,
                "mean": row["mean_pool"] if row else None,
                "pal":  row["pal_test"]  if row else None,
            }
            valid_accs = {k: v for k, v in acc_vals.items() if v is not None}
            best_key = max(valid_accs, key=valid_accs.__getitem__) if valid_accs else None

            cls_s  = _fmt_pct(row["cls"],       bold=(best_key == "cls"))  if row else "--"
            mean_s = _fmt_pct(row["mean_pool"],  bold=(best_key == "mean")) if row else "--"
            pal_s  = _fmt_pct(row["pal_test"],   bold=(best_key == "pal"))  if row else "--"
            time_s = _fmt_time(row["pal_fit_s"]) if row else "--"

            # Dataset label and avg tokens only on the first sub-row
            ds_col  = ds_label  if sub_idx == 0 else ""
            tok_col = avg_tok_s if sub_idx == 0 else ""

            if multi:
                lines.append(
                    rf"    {ds_col} & {tok_col} & {run_label}"
                    rf" & {cls_s} & {mean_s} & {pal_s} & {time_s} \\"
                )
            else:
                lines.append(
                    rf"    {ds_col} & {tok_col}"
                    rf" & {cls_s} & {mean_s} & {pal_s} & {time_s} \\"
                )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LaTeX table for text PAL pooling results")
    p.add_argument(
        "--results-dirs", nargs="+", type=Path,
        default=[Path("results/text_pal_pooling")],
        help="One or more results directories (one per training-set variant)",
    )
    p.add_argument(
        "--pal-variants", nargs="+", default=None,
        help=f"PAL variant suffix per run (default: {DEFAULT_PAL_VARIANT} for all). "
             "Use this when runs share the same directory but differ by suffix, "
             f"e.g. --pal-variants liw_tvf liw_tvf_ds_ms",
    )
    p.add_argument(
        "--run-labels", nargs="+", default=None,
        help="Display label for each run (default: Full, 4k, ...)",
    )
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--metric", choices=["acc", "auroc"], default="acc")
    p.add_argument("--output", "-o", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dirs = args.results_dirs

    n_runs = len(dirs)

    if args.run_labels is not None:
        if len(args.run_labels) != n_runs:
            raise SystemExit(
                f"--run-labels ({len(args.run_labels)}) must match "
                f"--results-dirs ({n_runs})"
            )
        labels = args.run_labels
    else:
        default_labels = ["Full", "4k", "2k", "1k", "500"]
        labels = default_labels[:n_runs]

    pal_variants = args.pal_variants
    if pal_variants is not None and len(pal_variants) != n_runs:
        raise SystemExit(
            f"--pal-variants ({len(pal_variants)}) must match "
            f"--results-dirs ({n_runs})"
        )

    table = make_table(dirs, labels, args.datasets, args.metric, pal_variants=pal_variants)
    print(table)
    if args.output:
        args.output.write_text(table + "\n")
        print(f"\nSaved → {args.output}")
