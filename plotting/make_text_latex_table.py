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

    # two training sets (full + subsampled)
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

PAL_VARIANT = "liw_tvf"

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


def load_row(results_dir: Path, dataset: str, metric: str) -> Optional[dict]:
    """Load CLS, mean-pool, and PAL values for one dataset from one results directory."""
    path = results_dir / f"{dataset}_{PAL_VARIANT}" / "results.json"
    if not path.exists():
        return None
    data = _load(path)
    stages = data.get("stages", [])
    bsl = data.get("baselines", {})

    test_key_cls  = "cls_token"  if metric == "acc" else "cls_token_auroc"
    test_key_mean = "mean_pool"  if metric == "acc" else "mean_pool_auroc"

    cls_val  = bsl.get(test_key_cls)
    mean_val = bsl.get(test_key_mean)
    pal_info = _best_val_info(stages, metric)

    return {
        "cls":       cls_val,
        "mean_pool": mean_val,
        "pal_test":  pal_info[0] if pal_info else None,
        "pal_fit_s": pal_info[1] if pal_info else None,
    }


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v * 100:.2f}" if v is not None else "--"


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
) -> str:
    metric_label = "Accuracy (\\%)" if metric == "acc" else "AUROC (\\%)"
    multi = len(results_dirs) > 1

    # Collect data: dataset -> list of (label, row|None)
    dataset_runs: list[tuple[str, list[tuple[str, Optional[dict]]]]] = []
    for ds in datasets:
        runs = []
        for label, rdir in zip(run_labels, results_dirs):
            row = load_row(rdir, ds, metric)
            if row is None:
                print(f"[warn] no results for '{ds}' in {rdir}")
            runs.append((label, row))
        if any(r is not None for _, r in runs):
            dataset_runs.append((ds, runs))

    if not dataset_runs:
        raise SystemExit("No results found.")

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Text classification results. "
        r"PAL accuracy is taken at the best-validation iteration; "
        r"fit time is cumulative up to that iteration.}"
    )
    lines.append(r"  \label{tab:text_results}")
    lines.append(r"  \begin{tabular}{llccccc}")
    lines.append(r"    \toprule")
    if multi:
        lines.append(
            rf"    Dataset & Train set & Avg.\ tokens & CLS token & Mean pool"
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

        ds_label = DATASET_LABELS.get(ds, ds)
        avg_tok  = avg_token_length(ds)
        avg_tok_s = f"{avg_tok:.0f}" if avg_tok is not None else "--"

        for sub_idx, (run_label, row) in enumerate(runs):
            cls_s  = _fmt_pct(row["cls"])       if row else "--"
            mean_s = _fmt_pct(row["mean_pool"])  if row else "--"
            pal_s  = _fmt_pct(row["pal_test"])   if row else "--"
            time_s = _fmt_time(row["pal_fit_s"]) if row else "--"

            # Dataset label and avg tokens only on the first sub-row
            ds_col  = ds_label  if sub_idx == 0 else ""
            tok_col = avg_tok_s if sub_idx == 0 else ""

            if multi:
                lines.append(
                    rf"    {ds_col} & {run_label} & {tok_col}"
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
        "--run-labels", nargs="+", default=None,
        help="Display label for each results directory (default: Full, 4k, ...)",
    )
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--metric", choices=["acc", "auroc"], default="acc")
    p.add_argument("--output", "-o", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dirs = args.results_dirs

    if args.run_labels is not None:
        if len(args.run_labels) != len(dirs):
            raise SystemExit(
                f"--run-labels ({len(args.run_labels)}) must match "
                f"--results-dirs ({len(dirs)})"
            )
        labels = args.run_labels
    else:
        default_labels = ["Full", "4k", "2k", "1k", "500"]
        labels = default_labels[: len(dirs)]

    table = make_table(dirs, labels, args.datasets, args.metric)
    print(table)
    if args.output:
        args.output.write_text(table + "\n")
        print(f"\nSaved → {args.output}")
