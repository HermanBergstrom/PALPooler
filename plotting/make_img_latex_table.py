"""Generate a LaTeX table from image PAL pooling results.

Columns: Dataset | n patches | CLS token | Mean pool | PAL acc | PAL fit time

PAL is evaluated at the iteration with the highest val_accuracy (if available),
otherwise at the last iteration (for runs with model_selection=last_iteration).
Fit time is cumulative fit_time_s up to (and including) that iteration.

When multiple result directories/variants are supplied (--results-dirs or
--pal-variants), each dataset gets one sub-row per run, with a thin rule between
dataset groups.  n_patches is only printed on the first sub-row.

n_patches is read from the results.json dataset field; it is the same across
seeds so the value from the first available seed is used.

Usage
-----
    # single variant
    python plotting/make_img_latex_table.py

    # two variants from the same directory
    python plotting/make_img_latex_table.py \\
        --results-dirs results/img_pal_pooling results/img_pal_pooling \\
        --pal-variants tvf_ms notvf_ms \\
        --run-labels TVF No-TVF

    python plotting/make_img_latex_table.py --metric auroc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


DATASETS = ["butterfly", "rsna"]
DATASET_LABELS: dict[str, str] = {
    "butterfly": "Butterfly",
    "rsna":      "RSNA",
}

DEFAULT_PAL_VARIANT = "tvf_ms"
DEFAULT_PAL_VARIANTS = ["tvf_ms", "tvf_ms_ds"]
DEFAULT_RUN_LABELS   = ["Full", "DS"]


def _load(path: Path) -> dict:
    raw = path.read_text()
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw)


def _iter_stages(stages: list[dict]) -> list[dict]:
    return [s for s in stages if s.get("tag", "").startswith("iter_")]


def _best_val_info(stages: list[dict], metric: str) -> Optional[tuple[float, float]]:
    """Return (test_metric, cumulative_fit_time_s) at best-val or last iteration."""
    test_key = "test_accuracy" if metric == "acc" else "test_auroc"
    iters = _iter_stages(stages)
    if not iters:
        return None

    val_iters = [s for s in iters if s.get("val_accuracy") is not None]
    if val_iters:
        best_idx = max(range(len(val_iters)), key=lambda i: val_iters[i]["val_accuracy"])
        chosen = val_iters[best_idx]
        # cumulative fit time over ALL iters up to and including this one in iters
        chosen_tag = chosen["tag"]
        idx_in_all = next(i for i, s in enumerate(iters) if s["tag"] == chosen_tag)
        cumulative_fit = sum(s.get("fit_time_s", 0.0) or 0.0 for s in iters[: idx_in_all + 1])
    else:
        # fall back to last iteration
        chosen = iters[-1]
        cumulative_fit = sum(s.get("fit_time_s", 0.0) or 0.0 for s in iters)

    return chosen.get(test_key), cumulative_fit


def _load_single(results_json: Path, metric: str) -> Optional[dict]:
    if not results_json.exists():
        return None
    data = _load(results_json)
    stages = data.get("stages", [])
    bsl = data.get("baselines", {})
    ds_info = data.get("dataset", {})

    test_key_cls  = "cls_token"  if metric == "acc" else "cls_token_auroc"
    test_key_mean = "mean_pool"  if metric == "acc" else "mean_pool_auroc"

    pal_info = _best_val_info(stages, metric)
    return {
        "cls":        bsl.get(test_key_cls),
        "mean_pool":  bsl.get(test_key_mean),
        "pal_test":   pal_info[0] if pal_info else None,
        "pal_fit_s":  pal_info[1] if pal_info else None,
        "n_patches":  ds_info.get("n_patches"),
        "n_train":    ds_info.get("n_train"),
    }


def _mean(vals: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else None


def load_row(results_dir: Path, dataset: str, metric: str, pal_variant: str = DEFAULT_PAL_VARIANT) -> Optional[dict]:
    """Load CLS, mean-pool, and PAL values for one dataset.

    Averages across seed_* subdirectories when present.
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
        n_patches = next((r["n_patches"] for r in seed_rows if r["n_patches"] is not None), None)
        n_train   = next((r["n_train"]   for r in seed_rows if r["n_train"]   is not None), None)
        return {
            "cls":       _mean([r["cls"]       for r in seed_rows]),
            "mean_pool": _mean([r["mean_pool"]  for r in seed_rows]),
            "pal_test":  _mean([r["pal_test"]   for r in seed_rows]),
            "pal_fit_s": _mean([r["pal_fit_s"]  for r in seed_rows]),
            "n_patches": n_patches,
            "n_train":   n_train,
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
                print(f"[warn] no results for '{ds}' in {rdir} (variant={variant})")
            runs.append((label, row))
        if any(r is not None for _, r in runs):
            dataset_runs.append((ds, runs))

    if not dataset_runs:
        raise SystemExit("No results found.")

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Image classification results. "
        r"PAL accuracy is taken at the best-validation iteration (or last "
        r"iteration when no validation split is used); "
        r"fit time is cumulative up to that iteration.}"
    )
    lines.append(r"  \label{tab:img_results}")
    lines.append(r"  \begin{tabular}{llclccccc}" if multi else r"  \begin{tabular}{llcccccc}")
    lines.append(r"    \toprule")
    if multi:
        lines.append(
            rf"    Dataset & $n_\text{{train}}$ & Patches & Run & CLS token & Mean pool"
            rf" & PAL ({metric_label}) & PAL fit time \\"
        )
    else:
        lines.append(
            rf"    Dataset & $n_\text{{train}}$ & Patches & CLS token & Mean pool"
            rf" & PAL ({metric_label}) & PAL fit time \\"
        )
    lines.append(r"    \midrule")

    n_cols = 8 if multi else 7

    for ds_idx, (ds, runs) in enumerate(dataset_runs):
        if ds_idx > 0:
            lines.append(rf"    \cmidrule{{1-{n_cols}}}")

        ds_label = DATASET_LABELS.get(ds, ds)
        # n_patches and n_train are dataset-level; read from first available row
        n_patches = next(
            (r["n_patches"] for _, r in runs if r is not None and r.get("n_patches") is not None),
            None,
        )
        n_train = next(
            (r["n_train"] for _, r in runs if r is not None and r.get("n_train") is not None),
            None,
        )
        patches_s = str(n_patches) if n_patches is not None else "--"
        train_s   = str(n_train)   if n_train   is not None else "--"

        for sub_idx, (run_label, row) in enumerate(runs):
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

            ds_col      = ds_label  if sub_idx == 0 else ""
            train_col   = train_s   if sub_idx == 0 else ""
            patches_col = patches_s if sub_idx == 0 else ""

            if multi:
                lines.append(
                    rf"    {ds_col} & {train_col} & {patches_col} & {run_label}"
                    rf" & {cls_s} & {mean_s} & {pal_s} & {time_s} \\"
                )
            else:
                lines.append(
                    rf"    {ds_col} & {train_col} & {patches_col}"
                    rf" & {cls_s} & {mean_s} & {pal_s} & {time_s} \\"
                )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LaTeX table for image PAL pooling results")
    p.add_argument(
        "--results-dirs", nargs="+", type=Path,
        default=[Path("results/img_pal_pooling")] * len(DEFAULT_PAL_VARIANTS),
        help="One or more results directories (one per run variant)",
    )
    p.add_argument(
        "--pal-variants", nargs="+", default=None,
        help=f"PAL variant suffix per run (default: {DEFAULT_PAL_VARIANTS}). "
             "Use when runs share the same directory but differ by suffix, "
             f"e.g. --pal-variants tvf_ms notvf_ms",
    )
    p.add_argument(
        "--run-labels", nargs="+", default=None,
        help=f"Display label for each run (default: {DEFAULT_RUN_LABELS})",
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
        labels = (DEFAULT_RUN_LABELS + ["Run3", "Run4", "Run5"])[:n_runs]

    pal_variants = args.pal_variants
    if pal_variants is None:
        pal_variants = (DEFAULT_PAL_VARIANTS + [DEFAULT_PAL_VARIANT] * 3)[:n_runs]
    elif len(pal_variants) != n_runs:
        raise SystemExit(
            f"--pal-variants ({len(pal_variants)}) must match "
            f"--results-dirs ({n_runs})"
        )

    table = make_table(dirs, labels, args.datasets, args.metric, pal_variants=pal_variants)
    print(table)
    if args.output:
        args.output.write_text(table + "\n")
        print(f"\nSaved → {args.output}")
