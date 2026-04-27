"""Generate a unified LaTeX table combining text and image PAL pooling results.

The table has two sections:
  1. Text classification datasets (imdb, 20news, ag_news, yelp)
  2. Image classification datasets (rsna, butterfly, coco, open-images)

Each dataset shows one sub-row per training-fraction variant (20% and 100%).
The "Info" column shows avg. token length for text and patch count for image.

PAL is evaluated at the iteration with the highest val_accuracy (or last
iteration when no validation split is used).  Fit time is cumulative up to
that iteration.

Usage
-----
    # default: pgs16 only, both text + image datasets
    python plotting/make_latex_table.py

    # show both pgs variants for image
    python plotting/make_latex_table.py --img-pgs-variants pgs16 pgs1

    # AUROC metric
    python plotting/make_latex_table.py --metric auroc

    # save to file
    python plotting/make_latex_table.py -o paper/table.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


TEXT_DATASETS: list[str] = ["imdb", "20news", "ag_news", "yelp"]
IMG_DATASETS:  list[str] = ["rsna", "butterfly", "coco", "open-images"]

DATASET_LABELS: dict[str, str] = {
    "imdb":        "IMDB",
    "20news":      "20 Newsgroups",
    "ag_news":     "AG News",
    "yelp":        "Yelp",
    "rsna":        "RSNA",
    "butterfly":   "Butterfly",
    "coco":        "COCO",
    "open-images": "Open Images",
}

# (experiment-dir suffix, display fraction string shown in table)
NTRAIN_VARIANTS: list[tuple[str, str]] = [("n02", "20"), ("nfull", "100")]

DEFAULT_IMG_PGS_VARIANTS: list[str] = ["pgs16"]

HDF5_BASE = Path("/scratch/hermanb/temp_datasets/extracted_features")
_AVG_TOKENS_FALLBACK: dict[str, float] = {
    "imdb":    273.3,
    "20news":  174.5,
    "ag_news":  53.1,
    "yelp":    129.2,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _avg_token_length(dataset: str) -> Optional[float]:
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


def _load_json(path: Path) -> dict:
    raw = path.read_text()
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw)


def _best_val_info(
    stages: list[dict], metric: str
) -> Optional[tuple[Optional[float], float]]:
    """Return (test_metric, cumulative_fit_s) at best-val or last iteration."""
    test_key = "test_accuracy" if metric == "acc" else "test_auroc"
    iters = [s for s in stages if s.get("tag", "").startswith("iter_")]
    if not iters:
        return None
    val_iters = [s for s in iters if s.get("val_accuracy") is not None]
    if val_iters:
        best_idx = max(range(len(val_iters)), key=lambda i: val_iters[i]["val_accuracy"])
        chosen_tag = val_iters[best_idx]["tag"]
        idx_in_all = next(i for i, s in enumerate(iters) if s["tag"] == chosen_tag)
        cum_fit = sum(s.get("fit_time_s", 0.0) or 0.0 for s in iters[: idx_in_all + 1])
        return val_iters[best_idx].get(test_key), cum_fit
    cum_fit = sum(s.get("fit_time_s", 0.0) or 0.0 for s in iters)
    return iters[-1].get(test_key), cum_fit


def _parse_results_json(path: Path, metric: str) -> Optional[dict]:
    if not path.exists():
        return None
    data = _load_json(path)
    bsl     = data.get("baselines", {})
    ds_info = data.get("dataset", {})
    cls_key  = "cls_token"  if metric == "acc" else "cls_token_auroc"
    mean_key = "mean_pool"  if metric == "acc" else "mean_pool_auroc"
    pal_info = _best_val_info(data.get("stages", []), metric)
    return {
        "cls":       bsl.get(cls_key),
        "mean_pool": bsl.get(mean_key),
        "pal_test":  pal_info[0] if pal_info else None,
        "pal_fit_s": pal_info[1] if pal_info else None,
        "n_patches": ds_info.get("n_patches"),
    }


def _mean(vals: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else None


def _std(vals: list[Optional[float]]) -> Optional[float]:
    valid = [v for v in vals if v is not None]
    if len(valid) < 2:
        return None
    mu = sum(valid) / len(valid)
    return (sum((x - mu) ** 2 for x in valid) / (len(valid) - 1)) ** 0.5


def _load_exp(exp_dir: Path, metric: str) -> Optional[dict]:
    """Average (and std) across seed_* subdirs when present; otherwise read results.json directly."""
    if not exp_dir.exists():
        return None
    seed_dirs = sorted(exp_dir.glob("seed_*"))
    if seed_dirs:
        rows = [_parse_results_json(sd / "results.json", metric) for sd in seed_dirs]
        rows = [r for r in rows if r is not None]
        if not rows:
            return None
        n_patches = next(
            (r["n_patches"] for r in rows if r.get("n_patches") is not None), None
        )
        return {
            "cls":           _mean([r["cls"]       for r in rows]),
            "cls_std":       _std( [r["cls"]       for r in rows]),
            "mean_pool":     _mean([r["mean_pool"]  for r in rows]),
            "mean_pool_std": _std( [r["mean_pool"]  for r in rows]),
            "pal_test":      _mean([r["pal_test"]   for r in rows]),
            "pal_test_std":  _std( [r["pal_test"]   for r in rows]),
            "pal_fit_s":     _mean([r["pal_fit_s"]  for r in rows]),
            "n_patches":     n_patches,
        }
    return _parse_results_json(exp_dir / "results.json", metric)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_pct(v: Optional[float], std: Optional[float] = None, bold: bool = False) -> str:
    if v is None:
        return "--"
    mean_s = f"{v * 100:.2f}"
    if std is not None:
        std_s = f"{std * 100:.2f}"
        inner = rf"{mean_s} {{\scriptsize $\pm${std_s}}}"
    else:
        inner = mean_s
    return rf"\textbf{{{inner}}}" if bold else inner


def _fmt_time(s: Optional[float]) -> str:
    if s is None:
        return "--"
    return f"{s:.0f}s" if s < 60 else f"{s / 60:.1f}m"


def _data_cols(row: Optional[dict]) -> tuple[str, str, str, str]:
    """Return (cls_s, mean_s, pal_s, time_s) with the best metric bolded."""
    if row is None:
        return "--", "--", "--", "--"
    acc_vals = {
        k: row[k] for k in ("cls", "mean_pool", "pal_test") if row.get(k) is not None
    }
    best_key = max(acc_vals, key=acc_vals.__getitem__) if acc_vals else None
    return (
        _fmt_pct(row.get("cls"),       row.get("cls_std"),       bold=(best_key == "cls")),
        _fmt_pct(row.get("mean_pool"),  row.get("mean_pool_std"), bold=(best_key == "mean_pool")),
        _fmt_pct(row.get("pal_test"),   row.get("pal_test_std"),  bold=(best_key == "pal_test")),
        _fmt_time(row.get("pal_fit_s")),
    )


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def make_table(
    text_results_dir: Path,
    img_results_dir: Path,
    metric: str,
    text_datasets: list[str],
    img_datasets: list[str],
    img_pgs_variants: list[str],
    ntrain_variants: list[tuple[str, str]],
) -> str:
    n_cols = 7

    # Sort text datasets by descending avg token length
    text_datasets = sorted(
        text_datasets, key=lambda d: _avg_token_length(d) or 0, reverse=True
    )

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Text and image classification results. "
        r"PAL accuracy is taken at the best-validation iteration "
        r"(or last iteration when no validation split is used); "
        r"fit time is cumulative up to that iteration. "
        r"Train (\%) is the fraction of the full training set used. "
        r"Info: avg.\ token length for text datasets, "
        r"number of patches for image datasets.}"
    )
    lines.append(r"  \label{tab:combined_results}")
    lines.append(r"  \begin{tabular}{llccccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Dataset & Info & Train (\%) & CLS token & Mean pool & PAL & Fit time \\"
    )
    lines.append(r"    \midrule")

    # === Text section ===
    lines.append(rf"    \multicolumn{{{n_cols}}}{{l}}{{\textit{{Text classification}}}} \\")
    lines.append(r"    \addlinespace[2pt]")

    for ds_idx, ds in enumerate(text_datasets):
        if ds_idx > 0:
            lines.append(rf"    \cmidrule{{1-{n_cols}}}")

        ds_label = DATASET_LABELS.get(ds, ds)
        avg_tok  = _avg_token_length(ds)
        info_s   = f"{avg_tok:.0f}" if avg_tok is not None else "--"

        for sub_idx, (ntrain_suffix, frac_pct) in enumerate(ntrain_variants):
            row = _load_exp(text_results_dir / f"{ds}_{ntrain_suffix}", metric)
            if row is None:
                print(f"[warn] no text results for '{ds}' variant '{ntrain_suffix}'")
            cls_s, mean_s, pal_s, time_s = _data_cols(row)
            ds_col   = ds_label if sub_idx == 0 else ""
            info_col = info_s   if sub_idx == 0 else ""
            lines.append(
                rf"    {ds_col} & {info_col} & {frac_pct}\% & {cls_s} & {mean_s} & {pal_s} & {time_s} \\"
            )

    # === Image section ===
    lines.append(r"    \midrule")
    lines.append(rf"    \multicolumn{{{n_cols}}}{{l}}{{\textit{{Image classification}}}} \\")
    lines.append(r"    \addlinespace[2pt]")

    for ds_idx, ds in enumerate(img_datasets):
        if ds_idx > 0:
            lines.append(rf"    \cmidrule{{1-{n_cols}}}")

        ds_label     = DATASET_LABELS.get(ds, ds)
        first_ds_row = True

        for pgs_variant in img_pgs_variants:
            # Resolve n_patches for this (dataset, pgs_variant) from first available seed
            n_patches = None
            for ntrain_suffix, _ in ntrain_variants:
                r = _load_exp(
                    img_results_dir / f"{ds}_{pgs_variant}_{ntrain_suffix}", metric
                )
                if r and r.get("n_patches") is not None:
                    n_patches = r["n_patches"]
                    break
            info_s = str(n_patches) if n_patches is not None else "--"

            for sub_idx, (ntrain_suffix, frac_pct) in enumerate(ntrain_variants):
                row = _load_exp(
                    img_results_dir / f"{ds}_{pgs_variant}_{ntrain_suffix}", metric
                )
                if row is None:
                    print(
                        f"[warn] no img results for '{ds}' "
                        f"pgs='{pgs_variant}' ntrain='{ntrain_suffix}'"
                    )
                cls_s, mean_s, pal_s, time_s = _data_cols(row)
                ds_col   = ds_label if first_ds_row else ""
                info_col = info_s   if sub_idx == 0  else ""
                first_ds_row = False
                lines.append(
                    rf"    {ds_col} & {info_col} & {frac_pct}\% & {cls_s} & {mean_s} & {pal_s} & {time_s} \\"
                )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a unified LaTeX table for PAL pooling results"
    )
    p.add_argument(
        "--text-results-dir", type=Path, default=Path("results/text_pal_pooling"),
        help="Root directory for text PAL pooling results",
    )
    p.add_argument(
        "--img-results-dir", type=Path, default=Path("results/img_pal_pooling"),
        help="Root directory for image PAL pooling results",
    )
    p.add_argument("--text-datasets", nargs="+", default=TEXT_DATASETS)
    p.add_argument("--img-datasets",  nargs="+", default=IMG_DATASETS)
    p.add_argument(
        "--img-pgs-variants", nargs="+", default=DEFAULT_IMG_PGS_VARIANTS,
        metavar="PGS",
        help="Patch-group-size variant label(s) for image experiments "
             "(e.g. pgs16 or pgs16 pgs1)",
    )
    p.add_argument(
        "--ntrain-variants", nargs="+", default=None,
        metavar="SUFFIX",
        help="Override ntrain variant labels (default: n02 nfull). "
             "Fractions are derived from the label: n02→20%%, nfull→100%%.",
    )
    p.add_argument("--metric", choices=["acc", "auroc"], default="acc")
    p.add_argument("--output", "-o", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ntrain_variants = NTRAIN_VARIANTS
    if args.ntrain_variants is not None:
        _pct_map = {"n02": "20", "nfull": "100"}
        ntrain_variants = [(v, _pct_map.get(v, v)) for v in args.ntrain_variants]

    table = make_table(
        text_results_dir=args.text_results_dir,
        img_results_dir=args.img_results_dir,
        metric=args.metric,
        text_datasets=args.text_datasets,
        img_datasets=args.img_datasets,
        img_pgs_variants=args.img_pgs_variants,
        ntrain_variants=ntrain_variants,
    )
    print(table)
    if args.output:
        args.output.write_text(table + "\n")
        print(f"\nSaved → {args.output}")
