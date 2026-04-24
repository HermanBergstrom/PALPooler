"""Plot imagenet n-train 500 results: CLS token, Mean pool, and PAL (single, best-val iteration).

Usage
-----
    python plotting/plot_imagenet_500.py
    python plotting/plot_imagenet_500.py --results-dir results/imagenet_sweep
    python plotting/plot_imagenet_500.py --metric auroc --output my_plot.png
    python plotting/plot_imagenet_500.py --datasets imagenet-beetle imagenet-birds
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "serif"

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

DATASETS = [
    "imagenet-beetle",
    "imagenet-birds",
    "imagenet-cats",
    "imagenet-feline",
    "imagenet-fruits",
    "imagenet-monkey",
    "imagenet-motor-vehicle",
    "imagenet-snake",
    "imagenet-sporting-dog",
    "imagenet-terrier",
    "imagenet-vessel",
    "imagenet-working-dog",
]

DATASET_LABELS: dict[str, str] = {
    ds: ds.replace("imagenet-", "").replace("-", " ").title() for ds in DATASETS
}

BARS = [
    # (key, label, color)
    ("cls_token", "CLS token", "#e07b39"),
    ("mean_pool", "Mean pool", "#c44e52"),
    ("pal",       "PAL",       "#55a868"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    with path.open() as f:
        raw = f.read()
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw)


def _mean_se(values: list) -> tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def load_dataset_data(results_dir: Path, dataset: str, metric: str) -> Optional[dict]:
    path = results_dir / f"{dataset}_single_sweep_500" / "seed_sweep_results.json"
    if not path.exists():
        return None

    with path.open() as f:
        raw = f.read()
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    data = json.loads(raw)

    runs = data.get("runs", [])
    if not runs:
        return None

    acc_key = "test_accuracy" if metric == "acc" else "test_auroc"
    bsl_key = "cls_token"    if metric == "acc" else "cls_token_auroc"
    mp_key  = "mean_pool"    if metric == "acc" else "mean_pool_auroc"

    cls_vals, mp_vals, pal_vals = [], [], []
    for run in runs:
        bsl = run.get("baselines", {})
        cls_vals.append(bsl.get(bsl_key))
        mp_vals.append(bsl.get(mp_key))

        stages = run.get("stages", [])
        val_iters = [
            s for s in stages
            if s.get("tag", "").startswith("iter_") and s.get("val_accuracy") is not None
        ]
        if val_iters:
            best = max(val_iters, key=lambda s: s["val_accuracy"])
            pal_vals.append(best.get(acc_key))

    return {
        "cls_token": _mean_se(cls_vals),
        "mean_pool": _mean_se(mp_vals),
        "pal":       _mean_se(pal_vals),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_dataset(ax: plt.Axes, data: dict, dataset: str, metric: str) -> None:
    scale = 100.0
    metric_label = "Accuracy (%)" if metric == "acc" else "AUROC (%)"

    bar_entries = [(label, data[key][0] * scale, data[key][1] * scale, color)
                   for key, label, color in BARS
                   if data.get(key) and data[key][0] is not None]

    if not bar_entries:
        ax.set_visible(False)
        return

    labels, values, ses, colors = zip(*bar_entries)
    x = np.arange(len(labels))
    values = np.array(values)
    ses = np.array(ses)

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.errorbar(x, values, yerr=ses,
                fmt="none", ecolor="#333333", elinewidth=1.2, capsize=3, zorder=4)

    bar_width = bars[0].get_width()
    for bar, val, se in zip(bars, values, ses):
        ax.text(bar.get_x() + bar_width / 2, val + se + 0.2,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=10, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.6, len(labels) - 0.4)

    y_min = max(0.0, min(values) - 2.0)
    y_max = max(values) + 4.0
    ax.set_ylim(y_min, y_max)

    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_imagenet_500(
    results_dir: Path,
    datasets: list[str],
    metric: str = "acc",
    output_path: Optional[Path] = None,
) -> None:
    dataset_data = []
    for ds in datasets:
        d = load_dataset_data(results_dir, ds, metric)
        if d is not None:
            dataset_data.append((ds, d))
        else:
            print(f"[warn] no results found for '{ds}' in {results_dir}")

    if not dataset_data:
        raise SystemExit("No results found — nothing to plot.")

    n = len(dataset_data)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)

    for idx, (ds, data) in enumerate(dataset_data):
        _plot_dataset(axes[idx // ncols][idx % ncols], data, ds, metric)

    for idx in range(len(dataset_data), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()

    if output_path is None:
        suf = f"_{metric}" if metric != "acc" else ""
        output_path = results_dir / f"imagenet_500_plot{suf}.pdf"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot imagenet n-train 500 PAL results")
    p.add_argument("--results-dir", type=Path, default=Path("results/imagenet_sweep"))
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--metric", choices=["acc", "auroc"], default="acc")
    p.add_argument("--output", "-o", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_imagenet_500(
        results_dir=args.results_dir,
        datasets=args.datasets,
        metric=args.metric,
        output_path=args.output,
    )
