"""Plot text PAL pooling experiment results comparing parameter combinations.

For each dataset, renders 6 bars:
    CLS token | Mean pool | PAL | PAL+LIW | PAL+TVF | PAL+LIW+TVF

where LIW = --use-length-importance-weights, TVF = --train-val-fraction.

For TVF variants the bar shows the final-iteration test accuracy. A small
horizontal tick is overlaid at the best-val-iteration test accuracy (the
iteration with the highest validation accuracy).

Usage
-----
    python plotting/plot_text_results.py
    python plotting/plot_text_results.py --results-dir results/text_pal_pooling
    python plotting/plot_text_results.py --metric auroc --output my_plot.png
    python plotting/plot_text_results.py --datasets imdb ag_news
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

DATASETS = ["imdb", "20news", "ag_news", "yelp"]
DATASET_LABELS: dict[str, str] = {
    "imdb":    "IMDB",
    "20news":  "20 Newsgroups",
    "ag_news": "AG News",
    "yelp":    "Yelp",
}

# (variant_suffix, display_label, bar_color, hatch)
VARIANTS: list[tuple[str, str, str, str]] = [
    ("noliw_notvf", "PAL",               "#55a868", ""),
    ("liw_notvf",    "PAL\n+LIW",          "#55a868", "///"),
    ("noliw_tvf",    "PAL\n+TVF",          "#8172b2", ""),
    ("liw_tvf",       "PAL\n+LIW\n+TVF",   "#8172b2", "///"),
]

# (baseline_key_in_json, display_label, color)
BASELINES: list[tuple[str, str, str]] = [
    ("cls_token", "CLS\ntoken", "#e07b39"),
    ("mean_pool", "Mean\npool",  "#4c72b0"),
]

BEST_VAL_COLOR = "#c44e52"   # red tick for best-val-iteration
BEST_VAL_WIDTH_FACTOR = 1.4  # tick extends this * bar_width around bar centre


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    with path.open() as f:
        raw = f.read()
    # results.json may contain NaN literals (not valid JSON); replace them
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw)


def _final_iter_acc(stages: list[dict], metric: str) -> Optional[float]:
    """Return the metric value of the last iter_* stage."""
    key = "test_accuracy" if metric == "acc" else "test_auroc"
    iters = [s for s in stages if s.get("tag", "").startswith("iter_")]
    if not iters:
        return None
    return iters[-1].get(key)


def _best_val_iter_acc(stages: list[dict], metric: str) -> Optional[float]:
    """Return test metric of the iter with the highest val_accuracy."""
    key = "test_accuracy" if metric == "acc" else "test_auroc"
    iters = [
        s for s in stages
        if s.get("tag", "").startswith("iter_") and s.get("val_accuracy") is not None
    ]
    if not iters:
        return None
    best = max(iters, key=lambda s: s["val_accuracy"])
    return best.get(key)


def load_dataset_data(
    results_dir: Path,
    dataset: str,
    metric: str,
) -> Optional[dict]:
    """
    Returns a dict with:
        baselines: {cls_token: float, mean_pool: float}
        variants:  {suffix: {"final": float, "best_val": float|None}}
    or None if no results found for this dataset.
    """
    out: dict = {"baselines": {}, "variants": {}}

    baseline_loaded = False
    for suffix, *_ in VARIANTS:
        path = results_dir / f"{dataset}_{suffix}" / "results.json"
        if not path.exists():
            continue
        data = _load(path)
        stages = data.get("stages", [])
        has_tvf = "tvf" in suffix

        final = _final_iter_acc(stages, metric)
        best_val = _best_val_iter_acc(stages, metric) if has_tvf else None
        out["variants"][suffix] = {"final": final, "best_val": best_val}

        if not baseline_loaded:
            bsl = data.get("baselines", {})
            bkey = "cls_token" if metric == "acc" else "cls_token_auroc"
            mpkey = "mean_pool" if metric == "acc" else "mean_pool_auroc"
            out["baselines"]["cls_token"] = bsl.get(bkey)
            out["baselines"]["mean_pool"] = bsl.get(mpkey)
            baseline_loaded = True

    if not baseline_loaded:
        return None
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_dataset(
    ax: plt.Axes,
    data: dict,
    dataset: str,
    metric: str,
) -> None:
    scale = 100.0  # convert to %
    metric_label = "Accuracy (%)" if metric == "acc" else "AUROC (%)"
    bars_info: list[tuple[str, float, str, str, Optional[float]]] = []
    # (label, value, color, hatch, best_val_or_None)

    for bkey, blabel, bcolor in BASELINES:
        v = data["baselines"].get(bkey)
        if v is not None:
            bars_info.append((blabel, v * scale, bcolor, "", None))

    for suffix, label, color, hatch in VARIANTS:
        vdata = data["variants"].get(suffix)
        if vdata is None or vdata["final"] is None:
            continue
        best_val = vdata["best_val"]
        bars_info.append((
            label,
            vdata["final"] * scale,
            color,
            hatch,
            best_val * scale if best_val is not None else None,
        ))

    if not bars_info:
        ax.set_visible(False)
        return

    n = len(bars_info)
    x = np.arange(n)
    labels, values, colors, hatches, best_vals = zip(*bars_info)

    bars = ax.bar(
        x, values,
        color=colors,
        hatch=hatches,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    bar_width = bars[0].get_width()

    # Value labels on top of each bar (or best_val tick if higher)
    for i, (bar, val, bv) in enumerate(zip(bars, values, best_vals)):
        label_y = val
        if bv is not None:
            label_y = max(val, bv)
            # Draw best-val tick
            tick_half = bar_width * BEST_VAL_WIDTH_FACTOR / 2
            cx = bar.get_x() + bar_width / 2
            ax.hlines(bv, cx - tick_half, cx + tick_half,
                      colors=BEST_VAL_COLOR, linewidths=2.0, zorder=5)
            # Connector from bar top to tick (dashed, only if tick is above bar)
            if bv > val:
                ax.vlines(cx, val, bv,
                          colors=BEST_VAL_COLOR, linewidths=0.8,
                          linestyles="dashed", zorder=4)

        ax.text(
            bar.get_x() + bar_width / 2,
            label_y + 0.15,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontsize=7, color="#333333",
        )

    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.6, n - 0.4)

    all_vals = list(values) + [bv for bv in best_vals if bv is not None]
    y_min = max(0.0, min(all_vals) - 1.5)
    y_max = max(all_vals) + 2.5
    ax.set_ylim(y_min, y_max)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_text_results(
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
            print(f"[warn] no results found for dataset '{ds}' in {results_dir}")

    if not dataset_data:
        raise SystemExit("No results found — nothing to plot.")

    n = len(dataset_data)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5), squeeze=False)

    for idx, (ds, data) in enumerate(dataset_data):
        ax = axes[idx // ncols][idx % ncols]
        _plot_dataset(ax, data, ds, metric)

    # Hide unused axes
    for idx in range(len(dataset_data), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Legend for best-val tick
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=BEST_VAL_COLOR, linewidth=2,
               label="Best-val iteration test acc (TVF variants)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    metric_label = "Accuracy" if metric == "acc" else "AUROC"
    fig.suptitle(f"Text PAL Pooling — {metric_label}", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path is None:
        suffix = f"_{metric}" if metric != "acc" else ""
        output_path = results_dir / f"text_results_plot{suffix}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot text PAL pooling experiment results"
    )
    p.add_argument(
        "--results-dir", type=Path,
        default=Path("results/text_pal_pooling"),
        help="Directory containing per-experiment subdirectories (default: results/text_pal_pooling)",
    )
    p.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        help=f"Datasets to include (default: {' '.join(DATASETS)})",
    )
    p.add_argument(
        "--metric", choices=["acc", "auroc"], default="acc",
        help="Metric to plot: 'acc' (default) or 'auroc'",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path for the figure (default: <results-dir>/text_results_plot.png)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_text_results(
        results_dir=args.results_dir,
        datasets=args.datasets,
        metric=args.metric,
        output_path=args.output,
    )
