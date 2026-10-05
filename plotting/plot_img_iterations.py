"""Two-panel image PAL pooling analysis plots.

Plot 1 — Accuracy over iterations
    One subplot per dataset.  Lines show mean ± shaded std across seeds for
    each combination of (pgs_variant, n_variant).  X-axis: Baseline → Iter 0
    → Iter 1 → Iter 2.

Plot 2 — pgs16 vs pgs1 comparison
    One subplot per n_variant (20% / 100%).  Grouped bars per dataset; within
    each group, pgs1 and pgs16 bars are shown side-by-side.  The bar height
    is the best-val-selected iteration accuracy (or last iteration if no val
    split), averaged across seeds.

Usage
-----
    python plotting/plot_img_iterations.py
    python plotting/plot_img_iterations.py --results-dir results/img_pal_pooling
    python plotting/plot_img_iterations.py --output-dir results/img_pal_pooling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 16
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["xtick.labelsize"] = 15
mpl.rcParams["ytick.labelsize"] = 15
mpl.rcParams["legend.fontsize"] = 17


DATASETS = ["butterfly", "rsna", "coco", "open-images"]
DATASET_LABELS: dict[str, str] = {
    "butterfly":   "Butterfly",
    "rsna":        "RSNA",
    "coco":        "COCO",
    "open-images": "Open Images",
}

N_VARIANTS = [
    ("nfull", "100% train"),
]

PGS_VARIANTS = [
    ("pgs1",  r"$1{\times}1 \rightarrow 1{\times}1 \rightarrow 1{\times}1$",  "#4c72b0"),
    ("pgs16", r"$4{\times}4 \rightarrow 2{\times}2 \rightarrow 1{\times}1$", "#55a868"),
]

ITER_LABELS = ["Baseline", "Iter 0", "Iter 1", "Iter 2"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    raw = path.read_text()
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw)


def _per_seed_accuracy(exp_dir: Path) -> Optional[list[list[Optional[float]]]]:
    """Return per-seed lists of [baseline, iter0, iter1, iter2] test accuracies."""
    if not exp_dir.exists():
        return None

    sweep = exp_dir / "seed_sweep_results.json"
    if sweep.exists():
        data = _load(sweep)
        runs = data.get("runs", [])
    else:
        runs = []
        for sd in sorted(exp_dir.glob("seed_*")):
            rj = sd / "results.json"
            if rj.exists():
                runs.append(_load(rj))

    if not runs:
        return None

    results = []
    for run in runs:
        stages = run.get("stages", [])
        baselines_d = run.get("baselines", {})

        baseline_acc = None
        iter_accs: list[Optional[float]] = [None, None, None]

        for s in stages:
            tag = s.get("tag", "")
            acc = s.get("test_accuracy")
            if tag == "baseline":
                baseline_acc = acc
            elif tag.startswith("iter_0"):
                iter_accs[0] = acc
            elif tag.startswith("iter_1"):
                iter_accs[1] = acc
            elif tag.startswith("iter_2"):
                iter_accs[2] = acc

        if baseline_acc is None:
            baseline_acc = baselines_d.get("mean_pool")

        results.append([baseline_acc] + iter_accs)

    return results if results else None


def _best_val_accuracy(exp_dir: Path) -> Optional[tuple[float, float]]:
    """Return (mean, std) of best-val-selected test accuracy across seeds."""
    if not exp_dir.exists():
        return None

    sweep = exp_dir / "seed_sweep_results.json"
    if sweep.exists():
        data = _load(sweep)
        runs = data.get("runs", [])
    else:
        runs = []
        for sd in sorted(exp_dir.glob("seed_*")):
            rj = sd / "results.json"
            if rj.exists():
                runs.append(_load(rj))

    if not runs:
        return None

    best_accs = []
    for run in runs:
        stages = run.get("stages", [])
        iters = [s for s in stages if s.get("tag", "").startswith("iter_")]
        if not iters:
            continue
        val_iters = [s for s in iters if s.get("val_accuracy") is not None]
        if val_iters:
            best = max(val_iters, key=lambda s: s["val_accuracy"])
        else:
            best = iters[-1]
        acc = best.get("test_accuracy")
        if acc is not None:
            best_accs.append(acc)

    if not best_accs:
        return None
    return float(np.mean(best_accs)), float(np.std(best_accs))


# ---------------------------------------------------------------------------
# Plot 1: accuracy over iterations
# ---------------------------------------------------------------------------

def plot_iterations(results_dir: Path, datasets: list[str], output_path: Path) -> None:
    n = len(datasets)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.2), squeeze=False)

    linestyles = ["-", "--"]

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        x = np.arange(len(ITER_LABELS))

        for pgs_key, pgs_label, color in PGS_VARIANTS:
            for ni, (n_key, n_label) in enumerate(N_VARIANTS):
                ls = linestyles[ni]
                exp_dir = results_dir / f"{ds}_{pgs_key}_{n_key}"
                seed_data = _per_seed_accuracy(exp_dir)
                if seed_data is None:
                    continue

                arr = np.array([[v if v is not None else np.nan for v in row]
                                for row in seed_data])
                means = np.nanmean(arr, axis=0) * 100
                stds  = np.nanstd(arr, axis=0) * 100

                label = f"{pgs_label}, {n_label}"
                ax.plot(x, means, marker="o", color=color, linestyle=ls,
                        linewidth=2.5, markersize=8, label=label)
                ax.fill_between(x, means - stds, means + stds,
                                alpha=0.15, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(ITER_LABELS)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend
    legend_handles = []
    for pgs_key, pgs_label, color in PGS_VARIANTS:
        for ni, (n_key, n_label) in enumerate(N_VARIANTS):
            ls = linestyles[ni]
            h = mlines.Line2D([], [], color=color, linestyle=ls, marker="o",
                              markersize=10, linewidth=2.5,
                              label=pgs_label)
            legend_handles.append(h)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: pgs16 vs pgs1 comparison
# ---------------------------------------------------------------------------

def plot_pgs_comparison(results_dir: Path, datasets: list[str], output_path: Path) -> None:
    n_n = len(N_VARIANTS)
    fig, axes = plt.subplots(1, n_n, figsize=(n_n * 6.5, 4.5), squeeze=False)

    bar_width = 0.35
    ds_x = np.arange(len(datasets))

    for col, (n_key, n_label) in enumerate(N_VARIANTS):
        ax = axes[0][col]

        for bar_offset, (pgs_key, pgs_label, color) in enumerate(PGS_VARIANTS):
            means, stds = [], []
            for ds in datasets:
                exp_dir = results_dir / f"{ds}_{pgs_key}_{n_key}"
                result = _best_val_accuracy(exp_dir)
                if result is not None:
                    means.append(result[0] * 100)
                    stds.append(result[1] * 100)
                else:
                    means.append(0.0)
                    stds.append(0.0)

            offset = (bar_offset - 0.5) * bar_width
            bars = ax.bar(ds_x + offset, means, bar_width,
                          color=color, label=pgs_label,
                          edgecolor="white", linewidth=0.8, zorder=3)
            ax.errorbar(ds_x + offset, means, yerr=stds,
                        fmt="none", ecolor="#333333",
                        elinewidth=1.2, capsize=4, zorder=4)

            for bar, m, s in zip(bars, means, stds):
                if m > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            m + s + 0.3, f"{m:.1f}",
                            ha="center", va="bottom", fontsize=7, color="#333333")

        ax.set_xticks(ds_x)
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=9)
        ax.set_ylabel("Test Accuracy (%)", fontsize=9)
        ax.set_title(f"pgs16 vs pgs1 — {n_label}", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, frameon=False)

        all_vals = [m for m in means if m > 0]
        if all_vals:
            ax.set_ylim(max(0, min(all_vals) - 3), max(all_vals) + 5)

    fig.suptitle("PAL Pooling — pgs16 vs pgs1 (Best-Val Selection)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot image PAL pooling iteration and pgs comparison")
    p.add_argument("--results-dir", type=Path, default=Path("results/img_pal_pooling"))
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to save plots (defaults to results-dir)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_dir = args.output_dir or args.results_dir

    plot_iterations(
        results_dir=args.results_dir,
        datasets=args.datasets,
        output_path=out_dir / "img_iterations_plot.pdf",
    )
    plot_pgs_comparison(
        results_dir=args.results_dir,
        datasets=args.datasets,
        output_path=out_dir / "img_pgs_comparison_plot.png",
    )
