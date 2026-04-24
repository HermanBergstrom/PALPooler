"""Plot imagenet sweep results: multi vs single patch-group-sizes × full vs n-train 500.

For each dataset, bars are grouped by training-set size so each group has its
own CLS/Mean-pool baseline:

    [CLS | Mean | PAL multi | PAL single]   [CLS | Mean | PAL multi | PAL single]
         Full data (all samples)                       n-train 500

Bars show mean across seeds; error bars show ±1 std. A best-val-iteration tick
(mean across seeds) is overlaid on every PAL bar.

Usage
-----
    python plotting/plot_imagenet_sweep.py
    python plotting/plot_imagenet_sweep.py --results-dir results/imagenet_sweep
    python plotting/plot_imagenet_sweep.py --metric auroc --output my_plot.png
    python plotting/plot_imagenet_sweep.py --datasets imagenet-beetle imagenet-birds
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

# Two groups, each is (group_label, [(dir_suffix, bar_label, color, hatch), ...])
GROUPS: list[tuple[str, list[tuple[str, str, str, str]]]] = [
    ("Full data", [
        ("multi_sweep",  "PAL\nmulti",   "#55a868", ""),
        ("single_sweep", "PAL\nsingle",  "#4c72b0", ""),
    ]),
    ("n-train 500", [
        ("multi_sweep_500",  "PAL\nmulti",   "#55a868", "///"),
        ("single_sweep_500", "PAL\nsingle",  "#4c72b0", "///"),
    ]),
]

BASELINE_SPECS: list[tuple[str, str, str]] = [
    ("cls_token", "CLS\ntoken", "#e07b39"),
    ("mean_pool", "Mean\npool",  "#c44e52"),
]

BEST_VAL_COLOR = "#333333"
BEST_VAL_WIDTH_FACTOR = 1.4
GROUP_GAP = 0.8  # extra spacing between groups (in bar-width units)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    with path.open() as f:
        raw = f.read()
    raw = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(raw)


def _mean_std(values: list) -> tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _load_variant(path: Path, metric: str) -> Optional[dict]:
    """Load seed_sweep_results.json and return aggregated stats, or None."""
    if not path.exists():
        return None
    data = _load(path)
    runs = data.get("runs", [])
    if not runs:
        return None

    acc_key  = "test_accuracy" if metric == "acc" else "test_auroc"
    bsl_key  = "cls_token"     if metric == "acc" else "cls_token_auroc"
    mp_key   = "mean_pool"     if metric == "acc" else "mean_pool_auroc"

    finals, best_vals, cls_vals, mp_vals = [], [], [], []
    for run in runs:
        stages = run.get("stages", [])
        iters  = [s for s in stages if s.get("tag", "").startswith("iter_")]
        if iters:
            finals.append(iters[-1].get(acc_key))
        val_iters = [s for s in iters if s.get("val_accuracy") is not None]
        if val_iters:
            best = max(val_iters, key=lambda s: s["val_accuracy"])
            best_vals.append(best.get(acc_key))
        bsl = run.get("baselines", {})
        cls_vals.append(bsl.get(bsl_key))
        mp_vals.append(bsl.get(mp_key))

    return {
        "final":    _mean_std(finals),
        "best_val": _mean_std(best_vals),
        "cls_token": _mean_std(cls_vals),
        "mean_pool": _mean_std(mp_vals),
    }


def load_dataset_data(results_dir: Path, dataset: str, metric: str) -> Optional[dict]:
    """
    Returns a list of group dicts, one per GROUPS entry:
        [{"label": str, "baselines": {...}, "variants": [...]}]
    Groups with no data at all are omitted. Returns None if nothing found.
    """
    groups_out = []
    for group_label, variants in GROUPS:
        # Use the first available variant directory to get baselines for this group
        baselines: dict[str, tuple] = {}
        pal_bars = []

        for suffix, bar_label, color, hatch in variants:
            path = results_dir / f"{dataset}_{suffix}" / "seed_sweep_results.json"
            vdata = _load_variant(path, metric)
            if vdata is None:
                continue
            if not baselines:
                baselines = {
                    "cls_token": vdata["cls_token"],
                    "mean_pool": vdata["mean_pool"],
                }
            pal_bars.append({
                "label":    bar_label,
                "color":    color,
                "hatch":    hatch,
                "final":    vdata["final"],
                "best_val": vdata["best_val"],
            })

        if pal_bars or baselines:
            groups_out.append({
                "label":     group_label,
                "baselines": baselines,
                "variants":  pal_bars,
            })

    return groups_out if groups_out else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_dataset(ax: plt.Axes, groups: list[dict], dataset: str, metric: str) -> None:
    scale = 100.0
    metric_label = "Accuracy (%)" if metric == "acc" else "AUROC (%)"

    # Build flat bar list with explicit x positions, inserting a gap between groups
    # Each entry: (x, label, mean, std, color, hatch, best_val|None)
    bar_entries = []
    group_label_positions = []   # (x_centre, group_label)
    x = 0.0

    for g, group in enumerate(groups):
        group_start_x = x
        bsl = group["baselines"]

        for bkey, blabel, bcolor in BASELINE_SPECS:
            pair = bsl.get(bkey)
            if pair and pair[0] is not None:
                bar_entries.append((x, blabel, pair[0] * scale, pair[1] * scale,
                                    bcolor, "", None))
                x += 1.0

        for vd in group["variants"]:
            if vd["final"][0] is None:
                continue
            m, s   = vd["final"]
            bv, _  = vd["best_val"]
            bar_entries.append((x, vd["label"], m * scale, s * scale,
                                vd["color"], vd["hatch"],
                                bv * scale if bv is not None else None))
            x += 1.0

        group_label_positions.append(((group_start_x + x - 1) / 2, group["label"]))
        x += GROUP_GAP  # gap before next group

    if not bar_entries:
        ax.set_visible(False)
        return

    xs, labels, values, stds, colors, hatches, best_vals = zip(*bar_entries)
    xs     = np.array(xs)
    values = np.array(values)
    stds   = np.array(stds)

    bars = ax.bar(xs, values, color=colors, hatch=hatches,
                  edgecolor="white", linewidth=0.8, zorder=3, width=0.8)
    ax.errorbar(xs, values, yerr=stds,
                fmt="none", ecolor="#333333", elinewidth=1.2, capsize=3, zorder=4)

    bar_width = bars[0].get_width()
    for bar, val, std, bv in zip(bars, values, stds, best_vals):
        label_y = val + std
        if bv is not None:
            label_y = max(label_y, bv)
            tick_half = bar_width * BEST_VAL_WIDTH_FACTOR / 2
            cx = bar.get_x() + bar_width / 2
            ax.hlines(bv, cx - tick_half, cx + tick_half,
                      colors=BEST_VAL_COLOR, linewidths=2.0, zorder=5)
            if bv > val + std:
                ax.vlines(cx, val + std, bv, colors=BEST_VAL_COLOR,
                          linewidths=0.8, linestyles="dashed", zorder=4)
        ax.text(bar.get_x() + bar_width / 2, label_y + 0.2, f"{val:.1f}",
                ha="center", va="bottom", fontsize=6.5, color="#333333")

    # x-axis ticks and limits
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)

    all_vals = list(values) + [bv for bv in best_vals if bv is not None]
    y_min = max(0.0, min(all_vals) - 2.0)
    y_max = max(all_vals) + 3.5
    ax.set_ylim(y_min, y_max)

    # Group labels drawn below the bars in data coordinates
    label_y = y_min - 0.012 * (y_max - y_min)
    for gx, glabel in group_label_positions:
        ax.text(gx, label_y, glabel, ha="center", va="top",
                fontsize=8, color="#555555", style="italic",
                clip_on=False)

    ax.set_ylabel(metric_label, fontsize=9)
    ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_imagenet_sweep(
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
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 5), squeeze=False)

    for idx, (ds, groups) in enumerate(dataset_data):
        ax = axes[idx // ncols][idx % ncols]
        _plot_dataset(ax, groups, ds, metric)

    for idx in range(len(dataset_data), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    from matplotlib.lines import Line2D
    fig.legend(
        handles=[Line2D([0], [0], color=BEST_VAL_COLOR, linewidth=2,
                        label="Best-val iteration test acc")],
        loc="lower center", ncol=1, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    metric_label = "Accuracy" if metric == "acc" else "AUROC"
    fig.suptitle(f"ImageNet PAL Pooling Sweep — {metric_label}", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path is None:
        suf = f"_{metric}" if metric != "acc" else ""
        output_path = results_dir / f"imagenet_sweep_plot{suf}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot imagenet PAL pooling sweep results")
    p.add_argument("--results-dir", type=Path, default=Path("results/imagenet_sweep"))
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--metric", choices=["acc", "auroc"], default="acc")
    p.add_argument("--output", "-o", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_imagenet_sweep(
        results_dir=args.results_dir,
        datasets=args.datasets,
        metric=args.metric,
        output_path=args.output,
    )
