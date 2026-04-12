"""Plot multimodal experiment results averaged over seeds.

Reads one or more ``multimodal_results.json`` files produced by
``multimodal_experiments.py`` and renders a grouped bar chart (one subplot
per dataset) showing mean ± 1 SE across seeds.

Conditions plotted
------------------
    tabular_only        mean_pool_img       cls_img
    mean_pool_img+tab   cls_img+tab
    pal_img             pal_img+tab
    pal_context_img     pal_context_img+tab

Usage
-----
    # Single dataset
    python plotting/plot_multimodal_results.py \\
        results/multimodal/pad-ufes/multimodal_results.json

    # Multiple datasets (one subplot each)
    python plotting/plot_multimodal_results.py \\
        results/multimodal/pad-ufes/multimodal_results.json \\
        results/multimodal/petfinder/multimodal_results.json

    # Auto-discover all datasets under a directory
    python plotting/plot_multimodal_results.py \\
        --results-dir results/multimodal

    # AUROC instead of accuracy
    python plotting/plot_multimodal_results.py ... --metric auroc

    # Split image-only and image+tabular into separate subplots
    python plotting/plot_multimodal_results.py ... --split-groups
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Condition catalogue
# ---------------------------------------------------------------------------

# Ordered list of (internal_key, display_label, color, hatch)
_CONDITIONS: list[tuple[str, str, str, str]] = [
    # Image-only methods
    ("tabular_only",        "Tabular\nonly",        "#a0a0a0", ""),
    ("mean_pool_img",       "Mean\npool",            "#4c72b0", ""),
    ("cls_img",             "CLS",                   "#e07b39", ""),
    ("pal_img",             "PAL",                   "#55a868", ""),
    ("pal_context_img",     "PAL ctx",               "#8172b2", ""),
    # Image + tabular methods
    ("mean_pool_img+tab",   "Mean pool\n+ Tab",      "#4c72b0", "//"),
    ("cls_img+tab",         "CLS\n+ Tab",            "#e07b39", "//"),
    ("pal_img+tab",         "PAL\n+ Tab",            "#55a868", "//"),
    ("pal_context_img+tab", "PAL ctx\n+ Tab",        "#8172b2", "//"),
]

_CONDITION_KEYS = [k for k, *_ in _CONDITIONS]

# Keys belonging to each group (used by --split-groups)
_IMG_ONLY_KEYS  = {"tabular_only", "mean_pool_img", "cls_img", "pal_img", "pal_context_img"}
_IMG_TAB_KEYS   = {"mean_pool_img+tab", "cls_img+tab", "pal_img+tab", "pal_context_img+tab"}


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def _load_records(json_path: Path) -> list[dict]:
    with json_path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Legacy single-record format
        data = [data]
    return data


def _aggregate(records: list[dict], metric: str) -> dict[str, tuple[float, float, int]]:
    """Return {condition: (mean, std, n_seeds)} for all present conditions."""
    values: dict[str, list[float]] = {k: [] for k in _CONDITION_KEYS}
    for record in records:
        for key in _CONDITION_KEYS:
            entry = record.get("results", {}).get(key)
            if entry is None:
                continue
            v = entry.get(metric)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values[key].append(float(v))
    return {
        k: (float(np.mean(vs)), float(np.std(vs) / np.sqrt(len(vs))), len(vs))
        for k, vs in values.items()
        if vs
    }


def _dataset_label(json_path: Path) -> str:
    """Derive a human-readable dataset name from the result file path."""
    return json_path.parent.name


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_dataset(
    ax: plt.Axes,
    agg: dict[str, tuple[float, float, int]],
    metric: str,
    title: str,
    n_seeds: int,
    condition_filter: set[str] | None = None,
) -> None:
    present = [
        (k, lbl, col, hatch) for k, lbl, col, hatch in _CONDITIONS
        if k in agg and (condition_filter is None or k in condition_filter)
    ]
    if not present:
        ax.set_visible(False)
        return

    keys   = [k   for k, *_ in present]
    labels = [lbl for _, lbl, *_ in present]
    colors = [col for _, _, col, _ in present]
    hatches= [h   for _, _, _, h in present]
    means  = [agg[k][0] * 100 for k in keys]
    stds   = [agg[k][1] * 100 for k in keys]

    # Insert a small gap between the image-only and image+tab groups,
    # but only when both groups are shown together.
    show_gap = condition_filter is None
    gap = 0.5  # extra spacing in bar-width units between the two groups
    x = np.array([
        i + (0.0 if (not show_gap or keys[i] in _IMG_ONLY_KEYS) else gap)
        for i in range(len(keys))
    ])
    bars = ax.bar(
        x, means,
        yerr=stds,
        color=colors,
        hatch=hatches,
        edgecolor="white",
        linewidth=0.8,
        error_kw=dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor="#333333"),
        zorder=3,
    )

    # Value labels on top of each bar
    for bar, mean, std in zip(bars, means, stds):
        y = mean + std + 0.1
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{mean:.2f}%",
            ha="center", va="bottom",
            fontsize=7.5, color="#333333",
        )

    ylabel = "Accuracy (%)" if metric == "acc" else "AUROC (%)"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}  (n={n_seeds} seed{'s' if n_seeds != 1 else ''}, ±1 SE)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    padding = 1.0  # in percentage points
    y_min = max(0.0, min(m - se for m, se in zip(means, stds)) - padding)
    y_max = max(m + se for m, se in zip(means, stds)) + padding
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_multimodal(
    json_paths: list[Path],
    metric: str = "acc",
    output_path: Path | None = None,
    split_groups: bool = False,
) -> None:
    datasets = [(p, _load_records(p)) for p in json_paths]
    n_datasets = len(datasets)
    metric_label = "Accuracy" if metric == "acc" else "AUROC"

    if split_groups:
        # Rows = datasets, cols = [image-only, image+tab]
        n_cols = 2
        fig_w = max(12, 10 * n_datasets)
        fig, axes = plt.subplots(n_datasets, n_cols, figsize=(fig_w, 5 * n_datasets), squeeze=False)
        group_defs = [
            (_IMG_ONLY_KEYS,  "Image only"),
            (_IMG_TAB_KEYS,   "Image + Tabular"),
        ]
        for row, (json_path, records) in enumerate(datasets):
            agg = _aggregate(records, metric)
            n_seeds = len(records)
            dataset_name = _dataset_label(json_path)
            for col, (condition_filter, group_label) in enumerate(group_defs):
                title = f"{dataset_name} — {group_label}  (n={n_seeds} seed{'s' if n_seeds != 1 else ''}, ±1 SE)"
                ax = axes[row][col]
                _plot_dataset(ax, agg, metric, title, n_seeds, condition_filter=condition_filter)
    else:
        fig_w = max(8, 9 * n_datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(fig_w, 5), squeeze=False)
        for ax, (json_path, records) in zip(axes[0], datasets):
            agg = _aggregate(records, metric)
            n_seeds = len(records)
            title = _dataset_label(json_path)
            _plot_dataset(ax, agg, metric, title, n_seeds)

    fig.suptitle(f"Multimodal experiment — {metric_label}", fontsize=14, y=1.01)
    fig.tight_layout()

    if output_path is None:
        suffix = f"_{metric}" if metric != "acc" else ""
        split_suffix = "_split" if split_groups else ""
        output_path = json_paths[0].parent / f"multimodal_plot{suffix}{split_suffix}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot multimodal experiment results averaged over seeds"
    )
    p.add_argument(
        "json_paths", type=Path, nargs="*",
        help="Path(s) to multimodal_results.json file(s). "
             "If omitted, --results-dir is used.",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Auto-discover all multimodal_results.json files under this directory "
             "(used when no positional paths are given; default: results/multimodal)",
    )
    p.add_argument(
        "--metric", choices=["acc", "auroc"], default="acc",
        help="Metric to plot: 'acc' (default) or 'auroc'",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path for the figure "
             "(default: <first_result_dir>/multimodal_plot[_auroc][_split].png)",
    )
    p.add_argument(
        "--split-groups", action="store_true",
        help="Put image-only and image+tabular conditions in separate subplots "
             "(useful when the two groups sit on different y-axis scales)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    json_paths: list[Path] = list(args.json_paths)

    if not json_paths:
        results_dir = args.results_dir or Path("results/multimodal")
        json_paths = sorted(results_dir.glob("*/multimodal_results.json"))
        if not json_paths:
            raise SystemExit(
                f"No multimodal_results.json files found under {results_dir}. "
                "Pass explicit paths or check --results-dir."
            )
        print(f"Auto-discovered {len(json_paths)} result file(s):")
        for p in json_paths:
            print(f"  {p}")

    plot_multimodal(json_paths, metric=args.metric, output_path=args.output, split_groups=args.split_groups)
