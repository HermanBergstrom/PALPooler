"""Plot cross-modal interaction ablation results, averaged over seeds.

Reads one or more ``baseline_results.json`` files produced by
``crossmodal_ablation.py`` and renders a grouped bar chart (one subplot per
dataset) showing mean ± 1 SE across seeds. A dashed reference line marks the
full-fusion ``fused`` score so the drop from disabling cross-modal interaction
is visible at a glance.

Conditions are laid out in three groups
---------------------------------------
    Single-modality        tabular_only  image_only  text_only
    Full fusion            fused                          (stock TabICL)
    Cross-modal disabled   fused_ablated                  (Method 1: modality_sizes)
                           ensemble_mean  ensemble_geomean (Method 2: late fusion)

The gap between ``fused`` and each "cross-modal disabled" bar estimates the value
of cross-modal interaction inside TabICL.

Usage
-----
    # Single dataset
    python plotting/plot_crossmodal_ablation.py \\
        results/crossmodal_ablation/petfinder/baseline_results.json

    # Multiple datasets (one subplot each)
    python plotting/plot_crossmodal_ablation.py \\
        results/crossmodal_ablation/petfinder/baseline_results.json \\
        results/crossmodal_ablation/dvm/baseline_results.json

    # Auto-discover all datasets under a directory
    python plotting/plot_crossmodal_ablation.py --results-dir results/crossmodal_ablation

    # AUROC instead of accuracy
    python plotting/plot_crossmodal_ablation.py ... --metric auroc

    # Emit an agent-friendly text table (markdown or csv), optionally to a file
    python plotting/plot_crossmodal_ablation.py --results-dir results/crossmodal_ablation --table
    python plotting/plot_crossmodal_ablation.py ... --no-plot --table-format csv \\
        --table-output results/crossmodal_ablation/summary.csv
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

# Ordered list of (internal_key, display_label, color, hatch). Order defines the
# left-to-right bar layout; hatching marks the cross-modal-disabled conditions.
_CONDITIONS: list[tuple[str, str, str, str]] = [
    # Single-modality references
    ("tabular_only",     "Tabular\nonly",      "#a0a0a0", ""),
    ("image_only",       "Image\nonly",        "#4c72b0", ""),
    ("text_only",        "Text\nonly",         "#dd8452", ""),
    # Full cross-modal fusion (the reference condition)
    ("fused",            "Fused\n(full)",      "#55a868", ""),
    # Cross-modal interaction disabled
    ("fused_ablated",    "Fused\nablated",     "#c44e52", "//"),
    ("ensemble_mean",    "Ensemble\nmean",     "#8172b2", "//"),
    ("ensemble_geomean", "Ensemble\ngeomean",  "#937860", "//"),
]

_CONDITION_KEYS = [k for k, *_ in _CONDITIONS]

# Group membership drives the extra spacing between blocks of bars.
_GROUPS: list[tuple[set[str], str]] = [
    ({"tabular_only", "image_only", "text_only"},              "Single-modality"),
    ({"fused"},                                                "Full fusion"),
    ({"fused_ablated", "ensemble_mean", "ensemble_geomean"},   "Cross-modal disabled"),
]


def _group_of(key: str) -> int:
    for gi, (keys, _label) in enumerate(_GROUPS):
        if key in keys:
            return gi
    return len(_GROUPS)


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def _load_records(json_path: Path) -> list[dict]:
    with json_path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]  # legacy single-record format
    return data


def _aggregate(records: list[dict], metric: str) -> dict[str, tuple[float, float, int]]:
    """Return {condition: (mean, se, n_seeds)} for all present conditions."""
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
) -> None:
    present = [(k, lbl, col, h) for k, lbl, col, h in _CONDITIONS if k in agg]
    if not present:
        ax.set_visible(False)
        return

    keys    = [k   for k, *_ in present]
    labels  = [lbl for _, lbl, *_ in present]
    colors  = [col for _, _, col, _ in present]
    hatches = [h   for _, _, _, h in present]
    means   = [agg[k][0] * 100 for k in keys]
    ses     = [agg[k][1] * 100 for k in keys]

    # Insert a gap in x whenever we cross a group boundary.
    gap = 0.7
    x_positions: list[float] = []
    cur = 0.0
    prev_group = _group_of(keys[0])
    for k in keys:
        g = _group_of(k)
        if g != prev_group:
            cur += gap
            prev_group = g
        x_positions.append(cur)
        cur += 1.0
    x = np.array(x_positions)

    bars = ax.bar(
        x, means,
        yerr=ses,
        color=colors,
        hatch=hatches,
        edgecolor="white",
        linewidth=0.8,
        error_kw=dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor="#333333"),
        zorder=3,
    )

    # Reference line at the full-fusion score.
    if "fused" in agg:
        fused_mean = agg["fused"][0] * 100
        ax.axhline(fused_mean, ls="--", lw=1.3, color="#55a868", alpha=0.8, zorder=2)
        ax.text(
            x[-1] + 0.4, fused_mean, "full fusion",
            va="center", ha="left", fontsize=7.5, color="#3d7a4c",
        )

    # Value labels on top of each bar; annotate the delta-from-fused for the
    # cross-modal-disabled bars so the interaction gap is explicit.
    fused_mean = agg["fused"][0] * 100 if "fused" in agg else None
    for bar, key, mean, se in zip(bars, keys, means, ses):
        label = f"{mean:.2f}%"
        if fused_mean is not None and _group_of(key) == 2:
            label += f"\nΔ{mean - fused_mean:+.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2, mean + se + 0.1, label,
            ha="center", va="bottom", fontsize=7.0, color="#333333",
        )

    ylabel = "Accuracy (%)" if metric == "acc" else "AUROC (%)"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}  (n={n_seeds} seed{'s' if n_seeds != 1 else ''}, ±1 SE)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(x[0] - 0.7, x[-1] + 1.2)
    padding = 1.0  # percentage points
    y_min = max(0.0, min(m - s for m, s in zip(means, ses)) - padding)
    y_max = max(m + s for m, s in zip(means, ses)) + padding * 2
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_crossmodal(
    json_paths: list[Path],
    metric: str = "acc",
    output_path: Path | None = None,
    ncols: int = 2,
) -> None:
    datasets = [(p, _load_records(p)) for p in json_paths]
    n_datasets = len(datasets)
    metric_label = "Accuracy" if metric == "acc" else "AUROC"

    n_cols = max(1, min(ncols, n_datasets))
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False
    )
    flat_axes = axes.flatten()
    for ax, (json_path, records) in zip(flat_axes, datasets):
        agg = _aggregate(records, metric)
        _plot_dataset(ax, agg, metric, _dataset_label(json_path), len(records))
    # Hide any unused axes in the last row.
    for ax in flat_axes[n_datasets:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Cross-modal interaction ablation — {metric_label}",
        fontsize=14, y=1.005,
    )
    fig.tight_layout()

    if output_path is None:
        suffix = f"_{metric}" if metric != "acc" else ""
        output_path = json_paths[0].parent / f"crossmodal_ablation_plot{suffix}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Table export (agent-friendly)
# ---------------------------------------------------------------------------

def _table_rows(json_paths: list[Path], metric: str) -> list[dict]:
    """Flatten aggregated results into one row per (dataset, condition).

    Each row carries the mean metric, its standard error, the delta vs. the
    ``fused`` full-fusion baseline, and the seed count — all as plain numbers.
    """
    rows: list[dict] = []
    for p in json_paths:
        records = _load_records(p)
        agg = _aggregate(records, metric)
        dataset = _dataset_label(p)
        fused_mean = agg["fused"][0] * 100 if "fused" in agg else None
        for key, *_ in _CONDITIONS:
            if key not in agg:
                continue
            mean, se, n_seeds = agg[key]
            delta = None if (fused_mean is None or key == "fused") else mean * 100 - fused_mean
            rows.append({
                "dataset":        dataset,
                "condition":      key,
                "mean_pct":       mean * 100,
                "se_pct":         se * 100,
                "delta_vs_fused": delta,
                "n_seeds":        n_seeds,
            })
    return rows


def _render(headers: list[str], rows: list[list[str]], fmt: str) -> str:
    """Render header + rows as a markdown or CSV table string."""
    if fmt == "csv":
        return "\n".join([",".join(headers)] + [",".join(r) for r in rows])
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def format_table(
    json_paths: list[Path],
    metric: str,
    fmt: str = "markdown",
    layout: str = "matrix",
    show_se: bool = False,
) -> str:
    """Render the results as a markdown or CSV table string.

    layout="matrix" (default): one row per dataset, one column per condition,
    cells hold the mean metric (%). layout="long": one row per (dataset,
    condition) with mean, SE, and delta-vs-fused.
    """
    metric_name = "acc" if metric == "acc" else "auroc"

    if layout == "long":
        rows_data = _table_rows(json_paths, metric)
        headers = ["dataset", "condition", f"{metric_name}_pct", "se_pct", "delta_vs_fused", "n_seeds"]
        rows = [[
            r["dataset"], r["condition"], f"{r['mean_pct']:.2f}", f"{r['se_pct']:.2f}",
            "" if r["delta_vs_fused"] is None else f"{r['delta_vs_fused']:+.2f}", str(r["n_seeds"]),
        ] for r in rows_data]
        return _render(headers, rows, fmt)

    # matrix: datasets × conditions
    per_ds = []
    for p in json_paths:
        records = _load_records(p)
        per_ds.append((_dataset_label(p), _aggregate(records, metric), len(records)))
    # Columns = canonical conditions present in at least one dataset.
    cols = [k for k, *_ in _CONDITIONS if any(k in agg for _, agg, _ in per_ds)]

    def _cell(agg: dict, k: str) -> str:
        if k not in agg:
            return ""
        mean, se, _ = agg[k]
        return f"{mean * 100:.2f}" + (f"±{se * 100:.2f}" if show_se else "")

    headers = [f"dataset \\ {metric_name}(%)"] + cols + ["n_seeds"]
    rows = [[ds] + [_cell(agg, k) for k in cols] + [str(n)] for ds, agg, n in per_ds]
    return _render(headers, rows, fmt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot cross-modal interaction ablation results averaged over seeds"
    )
    p.add_argument(
        "json_paths", type=Path, nargs="*",
        help="Path(s) to baseline_results.json file(s). If omitted, --results-dir is used.",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Auto-discover all baseline_results.json files under this directory "
             "(default: results/crossmodal_ablation)",
    )
    p.add_argument(
        "--metric", choices=["acc", "auroc"], default="acc",
        help="Metric to plot: 'acc' (default) or 'auroc'",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path for the figure "
             "(default: <first_result_dir>/crossmodal_ablation_plot[_auroc].png)",
    )
    p.add_argument(
        "--ncols", type=int, default=2,
        help="Number of subplot columns; datasets wrap onto multiple rows (default: 2)",
    )
    p.add_argument(
        "--table", action="store_true",
        help="Emit a text table (markdown by default) to stdout in addition to the figure. "
             "Handy for pasting/sending to another agent.",
    )
    p.add_argument(
        "--table-format", choices=["markdown", "csv"], default="markdown",
        help="Table format when --table is set (default: markdown)",
    )
    p.add_argument(
        "--table-layout", choices=["matrix", "long"], default="matrix",
        help="'matrix' (default): datasets as rows, conditions as columns, metric in cells. "
             "'long': one row per (dataset, condition) with SE and delta-vs-fused.",
    )
    p.add_argument(
        "--table-se", action="store_true",
        help="In matrix layout, append ±SE to each cell (default: mean only)",
    )
    p.add_argument(
        "--table-output", type=Path, default=None,
        help="Also write the --table output to this file (default: stdout only)",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip the figure and only produce the --table output",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    json_paths: list[Path] = list(args.json_paths)
    if not json_paths:
        results_dir = args.results_dir or Path("results/crossmodal_ablation")
        json_paths = sorted(results_dir.glob("*/baseline_results.json"))
        if not json_paths:
            raise SystemExit(
                f"No baseline_results.json files found under {results_dir}. "
                "Pass explicit paths or check --results-dir."
            )
        print(f"Auto-discovered {len(json_paths)} result file(s):")
        for p in json_paths:
            print(f"  {p}")

    if args.table or args.no_plot:
        table = format_table(
            json_paths, metric=args.metric, fmt=args.table_format,
            layout=args.table_layout, show_se=args.table_se,
        )
        print(table)
        if args.table_output is not None:
            args.table_output.parent.mkdir(parents=True, exist_ok=True)
            args.table_output.write_text(table + "\n")
            print(f"\nSaved table → {args.table_output}")

    if not args.no_plot:
        plot_crossmodal(json_paths, metric=args.metric, output_path=args.output, ncols=args.ncols)
