"""Plot the PCA ablation (raw vs. PCA features into TabICL), averaged over seeds.

Reads one or more ``pca_ablation_results.json`` files produced by
``pal_pooling/pca_ablation.py`` and renders a grouped bar chart (one subplot per
result directory, i.e. per dataset×modality) showing mean ± 1 SE across seeds.
A dashed reference line marks the ``raw`` (no-projection) score, and every PCA
bar is annotated with its delta from it — so both the cost of decorrelating the
axes (``pca_full``) and the cost of truncating them (``pca<k>``) are readable at
a glance.

Conditions are discovered from the results themselves and ordered
    raw  →  pca_full  →  pca<k> ascending
so adding a new ``--pca-dim`` to the ablation needs no change here. When a run
covers both poolings (``--pooling both``), each pooling gets its own bar series
(hatched for mean pooling) and a legend.

Usage
-----
    # Single result file
    python plotting/plot_pca_ablation.py \\
        results/pca_ablation/petfinder_image/pca_ablation_results.json

    # Auto-discover every dataset×modality under a directory
    python plotting/plot_pca_ablation.py --results-dir results/pca_ablation

    # AUROC instead of accuracy
    python plotting/plot_pca_ablation.py --results-dir results/pca_ablation --metric auroc

    # Text table (markdown or csv), optionally to a file, with or without the figure
    python plotting/plot_pca_ablation.py --results-dir results/pca_ablation --table
    python plotting/plot_pca_ablation.py --results-dir results/pca_ablation --no-plot \\
        --table-layout long --table-format csv \\
        --table-output results/pca_ablation/summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Condition catalogue
# ---------------------------------------------------------------------------

# The reference condition every PCA bar is compared against.
_BASELINE = "raw"

# Fixed colors for the two non-truncated conditions; truncated PCA conditions
# draw from _TRUNC_COLORS in ascending-dimension order. Colors are redundant
# with the x-tick labels (each bar is individually labelled), so they carry
# emphasis rather than identity.
_COLORS: dict[str, str] = {"raw": "#a0a0a0", "pca_full": "#55a868"}
_TRUNC_COLORS = ["#4c72b0", "#dd8452", "#8172b2", "#937860"]

# Mean pooling is hatched so the two poolings stay distinguishable without color.
_POOL_HATCH = {"cls": "", "mean": "//"}
_POOL_LABEL = {"cls": "CLS token", "mean": "mean-pooled"}


def _proj_sort_key(proj: str) -> tuple[int, int]:
    """Sort key placing raw first, then pca_full, then pca<k> by ascending k."""
    if proj == "raw":
        return (0, 0)
    if proj == "pca_full":
        return (1, 0)
    m = re.fullmatch(r"pca(\d+)", proj)
    return (2, int(m.group(1))) if m else (3, 0)


def _proj_label(proj: str) -> str:
    """Human-readable x-tick label for a projection condition."""
    if proj == "raw":
        return "raw\n(no PCA)"
    if proj == "pca_full":
        return "PCA\nfull"
    m = re.fullmatch(r"pca(\d+)", proj)
    return f"PCA\n{m.group(1)}" if m else proj


def _proj_colors(projections: list[str]) -> dict[str, str]:
    """Assign a color per projection: fixed for raw/pca_full, ordered for pca<k>."""
    colors = {}
    trunc_i = 0
    for proj in projections:
        if proj in _COLORS:
            colors[proj] = _COLORS[proj]
        else:
            colors[proj] = _TRUNC_COLORS[trunc_i % len(_TRUNC_COLORS)]
            trunc_i += 1
    return colors


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def _load_records(json_path: Path) -> list[dict]:
    with json_path.open() as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]  # legacy single-record format
    return data


def _aggregate(records: list[dict], metric: str) -> dict[tuple[str, str], dict]:
    """Return {(pooling, projection): {mean, se, n_seeds, feat_dim}}.

    Conditions are read from each entry's ``pooling`` / ``projection`` fields
    rather than parsed out of the tag, so tags stay free to change.
    """
    values: dict[tuple[str, str], list[float]] = {}
    feat_dims: dict[tuple[str, str], int] = {}
    for record in records:
        for tag, entry in record.get("results", {}).items():
            pooling = entry.get("pooling")
            proj    = entry.get("projection")
            if pooling is None or proj is None:
                continue  # not a pca-ablation result
            v = entry.get(metric)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            key = (pooling, proj)
            values.setdefault(key, []).append(float(v))
            if entry.get("feat_dim") is not None:
                feat_dims[key] = int(entry["feat_dim"])
    return {
        key: {
            "mean":     float(np.mean(vs)),
            "se":       float(np.std(vs) / np.sqrt(len(vs))),
            "n_seeds":  len(vs),
            "feat_dim": feat_dims.get(key),
        }
        for key, vs in values.items()
    }


def _axes_of(agg: dict[tuple[str, str], dict]) -> tuple[list[str], list[str]]:
    """Return (poolings, projections) present, each in canonical display order."""
    poolings = [p for p in ("cls", "mean") if any(k[0] == p for k in agg)]
    poolings += sorted({k[0] for k in agg} - set(poolings))
    projections = sorted({k[1] for k in agg}, key=_proj_sort_key)
    return poolings, projections


def _dataset_label(json_path: Path) -> str:
    """Derive a human-readable name (e.g. 'petfinder_image') from the file path."""
    return json_path.parent.name


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_dataset(
    ax: plt.Axes,
    agg: dict[tuple[str, str], dict],
    metric: str,
    title: str,
    n_seeds: int,
) -> None:
    if not agg:
        ax.set_visible(False)
        return

    poolings, projections = _axes_of(agg)
    colors = _proj_colors(projections)
    x = np.arange(len(projections), dtype=float)
    multi = len(poolings) > 1
    width = 0.8 / len(poolings)
    # Hatching only earns its keep when it separates two series; with a single
    # pooling it would be unexplained texture (no legend is drawn for one series).
    hatch_of = (lambda p: _POOL_HATCH.get(p, "")) if multi else (lambda p: "")
    annot_size = 6.2 if multi else 7.0

    all_means: list[float] = []
    all_ses: list[float] = []
    for pi, pooling in enumerate(poolings):
        offset = (pi - (len(poolings) - 1) / 2) * width
        # Baseline for this pooling's deltas: its own raw score.
        base = agg.get((pooling, _BASELINE))
        base_mean = base["mean"] * 100 if base else None

        for xi, proj in enumerate(projections):
            entry = agg.get((pooling, proj))
            if entry is None:
                continue
            mean, se = entry["mean"] * 100, entry["se"] * 100
            all_means.append(mean)
            all_ses.append(se)
            ax.bar(
                x[xi] + offset, mean,
                width=width * 0.92,  # 2px-equivalent gap between adjacent bars
                yerr=se,
                color=colors[proj],
                hatch=hatch_of(pooling),
                edgecolor="white",
                linewidth=0.8,
                error_kw=dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor="#333333"),
                zorder=3,
            )
            label = f"{mean:.2f}%"
            if base_mean is not None and proj != _BASELINE:
                label += f"\nΔ{mean - base_mean:+.2f}"
            if entry["feat_dim"] is not None:
                label += f"\nd={entry['feat_dim']}"
            ax.text(
                x[xi] + offset, mean + se + 0.1, label,
                ha="center", va="bottom", fontsize=annot_size, color="#333333",
            )

        # Reference line at raw. Only with a single pooling — two near-identical
        # dashed lines read as clutter, and the per-bar Δ already carries the gap.
        if base_mean is not None and not multi:
            ax.axhline(base_mean, ls="--", lw=1.3, color="#8a8a8a", alpha=0.8, zorder=2)

    if len(poolings) == 1:
        # One series needs no legend box — the title and x labels name it.
        base = agg.get((poolings[0], _BASELINE))
        if base is not None:
            ax.text(
                x[-1] + 0.45, base["mean"] * 100, "raw",
                va="center", ha="left", fontsize=7.5, color="#666666",
            )
    else:
        # Identity for ≥2 series must not be color-alone: hatch + legend.
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", edgecolor="white",
                          hatch=hatch_of(p), label=_POOL_LABEL.get(p, p))
            for p in poolings
        ]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")

    ylabel = "Accuracy (%)" if metric == "acc" else "AUROC (%)"
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}  (n={n_seeds} seed{'s' if n_seeds != 1 else ''}, ±1 SE)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([_proj_label(p) for p in projections], fontsize=9)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.9)
    padding = 1.0  # percentage points
    y_min = max(0.0, min(m - s for m, s in zip(all_means, all_ses)) - padding)
    y_max = max(m + s for m, s in zip(all_means, all_ses)) + padding * 3
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_pca_ablation(
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
        n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False
    )
    flat_axes = axes.flatten()
    for ax, (json_path, records) in zip(flat_axes, datasets):
        agg = _aggregate(records, metric)
        _plot_dataset(ax, agg, metric, _dataset_label(json_path), len(records))
    for ax in flat_axes[n_datasets:]:
        ax.set_visible(False)

    fig.suptitle(
        f"PCA ablation — raw vs. PCA features into TabICL ({metric_label})",
        fontsize=14, y=1.005,
    )
    fig.tight_layout()

    if output_path is None:
        suffix = f"_{metric}" if metric != "acc" else ""
        output_path = json_paths[0].parent / f"pca_ablation_plot{suffix}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Table export (agent-friendly)
# ---------------------------------------------------------------------------

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

    layout="matrix" (default): one row per (dataset, pooling), one column per
    projection, cells hold the mean metric (%). layout="long": one row per
    (dataset, pooling, projection) with mean, SE, feature dim, and delta-vs-raw.
    """
    metric_name = "acc" if metric == "acc" else "auroc"
    per_ds = [(_dataset_label(p), _aggregate(_load_records(p), metric), len(_load_records(p)))
              for p in json_paths]

    if layout == "long":
        headers = ["dataset", "pooling", "projection", f"{metric_name}_pct",
                   "se_pct", "delta_vs_raw", "feat_dim", "n_seeds"]
        rows: list[list[str]] = []
        for ds, agg, _ in per_ds:
            poolings, projections = _axes_of(agg)
            for pooling in poolings:
                base = agg.get((pooling, _BASELINE))
                base_mean = base["mean"] * 100 if base else None
                for proj in projections:
                    e = agg.get((pooling, proj))
                    if e is None:
                        continue
                    delta = None if (base_mean is None or proj == _BASELINE) \
                        else e["mean"] * 100 - base_mean
                    rows.append([
                        ds, pooling, proj,
                        f"{e['mean'] * 100:.2f}", f"{e['se'] * 100:.2f}",
                        "" if delta is None else f"{delta:+.2f}",
                        "" if e["feat_dim"] is None else str(e["feat_dim"]),
                        str(e["n_seeds"]),
                    ])
        return _render(headers, rows, fmt)

    # matrix: (dataset, pooling) rows × projection columns
    cols = sorted({k[1] for _, agg, _ in per_ds for k in agg}, key=_proj_sort_key)

    def _cell(agg: dict, key: tuple[str, str]) -> str:
        e = agg.get(key)
        if e is None:
            return ""
        return f"{e['mean'] * 100:.2f}" + (f"±{e['se'] * 100:.2f}" if show_se else "")

    headers = [f"dataset \\ {metric_name}(%)", "pooling"] + cols + ["n_seeds"]
    rows = []
    for ds, agg, n in per_ds:
        poolings, _ = _axes_of(agg)
        for pooling in poolings:
            rows.append([ds, pooling] + [_cell(agg, (pooling, c)) for c in cols] + [str(n)])
    return _render(headers, rows, fmt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot PCA ablation results (raw vs. PCA) averaged over seeds"
    )
    p.add_argument(
        "json_paths", type=Path, nargs="*",
        help="Path(s) to pca_ablation_results.json file(s). If omitted, --results-dir is used.",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Auto-discover all pca_ablation_results.json files under this directory "
             "(default: results/pca_ablation)",
    )
    p.add_argument(
        "--metric", choices=["acc", "auroc"], default="acc",
        help="Metric to plot: 'acc' (default) or 'auroc'",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path for the figure "
             "(default: <first_result_dir>/pca_ablation_plot[_auroc].png)",
    )
    p.add_argument(
        "--ncols", type=int, default=2,
        help="Number of subplot columns; datasets wrap onto multiple rows (default: 2)",
    )
    p.add_argument(
        "--table", action="store_true",
        help="Emit a text table (markdown by default) to stdout in addition to the figure",
    )
    p.add_argument(
        "--table-format", choices=["markdown", "csv"], default="markdown",
        help="Table format when --table is set (default: markdown)",
    )
    p.add_argument(
        "--table-layout", choices=["matrix", "long"], default="matrix",
        help="'matrix' (default): (dataset, pooling) rows × projection columns. "
             "'long': one row per condition with SE, feat_dim, and delta-vs-raw.",
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
        results_dir = args.results_dir or Path("results/pca_ablation")
        json_paths = sorted(results_dir.glob("*/pca_ablation_results.json"))
        if not json_paths:
            raise SystemExit(
                f"No pca_ablation_results.json files found under {results_dir}. "
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
        plot_pca_ablation(json_paths, metric=args.metric, output_path=args.output, ncols=args.ncols)
