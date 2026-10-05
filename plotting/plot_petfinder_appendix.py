"""Bar chart comparing CLS/Mean pooling vs PAL pooling on PetFinder.

Reads ``multimodal_results.json`` produced by ``multimodal_experiments.py`` and
plots mean ± 1 SE across seeds.

Key naming convention in the JSON:
    Image results:  cls_img, mean_pool_img, pal_img, cls_img+tab, …
    Text  results:  cls_text, mean_pool_text, pal_text, cls_text+tab, …

Flags
-----
--modality   img (default) | text
--pooling    cls (default) | mean | both

Colours follow ``plot_imagenet_500.py``; +tabular bars use hatch styling from
``plot_multimodal_results.py``.

Usage
-----
    python plotting/plot_petfinder_appendix.py
    python plotting/plot_petfinder_appendix.py \\
        results/petfinder_appendix_text/multimodal_results.json \\
        --modality text --pooling mean
    python plotting/plot_petfinder_appendix.py \\
        results/petfinder_appendix/multimodal_results.json \\
        --pooling both --metric auroc --output my_plot.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "serif"

# Colors
_COLOR_CLS  = "#e07b39"
_COLOR_MEAN = "#4c72b0"
_COLOR_PAL  = "#55a868"


# ---------------------------------------------------------------------------
# Conditions  (key, label, color, hatch)
# ---------------------------------------------------------------------------

def _build_conditions(
    pooling: str, modality: str
) -> tuple[list[tuple[str, str, str, str]], set[str], set[str]]:
    """Return (conditions, solo_keys, tab_keys) for the chosen pooling/modality."""
    mod = modality                      # "img" or "text"
    mean_key = f"mean_pool_{mod}"       # mean_pool_img / mean_pool_text
    cls_key  = f"cls_{mod}"
    pal_key  = f"pal_{mod}"

    all_conds: list[tuple[str, str, str, str]] = [
        (cls_key,          "CLS",           _COLOR_CLS,  ""),
        (mean_key,         "Mean",          _COLOR_MEAN, ""),
        (pal_key,          "PAL",           _COLOR_PAL,  ""),
        (f"{cls_key}+tab",  "CLS\n+ Tab",   _COLOR_CLS,  "//"),
        (f"{mean_key}+tab", "Mean\n+ Tab",  _COLOR_MEAN, "//"),
        (f"{pal_key}+tab",  "PAL\n+ Tab",   _COLOR_PAL,  "//"),
    ]
    solo_keys = {cls_key, mean_key, pal_key}
    tab_keys  = {f"{cls_key}+tab", f"{mean_key}+tab", f"{pal_key}+tab"}

    if pooling == "cls":
        keep = {cls_key, pal_key, f"{cls_key}+tab", f"{pal_key}+tab"}
    elif pooling == "mean":
        keep = {mean_key, pal_key, f"{mean_key}+tab", f"{pal_key}+tab"}
    else:  # "both"
        keep = {k for k, *_ in all_conds}

    conditions = [c for c in all_conds if c[0] in keep]
    return conditions, solo_keys & keep, tab_keys & keep


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def _load_records(json_path: Path) -> list[dict]:
    with json_path.open() as f:
        data = json.load(f)
    return [data] if isinstance(data, dict) else data


def _aggregate(
    records: list[dict],
    metric: str,
    conditions: list[tuple[str, str, str, str]],
) -> dict[str, tuple[float, float, int]]:
    """Return {key: (mean, se, n)} across seeds."""
    values: dict[str, list[float]] = {k: [] for k, *_ in conditions}
    for record in records:
        for key in values:
            entry = record.get("results", {}).get(key)
            if entry is None:
                continue
            v = entry.get(metric)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values[key].append(float(v))
    return {
        k: (float(np.mean(vs)), float(np.std(vs, ddof=1) / np.sqrt(len(vs))), len(vs))
        for k, vs in values.items()
        if vs
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    json_path: Path,
    metric: str = "acc",
    output_path: Optional[Path] = None,
    pooling: str = "cls",
    modality: str = "img",
) -> None:
    conditions, solo_keys, tab_keys = _build_conditions(pooling, modality)
    records = _load_records(json_path)
    agg = _aggregate(records, metric, conditions)
    n_seeds = len(records)

    present = [(k, lbl, col, hatch) for k, lbl, col, hatch in conditions if k in agg]
    if not present:
        available = list(records[0].get("results", records[0]).keys())
        raise SystemExit(
            f"None of the expected conditions found in {json_path}.\n"
            f"  Expected keys (sample): {[k for k, *_ in conditions]}\n"
            f"  Found keys in file:     {available}"
        )

    keys    = [k   for k, *_ in present]
    labels  = [lbl for _, lbl, *_ in present]
    colors  = [col for _, _, col, _ in present]
    hatches = [h   for _, _, _, h in present]
    means   = np.array([agg[k][0] * 100 for k in keys])
    ses     = np.array([agg[k][1] * 100 for k in keys])

    # Visual gap between solo and +tabular groups
    gap = 0.5
    x = np.array([
        i + (0.0 if keys[i] in solo_keys else gap)
        for i in range(len(keys))
    ])

    fig, ax = plt.subplots(figsize=(6, 4.5))

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

    # Value labels
    for bar, mean, se in zip(bars, means, ses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + se + 0.05,
            f"{mean:.2f}%",
            ha="center", va="bottom",
            fontsize=8.5, color="#333333",
        )

    solo_x = [x[i] for i, k in enumerate(keys) if k in solo_keys]
    tab_x  = [x[i] for i, k in enumerate(keys) if k in tab_keys]
    mod_label = "Image" if modality == "img" else "Text"

    ylabel = "Accuracy (%)" if metric == "acc" else "AUROC (%)"
    ax.set_ylabel(ylabel, fontsize=11)
    baseline_label = {"cls": "CLS", "mean": "Mean", "both": "CLS & Mean"}.get(pooling, pooling)
    ax.set_title(
        f"PetFinder — {baseline_label} vs PAL  (n={n_seeds} seed{'s' if n_seeds != 1 else ''}, ±1 SE)",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

    padding = 0.8
    y_min = max(0.0, float((means - ses).min()) - padding)
    y_max = float((means + ses).max()) + padding
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#cccccc", edgecolor="#555555", label=f"{mod_label} only"),
        Patch(facecolor="#cccccc", edgecolor="#555555", hatch="//", label=f"{mod_label} + Tabular"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left", framealpha=0.7)

    fig.tight_layout()

    if output_path is None:
        metric_suffix = f"_{metric}" if metric != "acc" else ""
        output_path = json_path.parent / f"petfinder_{modality}_{pooling}_vs_pal{metric_suffix}.pdf"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bar chart: CLS/Mean pooling vs PAL pooling on PetFinder"
    )
    p.add_argument(
        "json_path", type=Path, nargs="?",
        default=Path("results/petfinder_appendix/multimodal_results.json"),
        help="Path to multimodal_results.json",
    )
    p.add_argument(
        "--modality", choices=["img", "text"], default="img",
        help="Modality suffix used in result keys: 'img' (default) or 'text'",
    )
    p.add_argument(
        "--pooling", choices=["cls", "mean", "both"], default="cls",
        help="Baseline pooling: 'cls' (default), 'mean', or 'both' side-by-side",
    )
    p.add_argument("--metric", choices=["acc", "auroc"], default="acc")
    p.add_argument("--output", "-o", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_comparison(
        args.json_path,
        metric=args.metric,
        output_path=args.output,
        pooling=args.pooling,
        modality=args.modality,
    )
