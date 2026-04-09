"""Plot accuracy (or AUROC) vs. training-set size, averaged over seeds.

Reads seed_sweep_results.json and averages metrics across all seeds.

Produces lines for:
  - CLS token
  - Mean pooling
  - 1 iteration
  - 2 iterations
  - 3 iterations
  - Attn. pooling (UB)

X-axis is log-scaled; Y-axis is classification accuracy (0–1) or AUROC.
Error bars show ±1 std dev across seeds.

Usage:
    python pal_pooling/plot_seed_sweep.py seed_sweep_results.json
    python pal_pooling/plot_seed_sweep.py seed_sweep_results.json --output my_plot.pdf
    python pal_pooling/plot_seed_sweep.py seed_sweep_results.json --metric auroc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------

def _cls_token(run: dict, metric: str = "accuracy") -> float | None:
    b = run.get("baselines", {})
    key = "cls_token" if metric == "accuracy" else "cls_token_auroc"
    return b.get(key)


def _mean_pool(run: dict, metric: str = "accuracy") -> float | None:
    b = run.get("baselines", {})
    key = "mean_pool" if metric == "accuracy" else "mean_pool_auroc"
    return b.get(key)


def _stage_val(run: dict, stage_index: int, metric: str = "accuracy") -> float | None:
    stages = run.get("stages", [])
    if stage_index < len(stages):
        key = "test_accuracy" if metric == "accuracy" else "test_auroc"
        v = stages[stage_index].get(key)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    return None


def _attn_pool(run: dict, metric: str = "accuracy") -> float | None:
    v = run.get("baselines", {}).get("attn_pool")
    if v is None:
        return None
    key = "test_acc" if metric == "accuracy" else "test_auroc"
    return v.get(key)


# ---------------------------------------------------------------------------
# Time extractors  (return seconds for a single run, or None if missing)
# ---------------------------------------------------------------------------

def _stage_time_cumulative(run: dict, up_to_stage: int) -> float | None:
    """Cumulative time to reach up_to_stage.

    For each stage < up_to_stage: refine_time_s + eval_time_s.
    For up_to_stage itself:       fit_time_s   + eval_time_s.
    """
    stages = run.get("stages", [])
    total = 0.0
    for i in range(1, up_to_stage + 1):
        if i >= len(stages):
            return None
        s = stages[i]
        et = s.get("eval_time_s")
        stage_time_key = "fit_time_s" if i == up_to_stage else "refine_time_s"
        st = s.get(stage_time_key)
        if st is None or et is None:
            return None
        total += st + et
    return total


def _attn_time(run: dict) -> float | None:
    v = run.get("baselines", {}).get("attn_pool")
    if v is None:
        return None
    return v.get("time_to_best_s")


def _avg_time(entries: list[dict], time_extractor) -> str:
    """Return a human-readable average time string, or '' if no data.

    entries: list of dicts with 'run' key containing the run dict
    """
    times = [t for e in entries if (t := time_extractor(e["run"])) is not None]
    if not times:
        return ""
    avg = np.mean(times)
    if avg < 60:
        return f"{avg:.1f}s"
    return f"{avg / 60:.1f}min"


def _build_series(metric: str) -> list[tuple[str, object, str, str, bool, object]]:
    # (label, extractor, color, marker, is_attn, time_extractor)
    return [
        ("CLS token",          lambda r, m=metric: _cls_token(r, m),     "#e07b39", "D", False, None),
        ("Mean pooling",       lambda r, m=metric: _mean_pool(r, m),     "#4c72b0", "o", False, None),
        ("1 iteration",        lambda r, m=metric: _stage_val(r, 1, m),  "#55a868", "s", False, lambda r: _stage_time_cumulative(r, 1)),
        ("2 iterations",       lambda r, m=metric: _stage_val(r, 2, m),  "#c44e52", "^", False, lambda r: _stage_time_cumulative(r, 2)),
        ("3 iterations",       lambda r, m=metric: _stage_val(r, 3, m),  "#8172b2", "P", False, lambda r: _stage_time_cumulative(r, 3)),
        ("Attn. pooling (UB)", lambda r, m=metric: _attn_pool(r, m),     "#937860", "*", True,  _attn_time),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_seed_sweep(
    seed_sweep_path: Path,
    output_path: Path | None = None,
    exclude_cls: bool = False,
    exclude_attn: bool = False,
    metric: str = "accuracy",
) -> None:
    """Plot accuracy/AUROC vs n_train, averaged over seeds with error bars."""
    with seed_sweep_path.open() as f:
        data = json.load(f)

    # Group n_train_sweep entries by n_train across all seeds/runs
    # Structure: {n_train: [{seed: int, run: dict}, ...]}
    n_train_groups: dict[int, list[dict]] = defaultdict(list)

    for seed_run in data["runs"]:
        seed = seed_run["seed"]
        for n_train_entry in seed_run.get("n_train_sweep", []):
            n_train = n_train_entry["n_train"]
            n_train_groups[n_train].append({"seed": seed, "run": n_train_entry})

    sorted_n_trains = sorted(n_train_groups.keys())
    x = sorted_n_trains

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, extractor, color, marker, is_attn, time_extractor in _build_series(metric):
        if exclude_cls and label == "CLS token":
            continue
        if exclude_attn and is_attn:
            continue

        means, stds, valid_xs = [], [], []
        for n_train in sorted_n_trains:
            values = [
                v for entry in n_train_groups[n_train]
                if (v := extractor(entry["run"])) is not None
            ]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
                valid_xs.append(n_train)

        if not valid_xs:
            continue

        if time_extractor is not None:
            t_str = _avg_time(n_train_groups[sorted_n_trains[0]], time_extractor)
            legend_label = f"{label} (~{t_str})" if t_str else label
        else:
            legend_label = label

        ax.errorbar(
            valid_xs, means,
            yerr=stds,
            label=legend_label,
            color=color,
            marker=marker,
            markersize=8 if is_attn else 6,
            linewidth=1.8,
            linestyle="--" if is_attn else "-",
            capsize=4,
            capthick=1.2,
        )

    ylabel = "Test accuracy" if metric == "accuracy" else "Test AUROC"
    title  = (
        f"{'Accuracy' if metric == 'accuracy' else 'AUROC'} vs. training set size"
        f" (averaged over {len(data['seeds'])} seeds)"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Training set size", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels([str(xi) for xi in x], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0] - 0.01))
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()

    if output_path is None:
        suffix = "" if metric == "accuracy" else f"_{metric}"
        output_path = seed_sweep_path.parent / f"seed_sweep_plot{suffix}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot accuracy vs. n_train from seed_sweep_results.json, averaged over seeds"
    )
    p.add_argument("sweep_json", type=Path,
                   help="Path to seed_sweep_results.json produced by run_seed_sweep")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output path for the figure (default: <sweep_dir>/seed_sweep_plot.png)")
    p.add_argument("--no-cls", action="store_true",
                   help="Exclude the CLS token baseline from the plot")
    p.add_argument("--no-attn", action="store_true",
                   help="Exclude the attention pooling upper-bound line from the plot")
    p.add_argument("--metric", choices=["accuracy", "auroc"], default="accuracy",
                   help="Metric to plot: 'accuracy' (default) or 'auroc'")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_seed_sweep(
        args.sweep_json, args.output,
        exclude_cls=args.no_cls, exclude_attn=args.no_attn,
        metric=args.metric,
    )
