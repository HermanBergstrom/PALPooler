"""
Bar plots comparing pooling methods on PAD-UFES seed sweeps.
Supports results/pad_ufes_seeds and (when available) results/pad_ufes_cross_seeds.
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIRS = {
    "Seeds": "results/pad_ufes_seeds",
    "Cross Seeds": "results/pad_ufes_cross_seeds",
}

BAR_LABELS = [
    "CLS",
    "Mean Pooling",
    "Iter. Pool (Final)",
    "Iter. Pool (Best Val)",
]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def load_runs(sweep_json_path):
    with open(sweep_json_path) as f:
        raw = f.read().replace("NaN", "null")
    return json.loads(raw)["runs"]


def extract_accuracies(run):
    cls_acc = run["baselines"]["cls_token"]
    mean_acc = run["baselines"]["mean_pool"]

    iter_stages = [s for s in run["stages"] if s["tag"] != "baseline"]
    if not iter_stages:
        return None

    final_acc = iter_stages[-1]["test_accuracy"]

    # Best-val: stage with highest val_accuracy; fall back to final if no val scores
    stages_with_val = [s for s in iter_stages if s.get("val_accuracy") is not None]
    if stages_with_val:
        best_val_stage = max(stages_with_val, key=lambda s: s["val_accuracy"])
        best_val_acc = best_val_stage["test_accuracy"]
    else:
        best_val_acc = final_acc

    return cls_acc, mean_acc, final_acc, best_val_acc


def compute_stats(values):
    arr = np.array(values)
    mean = arr.mean()
    sem = arr.std(ddof=1) / math.sqrt(len(arr))
    return mean, sem


def plot_directory(ax, label, sweep_path):
    runs = load_runs(sweep_path)
    per_bar = [[], [], [], []]
    for run in runs:
        result = extract_accuracies(run)
        if result is None:
            continue
        for i, v in enumerate(result):
            if v is not None:
                per_bar[i].append(v)

    means, sems = [], []
    for vals in per_bar:
        m, s = compute_stats(vals)
        means.append(m)
        sems.append(s)

    x = np.arange(len(BAR_LABELS))
    bars = ax.bar(x, means, yerr=sems, capsize=4, color=COLORS, width=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(BAR_LABELS, rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(label)
    ax.grid(axis="y", zorder=0, alpha=0.4)
    ax.set_ylim(min(means) * 0.96, max(means) * 1.02)

    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main():
    available = {
        label: os.path.join(path, "seed_sweep_results.json")
        for label, path in RESULTS_DIRS.items()
        if os.path.exists(os.path.join(path, "seed_sweep_results.json"))
    }

    if not available:
        raise FileNotFoundError("No seed_sweep_results.json found in expected directories.")

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5), squeeze=False)
    for ax, (label, path) in zip(axes[0], available.items()):
        plot_directory(ax, label, path)

    fig.suptitle("PAD-UFES Pooling Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = "results/pad_ufes_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
