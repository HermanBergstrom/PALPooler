"""Patch-quality figure generation.

All functions here are pure matplotlib — no dataset loading, no TabICL.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pal_pooling.patch_pooling import compute_patch_entropy, compute_patch_pooling_weights


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _upscale_grid(flat: np.ndarray, n_side: int, patch_size: int) -> np.ndarray:
    """Upscale a flat [P] patch array to a pixel grid [H, W]."""
    return np.repeat(np.repeat(flat.reshape(n_side, n_side), patch_size, axis=0), patch_size, axis=1)


def _add_prob_overlay(
    ax: plt.Axes,
    fig: plt.Figure,
    img_rgb: np.ndarray,
    pixel_grid: np.ndarray,   # [H, W] raw values
    title: str,
    alpha: float,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> "plt.colorbar.Colorbar":
    """Overlay a heatmap on top of the image with a colourbar.

    vmin/vmax: if provided, use these for color scale; otherwise auto-scale.
    Returns the colorbar object so callers can set custom ticks.
    """
    ax.imshow(img_rgb)
    im = ax.imshow(pixel_grid, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    return fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Per-image figure
# ---------------------------------------------------------------------------

def visualise_image(
    image: Image.Image,
    patch_probs:       np.ndarray,            # [P, n_classes]  softmax distribution
    true_label:        int,
    idx_to_class:      dict[int, str],
    n_classes:         int,
    patch_size:        int   = 16,
    alpha:             float = 0.55,
    temperature:       float = 1.0,
    ridge_pred_logits: Optional[np.ndarray] = None,   # [P]  Ridge-predicted quality logits
    class_prior:       Optional[np.ndarray] = None,   # [n_classes]  empirical class frequencies
    weight_method:     str   = "correct_class_prob",  # method used for refinement (highlighted)
    show_pred_label:   bool  = False,                 # add a discrete per-patch predicted-class panel
    show_minority_prob: bool = False,                 # add a P(minority class) panel with image-local scale
    show_per_class_probs: bool = False,               # add one P(class k) panel per class (only when n_classes <= 10)
    binary_dist:       bool  = False,                 # collapse non-correct classes before computing weights
    weight_limits: Optional[dict[str, tuple[float, float]]] = None,  # shared vmin/vmax per method across images
) -> plt.Figure:
    """Figure with overlay panels showing per-patch softmax quality scores.

    Panels: original | P(true) | ccp weights | entropy weights | kl_div weights | wasserstein weights | js_div weights | tvd weights
            [+ Ridge weights when ridge_pred_logits is provided]
    kl_div, wasserstein, js_div, and tvd panels are only shown when class_prior is provided.
    The panel corresponding to weight_method is marked with ★ in its title.
    """
    P      = len(patch_probs)
    n_side = int(round(P ** 0.5))

    # Summary stats from softmax distribution (for suptitle)
    correct_probs     = patch_probs[:, true_label]
    mean_correct_prob = float(correct_probs.mean())
    patch_preds       = patch_probs.argmax(axis=1)
    unique, counts    = np.unique(patch_preds, return_counts=True)
    modal_class       = unique[counts.argmax()]
    consensus_frac    = counts.max() / P
    mean_entropy      = float(compute_patch_entropy(patch_probs).mean() / np.log(n_classes))

    img_rgb = np.array(image.resize((n_side * patch_size, n_side * patch_size)))

    def _up(vals: np.ndarray) -> np.ndarray:
        return _upscale_grid(vals, n_side, patch_size)

    def _mark(title: str, method: str) -> str:
        """Append ★ to panel title when method matches the active weight_method."""
        return f"{title}  ★" if method == weight_method else title

    def _dist_panels(dist: np.ndarray, label: str) -> list[tuple[str, Optional[np.ndarray], dict]]:
        """Build overlay panels for a [P, n_classes] distribution.

        Panels: original | P(true) | ccp weights | entropy weights | kl_div weights | wasserstein weights | js_div weights | tvd weights.
        The panel whose method matches weight_method is marked with ★.
        kl_div, wasserstein, js_div, and tvd panels are shown when class_prior is provided.
        """
        p_true    = dist[:, true_label]
        def _lim(method: str, arr: np.ndarray) -> tuple[float, float]:
            if weight_limits and method in weight_limits:
                return weight_limits[method]
            return float(arr.min()), float(arr.max())

        w_ccp     = compute_patch_pooling_weights(dist, true_label, temperature, "correct_class_prob", binary_dist=binary_dist)
        w_entropy = compute_patch_pooling_weights(dist, true_label, temperature, "entropy", binary_dist=binary_dist)
        ccp_lo, ccp_hi   = _lim("correct_class_prob", w_ccp)
        ent_lo,  ent_hi  = _lim("entropy", w_entropy)
        panels = [
            (f"Original image\n[{label}]", None, {}),
            (f"P(true class)  (mean={p_true.mean():.3f})",
             p_true, {"vmin": 0.0, "vmax": 1.0}),
            (_mark("Correct-class-prob weights", "correct_class_prob"),
             w_ccp, {"vmin": ccp_lo, "vmax": ccp_hi}),
            (_mark("Entropy weights", "entropy"),
             w_entropy, {"vmin": ent_lo, "vmax": ent_hi}),
        ]
        if class_prior is not None:
            w_kl = compute_patch_pooling_weights(
                dist, true_label, temperature, "kl_div", class_prior, binary_dist=binary_dist
            )
            kl_lo, kl_hi = _lim("kl_div", w_kl)
            panels.append((_mark("KL-div weights", "kl_div"),
                           w_kl, {"vmin": kl_lo, "vmax": kl_hi}))
            w_wass = compute_patch_pooling_weights(
                dist, true_label, temperature, "wasserstein", class_prior, binary_dist=binary_dist
            )
            wass_lo, wass_hi = _lim("wasserstein", w_wass)
            panels.append((_mark("Wasserstein weights", "wasserstein"),
                           w_wass, {"vmin": wass_lo, "vmax": wass_hi}))
            w_js = compute_patch_pooling_weights(
                dist, true_label, temperature, "js_div", class_prior, binary_dist=binary_dist
            )
            js_lo, js_hi = _lim("js_div", w_js)
            panels.append((_mark("JS-div weights", "js_div"),
                           w_js, {"vmin": js_lo, "vmax": js_hi}))
            w_tvd = compute_patch_pooling_weights(
                dist, true_label, temperature, "tvd", class_prior, binary_dist=binary_dist
            )
            tvd_lo, tvd_hi = _lim("tvd", w_tvd)
            panels.append((_mark("TVD weights", "tvd"),
                           w_tvd, {"vmin": tvd_lo, "vmax": tvd_hi}))
        if show_pred_label:
            pred_vals = dist.argmax(axis=1).astype(float)
            class_names = [idx_to_class[i] for i in range(n_classes)]
            panels.append((
                "Predicted label",
                pred_vals,
                {"cmap": "tab10", "vmin": -0.5, "vmax": n_classes - 0.5,
                 "tick_labels": class_names},
            ))
        if show_minority_prob and class_prior is not None:
            minority_idx  = int(np.argmin(class_prior))
            minority_name = idx_to_class[minority_idx]
            p_minority    = dist[:, minority_idx]
            min_lo, min_hi = _lim("minority_prob", p_minority)
            panels.append((
                f"P(minority: {minority_name!r})",
                p_minority,
                {"vmin": min_lo, "vmax": min_hi},
            ))
        return panels

    all_rows = [_dist_panels(patch_probs, "Softmax")]

    if ridge_pred_logits is not None:
        if weight_limits and "ridge" in weight_limits:
            ridge_lo, ridge_hi = weight_limits["ridge"]
        else:
            ridge_lo, ridge_hi = float(ridge_pred_logits.min()), float(ridge_pred_logits.max())
        ridge_panel = (
            f"Ridge pooling weights  (max={ridge_pred_logits.max():.4f})",
            ridge_pred_logits,
            {"cmap": "RdYlGn", "vmin": ridge_lo, "vmax": ridge_hi},
        )
        all_rows = [row + [ridge_panel] for row in all_rows]

    if show_per_class_probs and n_classes <= 10:
        per_class_row = [("Original image\n[per-class]", None, {})]
        for k in range(n_classes):
            p_k = patch_probs[:, k]
            star = "  ★" if k == true_label else ""
            per_class_row.append((
                f"P({idx_to_class[k]!r}){star}",
                p_k,
                {"vmin": 0.0, "vmax": 1.0, "cmap": "Blues"},
            ))
        all_rows.append(per_class_row)

    n_cols = max(len(r) for r in all_rows)
    fig, axes = plt.subplots(len(all_rows), n_cols,
                             figsize=(n_cols * 4.5, len(all_rows) * 5),
                             squeeze=False)

    fig.suptitle(
        f"True class: {idx_to_class[true_label]!r}  |  "
        f"mean P(true): {mean_correct_prob:.3f}  |  "
        f"modal pred: {idx_to_class[modal_class]!r} ({consensus_frac:.0%})  |  "
        f"mean entropy: {mean_entropy:.3f}",
        fontsize=11,
    )

    for row_idx, row_panels in enumerate(all_rows):
        for col_idx, (title, vals, kwargs) in enumerate(row_panels):
            ax = axes[row_idx, col_idx]
            if vals is None:
                ax.imshow(img_rgb)
                ax.set_title(title)
                ax.axis("off")
            else:
                tick_labels = kwargs.pop("tick_labels", None)
                cbar = _add_prob_overlay(ax, fig, img_rgb, _up(vals), title, alpha, **kwargs)
                if tick_labels is not None:
                    cbar.set_ticks(range(len(tick_labels)))
                    cbar.set_ticklabels(tick_labels)
        for col_idx in range(len(row_panels), n_cols):
            axes[row_idx, col_idx].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def summary_figure(results: list[dict]) -> plt.Figure:
    """Bar chart of per-image mean correct-class probability."""
    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.2), 4))
    if results:
        xs     = np.arange(len(results))
        probs  = [r["mean_correct_prob"] for r in results]
        labels = [r["class_name"]        for r in results]
        ax.bar(xs, probs, color="steelblue")
        ax.axhline(np.mean(probs), color="red", linestyle="--", label=f"mean={np.mean(probs):.3f}")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend()
    ax.set_ylabel("Mean P(true class) across patches")
    ax.set_ylim(0, 1)
    ax.set_title("Per-image patch prediction quality")
    fig.tight_layout()
    return fig
