"""Text token/word-level pooling visualizations.

All functions here are pure matplotlib — no dataset loading, no TabICL.
Each weight method gets its own panel showing the full text with words
color-coded by their pooling weight.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np

from pal_pooling.patch_pooling import compute_patch_pooling_weights


# ---------------------------------------------------------------------------
# Helper: aggregate token weights to word weights
# ---------------------------------------------------------------------------

def aggregate_token_to_word_weights(
    token_weights: np.ndarray,
    token_to_word: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum per-token weights into per-word weights.

    Tokens with token_to_word == -1 (special tokens) are excluded.

    Returns
    -------
    word_weights : np.ndarray [W]
    word_mask    : np.ndarray [W] bool — True for words that had at least one token
    """
    token_to_word = np.asarray(token_to_word)
    token_weights = np.asarray(token_weights)

    valid = token_to_word >= 0
    valid_word_ids = token_to_word[valid].astype(np.int64)
    valid_weights = token_weights[valid]

    if len(valid_word_ids) == 0:
        return np.array([]), np.array([], dtype=bool)

    n_words = int(valid_word_ids.max()) + 1
    word_weights = np.bincount(valid_word_ids, weights=valid_weights, minlength=n_words).astype(np.float32)
    word_mask = np.bincount(valid_word_ids, minlength=n_words).astype(bool)

    return word_weights, word_mask


# ---------------------------------------------------------------------------
# Core renderer: one axes = one text with per-word highlights
# ---------------------------------------------------------------------------

_CHAR_WIDTH  = 0.011   # approximate width per character in axes coordinates
_WORD_GAP    = 0.008   # horizontal gap between words
_LINE_HEIGHT = 0.10    # vertical distance between lines (axes coords)
_MARGIN_X    = 0.02
_MARGIN_Y    = 0.92
_MAX_WORDS   = 120     # truncate very long texts for readability


def _render_highlighted_text(
    ax: plt.Axes,
    text: str,
    word_weights: np.ndarray,
    word_mask: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdYlGn",
    title: str = "",
) -> None:
    """Render `text` onto `ax` with per-word background colors from `word_weights`."""
    words = text.split()[:_MAX_WORDS]

    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if not words:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center", transform=ax.transAxes)
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold")
        return

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)

    x, y = _MARGIN_X, _MARGIN_Y

    for word_idx, word in enumerate(words):
        word_w = len(word) * _CHAR_WIDTH + _WORD_GAP

        # Wrap to next line
        if x + word_w > 0.99 and x > _MARGIN_X:
            y -= _LINE_HEIGHT
            x = _MARGIN_X

        if y < 0.0:
            break

        # Color
        if word_idx < len(word_mask) and word_mask[word_idx]:
            face = cmap_obj(norm(word_weights[word_idx]))
            ink  = "black"
            alpha = 0.85
            edge = "none"
        else:
            face  = (0.92, 0.92, 0.92)
            ink   = "#999999"
            alpha = 0.5
            edge  = "none"

        text_w = len(word) * _CHAR_WIDTH
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.038),
            text_w + _WORD_GAP * 0.5,
            0.072,
            boxstyle="round,pad=0.002",
            facecolor=face,
            edgecolor=edge,
            alpha=alpha,
            transform=ax.transAxes,
            zorder=1,
        )
        ax.add_patch(rect)

        ax.text(
            x + (text_w + _WORD_GAP * 0.5) / 2,
            y,
            word,
            ha="center",
            va="center",
            fontsize=7.5,
            color=ink,
            transform=ax.transAxes,
            zorder=2,
        )

        x += word_w

    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)


# ---------------------------------------------------------------------------
# Per-text figure
# ---------------------------------------------------------------------------

def visualise_text(
    text: str,
    token_to_word: np.ndarray,
    token_ids: np.ndarray,
    token_probs: np.ndarray,          # [T, n_classes]
    true_label: int,
    idx_to_class: dict[int, str],
    n_classes: int,
    temperature: float = 1.0,
    ridge_weights: Optional[np.ndarray] = None,
    class_prior: Optional[np.ndarray] = None,
    weight_method: str = "correct_class_prob",
    binary_dist: bool = False,
) -> plt.Figure:
    """Figure with one highlighted-text panel per weight method.

    Each panel shows the full text with words color-coded by their pooling
    weight under that method.  The active `weight_method` is marked with ★.
    """
    T = len(token_probs)
    assert len(token_to_word) == T

    valid_mask = np.asarray(token_to_word) >= 0   # exclude CLS / SEP / padding
    correct_probs = token_probs[:, true_label]
    valid_correct = correct_probs[valid_mask]
    mean_correct_prob = float(valid_correct.mean()) if valid_mask.sum() > 0 else 0.0
    token_preds = token_probs[valid_mask].argmax(axis=1)
    unique, counts = np.unique(token_preds, return_counts=True)
    modal_class = int(unique[counts.argmax()]) if len(unique) > 0 else true_label
    consensus_frac = float(counts.max()) / valid_mask.sum() if valid_mask.sum() > 0 else 0.0

    def _mark(title: str, method: str) -> str:
        return f"{title}  ★" if method == weight_method else title

    def _pool(dist: np.ndarray, method: str, prior: Optional[np.ndarray] = None):
        # Only compute softmax over content tokens (token_to_word >= 0).
        # Including CLS / SEP / padding in the softmax dilutes real token weights.
        valid = token_to_word >= 0
        full_weights = np.zeros(len(token_to_word), dtype=np.float32)
        if valid.sum() > 0:
            full_weights[valid] = compute_patch_pooling_weights(
                dist[valid], true_label, temperature, method, prior, binary_dist=binary_dist
            )
        return aggregate_token_to_word_weights(full_weights, token_to_word)

    # Build panel list — each entry is (title, word_weights, word_mask)
    panels: list[dict] = []

    # Panel: mean P(true class) per word (mean over that word's tokens, not sum).
    # Summing would give multi-token words higher values simply because they have
    # more tokens, saturating the vmax=1.0 ceiling and distorting the display.
    raw_word_sum, raw_word_mask = aggregate_token_to_word_weights(correct_probs, token_to_word)
    token_counts, _ = aggregate_token_to_word_weights(
        np.ones(len(token_to_word), dtype=np.float32), token_to_word
    )
    raw_word_w = np.where(token_counts > 0, raw_word_sum / token_counts, 0.0).astype(np.float32)
    panels.append({
        "title": "mean P(true class) per word",
        "method": "__raw__",
        "word_weights": raw_word_w,
        "word_mask":    raw_word_mask,
        "vmin": 0.0,
        "vmax": 1.0,
    })

    # CCP weights
    w, m = _pool(token_probs, "correct_class_prob")
    panels.append({
        "title":  _mark("CCP pooling weights", "correct_class_prob"),
        "method": "correct_class_prob",
        "word_weights": w,
        "word_mask": m,
        "vmin": float(w[m].min()) if m.sum() > 0 else 0.0,
        "vmax": float(w[m].max()) if m.sum() > 0 else 1.0,
    })

    # Entropy weights
    w, m = _pool(token_probs, "entropy")
    panels.append({
        "title":  _mark("Entropy pooling weights", "entropy"),
        "method": "entropy",
        "word_weights": w,
        "word_mask": m,
        "vmin": float(w[m].min()) if m.sum() > 0 else 0.0,
        "vmax": float(w[m].max()) if m.sum() > 0 else 1.0,
    })

    # Divergence-based weights
    if class_prior is not None:
        for method in ("kl_div", "wasserstein", "js_div", "tvd"):
            w, m = _pool(token_probs, method, class_prior)
            panels.append({
                "title":  _mark(f"{method} pooling weights", method),
                "method": method,
                "word_weights": w,
                "word_mask": m,
                "vmin": float(w[m].min()) if m.sum() > 0 else 0.0,
                "vmax": float(w[m].max()) if m.sum() > 0 else 1.0,
            })

    # Ridge weights — softmax was computed over all T_max positions including padding;
    # renormalize over valid tokens before aggregating to word level.
    if ridge_weights is not None:
        valid = token_to_word >= 0
        ridge_valid = np.zeros_like(ridge_weights)
        if valid.sum() > 0:
            rv = ridge_weights[valid]
            s = rv.sum()
            ridge_valid[valid] = rv / s if s > 0 else rv
        w, m = aggregate_token_to_word_weights(ridge_valid, token_to_word)
        panels.append({
            "title":  "Ridge pooling weights",
            "method": "ridge",
            "word_weights": w,
            "word_mask": m,
            "vmin": float(w[m].min()) if m.sum() > 0 else 0.0,
            "vmax": float(w[m].max()) if m.sum() > 0 else 1.0,
        })

    n_panels = len(panels)
    panel_h  = 3.5           # inches per panel
    fig_h    = 1.2 + n_panels * panel_h

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, fig_h))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f'"{text[:80]}{"…" if len(text) > 80 else ""}"\n'
        f"True: {idx_to_class[true_label]!r}  |  "
        f"mean P(true): {mean_correct_prob:.3f}  |  "
        f"modal pred: {idx_to_class[modal_class]!r} ({consensus_frac:.0%})",
        fontsize=10,
        y=1.0,
    )

    for ax, panel in zip(axes, panels):
        _render_highlighted_text(
            ax,
            text,
            panel["word_weights"],
            panel["word_mask"],
            vmin=panel["vmin"],
            vmax=panel["vmax"],
            cmap="RdYlGn",
            title=panel["title"],
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Batch visualization
# ---------------------------------------------------------------------------

def visualise_text_batch(
    texts: list[str],
    token_to_word_batch: np.ndarray,    # [N, T_max]
    token_ids_batch: np.ndarray,        # [N, T_max]
    token_probs_batch: np.ndarray,      # [N, T_max, n_classes]
    true_labels: np.ndarray,            # [N]
    idx_to_class: dict[int, str],
    n_classes: int,
    ridge_weights_batch: Optional[np.ndarray] = None,
    class_prior: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    weight_method: str = "correct_class_prob",
    binary_dist: bool = False,
    max_samples: int = 3,
) -> list[plt.Figure]:
    """Generate per-text visualizations for a batch of samples.

    Returns one plt.Figure per sample (up to max_samples).
    """
    n_vis = min(len(texts), max_samples)
    figures = []

    for i in range(n_vis):
        token_ids   = token_ids_batch[i]
        valid_mask  = token_ids != 0
        if valid_mask.sum() == 0:
            continue

        last_valid    = int(np.where(valid_mask)[0][-1]) + 1
        token_to_word = token_to_word_batch[i, :last_valid]
        token_ids_i   = token_ids[:last_valid]
        token_probs   = token_probs_batch[i, :last_valid]
        ridge_weights = (
            ridge_weights_batch[i, :last_valid]
            if ridge_weights_batch is not None else None
        )

        fig = visualise_text(
            text=texts[i],
            token_to_word=token_to_word,
            token_ids=token_ids_i,
            token_probs=token_probs,
            true_label=int(true_labels[i]),
            idx_to_class=idx_to_class,
            n_classes=n_classes,
            temperature=temperature,
            ridge_weights=ridge_weights,
            class_prior=class_prior,
            weight_method=weight_method,
            binary_dist=binary_dist,
        )
        figures.append(fig)

    return figures
