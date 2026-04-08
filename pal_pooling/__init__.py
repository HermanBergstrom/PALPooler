"""pal_pooling — core algorithms and visualisation for quality-weighted patch pooling.

Submodules
----------
pal_pooler
    PALPooler class: sklearn-style fit/transform API for Ridge-based adaptive patch pooling.
patch_pooling
    Pure NumPy/sklearn algorithms: entropy, pooling weights, quality logits,
    patch grouping, Ridge pooling helpers, and the full refinement pass.
patch_visualisation
    Matplotlib figure generation: per-image heatmap overlays and summary charts.
"""

from pal_pooling.pal_pooler import PALPooler
from pal_pooling.patch_pooling import (
    compute_patch_entropy,
    compute_patch_pooling_weights,
    compute_patch_quality_logits,
    group_patches,
    refine_dataset_features,
)
from pal_pooling.patch_visualisation import (
    summary_figure,
    visualise_image,
)

__all__ = [
    "PALPooler",
    "compute_patch_entropy",
    "compute_patch_pooling_weights",
    "compute_patch_quality_logits",
    "group_patches",
    "refine_dataset_features",
    "summary_figure",
    "visualise_image",
]
