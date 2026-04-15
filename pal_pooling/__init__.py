"""pal_pooling — core algorithms and visualisation for quality-weighted token pooling.

Submodules
----------
pal_pooler
    PALPooler (base), ImagePALPooler, TextPALPooler, IterativePALPooler:
    sklearn-style fit/transform API for Ridge-based adaptive pooling.
patch_pooling
    Pure NumPy/sklearn algorithms for image patches: entropy, pooling weights,
    quality logits, patch grouping, Ridge pooling helpers, and the full
    refinement pass.
text_pooling
    Pure NumPy/sklearn algorithms for text tokens: token grouping (none /
    sentence), masked Ridge pooling, and the text refinement pass.
patch_visualisation
    Matplotlib figure generation: per-image heatmap overlays and summary charts.
"""

from pal_pooling.pal_pooler import (
    ImagePALPooler,
    IterativePALPooler,
    PALPooler,
    TextPALPooler,
    pooler_factory,
)
from pal_pooling.patch_pooling import (
    compute_patch_entropy,
    compute_patch_pooling_weights,
    compute_patch_quality_logits,
    group_patches,
    refine_dataset_features,
)
from pal_pooling.text_pooling import (
    group_text_tokens,
    refine_text_features,
)
from pal_pooling.patch_visualisation import (
    summary_figure,
    visualise_image,
)

__all__ = [
    # Pooler classes
    "PALPooler",
    "ImagePALPooler",
    "TextPALPooler",
    "IterativePALPooler",
    "pooler_factory",
    # Image patch algorithms
    "compute_patch_entropy",
    "compute_patch_pooling_weights",
    "compute_patch_quality_logits",
    "group_patches",
    "refine_dataset_features",
    # Text token algorithms
    "group_text_tokens",
    "refine_text_features",
    # Visualisation
    "summary_figure",
    "visualise_image",
]
