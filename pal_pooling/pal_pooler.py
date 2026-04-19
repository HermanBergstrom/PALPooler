"""PALPooler — Pseudo-Attention Label Pooler.

Class hierarchy
---------------
:class:`PALPooler`
    Abstract base class.  Holds shared fitted attributes, persistence
    (``save`` / ``load``), and ``score_tabicl``.

:class:`ImagePALPooler` (subclass of PALPooler)
    Single-stage Ridge-based adaptive patch pooling for image tokens
    (DINO / ViT patch embeddings).  This is the original ``PALPooler``
    implementation, renamed.  A backward-compatible alias
    ``PALPooler = ImagePALPooler`` is exported from ``__init__.py``.

:class:`TextPALPooler` (subclass of PALPooler)
    Single-stage Ridge-based adaptive token pooling for text
    (BERT embeddings).  Handles variable-length sequences with padding
    masks and two grouping modes: ``"none"`` (individual tokens) and
    ``"sentence"`` (mean-pooled sentence spans delimited by ``[SEP]``).

:class:`IterativePALPooler`
    Multi-stage pooler that chains :class:`ImagePALPooler` or
    :class:`TextPALPooler` stages automatically.  The modality is selected
    via ``modality="image"`` (default) or ``modality="text"``.

Typical image usage (backward-compatible)
------------------------------------------
>>> from pal_pooling import ImagePALPooler   # or: from pal_pooling import PALPooler
>>> pooler = IterativePALPooler(tabicl, refinement_cfg)
>>> pooler.fit(train_patches, train_labels)
>>> X_train = pooler.transform(train_patches)

Typical text usage
-------------------
>>> from pal_pooling import TextPALPooler, IterativePALPooler
>>> from pal_pooling.config import TextRefinementConfig
>>>
>>> text_cfg = TextRefinementConfig(
...     refine=True,
...     text_group_modes=["sentence", "none"],
...     temperature=[2.0, 1.0],
...     ridge_alpha=[10.0, 1.0],
...     weight_method="correct_class_prob",
...     ...
... )
>>> pooler = IterativePALPooler(tabicl, text_cfg, modality="text")
>>> pooler.fit(train_tokens, train_labels,
...            token_ids=train_token_ids,
...            attention_mask=train_attention_mask)
>>> X_train = pooler.transform(train_tokens,
...                            token_ids=train_token_ids,
...                            attention_mask=train_attention_mask)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from pal_pooling.config import RefinementConfig, TextRefinementConfig
from tabicl import TabICLClassifier

from pal_pooling.patch_pooling import (
    _ridge_pool_weights,
    group_patches,
    refine_dataset_features,
)
from pal_pooling.text_pooling import (
    _ridge_pool_weights_text,
    group_text_tokens,
    refine_text_features,
)


def _append_cls(grouped: np.ndarray, cls_tokens: Optional[np.ndarray]) -> np.ndarray:
    """Append a CLS token as one extra patch/group per sample.

    Parameters
    ----------
    grouped : np.ndarray, shape [N, G, D]
    cls_tokens : np.ndarray or None, shape [N, D]

    Returns
    -------
    np.ndarray, shape [N, G+1, D]  (or [N, G, D] when cls_tokens is None)
    """
    if cls_tokens is None:
        return grouped
    cls = np.asarray(cls_tokens, dtype=np.float32)[:, None, :]  # [N, 1, D]
    return np.concatenate([grouped, cls], axis=1)


def _append_cls_masked(
    grouped: np.ndarray,            # [N, G_max, D]
    group_mask: np.ndarray,         # [N, G_max]  bool
    cls_tokens: Optional[np.ndarray],  # [N, D] or None
) -> tuple[np.ndarray, np.ndarray]:
    """Append a CLS token as one extra valid group per sample (text variant).

    Returns updated (grouped, group_mask) with the CLS appended at position
    G_max (i.e. the last column), marked as valid in the mask.
    """
    if cls_tokens is None:
        return grouped, group_mask
    cls = np.asarray(cls_tokens, dtype=np.float32)[:, None, :]   # [N, 1, D]
    cls_valid = np.ones((grouped.shape[0], 1), dtype=bool)        # [N, 1]
    return (
        np.concatenate([grouped, cls], axis=1),
        np.concatenate([group_mask, cls_valid], axis=1),
    )


# ===========================================================================
# Base class
# ===========================================================================

class PALPooler:
    """Abstract base class for single-stage PAL poolers.

    Subclasses implement modality-specific ``fit``, ``transform``, and
    ``_group`` logic.  The base class provides shared fitted attributes,
    persistence, and ``score_tabicl``.

    Fitted attributes (available after ``fit``)
    -------------------------------------------
    ridge_model_ : Ridge or RidgeGPU
    feature_scaler_ : StandardScaler or None
    support_ : np.ndarray, shape [N, D]  — raw (pre-PCA) repooled support
    support_labels_ : np.ndarray, shape [N]
    scoring_clf_ : TabICLClassifier
    class_prior_ : np.ndarray or None
    embed_dim_ : int
    fit_time_s_ : float
    pool_time_s_ : float
    _pca_ : PCA or None  (internal; used for stage chaining)
    _support_projected_ : np.ndarray  (internal; used for stage chaining)
    """

    # Subclasses must set these in __init__
    tabicl: TabICLClassifier
    refinement_cfg: Union[RefinementConfig, TextRefinementConfig]
    seed: int
    gpu_ridge_device: str

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the fitted pooler to a ``joblib`` file."""
        import joblib
        self._check_fitted()
        joblib.dump(self, Path(path))
        print(f"[{type(self).__name__}] Saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PALPooler":
        """Load a previously saved pooler from a ``joblib`` file."""
        import joblib
        pooler = joblib.load(Path(path))
        if not isinstance(pooler, cls):
            raise TypeError(
                f"Loaded object is {type(pooler).__name__}, expected {cls.__name__}"
            )
        return pooler

    # ------------------------------------------------------------------
    # Shared evaluation
    # ------------------------------------------------------------------

    def score_tabicl(
        self,
        query_features: np.ndarray,   # [N_test, D]  already-pooled embeddings
        query_labels: np.ndarray,
        n_estimators: Optional[int] = None,
    ) -> tuple[float, float]:
        """Evaluate accuracy and AUROC on pre-pooled query features.

        Projects *query_features* into the internal PCA space, fits a fresh
        ``TabICLClassifier`` on ``_support_projected_``, and reports
        accuracy + AUROC.

        Subclasses expose modality-specific wrappers that call ``transform``
        first and then delegate here.

        Returns
        -------
        (accuracy, auroc) : tuple[float, float]
        """
        from sklearn.metrics import roc_auc_score

        self._check_fitted()
        n_est = n_estimators or getattr(self.tabicl, "n_estimators", 1)

        if self._pca_ is not None:
            query_proj = self._pca_.transform(query_features).astype(np.float32)
        else:
            query_proj = query_features.astype(np.float32)

        clf = TabICLClassifier(n_estimators=n_est, random_state=self.seed)
        clf.fit(self._support_projected_, self.support_labels_)
        proba = clf.predict_proba(query_proj)
        acc = float((np.argmax(proba, axis=1) == query_labels).mean())
        try:
            if proba.shape[1] == 2:
                auroc = float(roc_auc_score(query_labels, proba[:, 1]))
            else:
                auroc = float(
                    roc_auc_score(query_labels, proba, multi_class="ovr", average="macro")
                )
        except ValueError:
            auroc = float("nan")
        return acc, auroc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "ridge_model_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet.  Call fit() first."
            )


# ===========================================================================
# Image pooler
# ===========================================================================

class ImagePALPooler(PALPooler):
    """Single-stage Ridge-based adaptive patch pooling for image tokens.

    Fits a Ridge regression model that predicts per-patch quality logits from
    raw DINO patch embeddings, driven by pseudo-labels from a TabICL classifier
    evaluated on a support set.  At transform time the Ridge logits drive a
    softmax-weighted pooling that collapses ``[N, P, D] → [N, D]``.

    Parameters
    ----------
    tabicl : TabICLClassifier
    refinement_cfg : RefinementConfig
    seed : int
    gpu_ridge_device : str

    Fitted attributes
    -----------------
    n_patches_grouped_ : int  — P' after spatial grouping
    (all base-class attributes apply)
    """

    def __init__(
        self,
        tabicl: TabICLClassifier,
        refinement_cfg: RefinementConfig,
        seed: int = 42,
        gpu_ridge_device: str = "cuda",
    ) -> None:
        self.tabicl = tabicl
        self.refinement_cfg = refinement_cfg
        self.pca_dim = refinement_cfg.tabicl_pca_dim
        self.seed = seed
        self.gpu_ridge_device = gpu_ridge_device

    # ------------------------------------------------------------------
    # Core sklearn-style API
    # ------------------------------------------------------------------

    def fit(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
        initial_support: Optional[np.ndarray] = None,
        initial_pca: Optional[PCA] = None,
        context_features: Optional[np.ndarray] = None,
        tabular_probs: Optional[np.ndarray] = None,
        val_patches: Optional[np.ndarray] = None,           # [N_val, P, D]
        val_labels: Optional[np.ndarray] = None,            # [N_val]
        val_cls_tokens: Optional[np.ndarray] = None,        # [N_val, D]
        val_context_features: Optional[np.ndarray] = None,  # [N_val, D_ctx]
        val_tabular_probs: Optional[np.ndarray] = None,     # [N_val, n_cls]
    ) -> "ImagePALPooler":
        """Fit the Ridge quality model on image patch embeddings.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
        labels : np.ndarray, shape [N]
        cls_tokens : np.ndarray or None, shape [N, D]
        initial_support : np.ndarray or None
            Pre-built support from a previous stage (PCA-projected space).
        initial_pca : PCA or None
        context_features, tabular_probs, val_* : optional, see patch_pooling docs.
        """
        patches = np.asarray(patches, dtype=np.float32)
        labels  = np.asarray(labels)
        N, P, D = patches.shape
        self.embed_dim_ = D

        grouped = group_patches(patches, self.refinement_cfg.patch_group_sizes)
        grouped = _append_cls(grouped, cls_tokens)
        self.n_patches_grouped_ = grouped.shape[1]

        if initial_support is not None:
            support     = initial_support
            current_pca = initial_pca
        else:
            support_raw = grouped.mean(axis=1)
            if self.pca_dim is not None:
                n_comp      = min(self.pca_dim, N, D)
                current_pca = PCA(n_components=n_comp, random_state=self.seed)
                support     = current_pca.fit_transform(support_raw).astype(np.float32)
            else:
                current_pca = None
                support     = support_raw

        aoe_mask: Optional[np.ndarray] = None
        if self.refinement_cfg.aoe_class is not None:
            aoe_mask = (labels == self.refinement_cfg.aoe_class)

        val_grouped = None
        if val_patches is not None and val_labels is not None:
            _vp = np.asarray(val_patches, dtype=np.float32)
            val_grouped = group_patches(_vp, self.refinement_cfg.patch_group_sizes)
            val_grouped = _append_cls(val_grouped, val_cls_tokens)

        (refined_support, new_pca, weights_ridge, ridge_model, feature_scaler,
         scoring_clf, fit_time_s, pool_time_s, updated_class_prior) = refine_dataset_features(
            train_patches=grouped,
            train_labels=labels,
            support_features=support,
            pca=current_pca,
            seed=self.seed,
            aoe_mask=aoe_mask,
            gpu_ridge_device=self.gpu_ridge_device,
            tabicl=self.tabicl,
            refinement_cfg=self.refinement_cfg,
            context_features=context_features,
            tabular_probs=tabular_probs,
            val_patches=val_grouped,
            val_labels=val_labels,
            val_context_features=val_context_features,
            val_tabular_probs=val_tabular_probs,
        )

        repooled_raw = (
            weights_ridge[:, :, None] * grouped
        ).sum(axis=1).astype(np.float32)

        self.ridge_model_          = ridge_model
        self.feature_scaler_       = feature_scaler
        self._pca_                 = new_pca
        self._support_projected_   = refined_support
        self.support_              = repooled_raw
        self.support_labels_       = labels
        self.scoring_clf_          = scoring_clf
        self.class_prior_          = updated_class_prior
        self.fit_time_s_           = fit_time_s
        self.pool_time_s_          = pool_time_s
        return self

    def _fit_from_indices(self, data, labels, cls_tokens, init_sup, init_pca,
                          context_features, tabular_probs, val_tabular_probs,
                          train_idx, val_idx, **kw):
        tr = train_idx
        vl = val_idx
        patches = data[tr] if tr is not None else data
        _cls    = cls_tokens[tr] if (cls_tokens is not None and tr is not None) else cls_tokens
        _sup    = init_sup[tr] if (init_sup is not None and tr is not None and len(init_sup) == len(data)) else init_sup
        _ctx    = context_features[tr] if (context_features is not None and tr is not None) else context_features
        _tab    = tabular_probs[tr] if (tabular_probs is not None and tr is not None) else tabular_probs
        _val_patches = data[vl]              if vl is not None else None
        _val_labels  = labels[vl]            if vl is not None else None
        _val_cls     = cls_tokens[vl]        if (cls_tokens is not None and vl is not None) else None
        _val_ctx     = context_features[vl]  if (context_features is not None and vl is not None) else None
        _val_tab     = tabular_probs[vl]     if (tabular_probs is not None and vl is not None) else None
        self.fit(patches, labels[tr] if tr is not None else labels,
                 cls_tokens=_cls, initial_support=_sup, initial_pca=init_pca,
                 context_features=_ctx, tabular_probs=_tab,
                 val_patches=_val_patches, val_labels=_val_labels,
                 val_cls_tokens=_val_cls, val_context_features=_val_ctx,
                 val_tabular_probs=_val_tab)

    def _transform_subset(self, data, idx, cls_tokens, token_ids=None, attention_mask=None):
        """Pool data[idx] (or all data if idx is None)."""
        patches = data[idx] if idx is not None else data
        _cls    = cls_tokens[idx] if (cls_tokens is not None and idx is not None) else cls_tokens
        return self.transform(patches, cls_tokens=_cls)

    def _transform_all(self, data, cls_tokens, token_ids=None, attention_mask=None):
        return self.transform(data, cls_tokens=cls_tokens)

    def transform(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pool patches with the fitted Ridge model → [N, D]."""
        self._check_fitted()
        patches = np.asarray(patches, dtype=np.float32)
        grouped = group_patches(patches, self.refinement_cfg.patch_group_sizes)
        grouped = _append_cls(grouped, cls_tokens)
        weights = _ridge_pool_weights(grouped, self.ridge_model_, self.feature_scaler_)
        return (weights[:, :, None] * grouped).sum(axis=1)

    def fit_transform(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
        initial_support: Optional[np.ndarray] = None,
        initial_pca: Optional[PCA] = None,
    ) -> np.ndarray:
        """Fit then transform — returns [N, D]."""
        return self.fit(
            patches, labels, cls_tokens, initial_support, initial_pca
        ).transform(patches, cls_tokens)

    # ------------------------------------------------------------------
    # Score tabicl (image-specific wrapper)
    # ------------------------------------------------------------------

    def score_tabicl(  # type: ignore[override]
        self,
        query_patches: np.ndarray,
        query_labels: np.ndarray,
        n_estimators: Optional[int] = None,
        query_cls_tokens: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Pool *query_patches* then evaluate TabICL accuracy + AUROC."""
        self._check_fitted()
        query_raw = self.transform(query_patches, cls_tokens=query_cls_tokens)
        return super().score_tabicl(query_raw, query_labels, n_estimators)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def patch_weights(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-patch Ridge softmax weights → [N, P'] or [P'] for a single image."""
        self._check_fitted()
        single = patches.ndim == 2
        if single:
            patches = patches[None]
            if cls_tokens is not None:
                cls_tokens = np.asarray(cls_tokens, dtype=np.float32)
                if cls_tokens.ndim == 1:
                    cls_tokens = cls_tokens[None]
        patches = np.asarray(patches, dtype=np.float32)
        grouped = group_patches(patches, self.refinement_cfg.patch_group_sizes)
        grouped = _append_cls(grouped, cls_tokens)
        weights = _ridge_pool_weights(grouped, self.ridge_model_, self.feature_scaler_)
        return weights[0] if single else weights

    def patch_quality_logits(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw Ridge quality predictions before softmax → [N, P'] or [P']."""
        self._check_fitted()
        single = patches.ndim == 2
        if single:
            patches = patches[None]
            if cls_tokens is not None:
                cls_tokens = np.asarray(cls_tokens, dtype=np.float32)
                if cls_tokens.ndim == 1:
                    cls_tokens = cls_tokens[None]
        patches = np.asarray(patches, dtype=np.float32)
        N, P, D = patches.shape
        grouped = group_patches(patches, self.refinement_cfg.patch_group_sizes)
        grouped = _append_cls(grouped, cls_tokens)
        _, P_prime, _ = grouped.shape
        flat = grouped.reshape(N * P_prime, D)
        if self.feature_scaler_ is not None:
            flat = self.feature_scaler_.transform(flat)
        logits = (
            self.ridge_model_.predict(flat).reshape(N, P_prime).astype(np.float32)
        )
        return logits[0] if single else logits

    def __repr__(self) -> str:
        fitted = hasattr(self, "ridge_model_")
        status = (
            f"fitted: D={self.embed_dim_}  P'={self.n_patches_grouped_}  "
            f"fit={self.fit_time_s_:.1f}s  pool={self.pool_time_s_:.1f}s"
            if fitted else "not fitted"
        )
        return (
            f"ImagePALPooler("
            f"patch_group_sizes={self.refinement_cfg.patch_group_sizes}, "
            f"weight_method='{self.refinement_cfg.weight_method}', "
            f"temperature={self.refinement_cfg.temperature}, "
            f"ridge_alpha={self.refinement_cfg.ridge_alpha}, "
            f"pca_dim={self.refinement_cfg.tabicl_pca_dim}, "
            f"[{status}])"
        )


# ===========================================================================
# Text pooler
# ===========================================================================

class TextPALPooler(PALPooler):
    """Single-stage Ridge-based adaptive token pooling for text (BERT) embeddings.

    Handles variable-length sequences with padding masks and two grouping
    modes controlled by ``TextRefinementConfig.text_group_modes``.

    Parameters
    ----------
    tabicl : TabICLClassifier
    refinement_cfg : TextRefinementConfig
        Must have ``text_group_modes`` as a single string (one stage) or the
        caller sets ``text_group_mode`` directly via the ``fit`` internals.
    text_group_mode : str
        Overrides ``refinement_cfg.text_group_modes`` when provided as a
        scalar.  Used by :class:`IterativePALPooler` to pass the per-stage
        mode.
    seed : int
    gpu_ridge_device : str

    Fitted attributes
    -----------------
    n_groups_ : int  — G_max (max groups per sequence after grouping + optional CLS)
    group_mask_ : np.ndarray, shape [N_train, G_max]  — validity mask for training set
    text_group_mode_ : str  — grouping mode used at fit time
    (all base-class attributes apply)
    """

    def __init__(
        self,
        tabicl: TabICLClassifier,
        refinement_cfg: TextRefinementConfig,
        text_group_mode: Optional[str] = None,
        seed: int = 42,
        gpu_ridge_device: str = "cuda",
    ) -> None:
        self.tabicl = tabicl
        self.refinement_cfg = refinement_cfg
        self.pca_dim = refinement_cfg.tabicl_pca_dim
        self.seed = seed
        self.gpu_ridge_device = gpu_ridge_device
        # Allow per-stage override from IterativePALPooler.
        if text_group_mode is not None:
            self._stage_group_mode = text_group_mode
        elif isinstance(refinement_cfg.text_group_modes, str):
            self._stage_group_mode = refinement_cfg.text_group_modes
        else:
            # Default to first entry if a list was passed.
            self._stage_group_mode = refinement_cfg.text_group_modes[0]

    # ------------------------------------------------------------------
    # Core sklearn-style API
    # ------------------------------------------------------------------

    def fit(
        self,
        tokens: np.ndarray,               # [N, T_max, D]
        token_ids: np.ndarray,            # [N, T_max]
        attention_mask: np.ndarray,       # [N, T_max]  bool — True = valid
        labels: np.ndarray,               # [N]
        cls_tokens: Optional[np.ndarray] = None,   # [N, D]  appended if append_cls
        initial_support: Optional[np.ndarray] = None,
        initial_pca: Optional[PCA] = None,
        context_features: Optional[np.ndarray] = None,
        tabular_probs: Optional[np.ndarray] = None,
        val_tokens: Optional[np.ndarray] = None,     # [N_val, T_max, D]
        val_token_ids: Optional[np.ndarray] = None,  # [N_val, T_max]
        val_labels: Optional[np.ndarray] = None,     # [N_val]
    ) -> "TextPALPooler":
        """Fit the Ridge quality model on BERT token embeddings.

        Parameters
        ----------
        tokens : np.ndarray, shape [N, T_max, D]
            Padded BERT embeddings.
        token_ids : np.ndarray, shape [N, T_max]
            Integer token IDs (used for ``[CLS]`` / ``[SEP]`` detection).
        attention_mask : np.ndarray, shape [N, T_max]  bool
            ``True`` for real tokens, ``False`` for padding.  Currently used
            to validate inputs; the grouping logic also uses ``token_ids``
            directly for ``[CLS]``/padding detection.
        labels : np.ndarray, shape [N]
        cls_tokens : np.ndarray or None, shape [N, D]
            Pre-extracted ``[CLS]`` embedding per sequence.  Appended as an
            extra group only when ``refinement_cfg.append_cls`` is ``True``.
        initial_support, initial_pca : optional, for stage chaining.
        context_features, tabular_probs : optional side features.
        """
        tokens    = np.asarray(tokens,    dtype=np.float32)
        token_ids = np.asarray(token_ids)
        labels    = np.asarray(labels)
        N, T_max, D = tokens.shape
        self.embed_dim_       = D
        self.text_group_mode_ = self._stage_group_mode

        # Group tokens.
        grouped, group_mask = group_text_tokens(
            tokens, token_ids,
            mode=self.text_group_mode_,
            sep_token_id=self.refinement_cfg.sep_token_id,
            cls_token_id=self.refinement_cfg.cls_token_id,
        )  # [N, G_max, D], [N, G_max]

        # Optionally append [CLS] as an extra group.
        if self.refinement_cfg.append_cls:
            grouped, group_mask = _append_cls_masked(grouped, group_mask, cls_tokens)

        self.n_groups_  = grouped.shape[1]
        self.group_mask_ = group_mask

        # Build initial support.
        if initial_support is not None:
            support     = initial_support
            current_pca = initial_pca
        else:
            # Mean pool over valid groups per sequence.
            valid_counts = group_mask.sum(axis=1, keepdims=True).clip(min=1)  # [N, 1]
            support_raw  = (
                (grouped * group_mask[:, :, None]).sum(axis=1) / valid_counts
            ).astype(np.float32)  # [N, D]
            if self.pca_dim is not None:
                n_comp      = min(self.pca_dim, N, D)
                current_pca = PCA(n_components=n_comp, random_state=self.seed)
                support     = current_pca.fit_transform(support_raw).astype(np.float32)
            else:
                current_pca = None
                support     = support_raw

        # Per-stage refinement config with scalar temperature / alpha.
        stage_cfg = copy.deepcopy(self.refinement_cfg)
        # text_group_modes may be a list; refine_text_features takes the scalar.
        if not isinstance(stage_cfg.temperature, (int, float)):
            stage_cfg.temperature = stage_cfg.temperature[0]
        if not isinstance(stage_cfg.ridge_alpha, (int, float)):
            stage_cfg.ridge_alpha = stage_cfg.ridge_alpha[0]

        (refined_support, new_pca, weights_ridge, ridge_model, feature_scaler,
         scoring_clf, fit_time_s, pool_time_s,
         updated_class_prior, _group_mask_out) = refine_text_features(
            train_tokens=tokens,
            train_token_ids=token_ids,
            train_labels=labels,
            support_features=support,
            refinement_cfg=stage_cfg,
            pca=current_pca,
            text_group_mode=self.text_group_mode_,
            seed=self.seed,
            gpu_ridge_device=self.gpu_ridge_device,
            tabicl=self.tabicl,
            context_features=context_features,
            tabular_probs=tabular_probs,
            val_tokens=val_tokens,
            val_token_ids=val_token_ids,
            val_labels=val_labels,
        )

        # Re-pool in original D-space for raw support (same as image path).
        repooled_raw = (
            weights_ridge[:, :, None] * grouped
        ).sum(axis=1).astype(np.float32)  # [N, D]

        self.ridge_model_        = ridge_model
        self.feature_scaler_     = feature_scaler
        self._pca_               = new_pca
        self._support_projected_ = refined_support
        self.support_            = repooled_raw
        self.support_labels_     = labels
        self.scoring_clf_        = scoring_clf
        self.class_prior_        = updated_class_prior
        self.fit_time_s_         = fit_time_s
        self.pool_time_s_        = pool_time_s
        return self

    def _fit_from_indices(self, data, labels, cls_tokens, init_sup, init_pca,
                          context_features, tabular_probs, val_tabular_probs,
                          train_idx, val_idx, token_ids=None, attention_mask=None, **kw):
        tr = train_idx
        vl = val_idx
        _cls  = cls_tokens[tr] if (cls_tokens is not None and tr is not None) else cls_tokens
        _sup  = init_sup[tr] if (init_sup is not None and tr is not None and len(init_sup) == len(data)) else init_sup
        _ctx  = context_features[tr] if (context_features is not None and tr is not None) else context_features
        _tab  = tabular_probs[tr] if (tabular_probs is not None and tr is not None) else tabular_probs
        _val_tokens    = data[vl]       if (vl is not None) else None
        _val_token_ids = token_ids[vl]  if (token_ids is not None and vl is not None) else None
        _val_labels    = labels[vl]     if (vl is not None) else None
        self.fit(
            data[tr] if tr is not None else data,
            token_ids[tr] if (token_ids is not None and tr is not None) else token_ids,
            attention_mask[tr] if (attention_mask is not None and tr is not None) else attention_mask,
            labels[tr] if tr is not None else labels,
            cls_tokens=_cls,
            initial_support=_sup, initial_pca=init_pca,
            context_features=_ctx, tabular_probs=_tab,
            val_tokens=_val_tokens, val_token_ids=_val_token_ids, val_labels=_val_labels,
        )

    def _transform_subset(self, data, idx, cls_tokens, token_ids=None, attention_mask=None):
        _d   = data[idx] if idx is not None else data
        _ids = token_ids[idx] if (token_ids is not None and idx is not None) else token_ids
        _msk = attention_mask[idx] if (attention_mask is not None and idx is not None) else attention_mask
        _cls = cls_tokens[idx] if (cls_tokens is not None and idx is not None) else cls_tokens
        return self.transform(_d, _ids, _msk, _cls)

    def _transform_all(self, data, cls_tokens, token_ids=None, attention_mask=None):
        return self.transform(data, token_ids, attention_mask, cls_tokens)

    def transform(
        self,
        tokens: np.ndarray,           # [N, T_max, D]
        token_ids: np.ndarray,        # [N, T_max]
        attention_mask: np.ndarray,   # [N, T_max]
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pool tokens with the fitted Ridge model → [N, D]."""
        self._check_fitted()
        tokens    = np.asarray(tokens,    dtype=np.float32)
        token_ids = np.asarray(token_ids)
        grouped, group_mask = group_text_tokens(
            tokens, token_ids,
            mode=self.text_group_mode_,
            sep_token_id=self.refinement_cfg.sep_token_id,
            cls_token_id=self.refinement_cfg.cls_token_id,
        )
        if self.refinement_cfg.append_cls:
            grouped, group_mask = _append_cls_masked(grouped, group_mask, cls_tokens)
        weights = _ridge_pool_weights_text(
            grouped, group_mask, self.ridge_model_, self.feature_scaler_
        )
        return (weights[:, :, None] * grouped).sum(axis=1)

    def fit_transform(
        self,
        tokens: np.ndarray,
        token_ids: np.ndarray,
        attention_mask: np.ndarray,
        labels: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
        initial_support: Optional[np.ndarray] = None,
        initial_pca: Optional[PCA] = None,
    ) -> np.ndarray:
        """Fit then transform — returns [N, D]."""
        return self.fit(
            tokens, token_ids, attention_mask, labels, cls_tokens,
            initial_support, initial_pca,
        ).transform(tokens, token_ids, attention_mask, cls_tokens)

    # ------------------------------------------------------------------
    # Score tabicl (text-specific wrapper)
    # ------------------------------------------------------------------

    def score_tabicl(  # type: ignore[override]
        self,
        query_tokens: np.ndarray,
        query_token_ids: np.ndarray,
        query_attention_mask: np.ndarray,
        query_labels: np.ndarray,
        n_estimators: Optional[int] = None,
        query_cls_tokens: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Pool *query_tokens* then evaluate TabICL accuracy + AUROC."""
        self._check_fitted()
        query_raw = self.transform(
            query_tokens, query_token_ids, query_attention_mask, query_cls_tokens
        )
        return super().score_tabicl(query_raw, query_labels, n_estimators)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def token_weights(
        self,
        tokens: np.ndarray,
        token_ids: np.ndarray,
        attention_mask: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-group Ridge softmax weights → [N, G_max] or [G_max] for single input.

        Zero for padded/invalid positions.
        """
        self._check_fitted()
        single = tokens.ndim == 2
        if single:
            tokens         = tokens[None]
            token_ids      = token_ids[None]
            attention_mask = attention_mask[None]
            if cls_tokens is not None:
                cls_tokens = np.asarray(cls_tokens, dtype=np.float32)
                if cls_tokens.ndim == 1:
                    cls_tokens = cls_tokens[None]
        tokens    = np.asarray(tokens,    dtype=np.float32)
        token_ids = np.asarray(token_ids)
        grouped, group_mask = group_text_tokens(
            tokens, token_ids,
            mode=self.text_group_mode_,
            sep_token_id=self.refinement_cfg.sep_token_id,
            cls_token_id=self.refinement_cfg.cls_token_id,
        )
        if self.refinement_cfg.append_cls:
            grouped, group_mask = _append_cls_masked(grouped, group_mask, cls_tokens)
        weights = _ridge_pool_weights_text(
            grouped, group_mask, self.ridge_model_, self.feature_scaler_
        )
        return weights[0] if single else weights

    def token_quality_logits(
        self,
        tokens: np.ndarray,
        token_ids: np.ndarray,
        attention_mask: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw Ridge quality predictions before masked softmax → [N, G_max] or [G_max]."""
        self._check_fitted()
        single = tokens.ndim == 2
        if single:
            tokens         = tokens[None]
            token_ids      = token_ids[None]
            attention_mask = attention_mask[None]
            if cls_tokens is not None:
                cls_tokens = np.asarray(cls_tokens, dtype=np.float32)
                if cls_tokens.ndim == 1:
                    cls_tokens = cls_tokens[None]
        tokens    = np.asarray(tokens,    dtype=np.float32)
        token_ids = np.asarray(token_ids)
        N = tokens.shape[0]
        grouped, group_mask = group_text_tokens(
            tokens, token_ids,
            mode=self.text_group_mode_,
            sep_token_id=self.refinement_cfg.sep_token_id,
            cls_token_id=self.refinement_cfg.cls_token_id,
        )
        if self.refinement_cfg.append_cls:
            grouped, group_mask = _append_cls_masked(grouped, group_mask, cls_tokens)
        _, G_max, D = grouped.shape
        flat = grouped.reshape(N * G_max, D)
        if self.feature_scaler_ is not None:
            flat = self.feature_scaler_.transform(flat)
        logits = (
            self.ridge_model_.predict(flat).reshape(N, G_max).astype(np.float32)
        )
        return logits[0] if single else logits

    def __repr__(self) -> str:
        fitted = hasattr(self, "ridge_model_")
        status = (
            f"fitted: D={self.embed_dim_}  G_max={self.n_groups_}  "
            f"mode={self.text_group_mode_}  "
            f"fit={self.fit_time_s_:.1f}s  pool={self.pool_time_s_:.1f}s"
            if fitted else "not fitted"
        )
        return (
            f"TextPALPooler("
            f"text_group_mode='{self._stage_group_mode}', "
            f"weight_method='{self.refinement_cfg.weight_method}', "
            f"temperature={self.refinement_cfg.temperature}, "
            f"ridge_alpha={self.refinement_cfg.ridge_alpha}, "
            f"pca_dim={self.refinement_cfg.tabicl_pca_dim}, "
            f"[{status}])"
        )


# ===========================================================================
# Iterative pooler (image + text)
# ===========================================================================

class IterativePALPooler:
    """Multi-stage PAL pooler that chains refinement stages automatically.

    Works with both image tokens (:class:`ImagePALPooler`) and text tokens
    (:class:`TextPALPooler`).  The modality is controlled by the *modality*
    constructor parameter.

    Each stage fits a sub-pooler, refines the support, and passes the refined
    projected support to the next stage.  After fitting, all transform /
    scoring calls are delegated to the selected stage (controlled by
    ``refinement_cfg.model_selection``).

    Parameters
    ----------
    tabicl : TabICLClassifier
    refinement_cfg : RefinementConfig or TextRefinementConfig
        For images: ``RefinementConfig`` with ``patch_group_sizes`` as a
        list (one entry per stage).
        For text: ``TextRefinementConfig`` with ``text_group_modes`` as a
        list of strings (one entry per stage).
    modality : str
        ``"image"`` (default) or ``"text"``.
    seed : int
    gpu_ridge_device : str

    Fitted attributes
    -----------------
    stages_ : list of ImagePALPooler or TextPALPooler
    best_stage_idx_ : int
    stage_train_accuracies_ : list of float or None
    support_ : np.ndarray, shape [N, D]
    support_labels_ : np.ndarray, shape [N]
    """

    def __init__(
        self,
        tabicl: TabICLClassifier,
        refinement_cfg: Union[RefinementConfig, TextRefinementConfig],
        modality: str = "image",
        gpu_ridge_device: str = "cuda",
        seed: int = 42,
    ) -> None:
        if modality not in ("image", "text"):
            raise ValueError(f"modality must be 'image' or 'text', got {modality!r}")
        if refinement_cfg.model_selection not in ("last_iteration", "masked_train_accuracy"):
            raise ValueError(
                f"model_selection must be 'last_iteration' or 'masked_train_accuracy', "
                f"got {refinement_cfg.model_selection!r}"
            )
        self.tabicl           = tabicl
        self.refinement_cfg   = refinement_cfg
        self.modality         = modality
        self.seed             = seed
        self.gpu_ridge_device = gpu_ridge_device

        if modality == "image":
            self.patch_group_sizes = list(refinement_cfg.patch_group_sizes)
        else:
            modes = refinement_cfg.text_group_modes
            self.text_group_modes = [modes] if isinstance(modes, str) else list(modes)

        self.pca_dim = refinement_cfg.tabicl_pca_dim

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        *,
        # image-specific
        cls_tokens: Optional[np.ndarray] = None,
        stage_callback: Optional[Callable] = None,
        context_features: Optional[np.ndarray] = None,
        # text-specific
        token_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        tabular_probs: Optional[np.ndarray] = None,
    ) -> "IterativePALPooler":
        """Fit all stages sequentially.

        Parameters
        ----------
        data : np.ndarray
            ``[N, P, D]`` image patches (``modality="image"``) or
            ``[N, T_max, D]`` padded BERT tokens (``modality="text"``).
        labels : np.ndarray, shape [N]
        cls_tokens : np.ndarray or None, shape [N, D]
            CLS token embeddings.  For images: appended to patches when
            ``append_cls=True``.  For text: the ``[CLS]`` token embedding,
            appended to groups when ``append_cls=True``.
        token_ids : np.ndarray or None, shape [N, T_max]
            Required when ``modality="text"``.
        attention_mask : np.ndarray or None, shape [N, T_max]
            Required when ``modality="text"``.
        tabular_probs : np.ndarray or None, shape [N, n_classes]
            Pre-computed P(Y|X_tab) for divergence-based weight methods.
            For text modality these must be supplied directly (no
            auto-compute from context_features currently).
        stage_callback : callable or None
            Called after each image stage (image modality only).
        context_features : np.ndarray or None
            Optional side features for divergence weight methods.
        """
        if self.modality == "text" and (token_ids is None or attention_mask is None):
            raise ValueError(
                "token_ids and attention_mask are required for modality='text'"
            )
        return self._fit_stages(
            data, labels,
            cls_tokens=cls_tokens,
            token_ids=token_ids,
            attention_mask=attention_mask,
            stage_callback=stage_callback,
            context_features=context_features,
            tabular_probs=tabular_probs,
        )

    def transform(
        self,
        data: np.ndarray,
        *,
        cls_tokens: Optional[np.ndarray] = None,
        token_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pool using the selected fitted stage → [N, D].

        Parameters
        ----------
        data : patches or tokens, depending on modality.
        cls_tokens, token_ids, attention_mask : see ``fit``.
        """
        self._check_fitted()
        stage = self.stages_[self.best_stage_idx_]
        if self.modality == "image":
            return stage.transform(data, cls_tokens=cls_tokens)
        else:
            if token_ids is None or attention_mask is None:
                raise ValueError(
                    "token_ids and attention_mask are required for modality='text'"
                )
            return stage.transform(data, token_ids, attention_mask, cls_tokens)

    def fit_transform(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        *,
        cls_tokens: Optional[np.ndarray] = None,
        context_features: Optional[np.ndarray] = None,
        token_ids: Optional[np.ndarray] = None,
        attention_mask: Optional[np.ndarray] = None,
        tabular_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit all stages then transform with the selected stage → [N, D]."""
        return self.fit(
            data, labels,
            cls_tokens=cls_tokens,
            context_features=context_features,
            token_ids=token_ids,
            attention_mask=attention_mask,
            tabular_probs=tabular_probs,
        ).transform(data, cls_tokens=cls_tokens, token_ids=token_ids, attention_mask=attention_mask)

    # ------------------------------------------------------------------
    # Convenience delegations to selected stage
    # ------------------------------------------------------------------

    @property
    def support_(self) -> np.ndarray:
        self._check_fitted()
        return self.stages_[self.best_stage_idx_].support_

    @property
    def support_labels_(self) -> np.ndarray:
        self._check_fitted()
        return self.stages_[self.best_stage_idx_].support_labels_

    def patch_weights(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-patch Ridge softmax weights (image modality only)."""
        self._check_fitted()
        if self.modality != "image":
            raise RuntimeError("patch_weights is only available for modality='image'.")
        return self.stages_[self.best_stage_idx_].patch_weights(patches, cls_tokens=cls_tokens)

    def patch_quality_logits(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw Ridge quality logits (image modality only)."""
        self._check_fitted()
        if self.modality != "image":
            raise RuntimeError("patch_quality_logits is only available for modality='image'.")
        return self.stages_[self.best_stage_idx_].patch_quality_logits(patches, cls_tokens=cls_tokens)

    def token_weights(
        self,
        tokens: np.ndarray,
        token_ids: np.ndarray,
        attention_mask: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-group Ridge softmax weights (text modality only)."""
        self._check_fitted()
        if self.modality != "text":
            raise RuntimeError("token_weights is only available for modality='text'.")
        return self.stages_[self.best_stage_idx_].token_weights(
            tokens, token_ids, attention_mask, cls_tokens
        )

    def token_quality_logits(
        self,
        tokens: np.ndarray,
        token_ids: np.ndarray,
        attention_mask: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw Ridge quality logits (text modality only)."""
        self._check_fitted()
        if self.modality != "text":
            raise RuntimeError("token_quality_logits is only available for modality='text'.")
        return self.stages_[self.best_stage_idx_].token_quality_logits(
            tokens, token_ids, attention_mask, cls_tokens
        )

    def score_tabicl(
        self,
        query_data: np.ndarray,
        query_labels: np.ndarray,
        n_estimators: Optional[int] = None,
        query_cls_tokens: Optional[np.ndarray] = None,
        query_token_ids: Optional[np.ndarray] = None,
        query_attention_mask: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Evaluate accuracy and AUROC using the selected stage's support."""
        self._check_fitted()
        stage = self.stages_[self.best_stage_idx_]
        if self.modality == "image":
            return stage.score_tabicl(
                query_data, query_labels, n_estimators,
                query_cls_tokens=query_cls_tokens,
            )
        else:
            return stage.score_tabicl(
                query_data, query_token_ids, query_attention_mask,
                query_labels, n_estimators,
                query_cls_tokens=query_cls_tokens,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        import joblib
        self._check_fitted()
        joblib.dump(self, Path(path))
        print(f"[IterativePALPooler] Saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IterativePALPooler":
        import joblib
        pooler = joblib.load(Path(path))
        if not isinstance(pooler, cls):
            raise TypeError(
                f"Loaded object is {type(pooler).__name__}, expected IterativePALPooler"
            )
        return pooler

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_pca(
        self,
        support_raw: np.ndarray,
        N: int,
        D: int,
    ) -> tuple:
        """Fit PCA on support_raw if pca_dim is set, else pass through."""
        if self.pca_dim is not None:
            n_comp = min(self.pca_dim, N, D)
            pca = PCA(n_components=n_comp, random_state=self.seed)
            support = pca.fit_transform(support_raw).astype(np.float32)
        else:
            pca = None
            support = support_raw
        return support, pca

    def _maybe_compute_tabular_probs(
        self,
        context_features: Optional[np.ndarray],
        labels: np.ndarray,
        provided_tabular_probs: Optional[np.ndarray] = None,
    ) -> tuple:
        """Return P(Y|X_tab) for divergence weight methods.

        If *provided_tabular_probs* is not None it is returned unchanged
        (caller pre-computed it).  Otherwise, if *context_features* is
        available and the weight method is divergence-based, a TabICL
        classifier is fit once on context_features and used to produce
        train probabilities.

        Returns
        -------
        (tabular_probs, val_tabular_probs) : each np.ndarray or None
        """
        if provided_tabular_probs is not None:
            return provided_tabular_probs, None

        tabular_probs: Optional[np.ndarray] = None
        val_tabular_probs: Optional[np.ndarray] = None

        if (context_features is not None
                and self.refinement_cfg.weight_method in (
                    "kl_div", "wasserstein", "js_div", "tvd")
                and not getattr(self.refinement_cfg, "use_global_prior", False)):
            print("[IterativePALPooler] Computing tabular-only P(Y|X_tab) ...")
            _clf_tab = TabICLClassifier(
                n_estimators=self.refinement_cfg.tabicl_n_estimators,
                random_state=self.seed,
            )
            _clf_tab.fit(context_features, labels)
            N_tab = len(context_features)
            _tab_mask = (
                np.eye(N_tab, dtype=bool)
                if self.refinement_cfg.use_attn_masking else None
            )
            raw_tab = _clf_tab.predict_proba(
                context_features, attn_mask=_tab_mask
            ).astype(np.float32)
            n_cls_local = int(labels.max()) + 1
            if raw_tab.shape[1] != n_cls_local:
                tabular_probs = np.zeros(
                    (raw_tab.shape[0], n_cls_local), dtype=np.float32
                )
                tabular_probs[:, _clf_tab.classes_] = raw_tab
            else:
                tabular_probs = raw_tab

        return tabular_probs, val_tabular_probs

    # ------------------------------------------------------------------
    # Internal: shared fit loop
    # ------------------------------------------------------------------

    def _fit_loop(
        self,
        stage_items: list,
        make_stage,        # fn(iter_cfg, item) → PALPooler subclass
        fit_stage,         # fn(stage, initial_support, initial_pca, k) → None
        eval_transform,    # fn(stage) → np.ndarray [N, D]  (for masked acc)
        labels: np.ndarray,
        context_features: Optional[np.ndarray],
        initial_support: np.ndarray,
        initial_pca: Optional[PCA],
        stage_callback=None,   # fn(k, stage, item, pre_sup, pre_pca) → None
        post_fit=None,         # fn(k, stage) → None  — called after fit_stage, before chaining
        stage_key_label: str = "stage",
    ) -> "IterativePALPooler":
        n_stages = len(stage_items)
        temps  = self._expand_param(self.refinement_cfg.temperature, n_stages, "temperature")
        alphas = self._expand_param(self.refinement_cfg.ridge_alpha,  n_stages, "ridge_alpha")

        stages: list = []
        stage_train_accuracies: List[float] = []

        for k, item in enumerate(stage_items):
            print(
                f"[IterativePALPooler] Stage {k}/{n_stages - 1} "
                f"— {stage_key_label}={item}, "
                f"temperature={temps[k]}, ridge_alpha={alphas[k]}"
            )

            iter_cfg = copy.deepcopy(self.refinement_cfg)
            iter_cfg.temperature = temps[k]
            iter_cfg.ridge_alpha = alphas[k]

            pre_sup, pre_pca = initial_support, initial_pca

            stage = make_stage(iter_cfg, item)
            fit_stage(stage, initial_support, initial_pca, k)

            if post_fit is not None:
                post_fit(k, stage)

            stages.append(stage)

            if stage_callback is not None:
                stage_callback(k, stage, item, pre_sup, pre_pca)

            if self.refinement_cfg.model_selection == "masked_train_accuracy":
                train_raw = eval_transform(stage)
                acc = self._eval_masked_train_accuracy(
                    stage, train_raw, labels, context_features
                )
                stage_train_accuracies.append(acc)
                print(
                    f"[IterativePALPooler] Stage {k} masked train accuracy: {acc:.4f}"
                )

            initial_support = stage._support_projected_
            initial_pca     = stage._pca_

        return self._finalise(stages, stage_train_accuracies)

    # ------------------------------------------------------------------
    # Internal: generalized fit loop
    # ------------------------------------------------------------------

    def _fit_stages(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        cls_tokens=None,
        token_ids=None,         # text only
        attention_mask=None,    # text only
        stage_callback=None,
        context_features=None,
        tabular_probs=None,
    ) -> "IterativePALPooler":
        is_image = self.modality == "image"
        stage_items = self.patch_group_sizes if is_image else self.text_group_modes
        
        if not stage_items:
            name = "patch_group_sizes" if is_image else "text_group_modes"
            raise ValueError(f"{name} must contain at least one entry.")

        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels)
        N, _, D = data.shape

        val_tabular_probs = None
        if is_image:
            if cls_tokens is not None:
                _cls = np.asarray(cls_tokens, dtype=np.float32)
                support_raw = np.concatenate([data, _cls[:, None, :]], axis=1).mean(axis=1)
            else:
                support_raw = data.mean(axis=1)
            tabular_probs, val_tabular_probs = self._maybe_compute_tabular_probs(
                context_features, labels, provided_tabular_probs=tabular_probs
            )
        else:
            token_ids = np.asarray(token_ids)
            _valid = (token_ids != 0) & (token_ids != self.refinement_cfg.cls_token_id)
            valid_counts = _valid.sum(axis=1, keepdims=True).clip(min=1)
            support_raw = ((data * _valid[:, :, None]).sum(axis=1) / valid_counts).astype(np.float32)
            if self.refinement_cfg.append_cls and cls_tokens is not None:
                _cls = np.asarray(cls_tokens, dtype=np.float32)
                support_raw = np.concatenate([support_raw[:, None, :], _cls[:, None, :]], axis=1).mean(axis=1)
            
            tabular_probs, _ = self._maybe_compute_tabular_probs(
                context_features, labels, provided_tabular_probs=tabular_probs
            )
        
        initial_support, initial_pca = self._init_pca(support_raw, N, D)

        # Optionally split data per stage so each iteration uses a fresh independent
        # validation fold for both image and text modalities.
        tvf = getattr(self.refinement_cfg, "train_val_fraction", None)
        splits: Optional[list] = None
        if tvf is not None and tvf > 0.0:
            splits = []
            for _k in range(len(stage_items)):
                _rng  = np.random.RandomState(self.seed + _k)
                _n_val = int(N * tvf)
                _perm  = _rng.permutation(N)
                splits.append((_perm[_n_val:], _perm[:_n_val]))  # (train_idx, val_idx)
            print(f"[IterativePALPooler] train_val_fraction={tvf}: "
                  f"{int(N*tvf)} val / {N - int(N*tvf)} train per stage (fresh split each stage)")

        _stage_val_accs: List[Optional[float]] = []

        def make_stage(iter_cfg, item):
            if is_image:
                iter_cfg.patch_group_sizes = item
                return ImagePALPooler(
                    tabicl=self.tabicl, refinement_cfg=iter_cfg,
                    seed=self.seed, gpu_ridge_device=self.gpu_ridge_device,
                )
            else:
                iter_cfg.text_group_modes = item
                return TextPALPooler(
                    tabicl=self.tabicl, refinement_cfg=iter_cfg,
                    text_group_mode=item,
                    seed=self.seed, gpu_ridge_device=self.gpu_ridge_device,
                )

        def fit_stage(stage, init_sup, init_pca, k):
            train_idx, val_idx = splits[k] if splits is not None else (None, None)
            stage._fit_from_indices(
                data, labels, cls_tokens, init_sup, init_pca,
                context_features, tabular_probs, val_tabular_probs=None,
                train_idx=train_idx, val_idx=val_idx,
                token_ids=token_ids, attention_mask=attention_mask,
            )

        def post_fit(k, stage):
            """After fitting on the train fold, pool val fold and rebuild full-N support."""
            if splits is None:
                return
            train_idx, val_idx = splits[k]
            val_pooled = stage._transform_subset(data, val_idx, cls_tokens,
                                                 token_ids=token_ids, attention_mask=attention_mask)
            val_proj = (
                stage._pca_.transform(val_pooled).astype(np.float32)
                if stage._pca_ is not None else val_pooled.astype(np.float32)
            )
            # Evaluate val accuracy now: support is still train-only, which is correct.
            # Pass val_pooled (raw D-space) — score_tabicl applies PCA internally.
            val_acc, val_auroc = PALPooler.score_tabicl(stage, val_pooled, labels[val_idx])
            stage._val_accuracy_ = val_acc
            stage._val_auroc_    = val_auroc
            _stage_val_accs.append(val_acc)
            print(
                f"[IterativePALPooler] Stage {k} val accuracy: {val_acc:.4f}  "
                f"auroc: {val_auroc:.4f}"
            )

            D_proj = stage._support_projected_.shape[1]
            full_support = np.empty((N, D_proj), dtype=np.float32)
            full_support[train_idx] = stage._support_projected_
            full_support[val_idx]   = val_proj
            stage._support_projected_ = full_support
            stage.support_labels_     = labels

        def eval_transform(stage):
            return stage._transform_all(data, cls_tokens, token_ids=token_ids, attention_mask=attention_mask)

        _cb = None
        if stage_callback is not None and is_image:
            def _cb(k, stage, item, pre_sup, pre_pca):
                stage_callback(
                    stage_idx=k, stage=stage, group_size=item,
                    pre_refine_support=pre_sup, pre_refine_pca=pre_pca,
                    train_grouped=group_patches(data, item),
                )

        self._fit_loop(
            stage_items, make_stage, fit_stage, eval_transform,
            labels, context_features, initial_support, initial_pca,
            stage_callback=_cb,
            post_fit=post_fit if splits is not None else None,
            stage_key_label="patch_group_size" if is_image else "text_group_mode",
        )
        self.stage_val_accuracies_ = _stage_val_accs if _stage_val_accs else None
        return self

    # ------------------------------------------------------------------
    # Internal: shared finalisation + masked-accuracy helpers
    # ------------------------------------------------------------------

    def _finalise(
        self,
        stages: list,
        stage_train_accuracies: list,
    ) -> "IterativePALPooler":
        self.stages_ = stages
        self.stage_val_accuracies_ = None  # overwritten by _fit_stages when splits exist
        if self.refinement_cfg.model_selection == "masked_train_accuracy":
            self.best_stage_idx_        = int(np.argmax(stage_train_accuracies))
            self.stage_train_accuracies_ = stage_train_accuracies
            print(
                f"[IterativePALPooler] Best stage: {self.best_stage_idx_} "
                f"(masked train acc={stage_train_accuracies[self.best_stage_idx_]:.4f})"
            )
        else:
            self.best_stage_idx_        = len(stages) - 1
            self.stage_train_accuracies_ = None
        return self

    def _eval_masked_train_accuracy(
        self,
        stage,
        train_raw: np.ndarray,
        labels: np.ndarray,
        context_features=None,
    ) -> float:
        """Evaluate leave-one-out masked train accuracy for model selection.

        *train_raw* is the already-pooled [N, D] output of stage.transform(...)
        for the training set, passed in by _fit_loop so we don't re-run transform.
        """
        train_feat = (
            stage._pca_.transform(train_raw).astype(np.float32)
            if stage._pca_ is not None else train_raw.astype(np.float32)
        )
        N = len(labels)
        support_for_clf = stage._support_projected_
        query_for_clf   = train_feat
        if context_features is not None:
            ctx = context_features.astype(np.float32)
            support_for_clf = np.concatenate([support_for_clf, ctx], axis=1)
            query_for_clf   = np.concatenate([query_for_clf,  ctx], axis=1)
        n_est = getattr(self.tabicl, "n_estimators", 1)
        clf   = TabICLClassifier(n_estimators=n_est, random_state=self.seed)
        clf.fit(support_for_clf, stage.support_labels_)
        proba = clf.predict_proba(query_for_clf, blocked_indices=np.arange(N))
        return float((np.argmax(proba, axis=1) == labels).mean())

    @staticmethod
    def _expand_param(param, n_stages: int, name: str) -> list:
        if isinstance(param, list):
            if len(param) == 1:
                return param * n_stages
            if len(param) != n_stages:
                raise ValueError(
                    f"{name} list has {len(param)} entries but stage list "
                    f"has {n_stages} stages."
                )
            return param
        return [param] * n_stages

    def _check_fitted(self) -> None:
        if not hasattr(self, "stages_"):
            raise RuntimeError(
                "IterativePALPooler is not fitted yet.  Call fit() first."
            )

    def __repr__(self) -> str:
        fitted = hasattr(self, "stages_")
        is_image = self.modality == "image"
        
        if fitted:
            stage_strs = [
                f"{'g' if is_image else 'mode='}{s.refinement_cfg.patch_group_sizes if is_image else s.text_group_mode_}"
                f"(T={s.refinement_cfg.temperature}, α={s.refinement_cfg.ridge_alpha})"
                for s in self.stages_
            ]
        else:
            items = self.patch_group_sizes if is_image else self.text_group_modes
            n = len(items)
            temps  = self._expand_param(self.refinement_cfg.temperature, n, "temperature")
            alphas = self._expand_param(self.refinement_cfg.ridge_alpha,  n, "ridge_alpha")
            stage_strs = [
                f"{'g' if is_image else 'mode='}{item}(T={t}, α={a})"
                for item, t, a in zip(items, temps, alphas)
            ]
        best_note = f"  best={self.best_stage_idx_}" if fitted else ""
        return (
            f"IterativePALPooler(modality={self.modality!r}, "
            f"{'fitted' if fitted else 'not fitted'}: "
            f"[{', '.join(stage_strs)}]"
            f"  model_selection='{self.refinement_cfg.model_selection}'{best_note})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def pooler_factory(
    refinement_cfg: Union[RefinementConfig, TextRefinementConfig],
    seed: int,
    modality: str = "image",
) -> IterativePALPooler:
    """Convenience factory to build an IterativePALPooler from config dataclasses."""
    tabicl = TabICLClassifier(
        n_estimators=refinement_cfg.tabicl_n_estimators, random_state=seed
    )
    return IterativePALPooler(
        tabicl=tabicl,
        refinement_cfg=refinement_cfg,
        modality=modality,
        seed=seed,
    )
