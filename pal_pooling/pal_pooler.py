"""PALPooler — Pseudo-Attention Label Pooler.

Encapsulates a single stage of Ridge-based adaptive patch pooling.
Chain multiple instances via :class:`IterativePALPooler`, or manually by
passing ``_support_projected_`` and ``_pca_`` from one stage into
``initial_support`` / ``initial_pca`` on the next.

Typical single-stage usage
--------------------------
>>> from tabicl import TabICLClassifier
>>> from pal_pooling import PALPooler
>>>
>>> tabicl = TabICLClassifier(n_estimators=1, random_state=42)
>>> pooler = PALPooler(tabicl, patch_group_size=4, temperature=2.0, ridge_alpha=10.0)
>>> pooler.fit(train_patches, train_labels)          # [N, P, D], [N]
>>> X_train = pooler.transform(train_patches)        # [N, D]  raw DINO-space
>>> X_test  = pooler.transform(test_patches)         # [N_test, D]

Two-stage chained refinement (manual)
--------------------------------------
>>> stage0 = PALPooler(tabicl, patch_group_size=4, temperature=2.0, ridge_alpha=10.0)
>>> stage0.fit(train_patches, train_labels)
>>>
>>> stage1 = PALPooler(tabicl, patch_group_size=1, temperature=1.0, ridge_alpha=1.0)
>>> stage1.fit(train_patches, train_labels,
...            initial_support=stage0._support_projected_,
...            initial_pca=stage0._pca_)
>>>
>>> X_train = stage1.transform(train_patches)
>>> X_test  = stage1.transform(test_patches)

Two-stage chained refinement (IterativePALPooler)
--------------------------------------------------
>>> from pal_pooling import IterativePALPooler
>>>
>>> pooler = IterativePALPooler(
...     tabicl,
...     patch_group_sizes=[4, 1],
...     temperature=[2.0, 1.0],
...     ridge_alpha=[10.0, 1.0],
... )
>>> pooler.fit(train_patches, train_labels)
>>> X_train = pooler.transform(train_patches)
>>> X_test  = pooler.transform(test_patches)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from pal_pooling.config import RefinementConfig
from tabicl import TabICLClassifier

from pal_pooling.patch_pooling import (
    _ridge_pool_weights,
    group_patches,
    refine_dataset_features,
)


def _append_cls(grouped: np.ndarray, cls_tokens: Optional[np.ndarray]) -> np.ndarray:
    """Append a CLS token as one extra patch per image.

    Parameters
    ----------
    grouped : np.ndarray, shape [N, P', D]
        Spatially grouped patch embeddings.
    cls_tokens : np.ndarray or None, shape [N, D]
        Per-image CLS tokens.  When ``None`` the input is returned unchanged.

    Returns
    -------
    np.ndarray, shape [N, P'+1, D]  (or [N, P', D] when cls_tokens is None)
    """
    if cls_tokens is None:
        return grouped
    cls = np.asarray(cls_tokens, dtype=np.float32)[:, None, :]  # [N, 1, D]
    return np.concatenate([grouped, cls], axis=1)


class PALPooler:
    """Pseudo-Attention Label Pooler.

    Fits a Ridge regression model that predicts per-patch quality logits from
    raw DINO patch embeddings.  The Ridge model is trained to replicate quality
    signals produced by a TabICL classifier evaluated on a support set.  At
    transform time the Ridge logits drive a softmax-weighted pooling that
    collapses ``[N, P, D] → [N, D]``.

    A single ``PALPooler`` represents one stage of refinement at a fixed
    ``patch_group_size``.  Multi-stage refinement is achieved by chaining
    instances via :class:`IterativePALPooler`.

    The pooler intentionally outputs raw DINO-space features (``[N, D]``)
    so the caller can apply any dimensionality reduction they choose.
    PCA is used *internally* during refinement (to build a compact support
    for TabICL scoring) but the fitted projection is not part of the public
    API.

    Parameters
    ----------
    tabicl : TabICLClassifier
        Pre-initialized (unfitted) TabICL model whose ``n_estimators`` and
        ``random_state`` are forwarded to the internal quality scorer.
    refinement_cfg : RefinementConfig
        All pooling hyperparameters: ``patch_group_sizes``, ``temperature``,
        ``ridge_alpha``, ``weight_method``, ``tabicl_pca_dim``, etc.
        When used inside :class:`IterativePALPooler`, a per-stage copy with
        ``patch_group_sizes`` set to a single int is passed here.
    seed : int
        Random seed for PCA, subsampling, and TabICL initialisation.
    gpu_ridge_device : str
        Torch device string for GPU-accelerated Ridge (e.g. ``"cuda"``).
        Only relevant when ``refinement_cfg.gpu_ridge`` is ``True``.
    Fitted attributes (available after ``fit``)
    -------------------------------------------
    ridge_model_ : Ridge or RidgeGPU
        Fitted quality predictor.  The core learned component of the pooler.
    feature_scaler_ : StandardScaler or None
        Fitted scaler applied before Ridge prediction (``None`` unless
        ``normalize_features=True``).
    support_ : np.ndarray, shape [N, D]
        Refined support set embeddings in the original DINO feature space
        (raw, no PCA applied).  Apply your own dimensionality reduction
        before passing to a downstream classifier.
    support_labels_ : np.ndarray, shape [N]
        Labels corresponding to ``support_``.
    scoring_clf_ : TabICLClassifier
        Quality scorer fitted on the *input* support (used to generate Ridge
        training targets).  Retained for post-hoc visualisation.
    n_patches_grouped_ : int
        ``P'`` — number of patch groups after spatial grouping.
    embed_dim_ : int
        ``D`` — original patch embedding dimension.
    fit_time_s_ : float
        Seconds for the TabICL forward pass and Ridge fitting phase.
    pool_time_s_ : float
        Seconds for Ridge prediction over all images, repooling, and PCA refit.
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
        val_patches: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        val_cls_tokens: Optional[np.ndarray] = None,
        val_context_features: Optional[np.ndarray] = None,
        val_tabular_probs: Optional[np.ndarray] = None,
    ) -> "PALPooler":
        """Fit the Ridge quality model.

        Constructs a mean-pool support set from *patches* (or uses the
        provided *initial_support*), feeds it to TabICL to generate per-patch
        quality logit targets, then fits a Ridge model to predict those logits
        from raw patch features.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
            Raw patch embeddings for the training set.
        labels : np.ndarray, shape [N]
            Integer class labels.
        cls_tokens : np.ndarray or None, shape [N, D]
            Per-image CLS tokens.  When provided the CLS token is appended to
            the grouped patch tensor as one extra (ungrouped) patch before
            Ridge fitting and pooling.  Not affected by ``patch_group_size``.
        initial_support : np.ndarray or None
            Pre-built support (in the PCA-projected space of the previous
            stage) to use instead of the mean-pool baseline.  Pass
            ``prev_stage._support_projected_`` when chaining stages manually.
        initial_pca : PCA or None
            PCA corresponding to *initial_support*.  Pass
            ``prev_stage._pca_`` when chaining stages manually.
        context_features : np.ndarray or None, shape [N, D_context]
            Optional per-image side features (e.g. tabular) that are appended
            to both support rows and per-patch query rows before the TabICL
            scoring step.  They inform the quality scores but are never used
            as Ridge inputs — pooling weights are still derived from raw DINO
            features alone.
        tabular_probs : np.ndarray or None, shape [N, n_classes]
            Pre-computed P(Y|X_tab) for each training image.  When provided
            (and a divergence weight method is used), replaces the global class
            prior so each image's patches are scored relative to the
            tabular-informed marginal P(Y|X_tab_i).  Computed internally when
            *context_features* is given but *tabular_probs* is not; prefer
            passing it from :class:`IterativePALPooler` to avoid recomputation
            across stages.
        val_patches : np.ndarray or None, shape [N_val, P_val, D]
            Optional validation patches to be used for scoring and fitting the
            Ridge model instead of the training patches.
        val_labels : np.ndarray or None, shape [N_val]
        val_cls_tokens : np.ndarray or None, shape [N_val, D]
        val_context_features : np.ndarray or None, shape [N_val, D_context]
        val_tabular_probs : np.ndarray or None, shape [N_val, n_classes]

        Returns
        -------
        self
        """
        patches = np.asarray(patches, dtype=np.float32)
        labels = np.asarray(labels)
        N, P, D = patches.shape
        self.embed_dim_ = D

        grouped = group_patches(patches, self.refinement_cfg.patch_group_sizes)  # [N, P', D]
        grouped = _append_cls(grouped, cls_tokens)                               # [N, P'+1, D] or unchanged
        self.n_patches_grouped_ = grouped.shape[1]

        if initial_support is not None:
            support = initial_support
            current_pca = initial_pca
        else:
            # ── Initial mean pool ─────────────────────────────────────────────
            # By default the CLS token (if present) is included in the average.
            # To pool spatial patches only, replace `grouped` with:
            #   spatial = group_patches(patches, self.refinement_cfg.patch_group_sizes)
            #   support_raw = spatial.mean(axis=1)
            support_raw = grouped.mean(axis=1)  # [N, D]
            if self.pca_dim is not None:
                n_comp = min(self.pca_dim, N, D)
                current_pca = PCA(n_components=n_comp, random_state=self.seed)
                support = current_pca.fit_transform(support_raw).astype(np.float32)
            else:
                current_pca = None
                support = support_raw

        aoe_mask: Optional[np.ndarray] = None
        if self.refinement_cfg.aoe_class is not None:
            aoe_mask = (labels == self.refinement_cfg.aoe_class)

        val_grouped = None
        val_aoe_mask = None
        if val_patches is not None and val_labels is not None:
            val_patches = np.asarray(val_patches, dtype=np.float32)
            val_labels = np.asarray(val_labels)
            val_grouped = group_patches(val_patches, self.refinement_cfg.patch_group_sizes)
            val_grouped = _append_cls(val_grouped, val_cls_tokens)
            if self.refinement_cfg.aoe_class is not None:
                val_aoe_mask = (val_labels == self.refinement_cfg.aoe_class)

        (refined_support, new_pca, weights_ridge, ridge_model, feature_scaler, scoring_clf,
         fit_time_s, pool_time_s, updated_class_prior) = refine_dataset_features(
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
            val_aoe_mask=val_aoe_mask,
        )

        # Derive raw (pre-PCA) support in the original D-dimensional DINO space.
        # weights_ridge [N, P'] are the Ridge softmax weights already computed
        # inside refine_dataset_features; reuse them to avoid a second prediction.
        repooled_raw = (weights_ridge[:, :, None] * grouped).sum(axis=1).astype(np.float32)  # [N, D]
        raw_support = repooled_raw

        self.ridge_model_ = ridge_model
        self.feature_scaler_ = feature_scaler
        # Private: internal PCA and projected support for stage chaining and score_tabicl.
        self._pca_ = new_pca
        self._support_projected_ = refined_support
        # Public: raw DINO-space support (caller applies own dim reduction).
        self.support_ = raw_support
        self.support_labels_ = labels
        self.scoring_clf_ = scoring_clf
        self.class_prior_ = updated_class_prior
        self.fit_time_s_ = fit_time_s
        self.pool_time_s_ = pool_time_s

        return self

    def transform(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pool patches using the fitted Ridge model.

        Groups patches spatially at ``patch_group_size``, optionally appends
        the CLS token, computes softmax pooling weights from the Ridge model,
        and returns the weighted sum in the original DINO feature space.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
        cls_tokens : np.ndarray or None, shape [N, D]
            Per-image CLS tokens.  Must be provided at transform time if they
            were provided at fit time (the Ridge model was trained with them).

        Returns
        -------
        np.ndarray, shape [N, D]
            Quality-weighted pooled embeddings in the original DINO feature
            space.  Apply your own dimensionality reduction before passing
            to a downstream classifier.
        """
        self._check_fitted()
        patches = np.asarray(patches, dtype=np.float32)
        grouped = group_patches(patches, self.refinement_cfg.patch_group_sizes)      # [N, P', D]
        grouped = _append_cls(grouped, cls_tokens)                                   # [N, P'+1, D] or unchanged
        weights = _ridge_pool_weights(grouped, self.ridge_model_, self.feature_scaler_)  # [N, P'(+1)]
        return (weights[:, :, None] * grouped).sum(axis=1)           # [N, D]

    def fit_transform(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
        initial_support: Optional[np.ndarray] = None,
        initial_pca: Optional[PCA] = None,
    ) -> np.ndarray:
        """Fit and transform in one call.

        Equivalent to ``fit(...).transform(patches, cls_tokens)``.  Returns
        ``[N, D]`` raw pooled embeddings in the original DINO feature space.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
        labels : np.ndarray, shape [N]
        cls_tokens : np.ndarray or None, shape [N, D]
        initial_support, initial_pca : optional, see ``fit``

        Returns
        -------
        np.ndarray, shape [N, D]
        """
        return self.fit(patches, labels, cls_tokens, initial_support, initial_pca).transform(patches, cls_tokens)

    # ------------------------------------------------------------------
    # Visualisation support
    # ------------------------------------------------------------------

    def patch_weights(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-patch Ridge softmax weights.

        Useful for heatmap overlays: the returned array directly describes
        how much each (grouped) patch (and the CLS token, if present)
        contributes to the pooled embedding.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D] or [P, D]
        cls_tokens : np.ndarray or None, shape [N, D] or [D] for a single image

        Returns
        -------
        np.ndarray, shape [N, P'(+1)] or [P'(+1)]
            Softmax weights summing to 1 per image.  The last entry is the
            CLS-token weight when *cls_tokens* is provided.
        """
        self._check_fitted()
        single = patches.ndim == 2
        if single:
            patches = patches[None]
            if cls_tokens is not None:
                cls_tokens = np.asarray(cls_tokens, dtype=np.float32)
                if cls_tokens.ndim == 1:
                    cls_tokens = cls_tokens[None]   # [1, D]
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
        """Raw Ridge quality predictions before softmax normalisation.

        These are the values the Ridge model predicts from raw patch features.
        They have the same semantics as the targets in ``compute_patch_quality_logits``
        (negative log-probability scores, in ``(-∞, 0]`` for ``correct_class_prob``
        and ``entropy``).  Useful for debugging and understanding the Ridge fit.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D] or [P, D]
        cls_tokens : np.ndarray or None, shape [N, D] or [D] for a single image

        Returns
        -------
        np.ndarray, shape [N, P'(+1)] or [P'(+1)]
            Raw Ridge logits before softmax.  The last entry is the CLS-token
            logit when *cls_tokens* is provided.
        """
        self._check_fitted()
        single = patches.ndim == 2
        if single:
            patches = patches[None]
            if cls_tokens is not None:
                cls_tokens = np.asarray(cls_tokens, dtype=np.float32)
                if cls_tokens.ndim == 1:
                    cls_tokens = cls_tokens[None]   # [1, D]
        patches = np.asarray(patches, dtype=np.float32)
        N, P, D = patches.shape
        grouped = group_patches(patches, self.refinement_cfg.patch_group_sizes)   # [N, P', D]
        grouped = _append_cls(grouped, cls_tokens)                                # [N, P'+1, D] or unchanged
        _, P_prime, _ = grouped.shape
        flat = grouped.reshape(N * P_prime, D)
        if self.feature_scaler_ is not None:
            flat = self.feature_scaler_.transform(flat)
        logits = self.ridge_model_.predict(flat).reshape(N, P_prime).astype(np.float32)
        return logits[0] if single else logits

    def score_tabicl(
        self,
        query_patches: np.ndarray,
        query_labels: np.ndarray,
        n_estimators: Optional[int] = None,
        query_cls_tokens: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Evaluate test-set accuracy and AUROC using the fitted support.

        Pools *query_patches* with the fitted Ridge model, projects into the
        internal support space (via ``_pca_``), fits a fresh
        ``TabICLClassifier`` on ``_support_projected_``, and reports accuracy
        + AUROC on the query set.

        This mirrors the ``_compute_accuracy_from_features`` evaluation step
        in ``pal_experiment.py``.

        Parameters
        ----------
        query_patches : np.ndarray, shape [N_test, P, D]
        query_labels : np.ndarray, shape [N_test]
        n_estimators : int or None
            TabICL ensemble size.  Defaults to ``self.tabicl.n_estimators``.
        query_cls_tokens : np.ndarray or None, shape [N_test, D]
            CLS tokens for the query set.  Required when the model was fitted
            with ``cls_tokens``.

        Returns
        -------
        (accuracy, auroc) : tuple[float, float]
            AUROC is ``float("nan")`` when it cannot be computed (e.g. single
            class in the query set).
        """
        from sklearn.metrics import roc_auc_score

        self._check_fitted()
        n_est = n_estimators or getattr(self.tabicl, "n_estimators", 1)
        query_raw = self.transform(query_patches, cls_tokens=query_cls_tokens)  # [N, D]
        if self._pca_ is not None:
            query_features = self._pca_.transform(query_raw).astype(np.float32)
        else:
            query_features = query_raw

        clf = TabICLClassifier(n_estimators=n_est, random_state=self.seed)
        clf.fit(self._support_projected_, self.support_labels_)
        proba = clf.predict_proba(query_features)
        acc = float((np.argmax(proba, axis=1) == query_labels).mean())
        try:
            if proba.shape[1] == 2:
                auroc = float(roc_auc_score(query_labels, proba[:, 1]))
            else:
                auroc = float(roc_auc_score(query_labels, proba, multi_class="ovr", average="macro"))
        except ValueError:
            auroc = float("nan")
        return acc, auroc

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the entire fitted pooler to a ``joblib`` file.

        The saved file includes the Ridge model, scaler, and all
        hyperparameters.  Load with ``PALPooler.load(path)``.
        """
        import joblib
        self._check_fitted()
        joblib.dump(self, Path(path))
        print(f"[PALPooler] Saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PALPooler":
        """Load a previously saved ``PALPooler`` from a ``joblib`` file."""
        import joblib
        pooler = joblib.load(Path(path))
        if not isinstance(pooler, cls):
            raise TypeError(f"Loaded object is {type(pooler).__name__}, expected PALPooler")
        return pooler

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "ridge_model_"):
            raise RuntimeError("PALPooler is not fitted yet.  Call fit() first.")

    def __repr__(self) -> str:
        fitted = hasattr(self, "ridge_model_")
        status = (
            f"fitted: D={self.embed_dim_}  P'={self.n_patches_grouped_}  "
            f"fit={self.fit_time_s_:.1f}s  pool={self.pool_time_s_:.1f}s"
            if fitted else "not fitted"
        )
        return (
            f"PALPooler("
            f"patch_group_sizes={self.refinement_cfg.patch_group_sizes}, "
            f"weight_method='{self.refinement_cfg.weight_method}', "
            f"temperature={self.refinement_cfg.temperature}, "
            f"ridge_alpha={self.refinement_cfg.ridge_alpha}, "
            f"pca_dim={self.refinement_cfg.tabicl_pca_dim}, "
            f"[{status}])"
        )


class IterativePALPooler:
    """Multi-stage PAL pooler that chains refinement stages automatically.

    Each stage fits a :class:`PALPooler`, refines the support, and passes the
    refined projected support to the next stage as its starting point — exactly
    replicating the iterative loop in ``pal_experiment.py``.
    After fitting, all transform / scoring calls are delegated to the final
    stage.

    Parameters
    ----------
    tabicl : TabICLClassifier
        Pre-initialized (unfitted) TabICL model shared across all stages.
    refinement_cfg : RefinementConfig
        Pooling hyperparameters shared across all stages.
        ``patch_group_sizes`` must be a list with one entry per stage;
        ``temperature`` and ``ridge_alpha`` may be scalars (broadcast) or
        per-stage lists of the same length.
    seed : int
        Random seed forwarded to every stage.
    gpu_ridge_device : str
        Torch device string forwarded to every stage (e.g. ``"cuda"``).
    Fitted attributes (available after ``fit``)
    -------------------------------------------
    stages_ : list of PALPooler
        All fitted stage poolers, in order.
    support_ : np.ndarray, shape [N, D]
        Raw DINO-space support from the final stage (``stages_[-1].support_``).
    support_labels_ : np.ndarray, shape [N]
        Labels corresponding to ``support_``.
    """

    def __init__(
        self,
        tabicl: TabICLClassifier,
        refinement_cfg: RefinementConfig,
        gpu_ridge_device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.tabicl = tabicl
        self.refinement_cfg = refinement_cfg
        self.patch_group_sizes = list(refinement_cfg.patch_group_sizes)
        self.pca_dim = refinement_cfg.tabicl_pca_dim
        self.seed = seed
        self.gpu_ridge_device = gpu_ridge_device

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
        stage_callback: Optional[Callable] = None,
        context_features: Optional[np.ndarray] = None,
        val_patches: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        val_cls_tokens: Optional[np.ndarray] = None,
        val_context_features: Optional[np.ndarray] = None,
    ) -> "IterativePALPooler":
        """Fit all stages sequentially, passing the refined support forward.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
            Raw patch embeddings for the training set.
        labels : np.ndarray, shape [N]
            Integer class labels.
        cls_tokens : np.ndarray or None, shape [N, D]
            Per-image CLS tokens.  When provided the CLS token is appended to
            the grouped patch tensor as one extra (ungrouped) patch at every
            stage.  The same tokens are forwarded unchanged to each stage.
        context_features : np.ndarray or None, shape [N, D_context]
            Optional per-image side features (e.g. tabular) forwarded unchanged
            to every stage.  See :meth:`PALPooler.fit` for semantics.
        stage_callback : callable or None
            Optional hook called after each stage is fitted.  Signature::

                stage_callback(
                    stage_idx: int,
                    stage: PALPooler,
                    group_size: int,
                    pre_refine_support: np.ndarray,
                    pre_refine_pca: Optional[PCA],
                    train_grouped: np.ndarray,
                )

            ``pre_refine_support`` / ``pre_refine_pca`` are the support and PCA
            *before* this stage's refinement (i.e. the input to the stage).
            ``train_grouped`` is the spatially-grouped training patch tensor
            ``[N, P', D]`` used internally by this stage (CLS token excluded).

        Returns
        -------
        self
        """
        n_stages = len(self.patch_group_sizes)
        if n_stages == 0:
            raise ValueError("patch_group_sizes must contain at least one entry.")

        temps = self._expand_param(self.refinement_cfg.temperature, n_stages, "temperature")
        alphas = self._expand_param(self.refinement_cfg.ridge_alpha, n_stages, "ridge_alpha")

        stages: List[PALPooler] = []

        # Build the mean-pool baseline support up front so that stage 0's callback
        # receives it as pre_refine_support rather than None.
        patches = np.asarray(patches, dtype=np.float32)
        N, _P, D = patches.shape
        if cls_tokens is not None:
            cls_tokens = np.asarray(cls_tokens, dtype=np.float32)
            # Include CLS token in initial mean pool alongside spatial patches.
            # To exclude CLS from the initial mean pool, replace this with:
            #   support_raw = patches.mean(axis=1)
            support_raw = np.concatenate([patches, cls_tokens[:, None, :]], axis=1).mean(axis=1)
        else:
            support_raw = patches.mean(axis=1)  # [N, D]
        if self.pca_dim is not None:
            n_comp = min(self.pca_dim, N, D)
            initial_pca: Optional[PCA] = PCA(n_components=n_comp, random_state=self.seed)
            initial_support: Optional[np.ndarray] = initial_pca.fit_transform(support_raw).astype(np.float32)
        else:
            initial_pca = None
            initial_support = support_raw

        # Compute tabular-only P(Y|X_tab) once — reused as the per-image reference prior
        # across all stages to avoid repeated TabICL calls for divergence weight methods.
        tabular_probs: Optional[np.ndarray] = None
        if (context_features is not None
                and self.refinement_cfg.weight_method in ("kl_div", "wasserstein", "js_div", "tvd")
                and not getattr(self.refinement_cfg, "use_global_prior", False)):
            print("[IterativePALPooler] Computing tabular-only P(Y|X_tab) for per-image prior ...")
            _clf_tab = TabICLClassifier(
                n_estimators=self.refinement_cfg.tabicl_n_estimators, random_state=self.seed
            )
            _clf_tab.fit(context_features, labels)
            raw_tab_probs = _clf_tab.predict_proba(context_features).astype(np.float32)
            n_cls_local = int(labels.max()) + 1
            if raw_tab_probs.shape[1] != n_cls_local:
                tabular_probs = np.zeros((raw_tab_probs.shape[0], n_cls_local), dtype=np.float32)
                tabular_probs[:, _clf_tab.classes_] = raw_tab_probs
            else:
                tabular_probs = raw_tab_probs

        val_tabular_probs: Optional[np.ndarray] = None
        if val_context_features is not None and val_labels is not None and tabular_probs is not None:
            raw_vtab_probs = _clf_tab.predict_proba(val_context_features).astype(np.float32)
            if raw_vtab_probs.shape[1] != n_cls_local:
                val_tabular_probs = np.zeros((raw_vtab_probs.shape[0], n_cls_local), dtype=np.float32)
                val_tabular_probs[:, _clf_tab.classes_] = raw_vtab_probs
            else:
                val_tabular_probs = raw_vtab_probs

        for k, group_size in enumerate(self.patch_group_sizes):
            from pal_pooling.patch_pooling import group_patches

            print(f"[IterativePALPooler] Stage {k}/{n_stages - 1} "
                  f"— patch_group_size={group_size}, "
                  f"temperature={temps[k]}, ridge_alpha={alphas[k]}")

            iteration_k_refinement_cfg = copy.deepcopy(self.refinement_cfg)
            iteration_k_refinement_cfg.patch_group_sizes = group_size
            iteration_k_refinement_cfg.temperature = temps[k]
            iteration_k_refinement_cfg.ridge_alpha = alphas[k]

            # Snapshot support/PCA before refinement for the callback.
            pre_refine_support = initial_support
            pre_refine_pca = initial_pca

            # Compute grouped patches here (spatial only, without CLS) so the
            # callback receives them for visualisation purposes.
            train_grouped = group_patches(patches, group_size)  # [N, P', D]

            stage = PALPooler(
                tabicl=self.tabicl,
                refinement_cfg=iteration_k_refinement_cfg,
                seed=self.seed,
                gpu_ridge_device=self.gpu_ridge_device,
            )
            stage.fit(patches, labels, cls_tokens=cls_tokens,
                      initial_support=initial_support, initial_pca=initial_pca,
                      context_features=context_features, tabular_probs=tabular_probs,
                      val_patches=val_patches, val_labels=val_labels,
                      val_cls_tokens=val_cls_tokens,
                      val_context_features=val_context_features, val_tabular_probs=val_tabular_probs)
            stages.append(stage)

            if stage_callback is not None:
                stage_callback(
                    stage_idx=k,
                    stage=stage,
                    group_size=group_size,
                    pre_refine_support=pre_refine_support,
                    pre_refine_pca=pre_refine_pca,
                    train_grouped=train_grouped,
                )

            # Hand the internal projected support to the next stage.
            initial_support = stage._support_projected_
            initial_pca = stage._pca_

        self.stages_ = stages
        return self

    def transform(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Pool patches using the final fitted stage.

        Parameters
        ----------
        patches : np.ndarray, shape [N, P, D]
        cls_tokens : np.ndarray or None, shape [N, D]

        Returns
        -------
        np.ndarray, shape [N, D]
            Quality-weighted pooled embeddings in the original DINO feature space.
        """
        self._check_fitted()
        return self.stages_[-1].transform(patches, cls_tokens=cls_tokens)

    def fit_transform(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
        context_features: Optional[np.ndarray] = None,
        val_patches: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        val_cls_tokens: Optional[np.ndarray] = None,
        val_context_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit all stages then transform *patches* with the final stage.

        Returns
        -------
        np.ndarray, shape [N, D]
        """
        return self.fit(patches, labels, cls_tokens=cls_tokens,
                        context_features=context_features,
                        val_patches=val_patches, val_labels=val_labels,
                        val_cls_tokens=val_cls_tokens,
                        val_context_features=val_context_features).transform(patches, cls_tokens=cls_tokens)

    # ------------------------------------------------------------------
    # Convenience delegations to final stage
    # ------------------------------------------------------------------

    @property
    def support_(self) -> np.ndarray:
        """Raw DINO-space support from the final stage."""
        self._check_fitted()
        return self.stages_[-1].support_

    @property
    def support_labels_(self) -> np.ndarray:
        self._check_fitted()
        return self.stages_[-1].support_labels_

    def patch_weights(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-patch Ridge softmax weights from the final stage."""
        self._check_fitted()
        return self.stages_[-1].patch_weights(patches, cls_tokens=cls_tokens)

    def patch_quality_logits(
        self,
        patches: np.ndarray,
        cls_tokens: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Raw Ridge quality logits from the final stage."""
        self._check_fitted()
        return self.stages_[-1].patch_quality_logits(patches, cls_tokens=cls_tokens)

    def score_tabicl(
        self,
        query_patches: np.ndarray,
        query_labels: np.ndarray,
        n_estimators: Optional[int] = None,
        query_cls_tokens: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Evaluate accuracy and AUROC using the final stage's support.

        See :meth:`PALPooler.score_tabicl` for full documentation.
        """
        self._check_fitted()
        return self.stages_[-1].score_tabicl(
            query_patches, query_labels, n_estimators,
            query_cls_tokens=query_cls_tokens,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the entire fitted iterative pooler to a ``joblib`` file."""
        import joblib
        self._check_fitted()
        joblib.dump(self, Path(path))
        print(f"[IterativePALPooler] Saved → {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IterativePALPooler":
        """Load a previously saved ``IterativePALPooler`` from a ``joblib`` file."""
        import joblib
        pooler = joblib.load(Path(path))
        if not isinstance(pooler, cls):
            raise TypeError(
                f"Loaded object is {type(pooler).__name__}, expected IterativePALPooler"
            )
        return pooler

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_param(param, n_stages: int, name: str) -> list:
        if isinstance(param, list):
            if len(param) == 1:
                return param * n_stages
            else:
                if len(param) != n_stages:
                    raise ValueError(
                        f"{name} list has {len(param)} entries but patch_group_sizes "
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
        if fitted:
            stage_strs = [
                f"g{s.refinement_cfg.patch_group_sizes}(T={s.refinement_cfg.temperature}, α={s.refinement_cfg.ridge_alpha})"
                for s in self.stages_
            ]
        else:
            temps = self._expand_param(self.refinement_cfg.temperature, len(self.patch_group_sizes), "temperature")
            alphas = self._expand_param(self.refinement_cfg.ridge_alpha, len(self.patch_group_sizes), "ridge_alpha")
            stage_strs = [
                f"g{g}(T={t}, α={a})"
                for g, t, a in zip(self.patch_group_sizes, temps, alphas)
            ]
        return (
            f"IterativePALPooler("
            f"{'fitted' if fitted else 'not fitted'}: "
            f"[{', '.join(stage_strs)}])"
        )


def pooler_factory(refinement_cfg: RefinementConfig, seed: int) -> IterativePALPooler:
    """Convenience factory to build a PALPooler from config dataclasses."""
    tabicl = TabICLClassifier(n_estimators=refinement_cfg.tabicl_n_estimators, random_state=seed)
    
    pooler = IterativePALPooler(tabicl=tabicl, 
                                refinement_cfg=refinement_cfg, 
                                seed=seed)
    return pooler