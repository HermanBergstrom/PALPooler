"""Core text token quality scoring and pooling algorithms.

Functions here are pure NumPy/sklearn — no I/O, no visualisation.

The text analogues of the image patch functions in ``patch_pooling.py``.
Shared scoring utilities (``compute_patch_quality_logits``, ``RidgeGPU``,
etc.) live in ``patch_pooling.py`` and are reused unchanged — they operate on
arbitrary ``[T, n_classes]`` probability arrays regardless of modality.

Key differences from the image setting
---------------------------------------
* **Variable-length sequences**: each text has a different number of valid
  tokens.  Inputs are padded to ``[N, T_max, D]``; a boolean ``group_mask``
  tracks which positions are real.
* **Grouping modes** (``"none"`` / ``"sentence"``) replace the spatial
  ``patch_group_size`` integer.
* **Masked softmax**: pooling weights are computed only over valid groups.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from pal_pooling.config import TextRefinementConfig
from pal_pooling.patch_pooling import (
    RidgeGPU,
    compute_patch_quality_logits,
    compute_patch_quality_logits_gpu,
)
from tabicl import TabICLClassifier
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Text token grouping
# ---------------------------------------------------------------------------

def group_text_tokens(
    tokens: np.ndarray,          # [N, T_max, D]
    token_ids: np.ndarray,       # [N, T_max]  integer token IDs
    mode: str,                   # "none" | "sentence"
    sep_token_id: int = 102,     # BERT [SEP]
    cls_token_id: int = 101,     # BERT [CLS]  — always excluded from groups
) -> Tuple[np.ndarray, np.ndarray]:
    """Group BERT token embeddings and return a validity mask.

    The ``[CLS]`` token (first position where ``token_ids == cls_token_id``) is
    always excluded from the output groups.  Padding tokens (positions after
    the last ``[SEP]``) are excluded via the returned mask.

    Parameters
    ----------
    tokens : np.ndarray, shape [N, T_max, D]
        Padded BERT embeddings.
    token_ids : np.ndarray, shape [N, T_max]
        Integer token IDs aligned with *tokens*.
    mode : str
        ``"none"``     — individual non-CLS, non-padding tokens become groups.
        ``"sentence"`` — tokens are mean-pooled within each sentence, where
                         sentence boundaries are defined by ``[SEP]``.  The
                         ``[SEP]`` token itself is included in its sentence
                         before pooling.
    sep_token_id : int
        Token ID of the ``[SEP]`` marker (BERT default: 102).
    cls_token_id : int
        Token ID of the ``[CLS]`` marker (BERT default: 101); always excluded.

    Returns
    -------
    grouped : np.ndarray, shape [N, G_max, D]
        Grouped (or individual) token embeddings, padded to ``G_max``.
        Padding positions contain zeros.
    group_mask : np.ndarray, shape [N, G_max]  dtype bool
        ``True`` for valid groups, ``False`` for padding.

    Notes
    -----
    ``G_max`` equals ``T_max - 1`` for ``"none"`` mode (CLS excluded) and
    the maximum number of sentences across all samples for ``"sentence"`` mode.
    """
    if mode not in ("none", "sentence"):
        raise ValueError(f"text_group_mode must be 'none' or 'sentence', got {mode!r}")

    tokens = np.asarray(tokens, dtype=np.float32)
    token_ids = np.asarray(token_ids)
    N, T_max, D = tokens.shape

    if mode == "none":
        return _group_none(tokens, token_ids, cls_token_id, T_max, D)
    else:
        return _group_sentence(tokens, token_ids, sep_token_id, cls_token_id, D)


def _group_none(
    tokens: np.ndarray,    # [N, T_max, D]
    token_ids: np.ndarray, # [N, T_max]
    cls_token_id: int,
    T_max: int,
    D: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identity grouping: one group per non-CLS, non-padding token."""
    N = tokens.shape[0]
    # Valid positions: not [CLS] and not padding (token_id != 0).
    # [CLS] is assumed to be at position 0; we also guard by token ID in case
    # the sequence is unusual.
    is_cls     = token_ids == cls_token_id          # [N, T_max]
    is_padding = token_ids == 0                     # [N, T_max]
    valid      = ~is_cls & ~is_padding              # [N, T_max]

    G_max = int(valid.sum(axis=1).max()) if N > 0 else 0
    grouped    = np.zeros((N, G_max, D), dtype=np.float32)
    group_mask = np.zeros((N, G_max), dtype=bool)

    for i in range(N):
        idx = np.where(valid[i])[0]
        g   = len(idx)
        if g > 0:
            grouped[i, :g]    = tokens[i, idx]
            group_mask[i, :g] = True

    return grouped, group_mask


def _group_sentence(
    tokens: np.ndarray,    # [N, T_max, D]
    token_ids: np.ndarray, # [N, T_max]
    sep_token_id: int,
    cls_token_id: int,
    D: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sentence grouping: mean-pool tokens within each sentence.

    A 'sentence' is the span of tokens from (exclusive) the previous [SEP] /
    [CLS] up to and including the next [SEP].  The [CLS] token is excluded
    entirely.  Padding tokens (token_id == 0) terminate the sequence.
    """
    N = tokens.shape[0]

    # First pass: count max sentences across all samples.
    max_sentences = 0
    for i in range(N):
        n_sep = int((token_ids[i] == sep_token_id).sum())
        max_sentences = max(max_sentences, n_sep)

    if max_sentences == 0:
        # Fallback: treat the entire (non-CLS, non-padding) sequence as one sentence.
        max_sentences = 1

    grouped    = np.zeros((N, max_sentences, D), dtype=np.float32)
    group_mask = np.zeros((N, max_sentences), dtype=bool)

    for i in range(N):
        ids    = token_ids[i]                          # [T_max]
        embs   = tokens[i]                             # [T_max, D]

        # Collect sentence spans: start after [CLS] (pos 0), end at each [SEP].
        sep_positions = np.where(ids == sep_token_id)[0]
        if len(sep_positions) == 0:
            # No [SEP] found — treat all non-CLS, non-padding tokens as one group.
            valid = (ids != 0) & (ids != cls_token_id)
            span  = embs[valid]
            if len(span) > 0:
                grouped[i, 0]    = span.mean(axis=0)
                group_mask[i, 0] = True
            continue

        sent_idx  = 0
        start_tok = 1  # skip [CLS] at position 0

        for sep_pos in sep_positions:
            # span: [start_tok, sep_pos] inclusive (includes the [SEP] token)
            span_tokens = embs[start_tok : sep_pos + 1]  # may be empty
            # Exclude any padding tokens that snuck in before the first real SEP.
            span_ids    = ids[start_tok : sep_pos + 1]
            valid_in_span = span_ids != 0
            span_tokens   = span_tokens[valid_in_span]

            if len(span_tokens) > 0 and sent_idx < max_sentences:
                grouped[i, sent_idx]    = span_tokens.mean(axis=0)
                group_mask[i, sent_idx] = True
                sent_idx += 1

            start_tok = sep_pos + 1  # next sentence starts after this [SEP]

    return grouped, group_mask


# ---------------------------------------------------------------------------
# Masked Ridge softmax pooling helper
# ---------------------------------------------------------------------------

def _ridge_pool_weights_text(
    grouped:        np.ndarray,               # [N, G_max, D]
    group_mask:     np.ndarray,               # [N, G_max]  bool
    ridge_model:    Union[Ridge, "RidgeGPU"],
    feature_scaler: Optional[StandardScaler],
) -> np.ndarray:                              # [N, G_max]  softmax weights (0 for masked)
    """Compute per-group masked softmax pooling weights from a fitted Ridge model.

    Identical to ``_ridge_pool_weights`` from ``patch_pooling`` except that
    padded positions are forced to ``-inf`` before the softmax so they receive
    zero weight.
    """
    N, G_max, D = grouped.shape
    flat = grouped.reshape(N * G_max, D)
    if feature_scaler is not None:
        flat = feature_scaler.transform(flat)
    logits = ridge_model.predict(flat).reshape(N, G_max).astype(np.float32)

    # Mask out padding positions with -inf before softmax.
    logits[~group_mask] = -np.inf
    logits -= np.where(group_mask, logits, -np.inf).max(axis=1, keepdims=True)  # numerical stability
    exp_l  = np.exp(logits)
    exp_l[~group_mask] = 0.0
    row_sums = exp_l.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)  # guard all-masked rows
    return exp_l / row_sums


# ---------------------------------------------------------------------------
# Full text refinement pass
# ---------------------------------------------------------------------------

def refine_text_features(
    train_tokens:       np.ndarray,      # [N, T_max, D]  BERT embeddings (padded)
    train_token_ids:    np.ndarray,      # [N, T_max]  token IDs
    train_labels:       np.ndarray,      # [N]
    support_features:   np.ndarray,      # [N, d]  initial pooled (post-PCA) features
    refinement_cfg:     TextRefinementConfig,
    pca:                Optional[PCA],
    text_group_mode:    str,             # "none" | "sentence" — single stage value
    seed:               int = 42,
    gpu_ridge_device:   str = "cuda",
    tabicl:             Optional[TabICLClassifier] = None,
    context_features:   Optional[np.ndarray] = None,  # [N, D_context]
    tabular_probs:      Optional[np.ndarray] = None,  # [N, n_classes]
    val_tokens:         Optional[np.ndarray] = None,  # [N_val, T_max, D]
    val_token_ids:      Optional[np.ndarray] = None,  # [N_val, T_max]
    val_labels:         Optional[np.ndarray] = None,  # [N_val]
) -> tuple:
    """Refine mean-pooled text support features with Ridge-predicted quality-weighted pooling.

    Mirrors ``refine_dataset_features`` from ``patch_pooling.py`` for the text
    modality.  The core flow is identical:

    1. Fit a ``TabICLClassifier`` on *support_features*.
    2. Forward valid (non-padding) token groups through TabICL to get per-group
       quality-logit targets.
    3. Fit a ``Ridge`` model on ``(group_embedding, quality_logit)`` pairs.
    4. Predict Ridge quality logits for **all** groups of **all** sequences.
    5. Apply masked-softmax weights → re-pooled embeddings.
    6. Refit PCA on re-pooled embeddings.

    The key text-specific difference: sequences have variable numbers of valid
    groups.  Only valid groups (``group_mask == True``) are forwarded to TabICL
    and used as Ridge training targets; padded positions are never forwarded and
    receive zero weight during pooling.

    Parameters
    ----------
    train_tokens : np.ndarray, shape [N, T_max, D]
        Padded BERT token embeddings.
    train_token_ids : np.ndarray, shape [N, T_max]
        Token IDs aligned with *train_tokens*.
    train_labels : np.ndarray, shape [N]
    support_features : np.ndarray, shape [N, d]
        Initial (mean-pooled, optionally PCA-projected) support features.
    refinement_cfg : TextRefinementConfig
    pca : PCA or None
        PCA fitted on the baseline support features.
    text_group_mode : str
        Grouping mode for this stage (``"none"`` or ``"sentence"``).
    seed : int
    gpu_ridge_device : str
    tabicl : TabICLClassifier or None
    context_features : np.ndarray or None, shape [N, D_context]
    tabular_probs : np.ndarray or None, shape [N, n_classes]

    Returns
    -------
    refined : np.ndarray, shape [N, d]
    new_pca : PCA or None
    weights_ridge : np.ndarray, shape [N, G_max]   masked softmax weights
    ridge_model : Ridge or RidgeGPU
    feature_scaler : StandardScaler or None
    clf : TabICLClassifier
    fit_time_s : float
    pool_time_s : float
    class_prior : np.ndarray or None
    group_mask : np.ndarray, shape [N, G_max]  bool
    """
    if not isinstance(refinement_cfg.temperature, (int, float)):
        raise TypeError("refinement_cfg.temperature must be a float or int")
    if not isinstance(refinement_cfg.ridge_alpha, (int, float)):
        raise TypeError("refinement_cfg.ridge_alpha must be a float or int")

    t_start = time.perf_counter()
    N, T_max, D = train_tokens.shape

    # Group tokens according to the current stage's mode.
    grouped, group_mask = group_text_tokens(
        train_tokens, train_token_ids,
        mode=text_group_mode,
        sep_token_id=refinement_cfg.sep_token_id,
        cls_token_id=refinement_cfg.cls_token_id,
    )
    # grouped:    [N, G_max, D]
    # group_mask: [N, G_max]  bool
    G_max = grouped.shape[1]

    # When val tokens are provided, use them as the Ridge query source (mirrors
    # image path in refine_dataset_features).  The TabICL support is always the
    # train set; val tokens are forwarded as queries so the Ridge model sees
    # out-of-support signal.  Attention masking is disabled because val samples
    # are not in the support.
    if val_tokens is not None and val_labels is not None:
        val_grouped, val_group_mask = group_text_tokens(
            val_tokens, val_token_ids,
            mode=text_group_mode,
            sep_token_id=refinement_cfg.sep_token_id,
            cls_token_id=refinement_cfg.cls_token_id,
        )
        source_grouped    = val_grouped
        source_group_mask = val_group_mask
        source_labels     = val_labels
        _apply_attn_mask  = False
    else:
        source_grouped    = grouped
        source_group_mask = group_mask
        source_labels     = train_labels
        _apply_attn_mask  = refinement_cfg.use_attn_masking

    # Precompute empirical class prior.
    n_cls = int(train_labels.max()) + 1
    counts = np.bincount(train_labels.astype(np.int64), minlength=n_cls)
    empirical_prior = (counts / counts.sum()).astype(np.float32)

    _divergence_methods = ("kl_div", "wasserstein", "js_div", "tvd")
    class_prior: Optional[np.ndarray] = (
        empirical_prior if refinement_cfg.weight_method in _divergence_methods else None
    )

    # For current_pool_marginal, compute the prior by running the just-fitted classifier
    # on the current pooled support with a diagonal attention mask (each sequence cannot
    # attend to its own support row). This gives a sequence-level marginal that reflects
    # the actual discriminability of the pooled representation at this stage.
    if refinement_cfg.prior == "current_pool_marginal" and refinement_cfg.weight_method in _divergence_methods:
        print("[calibration] Computing current_pool_marginal prior from current pooled features ...")
        N_supp = len(support_for_clf)
        n_cls_prior = int(train_labels.max()) + 1
        blocked_diag_np = np.arange(N_supp)
        if hasattr(clf, "predict_proba_tensor"):
            import torch as _torch
            blocked_diag_t = _torch.from_numpy(blocked_diag_np).long()
            probs_supp_t = clf.predict_proba_tensor(support_for_clf, blocked_indices=blocked_diag_t)
            if probs_supp_t.shape[1] != n_cls_prior:
                cls_t = _torch.tensor(clf.classes_, device=probs_supp_t.device, dtype=_torch.long)
                full_t = _torch.zeros((N_supp, n_cls_prior), dtype=probs_supp_t.dtype, device=probs_supp_t.device)
                full_t[:, cls_t] = probs_supp_t
                probs_supp_t = full_t
            class_prior = probs_supp_t.mean(dim=0).cpu().numpy().astype(np.float32)
        else:
            probs_supp = clf.predict_proba(support_for_clf, blocked_indices=blocked_diag_np)
            if probs_supp.shape[1] != n_cls_prior:
                full = np.zeros((N_supp, n_cls_prior), dtype=probs_supp.dtype)
                full[:, clf.classes_] = probs_supp
                probs_supp = full
            class_prior = probs_supp.mean(axis=0).astype(np.float32)
        print(f"[calibration] current_pool_marginal prior: {np.round(class_prior, 4)}")

    # Fit the shared TabICL classifier on the (possibly context-augmented) support.
    if tabicl is not None:
        clf = tabicl
    else:
        clf = TabICLClassifier(
            n_estimators=refinement_cfg.tabicl_n_estimators, random_state=seed
        )
    support_for_clf = support_features
    if context_features is not None:
        support_for_clf = np.concatenate(
            [support_features, context_features.astype(np.float32)], axis=1
        )
    clf.fit(support_for_clf, train_labels)

    use_gpu = hasattr(clf, "predict_proba_tensor")

    _binary_dist = getattr(refinement_cfg, "binary_dist", False)

    # Collect (group_embedding, quality_logit) pairs for Ridge fitting.
    # Only valid groups (source_group_mask == True) are forwarded.
    # We flatten all valid groups across samples into a single query batch.

    # Build flat index arrays: which sample and which group position each row belongs to.
    N_source = source_grouped.shape[0]
    valid_sample_idx = []  # sample index for each valid group
    valid_group_idx  = []  # group position for each valid group
    for i in range(N_source):
        gidx = np.where(source_group_mask[i])[0]
        valid_sample_idx.extend([i] * len(gidx))
        valid_group_idx.extend(gidx.tolist())
    valid_sample_idx = np.array(valid_sample_idx, dtype=np.int64)
    valid_group_idx  = np.array(valid_group_idx,  dtype=np.int64)

    n_valid = len(valid_sample_idx)  # total valid groups across all sequences

    # Flat features for Ridge fitting: [n_valid, D]
    all_features = source_grouped[valid_sample_idx, valid_group_idx].astype(np.float32)  # [n_valid, D]

    # Build query features for TabICL: project via PCA and optionally append context.
    query_raw = all_features  # [n_valid, D]
    query_features = pca.transform(query_raw) if pca is not None else query_raw
    if context_features is not None:
        ctx_repeated = context_features[valid_sample_idx].astype(np.float32)
        query_features = np.concatenate([query_features, ctx_repeated], axis=1)

    # Attention masking: each group from sequence i must not attend to support row i.
    blocked_indices_np = valid_sample_idx if _apply_attn_mask else None

    # Subsampling cap (mirrors image path).
    exceeded = (
        refinement_cfg.max_query_rows is not None
        and n_valid > refinement_cfg.max_query_rows
    )
    if exceeded and refinement_cfg.use_random_subsampling:
        n_fwd = refinement_cfg.max_query_rows
        print(f"[text_sampling] Subsampling {n_fwd:,} / {n_valid:,} valid group rows "
              f"({100 * n_fwd / n_valid:.1f}%) for Ridge fitting")
        rng = np.random.RandomState(seed)
        sel  = np.sort(rng.choice(n_valid, size=n_fwd, replace=False))
        fit_sample_idx = valid_sample_idx[sel]
        fit_features   = all_features[sel]
        fit_qfeatures  = query_features[sel]
        fit_blocked    = fit_sample_idx if _apply_attn_mask else None
    elif exceeded:
        # Batched path: forward in batches of batch_size images (batch by sample,
        # not by flat group row, to keep per-image quality signals coherent).
        fit_sample_idx = valid_sample_idx
        fit_features   = all_features
        fit_qfeatures  = query_features
        fit_blocked    = blocked_indices_np
    else:
        fit_sample_idx = valid_sample_idx
        fit_features   = all_features
        fit_qfeatures  = query_features
        fit_blocked    = blocked_indices_np

    n_cls_expected = n_cls

    # Compute token_marginal prior if requested (must happen after getting probs but before using them).
    _compute_token_marginal = refinement_cfg.prior == "token_marginal" and refinement_cfg.weight_method in _divergence_methods

    if use_gpu:
        import torch as _torch
        _dev = clf.device_
        class_prior_t = (
            _torch.from_numpy(class_prior).to(_dev) if class_prior is not None else None
        )
        _fit_blocked_t = (
            _torch.from_numpy(fit_blocked).long() if fit_blocked is not None else None
        )
        probs_flat_t = clf.predict_proba_tensor(fit_qfeatures, blocked_indices=_fit_blocked_t)
        if probs_flat_t.shape[1] != n_cls_expected:
            cls_t  = _torch.tensor(clf.classes_, device=probs_flat_t.device, dtype=_torch.long)
            full_t = _torch.zeros(
                (probs_flat_t.shape[0], n_cls_expected),
                dtype=probs_flat_t.dtype, device=probs_flat_t.device
            )
            full_t[:, cls_t] = probs_flat_t
            probs_flat_t = full_t

        # Compute token_marginal from GPU probs
        if _compute_token_marginal:
            print("[calibration] Empirical label prior:    {0}".format(np.round(empirical_prior, 4)))
            mean_probs = probs_flat_t.mean(dim=0).cpu().numpy().astype(np.float32)
            print("[calibration] Marginal token predicted: {0}".format(np.round(mean_probs, 4)))
            class_prior = mean_probs
            class_prior_t = _torch.from_numpy(class_prior).to(_dev)

        # Build targets
        all_targets_t = _torch.empty(len(fit_features), dtype=_torch.float32, device=_dev)
        # Per-valid-group tabular prior slicing
        if tabular_probs is not None:
            active_tab_t = _torch.from_numpy(
                tabular_probs[fit_sample_idx].astype(np.float32)
            ).to(_dev)
        else:
            active_tab_t = None

        # Assign targets per sample (unique samples in fit_sample_idx)
        unique_samples, first_occurrence = np.unique(fit_sample_idx, return_index=True)
        # Build a mapping: flat row -> number of groups for that sample
        # We need to process groups per sample for compute_patch_quality_logits_gpu
        # which expects [P, n_classes] per sample.
        # Group by sample index.
        ptr = 0
        for s_local, s_global in enumerate(unique_samples):
            rows_for_s = np.where(fit_sample_idx == s_global)[0]
            start, end = rows_for_s[0], rows_for_s[-1] + 1
            prior_for_s = (
                active_tab_t[start] if active_tab_t is not None else class_prior_t
            )
            all_targets_t[start:end] = compute_patch_quality_logits_gpu(
                probs_flat_t[start:end],
                int(source_labels[s_global]),
                refinement_cfg.temperature,
                refinement_cfg.weight_method,
                prior_for_s,
                binary_dist=_binary_dist,
            )
        all_targets = all_targets_t
    else:
        _fit_blocked = fit_blocked
        probs_flat = clf.predict_proba(fit_qfeatures, blocked_indices=_fit_blocked)
        if probs_flat.shape[1] != n_cls_expected:
            full = np.zeros((probs_flat.shape[0], n_cls_expected), dtype=probs_flat.dtype)
            full[:, clf.classes_] = probs_flat
            probs_flat = full

        # Compute token_marginal from CPU probs
        if _compute_token_marginal:
            print("[calibration] Empirical label prior:    {0}".format(np.round(empirical_prior, 4)))
            mean_probs = probs_flat.mean(axis=0).astype(np.float32)
            print("[calibration] Marginal token predicted: {0}".format(np.round(mean_probs, 4)))
            class_prior = mean_probs

        all_targets = np.empty(len(fit_features), dtype=np.float32)
        if tabular_probs is not None:
            active_tab = tabular_probs[fit_sample_idx].astype(np.float32)
        else:
            active_tab = None

        unique_samples = np.unique(fit_sample_idx)
        for s_global in unique_samples:
            rows_for_s = np.where(fit_sample_idx == s_global)[0]
            start, end = rows_for_s[0], rows_for_s[-1] + 1
            prior_for_s = (
                active_tab[start] if active_tab is not None else class_prior
            )
            all_targets[start:end] = compute_patch_quality_logits(
                probs_flat[start:end],
                int(source_labels[s_global]),
                refinement_cfg.temperature,
                refinement_cfg.weight_method,
                prior_for_s,
                binary_dist=_binary_dist,
            )

    # --- Importance weights (optional) ---
    # Correct for length-proportional sampling bias so each sequence contributes
    # equally regardless of how many tokens it contributes to the Ridge fit.
    # Weights are normalized to mean=1 to keep the effective Ridge alpha stable.
    ridge_sample_weight: Optional[np.ndarray] = None
    _iw_basis = getattr(refinement_cfg, "length_importance_weight_basis", "none")
    if _iw_basis == "full_length":
        # 1/sqrt(L_full): downweights tokens from long sequences regardless of
        # how many were actually sampled.
        seq_lengths = source_group_mask.sum(axis=1).astype(np.float32)  # [N_source]
        raw_w = 1.0 / np.sqrt(seq_lengths[fit_sample_idx])              # [n_fit]
        n_fit = len(fit_sample_idx)
        ridge_sample_weight = raw_w * (n_fit / raw_w.sum())
        print(f"[text_ridge] Length importance weights (basis=full_length) "
              f"(min={ridge_sample_weight.min():.3f}, max={ridge_sample_weight.max():.3f})")
    elif _iw_basis == "full_length_clip":
        # 1/sqrt(max(L_full, floor)): same as full_length but clips the denominator
        # so very short sequences aren't excessively upweighted.
        floor = int(getattr(refinement_cfg, "length_importance_floor", 25))
        seq_lengths = source_group_mask.sum(axis=1).astype(np.float32)  # [N_source]
        clipped = np.maximum(seq_lengths[fit_sample_idx], floor)
        raw_w = 1.0 / np.sqrt(clipped)                                  # [n_fit]
        n_fit = len(fit_sample_idx)
        ridge_sample_weight = raw_w * (n_fit / raw_w.sum())
        print(f"[text_ridge] Length importance weights (basis=full_length_clip, floor={floor}) "
              f"(min={ridge_sample_weight.min():.3f}, max={ridge_sample_weight.max():.3f})")
    elif _iw_basis == "sampled_count":
        # 1/sqrt(n_sampled): downweights tokens from sequences that contribute
        # more rows to the actual fit batch.
        N_source = source_group_mask.shape[0]
        sampled_counts = np.bincount(fit_sample_idx, minlength=N_source).astype(np.float32)
        raw_w = 1.0 / np.sqrt(sampled_counts[fit_sample_idx])           # [n_fit]
        n_fit = len(fit_sample_idx)
        ridge_sample_weight = raw_w * (n_fit / raw_w.sum())
        print(f"[text_ridge] Length importance weights (basis=sampled_count) "
              f"(min={ridge_sample_weight.min():.3f}, max={ridge_sample_weight.max():.3f})")

    # --- Fit Ridge ---
    feature_scaler: Optional[StandardScaler] = None
    if refinement_cfg.normalize_features:
        feature_scaler = StandardScaler()
        fit_features = feature_scaler.fit_transform(fit_features)

    backend = "GPU" if gpu_ridge_device else "CPU"
    print(f"[text_ridge] Fitting Ridge(alpha={refinement_cfg.ridge_alpha}) on "
          f"{len(fit_features):,} group samples (D={D}, mode={text_group_mode}, "
          f"method={refinement_cfg.weight_method}, backend={backend}) ...")
    if gpu_ridge_device:
        ridge_model: Union[Ridge, RidgeGPU] = RidgeGPU(
            alpha=refinement_cfg.ridge_alpha, device=gpu_ridge_device
        )
    else:
        ridge_model = Ridge(alpha=refinement_cfg.ridge_alpha)
        import torch as _torch
        if isinstance(all_targets, _torch.Tensor):
            all_targets = all_targets.cpu().numpy()
    ridge_model.fit(fit_features, all_targets, sample_weight=ridge_sample_weight)
    print(f"[text_ridge] Train R²: {ridge_model.score(fit_features, all_targets):.4f}")

    t_fit_done = time.perf_counter()
    fit_time_s = t_fit_done - t_start

    # --- Pool all sequences with Ridge weights ---
    print("[text_ridge] Pooling support set with Ridge-predicted weights ...")
    weights_ridge = _ridge_pool_weights_text(
        grouped, group_mask, ridge_model, feature_scaler
    )  # [N, G_max]
    repooled_raw = (weights_ridge[:, :, None] * grouped).sum(axis=1).astype(np.float32)  # [N, D]

    if pca is not None:
        new_pca = PCA(n_components=pca.n_components_, random_state=seed)
        refined = new_pca.fit_transform(repooled_raw).astype(np.float32)
    else:
        refined = repooled_raw
        new_pca = None

    pool_time_s = time.perf_counter() - t_fit_done
    print(f"[text_timing] fit={fit_time_s:.1f}s  pool={pool_time_s:.1f}s  "
          f"total={fit_time_s + pool_time_s:.1f}s")

    return (
        refined, new_pca, weights_ridge, ridge_model,
        feature_scaler, clf, fit_time_s, pool_time_s,
        class_prior, group_mask,
    )
