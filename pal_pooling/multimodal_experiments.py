"""Multimodal experiments: image PALPooling + tabular features on PetFinder.

Evaluates IterativePALPooler (fitted on image patches only) whose pooled
image embeddings are PCA-reduced and concatenated with tabular features
before downstream TabICL classification.

Conditions evaluated
--------------------
    tabular_only        — raw tabular features → TabICL
    mean_pool_img       — mean-pooled patches + PCA → TabICL
    cls_img             — CLS token + PCA → TabICL
    mean_pool_img+tab   — mean-pool + PCA ‖ tabular → TabICL
    cls_img+tab         — CLS + PCA ‖ tabular → TabICL
    pal_img             — PALPool + PCA → TabICL
    pal_img+tab         — PALPool + PCA ‖ tabular → TabICL

Usage
-----
    python pal_pooling/multimodal_experiments.py \\
        [--n-estimators 1] [--pca-dim 128] [--seed 42] \\
        [--patch-group-sizes 1] [--temperature 1.0] [--ridge-alpha 1.0] \\
        [--output-dir results/multimodal]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from skrub import TableVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

from pal_pooling.config import (
    CBIS_DDSM_DATASET_PATH, CLOTHING_DATASET_PATH, DatasetConfig, DVM_DATASET_PATH,
    FEATURES_DIR, PAD_UFES_DATASET_PATH, PETFINDER_DATASET_PATH, RefinementConfig,
    SALARY_INDIA_DATASET_PATH, TextRefinementConfig, get_modality,
)
from pal_pooling.data_loading import _load_features
from pal_pooling.pal_pooler import IterativePALPooler, pooler_factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compute_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    n_estimators: int = 1,
    seed: int = 42,
) -> tuple[float, float]:
    """Fit TabICL on train_features and return (accuracy, auroc) on test_features."""
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(train_features, train_labels)
    proba = clf.predict_proba(test_features)
    acc = float((np.argmax(proba, axis=1) == test_labels).mean())
    try:
        if proba.shape[1] == 2:
            auroc = float(roc_auc_score(test_labels, proba[:, 1]))
        else:
            auroc = float(roc_auc_score(test_labels, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = float("nan")
    return acc, auroc


def _pca_project(
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    pca_dim: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Optional[PCA]]:
    """Fit PCA on train_raw, project both splits. Returns (train_proj, test_proj, pca)."""
    if pca_dim is None:
        return train_raw.astype(np.float32), test_raw.astype(np.float32), None
    N, D = train_raw.shape
    n_comp = min(pca_dim, N, D)
    pca = PCA(n_components=n_comp, random_state=seed)
    train_proj = pca.fit_transform(train_raw).astype(np.float32)
    test_proj = pca.transform(test_raw).astype(np.float32)
    return train_proj, test_proj, pca


def _concat_tabular(img_features: np.ndarray, tabular: np.ndarray) -> np.ndarray:
    """Concatenate image and tabular features along the feature axis."""
    return np.concatenate([img_features, tabular.astype(np.float32)], axis=1)


def _vectorize_tabular(
    tab_train: np.ndarray,
    tab_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a TableVectorizer on tab_train, transform both splits.

    Handles missing values (NaN) and mixed types via skrub's TableVectorizer.
    Returns (tab_train_out, tab_test_out) as float32 arrays.
    """
    col_names = [str(i) for i in range(tab_train.shape[1])]
    df_train = pd.DataFrame(tab_train, columns=col_names)
    df_test  = pd.DataFrame(tab_test,  columns=col_names)
    tv = TableVectorizer()
    tab_train_out = np.nan_to_num(tv.fit_transform(df_train).astype(np.float32))
    tab_test_out  = np.nan_to_num(tv.transform(df_test).astype(np.float32))
    print(f"[TableVectorizer] {tab_train.shape[1]} → {tab_train_out.shape[1]} features")
    return tab_train_out, tab_test_out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(
    dataset_name: str,
    dataset_path: Path,
    n_train: Optional[int],
    max_train: Optional[int],
    max_test: Optional[int],
    seed: int,
) -> tuple[
    np.ndarray, np.ndarray,   # train_patches, train_labels
    np.ndarray, np.ndarray,   # test_patches, test_labels
    np.ndarray, np.ndarray,   # cls_train, cls_test
    np.ndarray, np.ndarray,   # tab_train, tab_test
    dict,                      # idx_to_class
    dict,                      # extra_data
]:
    """Load dataset patch, CLS, and tabular features via data_loading._load_features.

    ``max_train`` / ``max_test`` limit how many samples are loaded; ``n_train``
    is an optional post-hoc subsample applied after loading.
    """
    _BACKBONE: dict[str, str] = {
        "petfinder":      "dinov3",
        "dvm":            "dinov3_local",
        "pad-ufes":       "dinov3_local",
        "cbis-ddsm-mass": "dinov3_local",
        "cbis-ddsm-calc": "dinov3_local",
        "clothing":       "electra",
        "salary":         "electra",
    }
    _SUPPORTS_TEXT = {"petfinder"}
    dataset_cfg = DatasetConfig(
        dataset=dataset_name,
        backbone=_BACKBONE.get(dataset_name, "dinov3_local"),
        features_dir=FEATURES_DIR,
        dataset_path=Path(dataset_path),
        n_train=max_train,
        n_test=max_test,
        n_val=None,
        n_sample=0,
        balance_train=False,
        balance_test=False,
    )

    (train_patches, train_labels, test_patches, test_labels,
     cls_train, cls_test, idx_to_class, _, _, extra_data) = _load_features(
        dataset_cfg, seed=seed, load_tabular=True,
        load_text=(dataset_name in _SUPPORTS_TEXT),
    )

    tab_train = extra_data["tab_train"]
    tab_test  = extra_data["tab_test"]

    tab_train, tab_test = _vectorize_tabular(tab_train, tab_test)
    extra_data["tab_train"] = tab_train
    extra_data["tab_test"]  = tab_test

    # Optional post-hoc n_train subsampling (separate from max_train loading limit).
    if n_train is not None and n_train < len(train_labels):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_labels), size=n_train, replace=False)
        idx.sort()
        train_patches = train_patches[idx]
        train_labels  = train_labels[idx]
        cls_train     = cls_train[idx]
        tab_train     = tab_train[idx]
        for k in ["text_train", "text_train_token_ids", "text_train_attn_mask", "text_cls_train"]:
            if extra_data.get(k) is not None:
                extra_data[k] = extra_data[k][idx]

    return (
        train_patches, train_labels,
        test_patches,  test_labels,
        cls_train, cls_test,
        tab_train, tab_test,
        idx_to_class,
        extra_data,   # NEW: contains text_train, text_test, etc. when available
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _run_single_seed(
    args: argparse.Namespace,
    seed: int,
    dataset_path: Path,
    output_dir: Path,
) -> dict:
    """Run all experiment conditions for a single seed. Returns the result record."""
    _set_global_seeds(seed)
    run_ts = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    # ── Load data ────────────────────────────────────────────────────────────
    (train_patches, train_labels,
     test_patches,  test_labels,
     cls_train, cls_test,
     tab_train, tab_test,
     idx_to_class,
     extra_data) = _load_dataset(
        dataset_name=args.dataset,
        dataset_path=dataset_path,
        n_train=args.n_train,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=seed,
    )

    # For text-primary datasets (clothing, salary) the token embeddings live in
    # train_patches / extra_data["train_token_ids"] rather than extra_data["text_train"].
    # Normalise both cases into a single set of text_* variables so the text
    # conditions block below can run unchanged.
    is_text_primary = get_modality(args.dataset) == "text" and extra_data.get("text_train") is None
    if is_text_primary:
        has_text           = True
        text_train         = train_patches.astype(np.float32)
        text_test          = test_patches.astype(np.float32)
        text_train_tok_ids = extra_data["train_token_ids"]
        text_test_tok_ids  = extra_data["test_token_ids"]
        text_train_attn    = extra_data["train_attention_mask"]
        text_test_attn     = extra_data["test_attention_mask"]
        text_cls_train     = cls_train
        text_cls_test      = cls_test
    else:
        has_text            = extra_data.get("text_train") is not None
        text_train          = extra_data.get("text_train")
        text_test           = extra_data.get("text_test")
        text_train_tok_ids  = extra_data.get("text_train_token_ids")
        text_test_tok_ids   = extra_data.get("text_test_token_ids")
        text_train_attn     = extra_data.get("text_train_attn_mask")
        text_test_attn      = extra_data.get("text_test_attn_mask")
        text_cls_train      = extra_data.get("text_cls_train")
        text_cls_test       = extra_data.get("text_cls_test")

    N_train, P, D = train_patches.shape
    n_classes = len(idx_to_class)
    pca_dim = None if args.no_pca else args.pca_dim

    _counts = np.bincount(train_labels.astype(np.int64), minlength=n_classes)
    _test_counts = np.bincount(test_labels.astype(np.int64), minlength=n_classes)
    print("[class balance]")
    print(f"  {'class':<8} {'train':>8} {'train %':>9} {'test':>8} {'test %':>9}")
    for _i in range(n_classes):
        print(
            f"  {idx_to_class[_i]:<8} {_counts[_i]:>8d} "
            f"{100*_counts[_i]/_counts.sum():>8.1f}% "
            f"{_test_counts[_i]:>8d} "
            f"{100*_test_counts[_i]/_test_counts.sum():>8.1f}%"
        )

    results: dict[str, dict] = {}

    def _eval(tag: str, train_feat: np.ndarray, test_feat: np.ndarray) -> None:
        t0 = time.perf_counter()
        acc, auroc = _compute_accuracy(
            train_feat, train_labels,
            test_feat,  test_labels,
            n_estimators=args.n_estimators,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "nan"
        print(f"[{tag}]  acc={acc:.4f}  auroc={auroc_str}  ({elapsed:.1f}s)  feat_dim={train_feat.shape[1]}")
        results[tag] = {
            "acc":      round(acc, 6),
            "auroc":    round(auroc, 6) if not np.isnan(auroc) else None,
            "time_s":   round(elapsed, 2),
            "feat_dim": int(train_feat.shape[1]),
        }

    # ── Baseline: tabular only ───────────────────────────────────────────────
    print("\n--- tabular_only ---")
    _eval("tabular_only", tab_train, tab_test)

    if not is_text_primary:
        # ── Baseline: mean-pool image only ───────────────────────────────────
        print("\n--- mean_pool_img ---")
        mean_train_raw = train_patches.astype(np.float32).mean(axis=1)  # [N, D]
        mean_test_raw  = test_patches.astype(np.float32).mean(axis=1)
        mean_train, mean_test, mean_pca = _pca_project(mean_train_raw, mean_test_raw, pca_dim, seed)
        _eval("mean_pool_img", mean_train, mean_test)

        # ── Baseline: CLS token image only ──────────────────────────────────
        print("\n--- cls_img ---")
        cls_train_proj, cls_test_proj, cls_pca = _pca_project(cls_train, cls_test, pca_dim, seed)
        _eval("cls_img", cls_train_proj, cls_test_proj)

        # ── Baseline: mean-pool image + tabular ──────────────────────────────
        print("\n--- mean_pool_img+tab ---")
        _eval("mean_pool_img+tab",
              _concat_tabular(mean_train, tab_train),
              _concat_tabular(mean_test,  tab_test))

        # ── Baseline: CLS + tabular ──────────────────────────────────────────
        print("\n--- cls_img+tab ---")
        _eval("cls_img+tab",
              _concat_tabular(cls_train_proj, tab_train),
              _concat_tabular(cls_test_proj,  tab_test))

    # ── Text conditions (gated on has_text) ────────────────────────────────
    if has_text:
        # ── Baseline: text mean-pool (exclude CLS at pos 0) -────────────────
        print("\n--- mean_pool_text ---")
        valid_mask_tr = text_train_attn.copy(); valid_mask_tr[:, 0] = False
        valid_mask_te = text_test_attn.copy();  valid_mask_te[:, 0] = False
        counts_tr = valid_mask_tr.sum(axis=1, keepdims=True).clip(min=1)
        counts_te = valid_mask_te.sum(axis=1, keepdims=True).clip(min=1)
        mean_text_raw_tr = (text_train * valid_mask_tr[:, :, None]).sum(axis=1) / counts_tr
        mean_text_raw_te = (text_test  * valid_mask_te[:, :, None]).sum(axis=1) / counts_te
        mean_text_tr, mean_text_te, _ = _pca_project(mean_text_raw_tr, mean_text_raw_te, pca_dim, seed)
        _eval("mean_pool_text", mean_text_tr, mean_text_te)

        # ── Baseline: text CLS ───────────────────────────────────────────────
        print("\n--- cls_text ---")
        cls_text_tr, cls_text_te, _ = _pca_project(text_cls_train, text_cls_test, pca_dim, seed)
        _eval("cls_text", cls_text_tr, cls_text_te)

        # ── Baselines with tabular ────────────────────────────────────────────
        _eval("mean_pool_text+tab", _concat_tabular(mean_text_tr, tab_train), _concat_tabular(mean_text_te, tab_test))
        _eval("cls_text+tab",       _concat_tabular(cls_text_tr, tab_train),  _concat_tabular(cls_text_te, tab_test))

        # ── Build TextRefinementConfig ────────────────────────────────────────
        text_refinement_cfg = TextRefinementConfig(
            refine=True,
            text_group_modes=args.text_group_modes,
            temperature=args.temperature,
            weight_method=args.text_weight_method or args.weight_method,
            ridge_alpha=args.ridge_alpha,
            normalize_features=args.normalize_features,
            batch_size=args.batch_size,
            max_query_rows=args.max_query_rows,
            use_random_subsampling=True,
            gpu_ridge=args.gpu_ridge,
            tabicl_n_estimators=args.n_estimators,
            tabicl_pca_dim=pca_dim,
            append_cls=False,
            use_global_prior=args.use_global_prior,
            use_attn_masking=args.use_attn_masking,
            prior=args.prior,
            model_selection=args.model_selection,
            length_importance_weight_basis=args.length_importance_weight_basis,
            train_val_fraction=args.train_val_fraction,
        )

        # ── Fit text PAL pooler (no tabular context) ──────────────────────────
        print("\n--- Fitting text IterativePALPooler (text-only) ---")
        pooler_text = pooler_factory(refinement_cfg=text_refinement_cfg, seed=seed, modality="text")
        pooler_text.fit(text_train, train_labels,
                        token_ids=text_train_tok_ids,
                        attention_mask=text_train_attn)
        pal_text_tr_raw = pooler_text.transform(text_train, token_ids=text_train_tok_ids, attention_mask=text_train_attn)
        pal_text_te_raw = pooler_text.transform(text_test,  token_ids=text_test_tok_ids,  attention_mask=text_test_attn)
        best_stage_txt  = pooler_text.stages_[pooler_text.best_stage_idx_]
        pal_text_pca    = best_stage_txt._pca_
        if pal_text_pca is not None:
            pal_text_tr = pal_text_pca.transform(pal_text_tr_raw).astype(np.float32)
            pal_text_te = pal_text_pca.transform(pal_text_te_raw).astype(np.float32)
        else:
            pal_text_tr = pal_text_tr_raw.astype(np.float32)
            pal_text_te = pal_text_te_raw.astype(np.float32)
        _eval("pal_text",     pal_text_tr, pal_text_te)
        _eval("pal_text+tab", _concat_tabular(pal_text_tr, tab_train), _concat_tabular(pal_text_te, tab_test))

        # ── Fit text PAL pooler (tabular context) ────────────────────────────
        print("\n--- Fitting text IterativePALPooler (tabular context) ---")
        pooler_text_ctx = pooler_factory(refinement_cfg=text_refinement_cfg, seed=seed, modality="text")
        pooler_text_ctx.fit(text_train, train_labels,
                            token_ids=text_train_tok_ids,
                            attention_mask=text_train_attn,
                            context_features=tab_train)
        pal_ctx_text_tr_raw = pooler_text_ctx.transform(text_train, token_ids=text_train_tok_ids, attention_mask=text_train_attn)
        pal_ctx_text_te_raw = pooler_text_ctx.transform(text_test,  token_ids=text_test_tok_ids,  attention_mask=text_test_attn)
        best_stage_txt_ctx  = pooler_text_ctx.stages_[pooler_text_ctx.best_stage_idx_]
        pal_ctx_text_pca    = best_stage_txt_ctx._pca_
        if pal_ctx_text_pca is not None:
            pal_ctx_text_tr = pal_ctx_text_pca.transform(pal_ctx_text_tr_raw).astype(np.float32)
            pal_ctx_text_te = pal_ctx_text_pca.transform(pal_ctx_text_te_raw).astype(np.float32)
        else:
            pal_ctx_text_tr = pal_ctx_text_tr_raw.astype(np.float32)
            pal_ctx_text_te = pal_ctx_text_te_raw.astype(np.float32)
        _eval("pal_context_text",     pal_ctx_text_tr, pal_ctx_text_te)
        _eval("pal_context_text+tab", _concat_tabular(pal_ctx_text_tr, tab_train), _concat_tabular(pal_ctx_text_te, tab_test))

    if not is_text_primary:
        # ── PALPool: fit on image patches only ───────────────────────────────
        print("\n--- Fitting IterativePALPooler (image-only) ---")
        refinement_cfg = RefinementConfig(
            refine=True,
            patch_size=16,
            patch_group_sizes=args.patch_group_sizes,
            temperature=args.temperature,
            weight_method=args.weight_method,
            ridge_alpha=args.ridge_alpha,
            normalize_features=args.normalize_features,
            batch_size=args.batch_size,
            max_query_rows=args.max_query_rows,
            use_random_subsampling=True,
            aoe_class=None,
            aoe_handling="filter",
            gpu_ridge=args.gpu_ridge,
            tabicl_n_estimators=args.n_estimators,
            tabicl_pca_dim=pca_dim,
            append_cls=False,
            use_global_prior=args.use_global_prior,
            use_attn_masking=args.use_attn_masking,
            prior=args.prior,
            model_selection=args.model_selection,
            train_val_fraction=args.train_val_fraction,
        )

        pooler = pooler_factory(refinement_cfg=refinement_cfg, seed=seed)
        t_fit = time.perf_counter()
        pooler.fit(train_patches, train_labels)
        fit_time_s = time.perf_counter() - t_fit
        print(f"[pal] Pooler fit in {fit_time_s:.1f}s")

        pal_train_raw = pooler.transform(train_patches)
        pal_test_raw  = pooler.transform(test_patches)

        best_stage = pooler.stages_[pooler.best_stage_idx_]
        pal_pca = best_stage._pca_

        if pal_pca is not None:
            pal_train_proj = pal_pca.transform(pal_train_raw).astype(np.float32)
            pal_test_proj  = pal_pca.transform(pal_test_raw).astype(np.float32)
        else:
            pal_train_proj = pal_train_raw.astype(np.float32)
            pal_test_proj  = pal_test_raw.astype(np.float32)

        print("\n--- pal_img ---")
        _eval("pal_img", pal_train_proj, pal_test_proj)

        print("\n--- pal_img+tab ---")
        _eval("pal_img+tab",
              _concat_tabular(pal_train_proj, tab_train),
              _concat_tabular(pal_test_proj,  tab_test))

        # ── Context-aware image PALPool ───────────────────────────────────────
        print("\n--- Fitting IterativePALPooler (tabular context) ---")
        pooler_ctx = pooler_factory(refinement_cfg=refinement_cfg, seed=seed)
        t_fit_ctx = time.perf_counter()

        ctx_features_train = tab_train
        if has_text and args.image_context_text:
            print("[info] Using text+tabular as context for image pooler (text fitted with tabular context)")
            text_pool_pca_tr, text_pool_pca_te, _ = _pca_project(pal_ctx_text_tr_raw, pal_ctx_text_te_raw, pca_dim, seed)
            ctx_features_train = np.concatenate([text_pool_pca_tr, tab_train], axis=1)

        pooler_ctx.fit(train_patches, train_labels, context_features=ctx_features_train)
        fit_time_ctx_s = time.perf_counter() - t_fit_ctx
        print(f"[pal_ctx] Pooler fit in {fit_time_ctx_s:.1f}s")

        pal_ctx_train_raw = pooler_ctx.transform(train_patches)
        pal_ctx_test_raw  = pooler_ctx.transform(test_patches)

        best_stage_ctx = pooler_ctx.stages_[pooler_ctx.best_stage_idx_]
        pal_ctx_pca = best_stage_ctx._pca_

        if pal_ctx_pca is not None:
            pal_ctx_train_proj = pal_ctx_pca.transform(pal_ctx_train_raw).astype(np.float32)
            pal_ctx_test_proj  = pal_ctx_pca.transform(pal_ctx_test_raw).astype(np.float32)
        else:
            pal_ctx_train_proj = pal_ctx_train_raw.astype(np.float32)
            pal_ctx_test_proj  = pal_ctx_test_raw.astype(np.float32)

        print("\n--- pal_context_img ---")
        _eval("pal_context_img", pal_ctx_train_proj, pal_ctx_test_proj)

        print("\n--- pal_context_img+tab ---")
        _eval("pal_context_img+tab",
              _concat_tabular(pal_ctx_train_proj, tab_train),
              _concat_tabular(pal_ctx_test_proj,  tab_test))

    # ── Combined image + text conditions (only when both modalities present) ──
    if has_text and not is_text_primary:
        print("\n--- Combined image + text conditions ---")
        # mean+mean
        _eval("mean_pool_img+mean_pool_text",
              np.concatenate([mean_train, mean_text_tr], axis=1),
              np.concatenate([mean_test,  mean_text_te], axis=1))
        _eval("mean_pool_img+mean_pool_text+tab",
              _concat_tabular(np.concatenate([mean_train, mean_text_tr], axis=1), tab_train),
              _concat_tabular(np.concatenate([mean_test,  mean_text_te], axis=1), tab_test))
        # cls+cls
        _eval("cls_img+cls_text",
              np.concatenate([cls_train_proj, cls_text_tr], axis=1),
              np.concatenate([cls_test_proj,  cls_text_te], axis=1))
        _eval("cls_img+cls_text+tab",
              _concat_tabular(np.concatenate([cls_train_proj, cls_text_tr], axis=1), tab_train),
              _concat_tabular(np.concatenate([cls_test_proj,  cls_text_te], axis=1), tab_test))
        # pal+pal
        _eval("pal_img+pal_text",
              np.concatenate([pal_train_proj, pal_text_tr], axis=1),
              np.concatenate([pal_test_proj,  pal_text_te], axis=1))
        _eval("pal_img+pal_text+tab",
              _concat_tabular(np.concatenate([pal_train_proj, pal_text_tr], axis=1), tab_train),
              _concat_tabular(np.concatenate([pal_test_proj,  pal_text_te], axis=1), tab_test))
        # pal_context+pal_context
        _eval("pal_context_img+pal_context_text",
              np.concatenate([pal_ctx_train_proj, pal_ctx_text_tr], axis=1),
              np.concatenate([pal_ctx_test_proj,  pal_ctx_text_te], axis=1))
        _eval("pal_context_img+pal_context_text+tab",
              _concat_tabular(np.concatenate([pal_ctx_train_proj, pal_ctx_text_tr], axis=1), tab_train),
              _concat_tabular(np.concatenate([pal_ctx_test_proj,  pal_ctx_text_te], axis=1), tab_test))

    total_time_s = time.perf_counter() - t_start
    record = {
        "run_timestamp": run_ts,
        "seed":          seed,
        "total_time_s":  round(total_time_s, 2),
        "args": {
            "dataset":            args.dataset,
            "dataset_path":       str(dataset_path),
            "n_train":            args.n_train,
            "n_estimators":       args.n_estimators,
            "pca_dim":            pca_dim,
            "patch_group_sizes":  args.patch_group_sizes,
            "text_group_modes":   args.text_group_modes,
            "temperature":        args.temperature,
            "ridge_alpha":        args.ridge_alpha,
            "weight_method":      args.weight_method,
            "text_weight_method": args.text_weight_method or args.weight_method,
            "image_context_text":   args.image_context_text,
            "train_val_fraction":   args.train_val_fraction,
            "seed":                 seed,
        },
        "dataset_info": {
            "n_train":   int(N_train),
            "n_test":    int(len(test_labels)),
            "n_patches": int(P),
            "embed_dim": int(D),
            "tab_dim":   int(tab_train.shape[1]),
            "n_classes": int(n_classes),
            "pca_dim":   int(pal_pca.n_components_) if (not is_text_primary and pal_pca is not None) else None,
        },
        "results": results,
    }

    print(f"\n[seed {seed}] Total time: {total_time_s:.1f}s")
    _print_summary(results)
    return record


def run_multimodal_experiment(args: argparse.Namespace) -> None:
    seeds = args.seeds

    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = {
            "dvm":            DVM_DATASET_PATH,
            "petfinder":      PETFINDER_DATASET_PATH,
            "pad-ufes":       PAD_UFES_DATASET_PATH,
            "cbis-ddsm-mass": CBIS_DDSM_DATASET_PATH,
            "cbis-ddsm-calc": CBIS_DDSM_DATASET_PATH,
            "clothing":       CLOTHING_DATASET_PATH,
            "salary":         SALARY_INDIA_DATASET_PATH,
        }[args.dataset]

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path("results/multimodal") / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "multimodal_results.json"

    # Load existing results so a re-run appends rather than overwrites.
    if combined_path.exists():
        with combined_path.open() as f:
            all_records: list[dict] = json.load(f)
        if not isinstance(all_records, list):
            # Legacy single-record file — wrap it.
            all_records = [all_records]
    else:
        all_records = []

    completed_seeds = {r.get("seed", r["args"]["seed"]) for r in all_records}

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {seed}  ({i + 1}/{len(seeds)})")
        print(f"{'='*60}")

        if seed in completed_seeds:
            print(f"[skip] seed {seed} already present in {combined_path}")
            continue

        record = _run_single_seed(args, seed, dataset_path, output_dir)
        all_records.append(record)

        with combined_path.open("w") as f:
            json.dump(all_records, f, indent=2)
        print(f"[saved] {combined_path}  ({len(all_records)} record(s) total)")

    print(f"\n[done] All seeds finished. Results → {combined_path}")


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print(f"{'Condition':<22} {'Acc':>8} {'AUROC':>8}")
    print("-" * 60)
    for tag, r in results.items():
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "  nan"
        print(f"  {tag:<20} {r['acc']:>8.4f} {auroc_str:>8}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multimodal experiment: PALPool image features + tabular (PetFinder or DVM)"
    )
    p.add_argument("--dataset",        type=str,   default="petfinder",
                   choices=["petfinder", "dvm", "pad-ufes", "cbis-ddsm-mass", "cbis-ddsm-calc",
                            "clothing", "salary"],
                   help="Dataset to run multimodal experiment on")
    p.add_argument("--dataset-path",   type=Path,  default=None,
                   help="Root directory of the dataset (defaults to config value)")
    p.add_argument("--n-train",        type=int,   default=None,
                   help="Subsample this many training images AFTER loading (default: use all)")
    p.add_argument("--max-train",      type=int,   default=None,
                   help="Load at most this many train+val images (fast initial subset)")
    p.add_argument("--max-test",       type=int,   default=None,
                   help="Load at most this many test images")
    p.add_argument("--n-estimators",   type=int,   default=1,
                   help="TabICL ensemble size (default: 1)")
    p.add_argument("--pca-dim",        type=int,   default=128,
                   help="PCA components for image features (default: 128)")
    p.add_argument("--no-pca",         action="store_true",
                   help="Disable PCA; use full 768-D image embeddings")
    p.add_argument("--patch-group-sizes", type=int, nargs="+", default=[1],
                   help="Patch group sizes for IterativePALPooler (default: [1])")
    p.add_argument("--text-group-modes", type=str, nargs="+", default=["none"],
                   choices=["none", "sentence"],
                   help="Token grouping modes for the text PAL pooler (default: ['none'])")
    p.add_argument("--temperature",    type=float, nargs="+", default=[1.0],
                   help="Softmax temperature(s) for patch pooling (default: [1.0])")
    p.add_argument("--ridge-alpha",    type=float, nargs="+", default=[1.0],
                   help="Ridge regularisation strength(s) (default: [1.0])")
    p.add_argument("--weight-method",  type=str,   default="correct_class_prob",
                   choices=["correct_class_prob", "entropy", "kl_div",
                            "wasserstein", "js_div", "tvd"],
                   help="Patch quality weight method (default: correct_class_prob)")
    p.add_argument("--text-weight-method", type=str, default=None,
                   choices=["correct_class_prob", "entropy", "kl_div",
                            "wasserstein", "js_div", "tvd"],
                   help="Weight method for text PAL pooler (default: same as --weight-method)")
    p.add_argument("--normalize-features", action="store_true",
                   help="Fit a StandardScaler on patches before Ridge fitting")
    p.add_argument("--batch-size",     type=int,   default=1000,
                   help="Images per TabICL forward pass during refinement (default: 1000)")
    p.add_argument("--max-query-rows", type=int,   default=None,
                   help="Cap on patch-group rows forwarded through TabICL")
    p.add_argument("--gpu-ridge",      action="store_true",
                   help="Solve Ridge on the GPU (requires PyTorch + CUDA)")
    p.add_argument("--use-global-prior", action="store_true",
                   help="Use the global class prior P(Y) as the divergence reference even when "
                        "context features are provided, instead of the per-image P(Y|X_tab).")
    p.add_argument("--use-attn-masking", action=argparse.BooleanOptionalAction, default=True,
                   help="Use attention masking in PALPooler (default: True)")
    p.add_argument("--prior", type=str, default="label_frequency",
                   choices=["label_frequency", "patch_marginal", "current_pool_marginal"],
                   help="Prior to use as the divergence reference (default: label_frequency)")
    p.add_argument("--model-selection", type=str, default="last_iteration",
                   choices=["last_iteration", "masked_train_accuracy", "validation_accuracy"],
                   help="Which stage to use at inference after iterative refinement. "
                        "'last_iteration' (default) always uses the final stage. "
                        "'masked_train_accuracy' evaluates every stage on the training set "
                        "with a diagonal attention mask and selects the best-performing one.")
    p.add_argument("--length-importance-weight-basis", type=str, default="none",
                   choices=["none", "full_length", "full_length_clip", "sampled_count"],
                   help="Basis for weighting tokens by sequence length in text PAL pooler (default: none)")
    p.add_argument("--train-val-fraction", type=float, default=None,
                   help="Fraction of training data held out as validation inside each PAL pooler "
                        "stage (e.g. 0.2). If None (default) no internal split is performed.")
    p.add_argument("--image-context-text", action="store_true",
                   help="When fitting image PAL pooler with context, include text+tabular features "
                        "(only for datasets with text support, e.g., petfinder)")
    p.add_argument("--seeds",          type=int,   nargs="+", default=[42],
                   help="One or more random seeds, e.g. --seeds 42 123 456. "
                        "Results are saved after every seed. (default: 42)")
    p.add_argument("--output-dir",     type=Path,  default=None,
                   help="Directory to save results (default: results/multimodal/<dataset>)")
    return p.parse_args()


if __name__ == "__main__":
    run_multimodal_experiment(_parse_args())
