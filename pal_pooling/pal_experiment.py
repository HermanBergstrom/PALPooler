"""PAL (Pseudo Attention Label) experiment runner.

Evaluates iterative multi-scale quality-weighted patch pooling against mean-pool and
CLS-token baselines, with optional attention-pooling and patch-quality visualisations.

Usage:
    python pal_pooling/pal_experiment.py \\
        [--n-sample 8] [--n-estimators 1] [--pca-dim 128] \\
        [--seed 42] [--output-dir patch_quality_results]
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLClassifier
from tqdm import tqdm
from pal_pooling.tabicl_gpu_adapter import TabICLGPUAdapter
from pal_pooling.patch_pooling import (
    _ridge_pool_weights,
    group_patches,
    refine_dataset_features,
)
from pal_pooling.patch_visualisation import summary_figure, visualise_image
from pal_pooling.text_visualisation import visualise_text_batch, visualise_text
from pal_pooling.config import DatasetConfig, ImagePALConfig, TextPALConfig, AttentionPoolConfig, RunConfig, ExperimentConfig, parse_args
from pal_pooling.data_loading import (
    _get_image_paths,
    _dicom_to_pil,
    _get_petfinder_image_paths,
    _get_dvm_image_paths,
    _get_pad_ufes_image_paths,
    _get_cbis_ddsm_image_paths,
    _get_rsna_image_paths,
    _load_features,
    _balance_classes,
)

from pal_pooling.pal_pooler import IterativePALPooler, pooler_factory

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _set_global_seeds(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility across runs on the same machine.

    Sets Python's ``random``, NumPy's global state, and PyTorch (CPU + all CUDA
    devices).  Also enables CuDNN deterministic mode so GPU convolution kernels
    produce identical results.

    Note: floating-point results may still differ across *machines* that use
    different BLAS implementations (OpenBLAS vs MKL) for PCA/Ridge, because
    those libraries may choose different factorisation paths.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _compute_accuracy_from_features(
    support_features: np.ndarray,   # [N_train, d]
    support_labels:   np.ndarray,   # [N_train]
    query_features:   np.ndarray,   # [N_test, d]  already projected into support space
    query_labels:     np.ndarray,   # [N_test]
    n_estimators:     int = 1,
    seed:             int = 42,
) -> tuple[float, float]:
    """Classify pre-projected query features against a support set.

    Returns:
        (accuracy, auroc)  — auroc is NaN if it cannot be computed (e.g. single class in test set).
    """
    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, support_labels)
    proba = clf.predict_proba(query_features)   # [N_test, n_classes]
    acc   = float((np.argmax(proba, axis=1) == query_labels).mean())
    try:
        if proba.shape[1] == 2:
            auroc = float(roc_auc_score(query_labels, proba[:, 1]))
        else:
            auroc = float(roc_auc_score(query_labels, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = float("nan")
    return acc, auroc


def _compute_accuracy(
    support_features: np.ndarray,   # [N_train, d]
    support_labels:   np.ndarray,   # [N_train]
    test_patches:     np.ndarray,   # [N_test, P, D]
    test_labels:      np.ndarray,   # [N_test]
    pca:              Optional[PCA],
    n_estimators:     int = 1,
    seed:             int = 42,
) -> tuple[float, float]:
    """Accuracy and AUROC of TabICL on the held-out test set using mean-pooled test queries."""
    test_query = test_patches.mean(axis=1)   # [N_test, D]
    if pca is not None:
        test_query = pca.transform(test_query).astype(np.float32)
    return _compute_accuracy_from_features(
        support_features, support_labels, test_query, test_labels, n_estimators, seed
    )


# ---------------------------------------------------------------------------
# Visual evaluation loop
# ---------------------------------------------------------------------------

def _run_visual_eval(
    tag:              str,
    support_features: np.ndarray,      # [N_train, d]
    train_labels:     np.ndarray,      # [N_train]
    split_configs:    list,            # list of (split_name, patches, labels, image_paths, sample_idx)
    idx_to_class:     dict[int, str],
    pca:              Optional[PCA],
    n_estimators:     int,
    patch_size:       int,
    seed:             int,
    output_dir:       Path,
    temperature:      float = 1.0,
    ridge_model:      Optional[Ridge] = None,
    feature_scaler:   Optional[StandardScaler] = None,
    open_image:       Optional[Callable[[Path], Image.Image]] = None,
    class_prior:      Optional[np.ndarray] = None,   # [n_classes] empirical class frequencies
    weight_method:     str  = "correct_class_prob",
    show_pred_label:   bool = False,
    show_minority_prob: bool = False,
    show_per_class_probs: bool = False,
    use_attn_masking:  bool = False,
    binary_dist:       bool = False,
) -> dict[str, float]:
    """Run the patch-quality visual evaluation for one support set variant.

    Saves per-image heatmaps and a summary bar chart under output_dir/tag/<split>/.
    Returns a dict mapping split_name → mean correct-class probability.
    """
    n_classes = int(train_labels.max()) + 1
    mean_probs: dict[str, float] = {}

    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, train_labels)

    for split_name, patches_all, labels_all, image_paths, sample_idx in split_configs:
        split_out_dir = output_dir / tag / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)

        if len(sample_idx) == 0:
            mean_probs[split_name] = 0.0
            continue

        # --- Batch forward pass for plotting speed ---
        sampled_patches = patches_all[sample_idx]  # [N_s, P, D]
        N_s, P_dim, D_dim = sampled_patches.shape
        flat_query = sampled_patches.reshape(N_s * P_dim, D_dim)
        if pca is not None:
            flat_query = pca.transform(flat_query)

        if split_name == "train_loo":
            mask = np.ones(len(support_features), dtype=bool)
            mask[sample_idx] = False
            clf.fit(support_features[mask], train_labels[mask])
            probs_all = clf.predict_proba(flat_query)
            # Restore clf state for potential next configs
            clf.fit(support_features, train_labels)
        elif split_name == "train" and use_attn_masking:
            # Each query image is in the support; block its patches from attending
            # to its own support row. Mask shape: (N_s * P_dim, N_train).
            N_train = len(support_features)
            attn_mask = np.zeros((N_s * P_dim, N_train), dtype=bool)
            for _i, _orig in enumerate(sample_idx):
                attn_mask[_i * P_dim:(_i + 1) * P_dim, _orig] = True
            probs_all = clf.predict_proba(flat_query, attn_mask=attn_mask)
        else:
            probs_all = clf.predict_proba(flat_query)
            
        probs_all = probs_all.reshape(N_s, P_dim, n_classes)

        ridge_pred_logits_all = None
        if ridge_model is not None:
            ridge_pred_logits_all = _ridge_pool_weights(
                sampled_patches, ridge_model, feature_scaler
            )
        # ---------------------------------------------

        results: list[dict] = []
        bar = tqdm(enumerate(sample_idx), total=len(sample_idx),
                   desc=f"[{tag}] {split_name}", unit="img")
        for i, img_idx in bar:
            true_label = int(labels_all[img_idx])
            class_name = idx_to_class[true_label]

            probs = probs_all[i]   # [P, n_classes]

            correct_probs     = probs[:, true_label]
            mean_correct_prob = float(correct_probs.mean())
            patch_preds       = probs.argmax(axis=1)
            unique, counts    = np.unique(patch_preds, return_counts=True)
            modal_class       = unique[counts.argmax()]

            bar.set_postfix(
                true=class_name,
                P_true=f"{mean_correct_prob:.3f}",
                modal=idx_to_class[modal_class],
            )

            results.append(
                dict(img_idx=img_idx, label=true_label, class_name=class_name,
                     mean_correct_prob=mean_correct_prob)
            )

            ridge_pred_logits = ridge_pred_logits_all[i] if ridge_pred_logits_all is not None else None

            _opener = open_image or (lambda p: Image.open(p).convert("RGB"))
            img = _opener(image_paths[img_idx])
            fig = visualise_image(
                img, probs, true_label, idx_to_class,
                n_classes=n_classes,
                patch_size=patch_size,
                temperature=temperature,
                ridge_pred_logits=ridge_pred_logits,
                class_prior=class_prior,
                weight_method=weight_method,
                show_pred_label=show_pred_label,
                show_minority_prob=show_minority_prob,
                show_per_class_probs=show_per_class_probs,
                binary_dist=binary_dist,
            )
            out_path = (
                split_out_dir
                / f"patch_quality_{i:02d}_img{img_idx}_{class_name.replace(' ', '_')}.png"
            )
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        sfig     = summary_figure(results)
        sum_path = split_out_dir / "summary.png"
        sfig.savefig(sum_path, dpi=150, bbox_inches="tight")
        plt.close(sfig)

        mean_prob = float(np.mean([r["mean_correct_prob"] for r in results]))
        mean_probs[split_name] = mean_prob
        tqdm.write(f"[{tag}] {split_name}  mean P(true)={mean_prob:.3f}  summary → {sum_path}")

    return mean_probs


# ---------------------------------------------------------------------------
# Visual evaluation loop (Text modality)
# ---------------------------------------------------------------------------

def _run_text_visual_eval(
    tag:              str,
    support_features: np.ndarray,      # [N_train, d]
    train_labels:     np.ndarray,      # [N_train]
    split_configs:    list,            # list of (split_name, texts, patches, labels, token_ids, token_to_word, sample_idx)
    idx_to_class:     dict[int, str],
    pca:              Optional[PCA],
    n_estimators:     int,
    seed:             int,
    output_dir:       Path,
    temperature:      float = 1.0,
    ridge_model:      Optional[Ridge] = None,
    feature_scaler:   Optional[StandardScaler] = None,
    class_prior:      Optional[np.ndarray] = None,   # [n_classes] empirical class frequencies
    weight_method:     str  = "correct_class_prob",
    binary_dist:       bool = False,
) -> dict[str, float]:
    """Run the text-token quality visual evaluation for one support set variant.

    Saves per-text visualization figures under output_dir/tag/<split>/.
    Returns a dict mapping split_name → mean correct-class probability.
    """
    n_classes = int(train_labels.max()) + 1
    mean_probs: dict[str, float] = {}

    clf = TabICLClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(support_features, train_labels)

    for split_name, texts_all, patches_all, labels_all, token_ids_all, token_to_word_all, sample_idx in split_configs:
        split_out_dir = output_dir / tag / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)

        if len(sample_idx) == 0:
            mean_probs[split_name] = 0.0
            continue

        # --- Batch forward pass for plotting speed ---
        sampled_texts       = [texts_all[i] for i in sample_idx]
        sampled_patches     = patches_all[sample_idx]      # [N_s, T_max, D]
        sampled_token_ids   = token_ids_all[sample_idx]    # [N_s, T_max]
        sampled_token_to_word = token_to_word_all[sample_idx]  # [N_s, T_max]
        N_s, T_max, D_dim = sampled_patches.shape

        # Reshape for batch forward pass: [N_s * T_max, D]
        flat_query = sampled_patches.reshape(N_s * T_max, D_dim)
        if pca is not None:
            flat_query = pca.transform(flat_query)

        # Get per-token predictions
        probs_all = clf.predict_proba(flat_query)  # [N_s * T_max, n_classes]
        probs_all = probs_all.reshape(N_s, T_max, n_classes)

        ridge_pred_weights_all = None
        if ridge_model is not None:
            ridge_pred_weights_all = _ridge_pool_weights(
                sampled_patches, ridge_model, feature_scaler
            )  # [N_s, T_max]

        results: list[dict] = []
        bar = tqdm(enumerate(sample_idx), total=len(sample_idx),
                   desc=f"[{tag}] {split_name}", unit="text")
        for i, text_idx in bar:
            true_label = int(labels_all[text_idx])
            class_name = idx_to_class[true_label]

            # Get this text's data
            text = sampled_texts[i]
            text_probs = probs_all[i]           # [T_max, n_classes]
            token_ids = sampled_token_ids[i]    # [T_max]
            token_to_word = sampled_token_to_word[i]  # [T_max]

            correct_probs     = text_probs[:, true_label]
            mean_correct_prob = float(correct_probs.mean())

            bar.set_postfix(
                true=class_name,
                P_true=f"{mean_correct_prob:.3f}",
            )

            results.append(
                dict(text_idx=text_idx, label=true_label, class_name=class_name,
                     mean_correct_prob=mean_correct_prob)
            )

            ridge_weights = ridge_pred_weights_all[i] if ridge_pred_weights_all is not None else None

            # Generate visualization figure
            fig = visualise_text(
                text, token_to_word, token_ids, text_probs, true_label, idx_to_class,
                n_classes=n_classes,
                temperature=temperature,
                ridge_weights=ridge_weights,
                class_prior=class_prior,
                weight_method=weight_method,
                binary_dist=binary_dist,
            )
            out_path = (
                split_out_dir
                / f"text_quality_{i:02d}_idx{text_idx}_{class_name.replace(' ', '_')}.png"
            )
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        mean_prob = float(np.mean([r["mean_correct_prob"] for r in results]))
        mean_probs[split_name] = mean_prob
        tqdm.write(f"[{tag}] {split_name}  mean P(true)={mean_prob:.3f}  figures → {split_out_dir}")

    return mean_probs


def _cfg_to_args_dict(cfg: ExperimentConfig) -> dict:
    """Convert ExperimentConfig to a JSON-serializable dict for provenance logging."""
    def _convert(v):
        if isinstance(v, dict):
            return {ik: _convert(iv) for ik, iv in v.items()}
        if isinstance(v, list):
            return [_convert(i) for i in v]
        if isinstance(v, Path):
            return str(v)
        return v

    return {k: _convert(v) for k, v in asdict(cfg).items() if k != "cli_args"}


def _save_results(
    output_dir:    Path,
    run_ts:        str,
    total_time_s:  float,
    train_patches: np.ndarray,
    test_labels:   np.ndarray,
    D:             int,
    n_classes:     int,
    pca:           Optional[PCA],
    cls_acc:       Optional[float],
    cls_auroc:     Optional[float],
    baseline_acc:  float,
    baseline_auroc: float,
    all_results:   list,
    cfg:           ExperimentConfig,
    attn_result:   Optional[dict] = None,
) -> None:
    """Serialise experiment results to output_dir/results.json."""
    def _fmt(v: float) -> Optional[float]:
        return round(v, 6) if not np.isnan(v) else None

    record: dict = {
        "run_timestamp": run_ts,
        "total_time_s":  round(total_time_s, 2),
        "args": _cfg_to_args_dict(cfg),
        "dataset": {
            "n_train":   int(train_patches.shape[0]),
            "n_test":    int(len(test_labels)),
            "n_patches": int(train_patches.shape[1]),
            "embed_dim": int(D),
            "n_classes": int(n_classes),
            "pca_dim":   int(pca.n_components_) if pca is not None else None,
        },
        "baselines": {
            "cls_token":    round(float(cls_acc), 6) if cls_acc is not None else None,
            "cls_token_auroc": _fmt(cls_auroc) if cls_auroc is not None else None,
            "mean_pool":    round(float(baseline_acc), 6),
            "mean_pool_auroc": _fmt(baseline_auroc),
            "attn_pool":    attn_result,
        },
        "stages": [
            {
                "tag":             stage_name,
                "test_accuracy":   round(float(acc), 6),
                "test_auroc":      _fmt(auroc),
                "val_accuracy":    round(float(val_acc), 6) if val_acc is not None else None,
                "delta_acc":       round(float(acc - baseline_acc), 6),
                "delta_auroc":     _fmt(auroc - baseline_auroc) if not np.isnan(auroc) and not np.isnan(baseline_auroc) else None,
                "mean_prob_train": round(float(mean_probs.get("train", float("nan"))), 6),
                "mean_prob_test":  round(float(mean_probs.get("test",  float("nan"))), 6),
                "fit_time_s":      round(fit_s,    2),
                "pool_time_s":     round(pool_s,   2),
                "refine_time_s":   round(refine_s, 2),
                "eval_time_s":     round(eval_s,   2),
            }
            for stage_name, acc, auroc, mean_probs, refine_s, eval_s, fit_s, pool_s, val_acc in all_results
        ],
    }
    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[results] Saved → {results_path}")


def _run_attn_only_text(
    train_tokens:        np.ndarray,       # [N_train, T_max, D]
    train_token_ids:     np.ndarray,       # [N_train, T_max]
    train_attention_mask: np.ndarray,      # [N_train, T_max] bool
    train_labels:        np.ndarray,
    test_tokens:         np.ndarray,       # [N_test, T_max, D]
    test_token_ids:      np.ndarray,
    test_attention_mask: np.ndarray,
    test_labels:         np.ndarray,
    D:                   int,
    output_dir:          Path,
    attn_cfg:            AttentionPoolConfig,
    seed:                int,
    cfg:                 ExperimentConfig,
) -> dict:
    """Train attention pooling head on text tokens (with masking support).

    Returns:
        dict with keys: test_acc, test_auroc, best_val_acc_raw, best_val_step,
        time_to_best_s, total_train_time_s.
    """
    import torch as _torch
    from pal_pooling.attention_pooling import train_attention_pooling_head, _pool_with_head
    from pal_pooling.frozen_tabicl import EpisodicTrainingConfig

    if attn_cfg.device == "auto":
        _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    else:
        _device = _torch.device(attn_cfg.device)

    training_cfg = EpisodicTrainingConfig(
        num_steps=attn_cfg.attn_steps,
        learning_rate=attn_cfg.attn_lr,
        max_step_samples=attn_cfg.attn_max_step_samples,
        seed=seed,
        log_every=50
    )
    print(f"\n[attn-pool-text]  Training attention head  "
          f"(steps={attn_cfg.attn_steps}  lr={attn_cfg.attn_lr}  device={_device}  "
          f"n_queries={attn_cfg.attn_num_queries}  n_heads={attn_cfg.attn_num_heads}  "
          f"n_train={len(train_labels)})")

    t_start = time.perf_counter()
    head, attn_history = train_attention_pooling_head(
        train_patches=_torch.from_numpy(train_tokens),
        y_train=train_labels,
        val_patches=_torch.from_numpy(test_tokens),
        y_val=test_labels,
        train_attention_mask=_torch.from_numpy(train_attention_mask),
        val_attention_mask=_torch.from_numpy(test_attention_mask),
        embed_dim=D,
        out_dim=None,
        num_queries=attn_cfg.attn_num_queries,
        num_heads=attn_cfg.attn_num_heads,
        device=_device,
        config=training_cfg,
    )
    total_time_s = time.perf_counter() - t_start

    best_val_acc_raw = max(attn_history["val_accuracy"]) if attn_history["val_accuracy"] else float("nan")
    time_to_best_s   = attn_history.get("time_to_best_s", float("nan"))
    best_val_step    = attn_history.get("best_val_step", 0)

    # Post-hoc evaluation: pool with best checkpoint → PCA (if used) → TabICLClassifier
    print(f"[attn-pool-text]  Evaluating best checkpoint (step {best_val_step}) with PCA={attn_cfg.tabicl_pca_dim} ...")
    train_pooled = _pool_with_head(head, _torch.from_numpy(train_tokens), _device, mask=_torch.from_numpy(train_attention_mask))
    test_pooled  = _pool_with_head(head, _torch.from_numpy(test_tokens),  _device, mask=_torch.from_numpy(test_attention_mask))
    if attn_cfg.tabicl_pca_dim is not None:
        n_comp_attn = min(attn_cfg.tabicl_pca_dim, len(train_labels), train_pooled.shape[1])
        attn_pca    = PCA(n_components=n_comp_attn, random_state=seed)
        train_pooled = attn_pca.fit_transform(train_pooled).astype(np.float32)
        test_pooled  = attn_pca.transform(test_pooled).astype(np.float32)
    test_acc, test_auroc = _compute_accuracy_from_features(
        train_pooled, train_labels, test_pooled, test_labels,
        n_estimators=attn_cfg.tabicl_n_estimators, seed=seed,
    )

    attn_result = {
        "test_acc":           round(test_acc, 6),
        "test_auroc":         round(test_auroc, 6) if not np.isnan(test_auroc) else None,
        "best_val_acc_raw":   round(best_val_acc_raw, 6),
        "best_val_step":      best_val_step,
        "time_to_best_s":     time_to_best_s,
        "total_train_time_s": round(total_time_s, 2),
    }
    print(f"[attn-pool-text]  test acc (PCA={attn_cfg.tabicl_pca_dim}): {test_acc:.4f}  auroc: {test_auroc:.4f}  "
          f"(best train val: {best_val_acc_raw:.4f}  "
          f"step {best_val_step}/{attn_cfg.attn_steps}  time_to_best={time_to_best_s:.1f}s)")

    record = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "args": _cfg_to_args_dict(cfg),
        "dataset": {"n_train": int(len(train_labels)), "n_test": int(len(test_labels)), "embed_dim": int(D)},
        "attn_pool": attn_result,
    }
    attn_path = output_dir / "attn_pool_text_results.json"
    with attn_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"[attn-pool-text]  Saved → {attn_path}")
    return attn_result


def _run_attn_only(
    train_patches:        np.ndarray,
    train_labels:         np.ndarray,
    test_patches:         np.ndarray,
    test_labels:          np.ndarray,
    D:                    int,
    output_dir:           Path,
    attn_cfg:             AttentionPoolConfig,
    seed:                 int,
    cfg:                  ExperimentConfig,
) -> dict:
    """Train the attention pooling head, save results to attn_pool_results.json, and return the result dict.

    Checkpoint selection uses the best validation accuracy from the training loop
    (full 768-dim features via the frozen TabICL backbone).  The reported test
    accuracy is evaluated post-hoc: pool train with the best head, fit PCA if
    pca_dim is set, then call TabICLClassifier on the PCA-reduced features —
    matching the dimensionality used by all other baselines.

    Returns:
        dict with keys: test_acc, test_auroc, best_val_acc_raw, best_val_step,
        time_to_best_s, total_train_time_s.
    """
    import torch as _torch
    from pal_pooling.attention_pooling import train_attention_pooling_head, _pool_with_head
    from pal_pooling.frozen_tabicl import EpisodicTrainingConfig

    if attn_cfg.device == "auto":
        _device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    else:
        _device = _torch.device(attn_cfg.device)

    training_cfg = EpisodicTrainingConfig(
        num_steps=attn_cfg.attn_steps,
        learning_rate=attn_cfg.attn_lr,
        max_step_samples=attn_cfg.attn_max_step_samples,
        seed=seed,
        #log_every=max(1, attn_cfg.attn_steps // 10),
        log_every=50
    )
    print(f"\n[attn-pool]  Training attention head  "
          f"(steps={attn_cfg.attn_steps}  lr={attn_cfg.attn_lr}  device={_device}  "
          f"n_queries={attn_cfg.attn_num_queries}  n_heads={attn_cfg.attn_num_heads}  "
          f"n_train={len(train_labels)})")

    t_start = time.perf_counter()
    head, attn_history = train_attention_pooling_head(
        train_patches=_torch.from_numpy(train_patches),
        y_train=train_labels,
        val_patches=_torch.from_numpy(test_patches),
        y_val=test_labels,
        embed_dim=D,
        out_dim=None,
        num_queries=attn_cfg.attn_num_queries,
        num_heads=attn_cfg.attn_num_heads,
        device=_device,
        config=training_cfg,
    )
    total_time_s = time.perf_counter() - t_start

    best_val_acc_raw = max(attn_history["val_accuracy"]) if attn_history["val_accuracy"] else float("nan")
    time_to_best_s   = attn_history.get("time_to_best_s", float("nan"))
    best_val_step    = attn_history.get("best_val_step", 0)

    # Post-hoc evaluation: pool with best checkpoint → PCA (if used) → TabICLClassifier
    print(f"[attn-pool]  Evaluating best checkpoint (step {best_val_step}) with PCA={attn_cfg.tabicl_pca_dim} ...")
    train_pooled = _pool_with_head(head, _torch.from_numpy(train_patches), _device)
    test_pooled  = _pool_with_head(head, _torch.from_numpy(test_patches),  _device)
    if attn_cfg.tabicl_pca_dim is not None:
        n_comp_attn = min(attn_cfg.tabicl_pca_dim, len(train_labels), train_pooled.shape[1])
        attn_pca    = PCA(n_components=n_comp_attn, random_state=seed)
        train_pooled = attn_pca.fit_transform(train_pooled).astype(np.float32)
        test_pooled  = attn_pca.transform(test_pooled).astype(np.float32)
    test_acc, test_auroc = _compute_accuracy_from_features(
        train_pooled, train_labels, test_pooled, test_labels,
        n_estimators=attn_cfg.tabicl_n_estimators, seed=seed,
    )

    attn_result = {
        "test_acc":           round(test_acc, 6),
        "test_auroc":         round(test_auroc, 6) if not np.isnan(test_auroc) else None,
        "best_val_acc_raw":   round(best_val_acc_raw, 6),
        "best_val_step":      best_val_step,
        "time_to_best_s":     time_to_best_s,
        "total_train_time_s": round(total_time_s, 2),
    }
    print(f"[attn-pool]  test acc (PCA={attn_cfg.tabicl_pca_dim}): {test_acc:.4f}  auroc: {test_auroc:.4f}  "
          f"(best train val: {best_val_acc_raw:.4f}  "
          f"step {best_val_step}/{attn_cfg.attn_steps}  time_to_best={time_to_best_s:.1f}s)")

    record = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "args": _cfg_to_args_dict(cfg),
        "dataset": {"n_train": int(len(train_labels)), "n_test": int(len(test_labels)), "embed_dim": int(D)},
        "attn_pool": attn_result,
    }
    attn_path = output_dir / "attn_pool_results.json"
    with attn_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"[attn-pool]  Saved → {attn_path}")
    return attn_result


def _merge_attn_into_results(output_dir: Path) -> bool:
    """Patch baselines.attn_pool in results.json from attn_pool_results.json.

    Returns True if both files exist and the merge succeeded.
    """
    attn_path    = output_dir / "attn_pool_results.json"
    results_path = output_dir / "results.json"
    if not attn_path.exists() or not results_path.exists():
        return False
    with attn_path.open() as f:
        attn_data = json.load(f)
    with results_path.open() as f:
        results_data = json.load(f)
    results_data.setdefault("baselines", {})["attn_pool"] = attn_data.get("attn_pool")
    with results_path.open("w") as f:
        json.dump(results_data, f, indent=2)
    print(f"[merge] attn_pool → {results_path}")
    return True


def _make_stage_callback(
    cfg: ExperimentConfig,
    test_patches: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_image_paths: list,
    test_image_paths: list,
    train_sample_idx: np.ndarray,
    test_sample_idx: np.ndarray,
    idx_to_class: dict,
    open_image,
    class_prior: np.ndarray,
    output_dir: Path,
    all_results: list,
    cls_test_feats: Optional[np.ndarray] = None,
) -> tuple:
    """Build the per-stage callback for :class:`IterativePALPooler`.

    Returns ``(callback, last_stage_data)`` where ``last_stage_data`` is a
    dict that is populated with the final stage's data after ``fit`` completes.
    The caller can use it for the post-loop final-visualisation pass.

    The returned callback has the signature required by
    :meth:`IterativePALPooler.fit`::

        callback(stage_idx, stage, group_size,
                 pre_refine_support, pre_refine_pca, train_grouped)

    It runs (conditionally) pre- and post-refinement visualisations, evaluates
    test-set accuracy for the stage, appends a result tuple to ``all_results``,
    and saves the Ridge model to ``output_dir``.
    """
    last_stage_data: dict = {}

    def callback(stage_idx, stage, group_size, pre_refine_support, pre_refine_pca, train_grouped):
        group_side   = int(round(group_size ** 0.5))
        eff_patch_sz = cfg.refinement.patch_size * group_side
        tag          = f"iter_{stage_idx}_g{group_size}"

        test_grouped   = group_patches(test_patches, group_size)  # [N_test, P', D]
        ridge_model    = stage.ridge_model_
        feature_scaler = stage.feature_scaler_
        fit_time_s     = stage.fit_time_s_
        pool_time_s    = stage.pool_time_s_
        new_support    = stage._support_projected_
        new_pca        = stage._pca_

        # Pre-refinement visualisation (driven by the input support).
        if cfg.dataset.n_sample > 0 and not cfg.run.post_refinement_viz:
            split_configs_iter = [
                ("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
                ("test",  test_grouped,  test_labels,  test_image_paths,  test_sample_idx),
            ]
            if cfg.run.viz_loo_train:
                split_configs_iter.append(("train_loo", train_grouped, train_labels, train_image_paths, train_sample_idx))
            
            # Use stage.class_prior_ if available, falling back to class_prior.
            active_class_prior = getattr(stage, 'class_prior_', class_prior)

            iter_mean_probs = _run_visual_eval(
                tag, pre_refine_support, train_labels, split_configs_iter, idx_to_class,
                pca=pre_refine_pca, n_estimators=cfg.refinement.tabicl_n_estimators,
                patch_size=eff_patch_sz, seed=cfg.seed, output_dir=output_dir,
                temperature=stage.refinement_cfg.temperature,
                ridge_model=None, feature_scaler=None, open_image=open_image,
                class_prior=active_class_prior, weight_method=cfg.refinement.weight_method,
                show_pred_label=cfg.run.show_pred_label,
                show_minority_prob=cfg.run.show_minority_prob,
                show_per_class_probs=cfg.run.show_per_class_probs,
                use_attn_masking=cfg.refinement.use_attn_masking,
                binary_dist=cfg.refinement.binary_dist,
            )
        else:
            iter_mean_probs = {}

        # Save Ridge model to disk.
        ridge_path = output_dir / f"ridge_quality_model_{tag}.joblib"
        joblib.dump(ridge_model, ridge_path)
        print(f"[ridge] Model saved → {ridge_path}")

        # Post-refinement visualisation (driven by Ridge pooling weights).
        if cfg.dataset.n_sample > 0 and cfg.run.post_refinement_viz:
            split_configs_post = [
                ("train", train_grouped, train_labels, train_image_paths, train_sample_idx),
                ("test",  test_grouped,  test_labels,  test_image_paths,  test_sample_idx),
            ]
            if cfg.run.viz_loo_train:
                split_configs_post.append(("train_loo", train_grouped, train_labels, train_image_paths, train_sample_idx))
            
            # Use stage.class_prior_ if available, falling back to class_prior.
            active_class_prior = getattr(stage, 'class_prior_', class_prior)

            iter_mean_probs = _run_visual_eval(
                f"{tag}_post", pre_refine_support, train_labels, split_configs_post, idx_to_class,
                pca=pre_refine_pca, n_estimators=cfg.refinement.tabicl_n_estimators,
                patch_size=eff_patch_sz, seed=cfg.seed, output_dir=output_dir,
                temperature=stage.refinement_cfg.temperature,
                ridge_model=ridge_model, feature_scaler=feature_scaler, open_image=open_image,
                class_prior=active_class_prior, weight_method=cfg.refinement.weight_method,
                show_pred_label=cfg.run.show_pred_label,
                show_minority_prob=cfg.run.show_minority_prob,
                show_per_class_probs=cfg.run.show_per_class_probs,
                use_attn_masking=cfg.refinement.use_attn_masking,
                binary_dist=cfg.refinement.binary_dist,
            )

        # Pool test queries with Ridge and evaluate accuracy.
        # Append CLS token to test patches before pooling when append_cls is enabled.
        test_tokens = test_grouped
        if cls_test_feats is not None:
            test_tokens = np.concatenate(
                [test_grouped, cls_test_feats.astype(np.float32)[:, None, :]], axis=1
            )
        w_ridge       = _ridge_pool_weights(test_tokens, ridge_model, feature_scaler)
        test_repooled = (w_ridge[:, :, None] * test_tokens).sum(axis=1)  # [N_test, D]
        test_query    = (
            new_pca.transform(test_repooled).astype(np.float32)
            if new_pca is not None else test_repooled
        )
        t_eval_start = time.perf_counter()
        iter_acc, iter_auroc = _compute_accuracy_from_features(
            new_support, train_labels, test_query, test_labels,
            n_estimators=cfg.refinement.tabicl_n_estimators, seed=cfg.seed,
        )
        eval_time_s   = time.perf_counter() - t_eval_start
        refine_time_s = fit_time_s + pool_time_s
        print(f"[{tag}] test accuracy (quality-pooled queries): {iter_acc:.4f}  auroc: {iter_auroc:.4f}  "
              f"(fit {fit_time_s:.1f}s, pool {pool_time_s:.1f}s, eval {eval_time_s:.1f}s)")

        all_results.append((tag, iter_acc, iter_auroc, iter_mean_probs, refine_time_s, eval_time_s, fit_time_s, pool_time_s, getattr(stage, '_val_accuracy_', None)))

        # Persist state needed by the post-loop final-visualisation block.
        last_stage_data.update(
            tag=tag, eff_patch_sz=eff_patch_sz,
            train_grouped=train_grouped, test_grouped=test_grouped,
            ridge_model=ridge_model, feature_scaler=feature_scaler,
            pre_refine_support=pre_refine_support, pre_refine_pca=pre_refine_pca,
            temperature=stage.refinement_cfg.temperature,
            class_prior=getattr(stage, 'class_prior_', class_prior),
        )

    return callback, last_stage_data


def run_pal_experiment(
    cfg: ExperimentConfig,
) -> None:
    _set_global_seeds(cfg.seed)

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_start = time.perf_counter()
    run_ts = datetime.now(timezone.utc).isoformat()

    # --- Load pre-extracted patch (and CLS) features ---
    (train_patches, train_labels,
     test_patches,  test_labels,
     cls_train_feats, cls_test_feats,
     idx_to_class, train_sub_idx, test_sub_idx, extra_data) = _load_features(
        dataset_cfg=cfg.dataset,
        seed=cfg.seed,
    )

    bal_rng = np.random.RandomState(cfg.seed + 1)   # separate RNG so balancing doesn't shift other draws
    bal_train_keep_idx: Optional[np.ndarray] = None
    if cfg.dataset.balance_train:
        train_patches, train_labels, cls_train_feats, bal_train_keep_idx = _balance_classes(
            train_patches, train_labels, cls_train_feats, bal_rng
        )
    bal_test_keep_idx: Optional[np.ndarray] = None
    if cfg.dataset.balance_test:
        test_patches, test_labels, cls_test_feats, bal_test_keep_idx = _balance_classes(
            test_patches, test_labels, cls_test_feats, bal_rng
        )

    val_patches: Optional[np.ndarray] = None
    val_labels: Optional[np.ndarray] = None
    cls_val_feats: Optional[np.ndarray] = None

    # Train/val splitting is now handled internally by IterativePALPooler
    # (a fresh random split is drawn at each stage for both modalities).

    N_train, _P, D = train_patches.shape
    n_classes      = int(train_labels.max()) + 1

    # Empirical class prior (used by kl_div weight method and visualisation panels).
    _counts      = np.bincount(train_labels.astype(np.int64), minlength=n_classes)
    class_prior  = (_counts / _counts.sum()).astype(np.float32)

    _test_counts = np.bincount(test_labels.astype(np.int64), minlength=n_classes)
    print("[class balance]")
    print(f"  {'class':<8} {'train':>8} {'train %':>9} {'test':>8} {'test %':>9}")
    for _i, (_cls_name, _tr, _te) in enumerate(zip(idx_to_class, _counts, _test_counts)):
        print(f"  {str(_cls_name):<8} {_tr:>8d} {100*_tr/_counts.sum():>8.1f}% {_te:>8d} {100*_te/_test_counts.sum():>8.1f}%")
    print(f"  {'TOTAL':<8} {_counts.sum():>8d} {'100.0%':>9} {_test_counts.sum():>8d} {'100.0%':>9}")

    # -----------------------------------------------------------------------
    # TEXT MODALITY BRANCH — runs the full experiment and returns early.
    # -----------------------------------------------------------------------
    if cfg.dataset.modality == "text":
        text_cfg = cfg.refinement  # already a TextPALConfig from parse_args()

        train_token_ids      = extra_data["train_token_ids"]       # [N_train, T_max]
        train_attention_mask = extra_data["train_attention_mask"]  # [N_train, T_max]
        test_token_ids       = extra_data["test_token_ids"]
        test_attention_mask  = extra_data["test_attention_mask"]
        # val_token_ids / val_attention_mask are never set for text: splitting is
        # handled internally by IterativePALPooler (see TextPALConfig.train_val_fraction).

        # --- Masked mean-pool baseline (exclude [CLS] and padding) ---
        _cls_id  = text_cfg.cls_token_id   # 101
        _valid_tr = (train_token_ids != 0) & (train_token_ids != _cls_id)
        _valid_te = (test_token_ids  != 0) & (test_token_ids  != _cls_id)
        _cnts_tr  = _valid_tr.sum(axis=1, keepdims=True).clip(min=1)
        _cnts_te  = _valid_te.sum(axis=1, keepdims=True).clip(min=1)
        baseline_support_raw = (
            (train_patches.astype(np.float32) * _valid_tr[:, :, None]).sum(axis=1) / _cnts_tr
        )  # [N_train, D]
        test_mean_pool = (
            (test_patches.astype(np.float32) * _valid_te[:, :, None]).sum(axis=1) / _cnts_te
        )  # [N_test, D]

        pca: Optional[PCA] = None
        if text_cfg.tabicl_pca_dim is not None:
            n_comp = min(text_cfg.tabicl_pca_dim, N_train, D)
            pca    = PCA(n_components=n_comp, random_state=cfg.seed)
            baseline_support = pca.fit_transform(baseline_support_raw).astype(np.float32)
            test_mean_proj   = pca.transform(test_mean_pool).astype(np.float32)
            print(f"[info] PCA: {D}D → {n_comp}D")
        else:
            baseline_support = baseline_support_raw
            test_mean_proj   = test_mean_pool

        # CLS token baseline
        cls_acc:   Optional[float] = None
        cls_auroc: Optional[float] = None
        if cls_train_feats is not None and cls_test_feats is not None:
            cls_pca: Optional[PCA] = None
            if text_cfg.tabicl_pca_dim is not None:
                n_cls = min(text_cfg.tabicl_pca_dim, len(cls_train_feats), cls_train_feats.shape[1])
                cls_pca     = PCA(n_components=n_cls, random_state=cfg.seed)
                cls_support = cls_pca.fit_transform(cls_train_feats).astype(np.float32)
                cls_test_q  = cls_pca.transform(cls_test_feats).astype(np.float32)
            else:
                cls_support = cls_train_feats
                cls_test_q  = cls_test_feats
            cls_acc, cls_auroc = _compute_accuracy_from_features(
                cls_support, train_labels, cls_test_q, test_labels,
                n_estimators=text_cfg.tabicl_n_estimators, seed=cfg.seed,
            )

        # The baseline uses the full train set as support (train_patches is never split
        # for text; val splitting is handled internally by IterativePALPooler).
        baseline_acc, baseline_auroc = _compute_accuracy_from_features(
            baseline_support, train_labels, test_mean_proj, test_labels,
            n_estimators=text_cfg.tabicl_n_estimators, seed=cfg.seed,
        )
        if cls_acc is not None:
            print(f"\n[cls-token]  test accuracy: {cls_acc:.4f}  auroc: {cls_auroc:.4f}")
        else:
            print("\n[cls-token]  test accuracy: N/A")
        print(f"[mean-pool]  test accuracy: {baseline_acc:.4f}  auroc: {baseline_auroc:.4f}")

        # --- Attention pooling for text (if enabled) ---
        attn_result: Optional[dict] = None
        if cfg.attention.attn_pool:
            attn_result = _run_attn_only_text(
                train_tokens=train_patches,
                train_token_ids=train_token_ids,
                train_attention_mask=train_attention_mask,
                train_labels=train_labels,
                test_tokens=test_patches,
                test_token_ids=test_token_ids,
                test_attention_mask=test_attention_mask,
                test_labels=test_labels,
                D=D,
                output_dir=output_dir,
                attn_cfg=cfg.attention,
                seed=cfg.seed,
                cfg=cfg,
            )

        all_results: list = [
            ("baseline", baseline_acc, baseline_auroc, {}, 0.0, 0.0, 0.0, 0.0, None)
        ]

        # --- Text visualizations (when n_sample > 0) ---
        if cfg.dataset.n_sample > 0:
            # Load and subsample texts and token_to_word mappings (use local copies for visualization)
            vis_train_texts           = extra_data.get("train_texts", [f"text_{i}" for i in range(len(train_labels))])
            vis_test_texts            = extra_data.get("test_texts", [f"text_{i}" for i in range(len(test_labels))])
            vis_train_token_to_word   = extra_data.get("train_token_to_word", np.zeros((len(train_labels), train_token_ids.shape[1]), dtype=np.int32))
            vis_test_token_to_word    = extra_data.get("test_token_to_word", np.zeros((len(test_labels), test_token_ids.shape[1]), dtype=np.int32))

            # Create visualization copies (don't modify originals which are used for refinement)
            vis_train_token_ids = train_token_ids.copy()
            vis_test_token_ids  = test_token_ids.copy()
            vis_train_patches   = train_patches.copy()
            vis_test_patches    = test_patches.copy()

            # Apply same subsampling that was applied to patches
            if bal_train_keep_idx is not None:
                vis_train_texts = [vis_train_texts[i] for i in bal_train_keep_idx]
                vis_train_token_to_word = vis_train_token_to_word[bal_train_keep_idx]
                vis_train_token_ids = vis_train_token_ids[bal_train_keep_idx]
                vis_train_patches = vis_train_patches[bal_train_keep_idx]
            if bal_test_keep_idx is not None:
                vis_test_texts = [vis_test_texts[i] for i in bal_test_keep_idx]
                vis_test_token_to_word = vis_test_token_to_word[bal_test_keep_idx]
                vis_test_token_ids = vis_test_token_ids[bal_test_keep_idx]
                vis_test_patches = vis_test_patches[bal_test_keep_idx]
            if val_keep_idx is not None and train_keep_idx is not None:
                # Note: for visualization, we only use train/test, not val
                vis_train_texts = [vis_train_texts[i] for i in train_keep_idx]
                vis_train_token_to_word = vis_train_token_to_word[train_keep_idx]
                vis_train_token_ids = vis_train_token_ids[train_keep_idx]
                vis_train_patches = vis_train_patches[train_keep_idx]

            # Sample indices for visualization
            rng              = np.random.RandomState(cfg.seed)
            train_sample_idx = rng.choice(len(train_labels), size=min(cfg.dataset.n_sample, len(train_labels)), replace=False)
            test_sample_idx  = rng.choice(len(test_labels),  size=min(cfg.dataset.n_sample, len(test_labels)),  replace=False)

            split_configs_baseline = [
                ("train", vis_train_texts, vis_train_patches, train_labels, vis_train_token_ids, vis_train_token_to_word, train_sample_idx),
                ("test",  vis_test_texts,  vis_test_patches,  test_labels,  vis_test_token_ids,  vis_test_token_to_word,  test_sample_idx),
            ]

            _run_text_visual_eval(
                "baseline", baseline_support, train_labels, split_configs_baseline, idx_to_class,
                pca=pca, n_estimators=text_cfg.tabicl_n_estimators,
                seed=cfg.seed, output_dir=output_dir,
                temperature=1.0, ridge_model=None, feature_scaler=None,
                class_prior=class_prior, weight_method=text_cfg.weight_method,
                binary_dist=text_cfg.binary_dist,
            )


        # --- Iterative text PAL refinement ---
        _refine_dev = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if _refine_dev.startswith("cuda") and torch.cuda.is_available():
            tabicl_clf = TabICLGPUAdapter(n_estimators=text_cfg.tabicl_n_estimators, random_state=cfg.seed)
            _ridge_dev = _refine_dev if text_cfg.gpu_ridge else ""
            ridge_backend = f"RidgeGPU on {_refine_dev}" if text_cfg.gpu_ridge else "sklearn Ridge (CPU)"
            print(f"[refinement] Using TabICLGPUAdapter + {ridge_backend}")
        else:
            tabicl_clf = TabICLClassifier(n_estimators=text_cfg.tabicl_n_estimators, random_state=cfg.seed)
            _ridge_dev = ""
            print("[refinement] Using TabICLClassifier + sklearn Ridge (CPU)")

        pal_pooler = IterativePALPooler(
            tabicl=tabicl_clf, refinement_cfg=text_cfg, modality="text", seed=cfg.seed,
            gpu_ridge_device=_ridge_dev,
        )

        text_cls_train = cls_train_feats if text_cfg.append_cls else None
        text_cls_test  = cls_test_feats  if text_cfg.append_cls else None

        pal_pooler.fit(
            train_patches, train_labels,
            token_ids=train_token_ids,
            attention_mask=train_attention_mask,
            cls_tokens=text_cls_train,
        )

        # Evaluate each stage on the test set.
        # stage._support_projected_ already contains all N train samples (internal val
        # fold is pooled through Ridge and merged back by IterativePALPooler._fit_stages).
        for k, stage in enumerate(pal_pooler.stages_):
            tag = f"iter_{k}_mode_{stage.text_group_mode_}"
            t_eval = time.perf_counter()
            test_pooled = stage.transform(
                test_patches, test_token_ids, test_attention_mask, text_cls_test
            )
            test_query = (
                stage._pca_.transform(test_pooled).astype(np.float32)
                if stage._pca_ is not None else test_pooled
            )
            iter_acc, iter_auroc = _compute_accuracy_from_features(
                stage._support_projected_, stage.support_labels_, test_query, test_labels,
                n_estimators=text_cfg.tabicl_n_estimators, seed=cfg.seed,
            )
            eval_time_s = time.perf_counter() - t_eval
            print(f"[{tag}] test accuracy: {iter_acc:.4f}  auroc: {iter_auroc:.4f}  "
                  f"(fit {stage.fit_time_s_:.1f}s, pool {stage.pool_time_s_:.1f}s, "
                  f"eval {eval_time_s:.1f}s)")
            all_results.append((
                tag, iter_acc, iter_auroc, {},
                stage.fit_time_s_ + stage.pool_time_s_, eval_time_s,
                stage.fit_time_s_, stage.pool_time_s_,
                getattr(stage, '_val_accuracy_', None),
            ))

            # Per-stage visualisation (mirrors the image-modality stage callback).
            if cfg.dataset.n_sample > 0:
                active_class_prior = getattr(stage, 'class_prior_', class_prior)
                split_configs_stage = [
                    ("train", vis_train_texts, train_patches, train_labels,
                     vis_train_token_ids, vis_train_token_to_word, train_sample_idx),
                    ("test",  vis_test_texts,  test_patches,  test_labels,
                     vis_test_token_ids,  vis_test_token_to_word,  test_sample_idx),
                ]
                if not cfg.run.post_refinement_viz:
                    _run_text_visual_eval(
                        tag, stage._support_projected_, train_labels,
                        split_configs_stage, idx_to_class,
                        pca=stage._pca_, n_estimators=text_cfg.tabicl_n_estimators,
                        seed=cfg.seed, output_dir=output_dir,
                        temperature=stage.refinement_cfg.temperature,
                        ridge_model=None, feature_scaler=None,
                        class_prior=active_class_prior,
                        weight_method=text_cfg.weight_method,
                        binary_dist=text_cfg.binary_dist,
                    )
                else:
                    _run_text_visual_eval(
                        tag, stage._support_projected_, train_labels,
                        split_configs_stage, idx_to_class,
                        pca=stage._pca_, n_estimators=text_cfg.tabicl_n_estimators,
                        seed=cfg.seed, output_dir=output_dir,
                        temperature=stage.refinement_cfg.temperature,
                        ridge_model=stage.ridge_model_,
                        feature_scaler=stage.feature_scaler_,
                        class_prior=active_class_prior,
                        weight_method=text_cfg.weight_method,
                        binary_dist=text_cfg.binary_dist,
                    )

        # Final post-stage visualisation with trained pooler weights (last stage).
        # Runs only when --post-refinement-viz is off, mirroring the image modality.
        if cfg.dataset.n_sample > 0 and not cfg.run.post_refinement_viz and pal_pooler.stages_:
            last_stage = pal_pooler.stages_[-1]
            last_tag   = f"iter_{len(pal_pooler.stages_) - 1}_mode_{last_stage.text_group_mode_}"
            active_class_prior = getattr(last_stage, 'class_prior_', class_prior)
            split_configs_final = [
                ("train", vis_train_texts, train_patches, train_labels,
                 vis_train_token_ids, vis_train_token_to_word, train_sample_idx),
                ("test",  vis_test_texts,  test_patches,  test_labels,
                 vis_test_token_ids,  vis_test_token_to_word,  test_sample_idx),
            ]
            _run_text_visual_eval(
                f"{last_tag}_post", last_stage._support_projected_, train_labels,
                split_configs_final, idx_to_class,
                pca=last_stage._pca_, n_estimators=text_cfg.tabicl_n_estimators,
                seed=cfg.seed, output_dir=output_dir,
                temperature=last_stage.refinement_cfg.temperature,
                ridge_model=last_stage.ridge_model_,
                feature_scaler=last_stage.feature_scaler_,
                class_prior=active_class_prior,
                weight_method=text_cfg.weight_method,
                binary_dist=text_cfg.binary_dist,
            )

        total_time_s = time.perf_counter() - experiment_start

        # Summary table
        col_w = max(len(r[0]) for r in all_results) + 2
        _sep_w = col_w + 74
        print("\n" + "=" * _sep_w)
        print("ITERATIVE TEXT REFINEMENT SUMMARY")
        print("=" * _sep_w)
        print(f"  {'Stage':<{col_w}}  {'Test Acc':>10}  {'AUROC':>8}  {'Δ Acc':>8}  "
              f"{'Val Acc':>8}  {'Fit(s)':>8}  {'Pool(s)':>8}  {'Eval(s)':>8}")
        print("-" * _sep_w)
        _text_val_accs = [r[8] for r in all_results if r[8] is not None]
        _best_text_val = max(_text_val_accs) if _text_val_accs else None
        for stage_name, acc, auroc, _, refine_s, eval_s, fit_s, pool_s, _val in all_results:
            delta_str = "" if stage_name == "baseline" else f"{acc - baseline_acc:+.4f}"
            fit_str   = "-" if stage_name == "baseline" else f"{fit_s:.1f}"
            pool_str  = "-" if stage_name == "baseline" else f"{pool_s:.1f}"
            eval_str  = "-" if stage_name == "baseline" else f"{eval_s:.1f}"
            auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "  N/A"
            val_str   = f"{_val:.4f}" if _val is not None else "-"
            row = (f"  {stage_name:<{col_w}}  {acc:>10.4f}  {auroc_str:>8}  {delta_str:>8}"
                   f"  {val_str:>8}  {fit_str:>8}  {pool_str:>8}  {eval_str:>8}")
            if _best_text_val is not None and _val == _best_text_val:
                print(f"\033[1;32m{row}\033[0m")
            else:
                print(row)
        print("=" * _sep_w)
        print(f"  Total wall time: {total_time_s:.1f}s")

        _save_results(
            output_dir=output_dir, run_ts=run_ts,
            total_time_s=total_time_s,
            train_patches=train_patches, test_labels=test_labels, D=D,
            n_classes=n_classes, pca=pca,
            cls_acc=cls_acc, cls_auroc=cls_auroc,
            baseline_acc=baseline_acc, baseline_auroc=baseline_auroc,
            all_results=all_results, cfg=cfg, attn_result=attn_result,
        )
        return
    # -----------------------------------------------------------------------
    # END TEXT BRANCH
    # -----------------------------------------------------------------------

    # --- Resolve absence-of-evidence class ---
    aoe_mask: Optional[np.ndarray] = None
    val_aoe_mask: Optional[np.ndarray] = None
    if cfg.refinement.aoe_class is not None:
        class_to_idx = {v: k for k, v in idx_to_class.items()}
        try:
            aoe_class_idx = int(cfg.refinement.aoe_class)
        except (ValueError, TypeError):
            aoe_class_idx = None
        if aoe_class_idx is not None:
            if aoe_class_idx not in idx_to_class:
                raise ValueError(f"aoe_class index {aoe_class_idx} not in [0, {n_classes - 1}]")
        else:
            name = str(cfg.refinement.aoe_class)
            if name not in class_to_idx:
                raise ValueError(
                    f"aoe_class '{name}' not found in class names. "
                    f"Available: {sorted(class_to_idx)}"
                )
            aoe_class_idx = class_to_idx[name]
        aoe_class_name = idx_to_class[aoe_class_idx]
        aoe_mask = (train_labels == aoe_class_idx)
        if val_labels is not None:
            val_aoe_mask = (val_labels == aoe_class_idx)
        print(f"[aoe] Absence-of-evidence class: '{aoe_class_name}' (index {aoe_class_idx}), "
              f"{int(aoe_mask.sum())} training images excluded from Ridge fitting")

    # --- Attn-pool-only fast path: skip all feature refinement stages ---
    if cfg.attention.attn_pool_only:
        _run_attn_only(
            train_patches=train_patches, train_labels=train_labels,
            test_patches=test_patches,   test_labels=test_labels,
            D=D, output_dir=output_dir, attn_cfg=cfg.attention,
            seed=cfg.seed, cfg=cfg,
        )
        _merge_attn_into_results(output_dir)
        return

    n_stages       = len(cfg.refinement.patch_group_sizes)

    # Normalise temperature / ridge_alpha → one value per stage.
    # Scalar or single-element list → broadcast to all stages.
    # Multi-element list → must match n_stages exactly.
    def _broadcast(val, label: str) -> list:
        if isinstance(val, (int, float)):
            return [float(val)] * n_stages
        vals = list(val)
        if len(vals) == 1:
            return vals * n_stages
        if len(vals) != n_stages:
            raise ValueError(
                f"{label}: {len(vals)} value(s) given for {n_stages} stage(s) in "
                f"--patch-group-sizes; pass a single value (broadcast to all stages) "
                f"or exactly {n_stages} value(s)."
            )
        return vals

    temperatures = _broadcast(cfg.refinement.temperature,  "--temperature")
    ridge_alphas = _broadcast(cfg.refinement.ridge_alpha,  "--ridge-alpha")

    # --- Baseline support: mean-pool original patches → optional PCA ---
    # Cast to float32 before mean to avoid float16 accumulation errors.
    # When append_cls is enabled the CLS token is treated as one extra patch
    # in the mean pool so the baseline is comparable to the PAL stages.
    if cfg.refinement.append_cls and cls_train_feats is not None:
        baseline_support_raw = np.concatenate(
            [train_patches.astype(np.float32), cls_train_feats[:, None, :]], axis=1
        ).mean(axis=1)
    else:
        baseline_support_raw = train_patches.astype(np.float32).mean(axis=1)   # [N_train, D]
    pca: Optional[PCA] = None
    if cfg.refinement.tabicl_pca_dim is not None:
        n_comp = min(cfg.refinement.tabicl_pca_dim, N_train, D)
        pca    = PCA(n_components=n_comp, random_state=cfg.seed)
        baseline_support = pca.fit_transform(baseline_support_raw).astype(np.float32)
        print(f"[info] PCA: {D}D → {n_comp}D")
    else:
        baseline_support = baseline_support_raw

    # --- CLS token baseline ---
    cls_acc:   Optional[float] = None
    cls_auroc: Optional[float] = None
    if cls_train_feats is not None and cls_test_feats is not None:
        cls_pca: Optional[PCA] = None
        if cfg.refinement.tabicl_pca_dim is not None:
            n_comp_cls  = min(cfg.refinement.tabicl_pca_dim, len(cls_train_feats), cls_train_feats.shape[1])
            cls_pca     = PCA(n_components=n_comp_cls, random_state=cfg.seed)
            cls_support = cls_pca.fit_transform(cls_train_feats).astype(np.float32)
            cls_test_q  = cls_pca.transform(cls_test_feats).astype(np.float32)
        else:
            cls_support = cls_train_feats
            cls_test_q  = cls_test_feats
        cls_acc, cls_auroc = _compute_accuracy_from_features(
            cls_support, train_labels, cls_test_q, test_labels,
            n_estimators=cfg.refinement.tabicl_n_estimators, seed=cfg.seed,
        )

    # --- Image paths + opener for visualisation (only loaded when needed) ---
    train_image_paths: list = []
    test_image_paths:  list = []
    val_image_paths:   list = []
    open_image: Optional[Callable] = None
    if cfg.dataset.n_sample > 0:
        if cfg.dataset.dataset == "butterfly":
            train_image_paths, _, idx_to_class = _get_image_paths(cfg.dataset.dataset_path, split="train", seed=cfg.seed)
            test_image_paths,  _, _            = _get_image_paths(cfg.dataset.dataset_path, split="test",  seed=cfg.seed)
        elif cfg.dataset.dataset == "rsna":
            train_image_paths, _, _ = _get_rsna_image_paths(cfg.dataset.dataset_path, cfg.dataset.features_dir, split="train", backbone=cfg.dataset.backbone)
            test_image_paths,  _, _ = _get_rsna_image_paths(cfg.dataset.dataset_path, cfg.dataset.features_dir, split="test",  backbone=cfg.dataset.backbone)
            open_image = _dicom_to_pil
        elif cfg.dataset.dataset == "petfinder":
            train_image_paths, test_image_paths = _get_petfinder_image_paths(cfg.dataset.dataset_path)
        elif cfg.dataset.dataset == "dvm":
            train_image_paths, test_image_paths = _get_dvm_image_paths(cfg.dataset.dataset_path)
        elif cfg.dataset.dataset == "pad-ufes":
            train_image_paths, test_image_paths = _get_pad_ufes_image_paths(cfg.dataset.dataset_path, seed=cfg.seed)
        elif cfg.dataset.dataset in ("cbis-ddsm-mass", "cbis-ddsm-calc"):
            kind = "mass" if cfg.dataset.dataset == "cbis-ddsm-mass" else "calc"
            train_image_paths, test_image_paths = _get_cbis_ddsm_image_paths(cfg.dataset.dataset_path, kind=kind)

        # Keep train_image_paths aligned with train_patches by applying the same
        # index selections that _load_features and _balance_classes applied.
        if train_sub_idx is not None:
            train_image_paths = [train_image_paths[i] for i in train_sub_idx]
        if bal_train_keep_idx is not None:
            train_image_paths = [train_image_paths[i] for i in bal_train_keep_idx]

        if test_sub_idx is not None:
            test_image_paths = [test_image_paths[i] for i in test_sub_idx]
        if bal_test_keep_idx is not None:
            test_image_paths = [test_image_paths[i] for i in bal_test_keep_idx]

    rng              = np.random.RandomState(cfg.seed)
    train_sample_idx = rng.choice(len(train_labels), size=min(cfg.dataset.n_sample, len(train_labels)), replace=False)
    test_sample_idx  = rng.choice(len(test_labels),  size=min(cfg.dataset.n_sample, len(test_labels)),  replace=False)

    # --- Baseline: accuracy + visual eval at original patch resolution ---
    baseline_acc, baseline_auroc = _compute_accuracy(
        baseline_support, train_labels, test_patches, test_labels,
        pca=pca, n_estimators=cfg.refinement.tabicl_n_estimators, seed=cfg.seed,
    )
    if cls_acc is not None:
        print(f"\n[cls-token]  test accuracy: {cls_acc:.4f}  auroc: {cls_auroc:.4f}")
    else:
        print("\n[cls-token]  test accuracy: N/A (files not found)")
    print(f"[mean-pool]  test accuracy: {baseline_acc:.4f}  auroc: {baseline_auroc:.4f}")

    # --- Attention pooling upper-bound baseline ---
    attn_result: Optional[dict] = None
    if cfg.attention.attn_pool:
        attn_result = _run_attn_only(
            train_patches=train_patches, train_labels=train_labels,
            test_patches=test_patches,   test_labels=test_labels,
            D=D, output_dir=output_dir, attn_cfg=cfg.attention,
            seed=cfg.seed, cfg=cfg,
        )

    if cfg.dataset.n_sample > 0 and not cfg.run.post_refinement_viz:
        split_configs_orig = [
            ("train", train_patches, train_labels, train_image_paths, train_sample_idx),
            ("test",  test_patches,  test_labels,  test_image_paths,  test_sample_idx),
        ]
        if cfg.run.viz_loo_train:
            split_configs_orig.append(("train_loo", train_patches, train_labels, train_image_paths, train_sample_idx))
        baseline_mean_probs = _run_visual_eval(
            "baseline", baseline_support, train_labels, split_configs_orig, idx_to_class,
            pca=pca, n_estimators=cfg.refinement.tabicl_n_estimators, patch_size=cfg.refinement.patch_size,
            seed=cfg.seed, output_dir=output_dir,
            temperature=temperatures[0],
            ridge_model=None, feature_scaler=None, open_image=open_image,
            class_prior=class_prior, weight_method=cfg.refinement.weight_method,
            show_pred_label=cfg.run.show_pred_label,
            show_minority_prob=cfg.run.show_minority_prob,
            show_per_class_probs=cfg.run.show_per_class_probs,
            use_attn_masking=cfg.refinement.use_attn_masking,
            binary_dist=cfg.refinement.binary_dist,
        )
    else:
        baseline_mean_probs = {}

    # ---------------------------------------------------------------------------
    # Iterative multi-scale refinement
    # ---------------------------------------------------------------------------
    # Each stage groups the original DINO patches at a given resolution, visualises
    # patch quality scores under the *current* (pre-refinement) support, refines the
    # support via quality-weighted pooling, then evaluates accuracy using the same
    # clf and pooling that drove the refinement (ensuring query pooling matches
    # how training embeddings were constructed).
    # ---------------------------------------------------------------------------

    all_results: list[tuple[str, float, float, dict, float, float]] = [
        ("baseline", baseline_acc, baseline_auroc, baseline_mean_probs, 0.0, 0.0, 0.0, 0.0, None)
    ]

    stage_callback, _last_stage_data = _make_stage_callback(
        cfg=cfg,
        test_patches=test_patches,
        train_labels=train_labels,
        test_labels=test_labels,
        cls_test_feats=cls_test_feats if cfg.refinement.append_cls else None,
        train_image_paths=train_image_paths,
        test_image_paths=test_image_paths,
        train_sample_idx=train_sample_idx,
        test_sample_idx=test_sample_idx,
        idx_to_class=idx_to_class,
        open_image=open_image,
        class_prior=class_prior,
        output_dir=output_dir,
        all_results=all_results,
    )

    _refine_dev = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if _refine_dev.startswith("cuda") and torch.cuda.is_available():
        tabicl_clf = TabICLGPUAdapter(n_estimators=cfg.refinement.tabicl_n_estimators, random_state=cfg.seed)
        _ridge_dev = _refine_dev if cfg.refinement.gpu_ridge else ""
        ridge_backend = f"RidgeGPU on {_refine_dev}" if cfg.refinement.gpu_ridge else "sklearn Ridge (CPU)"
        print(f"[refinement] Using TabICLGPUAdapter + {ridge_backend}")
    else:
        tabicl_clf = TabICLClassifier(n_estimators=cfg.refinement.tabicl_n_estimators, random_state=cfg.seed)
        _ridge_dev = ""
        print("[refinement] Using TabICLClassifier + sklearn Ridge (CPU)")
    pal_pooler = IterativePALPooler(
        tabicl=tabicl_clf, refinement_cfg=cfg.refinement, seed=cfg.seed,
        gpu_ridge_device=_ridge_dev,
    )
    train_cls = cls_train_feats if cfg.refinement.append_cls else None

    pal_pooler.fit(train_patches, train_labels, cls_tokens=train_cls, stage_callback=stage_callback)

    # -- Final post-all-refinement visualisation (only when --post-refinement-viz is off) --
    # Produces Ridge-weight figures for the last refinement stage, giving you the quality
    # heatmaps even when per-stage post-refinement viz was skipped.

    if cfg.dataset.n_sample > 0 and not cfg.run.post_refinement_viz and _last_stage_data:
        split_configs_final = [
            ("train", _last_stage_data["train_grouped"], train_labels, train_image_paths, train_sample_idx),
            ("test",  _last_stage_data["test_grouped"],  test_labels,  test_image_paths,  test_sample_idx),
        ]
        if cfg.run.viz_loo_train:
            split_configs_final.append(("train_loo", _last_stage_data["train_grouped"], train_labels, train_image_paths, train_sample_idx))
        _run_visual_eval(
            f"{_last_stage_data['tag']}_post", _last_stage_data["pre_refine_support"], train_labels,
            split_configs_final, idx_to_class,
            pca=_last_stage_data["pre_refine_pca"], n_estimators=cfg.refinement.tabicl_n_estimators,
            patch_size=_last_stage_data["eff_patch_sz"],
            seed=cfg.seed, output_dir=output_dir,
            temperature=_last_stage_data["temperature"],
            ridge_model=_last_stage_data["ridge_model"], feature_scaler=_last_stage_data["feature_scaler"],
            open_image=open_image, class_prior=_last_stage_data.get("class_prior", class_prior), weight_method=cfg.refinement.weight_method,
            show_pred_label=cfg.run.show_pred_label,
            show_minority_prob=cfg.run.show_minority_prob,
            show_per_class_probs=cfg.run.show_per_class_probs,
            use_attn_masking=cfg.refinement.use_attn_masking,
            binary_dist=cfg.refinement.binary_dist,
        )

    total_time_s = time.perf_counter() - experiment_start

    # --- Summary table ---
    col_w = max(len(r[0]) for r in all_results) + 2
    _sep_w = col_w + 74
    print("\n" + "=" * _sep_w)
    print("ITERATIVE REFINEMENT SUMMARY")
    print("=" * _sep_w)
    print(f"  {'Stage':<{col_w}}  {'Test Acc':>10}  {'AUROC':>8}  {'Δ Acc':>8}  "
          f"{'Val Acc':>8}  {'Fit(s)':>8}  {'Pool(s)':>8}  {'Eval(s)':>8}")
    print("-" * _sep_w)
    _img_val_accs = [r[8] for r in all_results if r[8] is not None]
    _best_img_val = max(_img_val_accs) if _img_val_accs else None
    for stage_name, acc, auroc, mean_probs, refine_s, eval_s, fit_s, pool_s, _val in all_results:
        delta_str = "" if stage_name == "baseline" else f"{acc - baseline_acc:+.4f}"
        fit_str   = "-" if stage_name == "baseline" else f"{fit_s:.1f}"
        pool_str  = "-" if stage_name == "baseline" else f"{pool_s:.1f}"
        eval_str  = "-" if stage_name == "baseline" else f"{eval_s:.1f}"
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "  N/A"
        val_str   = f"{_val:.4f}" if _val is not None else "-"
        row = (
            f"  {stage_name:<{col_w}}  {acc:>10.4f}  {auroc_str:>8}  {delta_str:>8}"
            f"  {val_str:>8}  {fit_str:>8}  {pool_str:>8}  {eval_str:>8}"
        )
        if _best_img_val is not None and _val == _best_img_val:
            print(f"\033[1;32m{row}\033[0m")
        else:
            print(row)
    print("=" * _sep_w)
    print(f"  Total wall time: {total_time_s:.1f}s")

    _save_results(
        output_dir=output_dir, run_ts=run_ts,
        total_time_s=total_time_s,
        train_patches=train_patches, test_labels=test_labels, D=D,
        n_classes=n_classes, pca=pca,
        cls_acc=cls_acc, cls_auroc=cls_auroc,
        baseline_acc=baseline_acc, baseline_auroc=baseline_auroc,
        all_results=all_results,
        cfg=cfg,
        attn_result=attn_result,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_n_train_sweep(
    cfg: ExperimentConfig,
    #n_train_values: list[int],
    #base_output_dir: Path,
    #**kwargs,
) -> None:
    """Run patch-quality eval for each value in n_train_values.

    Each run is saved under ``base_output_dir/n_train_{value}/``.
    A consolidated ``sweep_results.json`` is written to ``base_output_dir``.

    Args:
        n_train_values: Ordered list of support-set sizes to evaluate.
        base_output_dir: Root directory; per-run sub-dirs are created here.
        **kwargs: Forwarded verbatim to ``run_pal_experiment`` (except
            ``n_train`` and ``output_dir``, which are set per-run).
    """

    base_output_dir = Path(cfg.run.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    sweep_start = time.perf_counter()
    sweep_ts    = datetime.now(timezone.utc).isoformat()

    sweep_runs: list[dict] = []

    for n_train in cfg.run.n_train_sweep:
        run_dir = base_output_dir / f"n_train_{n_train}"
        print(f"\n{'='*60}")
        print(f"  SWEEP  n_train={n_train}  →  {run_dir}")
        print(f"{'='*60}")

        # Override n_train / output_dir; record in cli_args for provenance
        run_cfg = copy.deepcopy(cfg)
        run_cfg.dataset.n_train = n_train
        run_cfg.run.output_dir = str(run_dir)

        run_pal_experiment(
            run_cfg
        )

        # Read back results.json (may have been created or merged into)
        results_path  = run_dir / "results.json"
        attn_path     = run_dir / "attn_pool_results.json"
        run_summary: dict = {"n_train": n_train, "output_dir": str(run_dir)}
        if results_path.exists():
            with results_path.open() as f:
                run_data = json.load(f)
            run_summary["baselines"]    = run_data.get("baselines", {})
            run_summary["stages"]       = run_data.get("stages", [])
            run_summary["total_time_s"] = run_data.get("total_time_s")
        elif attn_path.exists():
            # attn-only run with no prior results.json
            with attn_path.open() as f:
                attn_data = json.load(f)
            run_summary["baselines"] = {"attn_pool": attn_data.get("attn_pool")}
        sweep_runs.append(run_summary)

    sweep_total = time.perf_counter() - sweep_start

    sweep_record = {
        "sweep_timestamp": sweep_ts,
        "n_train_values":  cfg.run.n_train_sweep,
        "total_sweep_time_s": round(sweep_total, 2),
        "runs": sweep_runs,
    }
    sweep_path = base_output_dir / "sweep_results.json"
    with sweep_path.open("w") as f:
        json.dump(sweep_record, f, indent=2)
    print(f"\n[sweep] Done — {len(cfg.run.n_train_sweep)} runs in {sweep_total:.1f}s")
    print(f"[sweep] Results → {sweep_path}")

def run_seed_sweep(cfg: ExperimentConfig) -> None:
    """Run the experiment for each seed in ``cfg.run.seeds``, saving results continuously.

    After each seed completes the consolidated ``seed_sweep_results.json`` is written
    to ``cfg.run.output_dir`` so partial results survive an interrupted run.  On the
    next invocation with the same output directory any already-completed seeds are
    skipped automatically.

    If ``cfg.run.n_train_sweep`` is also set, each seed runs a full n_train sweep
    (one sub-directory per training-set size, nested under the seed directory).
    """
    base_output_dir = Path(cfg.run.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path  = base_output_dir / "seed_sweep_results.json"
    sweep_ts    = datetime.now(timezone.utc).isoformat()
    sweep_start = time.perf_counter()

    # Resume support: load any already-finished seeds from a prior run.
    seed_results: list[dict] = []
    if sweep_path.exists():
        with sweep_path.open() as f:
            prior = json.load(f)
        seed_results = prior.get("runs", [])
        completed_seeds = {r["seed"] for r in seed_results}
        print(f"[seed-sweep] Resuming — {len(completed_seeds)} seed(s) already done: "
              f"{sorted(completed_seeds)}")
    else:
        completed_seeds: set = set()

    for seed in cfg.run.seeds:
        if seed in completed_seeds:
            print(f"\n[seed-sweep] Skipping seed={seed} (already completed)")
            continue

        seed_dir = base_output_dir / f"seed_{seed}"
        print(f"\n{'='*60}")
        print(f"  SEED SWEEP  seed={seed}  →  {seed_dir}")
        print(f"{'='*60}")

        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = seed
        run_cfg.run.output_dir = seed_dir

        t0 = time.perf_counter()
        if cfg.run.n_train_sweep is not None:
            run_n_train_sweep(run_cfg)
        else:
            run_pal_experiment(run_cfg)
        elapsed = time.perf_counter() - t0

        # Collect a compact per-seed summary from the saved result files.
        run_summary: dict = {
            "seed":       seed,
            "output_dir": str(seed_dir),
            "time_s":     round(elapsed, 2),
        }
        if cfg.run.n_train_sweep is not None:
            ntrain_sweep_path = seed_dir / "sweep_results.json"
            if ntrain_sweep_path.exists():
                with ntrain_sweep_path.open() as f:
                    run_summary["n_train_sweep"] = json.load(f).get("runs", [])
        else:
            results_path = seed_dir / "results.json"
            if results_path.exists():
                with results_path.open() as f:
                    data = json.load(f)
                run_summary["baselines"] = data.get("baselines", {})
                run_summary["stages"]    = data.get("stages", [])

        seed_results.append(run_summary)

        # Write continuously so a crash after seed K leaves K seeds intact.
        sweep_record = {
            "sweep_timestamp":    sweep_ts,
            "seeds":              cfg.run.seeds,
            "total_sweep_time_s": round(time.perf_counter() - sweep_start, 2),
            "runs":               seed_results,
        }
        with sweep_path.open("w") as f:
            json.dump(sweep_record, f, indent=2)
        print(f"[seed-sweep] seed={seed} done ({elapsed:.1f}s) — "
              f"{len(seed_results)}/{len(cfg.run.seeds)} seeds saved → {sweep_path}")

    total = time.perf_counter() - sweep_start
    print(f"\n[seed-sweep] All {len(cfg.run.seeds)} seed(s) finished in {total:.1f}s")
    print(f"[seed-sweep] Results → {sweep_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    cfg = parse_args()

    if cfg.run.n_train_sweep is not None and cfg.dataset.n_train is not None:
        raise SystemExit("error: --n-train-sweep and --n-train are mutually exclusive")

    if cfg.run.seeds is not None:
        run_seed_sweep(cfg)
    elif cfg.run.n_train_sweep is not None:
        run_n_train_sweep(cfg=cfg)
    else:
        run_pal_experiment(cfg)
