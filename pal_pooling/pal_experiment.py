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
from pal_pooling.config import DatasetConfig, RefinementConfig, AttentionPoolConfig, RunConfig, ExperimentConfig, parse_args
from pal_pooling.data_loading import (
    _get_image_paths,
    _dicom_to_pil,
    _get_petfinder_image_paths,
    _get_dvm_image_paths,
    _get_pad_ufes_image_paths,
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
# Main experiment
# ---------------------------------------------------------------------------

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
                "delta_acc":       round(float(acc - baseline_acc), 6),
                "delta_auroc":     _fmt(auroc - baseline_auroc) if not np.isnan(auroc) and not np.isnan(baseline_auroc) else None,
                "mean_prob_train": round(float(mean_probs.get("train", float("nan"))), 6),
                "mean_prob_test":  round(float(mean_probs.get("test",  float("nan"))), 6),
                "fit_time_s":      round(fit_s,    2),
                "pool_time_s":     round(pool_s,   2),
                "refine_time_s":   round(refine_s, 2),
                "eval_time_s":     round(eval_s,   2),
            }
            for stage_name, acc, auroc, mean_probs, refine_s, eval_s, fit_s, pool_s in all_results
        ],
    }
    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[results] Saved → {results_path}")


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
    val_patches: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    val_image_paths: Optional[list],
    val_sample_idx: Optional[np.ndarray],
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
        val_grouped    = group_patches(val_patches, group_size) if val_patches is not None else None
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
            if val_grouped is not None:
                split_configs_iter.append(("val", val_grouped, val_labels, val_image_paths, val_sample_idx))
            if cfg.run.viz_loo_train:
                split_configs_iter.append(("train_loo", train_grouped, train_labels, train_image_paths, train_sample_idx))
            iter_mean_probs = _run_visual_eval(
                tag, pre_refine_support, train_labels, split_configs_iter, idx_to_class,
                pca=pre_refine_pca, n_estimators=cfg.refinement.tabicl_n_estimators,
                patch_size=eff_patch_sz, seed=cfg.seed, output_dir=output_dir,
                temperature=stage.refinement_cfg.temperature,
                ridge_model=None, feature_scaler=None, open_image=open_image,
                class_prior=class_prior, weight_method=cfg.refinement.weight_method,
                show_pred_label=cfg.run.show_pred_label,
                show_minority_prob=cfg.run.show_minority_prob,
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
            if val_grouped is not None:
                split_configs_post.append(("val", val_grouped, val_labels, val_image_paths, val_sample_idx))
            if cfg.run.viz_loo_train:
                split_configs_post.append(("train_loo", train_grouped, train_labels, train_image_paths, train_sample_idx))
            iter_mean_probs = _run_visual_eval(
                f"{tag}_post", pre_refine_support, train_labels, split_configs_post, idx_to_class,
                pca=pre_refine_pca, n_estimators=cfg.refinement.tabicl_n_estimators,
                patch_size=eff_patch_sz, seed=cfg.seed, output_dir=output_dir,
                temperature=stage.refinement_cfg.temperature,
                ridge_model=ridge_model, feature_scaler=feature_scaler, open_image=open_image,
                class_prior=class_prior, weight_method=cfg.refinement.weight_method,
                show_pred_label=cfg.run.show_pred_label,
                show_minority_prob=cfg.run.show_minority_prob,
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

        all_results.append((tag, iter_acc, iter_auroc, iter_mean_probs, refine_time_s, eval_time_s, fit_time_s, pool_time_s))

        # Persist state needed by the post-loop final-visualisation block.
        last_stage_data.update(
            tag=tag, eff_patch_sz=eff_patch_sz,
            train_grouped=train_grouped, test_grouped=test_grouped,
            val_grouped=val_grouped,
            ridge_model=ridge_model, feature_scaler=feature_scaler,
            pre_refine_support=pre_refine_support, pre_refine_pca=pre_refine_pca,
            temperature=stage.refinement_cfg.temperature,
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
     idx_to_class, train_sub_idx, test_sub_idx) = _load_features(
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
    val_keep_idx: Optional[np.ndarray] = None
    train_keep_idx: Optional[np.ndarray] = None
    
    if cfg.dataset.train_val_fraction is not None and cfg.dataset.train_val_fraction > 0.0:
        split_rng = np.random.RandomState(cfg.seed + 2)
        n_val_split = int(len(train_labels) * cfg.dataset.train_val_fraction)
        perm = split_rng.permutation(len(train_labels))
        val_keep_idx = perm[:n_val_split]
        train_keep_idx = perm[n_val_split:]
        
        val_patches = train_patches[val_keep_idx]
        val_labels = train_labels[val_keep_idx]
        if cls_train_feats is not None:
            cls_val_feats = cls_train_feats[val_keep_idx]
        
        train_patches = train_patches[train_keep_idx]
        train_labels = train_labels[train_keep_idx]
        if cls_train_feats is not None:
            cls_train_feats = cls_train_feats[train_keep_idx]
            
        print(f"[val_split] Held out {n_val_split} images for validation during refinement (fraction={cfg.dataset.train_val_fraction})")

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

        # Keep train_image_paths aligned with train_patches by applying the same
        # index selections that _load_features and _balance_classes applied.
        if train_sub_idx is not None:
            train_image_paths = [train_image_paths[i] for i in train_sub_idx]
        if bal_train_keep_idx is not None:
            train_image_paths = [train_image_paths[i] for i in bal_train_keep_idx]
        
        val_image_paths: list = []
        if val_keep_idx is not None and train_keep_idx is not None:
            val_image_paths = [train_image_paths[i] for i in val_keep_idx]
            train_image_paths = [train_image_paths[i] for i in train_keep_idx]
        
        if test_sub_idx is not None:
            test_image_paths = [test_image_paths[i] for i in test_sub_idx]
        if bal_test_keep_idx is not None:
            test_image_paths = [test_image_paths[i] for i in bal_test_keep_idx]

    rng              = np.random.RandomState(cfg.seed)
    train_sample_idx = rng.choice(len(train_labels), size=min(cfg.dataset.n_sample, len(train_labels)), replace=False)
    test_sample_idx  = rng.choice(len(test_labels),  size=min(cfg.dataset.n_sample, len(test_labels)),  replace=False)
    val_sample_idx   = None
    if val_labels is not None:
        val_sample_idx = rng.choice(len(val_labels), size=min(cfg.dataset.n_sample, len(val_labels)), replace=False)

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
        if val_patches is not None:
            split_configs_orig.append(("val", val_patches, val_labels, val_image_paths, val_sample_idx))
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
        )
    else:
        baseline_mean_probs = {}

    if not cfg.refinement.refine:
        _save_results(
            output_dir=output_dir, run_ts=run_ts,
            total_time_s=time.perf_counter() - experiment_start,
            train_patches=train_patches, test_labels=test_labels, D=D,
            n_classes=n_classes, pca=pca,
            cls_acc=cls_acc, cls_auroc=cls_auroc,
            baseline_acc=baseline_acc, baseline_auroc=baseline_auroc,
            all_results=[("baseline", baseline_acc, baseline_auroc, baseline_mean_probs, 0.0, 0.0, 0.0, 0.0)],
            cfg=cfg,
            attn_result=attn_result,
        )
        return

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
        ("baseline", baseline_acc, baseline_auroc, baseline_mean_probs, 0.0, 0.0, 0.0, 0.0)
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
        val_patches=val_patches,
        val_labels=val_labels,
        val_image_paths=val_image_paths,
        val_sample_idx=val_sample_idx,
        idx_to_class=idx_to_class,
        open_image=open_image,
        class_prior=class_prior,
        output_dir=output_dir,
        all_results=all_results,
    )

    _refine_dev = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if _refine_dev.startswith("cuda") and torch.cuda.is_available():
        tabicl_clf = TabICLGPUAdapter(n_estimators=cfg.refinement.tabicl_n_estimators, random_state=cfg.seed)
        print(f"[refinement] Using TabICLGPUAdapter + RidgeGPU on {_refine_dev}")
    else:
        tabicl_clf = TabICLClassifier(n_estimators=cfg.refinement.tabicl_n_estimators, random_state=cfg.seed)
        _refine_dev = ""   # signal to PALPooler/RidgeGPU to use CPU sklearn Ridge
        print("[refinement] Using TabICLClassifier + sklearn Ridge (CPU)")
    pal_pooler = IterativePALPooler(
        tabicl=tabicl_clf, refinement_cfg=cfg.refinement, seed=cfg.seed,
        gpu_ridge_device=_refine_dev,
    )
    train_cls = cls_train_feats if cfg.refinement.append_cls else None
    fit_kwargs = {}
    if val_patches is not None:
        fit_kwargs["val_patches"] = val_patches
        fit_kwargs["val_labels"] = val_labels
        if cfg.refinement.append_cls:
            fit_kwargs["val_cls_tokens"] = cls_val_feats

    pal_pooler.fit(train_patches, train_labels, cls_tokens=train_cls, stage_callback=stage_callback, **fit_kwargs)

    # -- Final post-all-refinement visualisation (only when --post-refinement-viz is off) --
    # Produces Ridge-weight figures for the last refinement stage, giving you the quality
    # heatmaps even when per-stage post-refinement viz was skipped.

    if cfg.dataset.n_sample > 0 and not cfg.run.post_refinement_viz and cfg.refinement.refine and _last_stage_data:
        split_configs_final = [
            ("train", _last_stage_data["train_grouped"], train_labels, train_image_paths, train_sample_idx),
            ("test",  _last_stage_data["test_grouped"],  test_labels,  test_image_paths,  test_sample_idx),
        ]
        if _last_stage_data.get("val_grouped") is not None:
            split_configs_final.append(("val", _last_stage_data["val_grouped"], val_labels, val_image_paths, val_sample_idx))
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
            open_image=open_image, class_prior=class_prior, weight_method=cfg.refinement.weight_method,
            show_pred_label=cfg.run.show_pred_label,
            show_minority_prob=cfg.run.show_minority_prob,
        )

    total_time_s = time.perf_counter() - experiment_start

    # --- Summary table ---
    col_w = max(len(r[0]) for r in all_results) + 2
    print("\n" + "=" * (col_w + 78))
    print("ITERATIVE REFINEMENT SUMMARY")
    print("=" * (col_w + 70))
    print(f"  {'Stage':<{col_w}}  {'Test Acc':>10}  {'AUROC':>8}  {'Δ Acc':>8}  "
          f"{'P(true)/train':>14}  {'P(true)/test':>13}  {'Fit(s)':>8}  {'Pool(s)':>8}  {'Eval(s)':>8}")
    print("-" * (col_w + 78))
    for stage_name, acc, auroc, mean_probs, refine_s, eval_s, fit_s, pool_s in all_results:
        delta_str = "" if stage_name == "baseline" else f"{acc - baseline_acc:+.4f}"
        fit_str   = "-" if stage_name == "baseline" else f"{fit_s:.1f}"
        pool_str  = "-" if stage_name == "baseline" else f"{pool_s:.1f}"
        eval_str  = "-" if stage_name == "baseline" else f"{eval_s:.1f}"
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "  N/A"
        print(
            f"  {stage_name:<{col_w}}  {acc:>10.4f}  {auroc_str:>8}  {delta_str:>8}"
            f"  {mean_probs.get('train', float('nan')):>14.3f}"
            f"  {mean_probs.get('test', float('nan')):>13.3f}"
            f"  {fit_str:>8}  {pool_str:>8}  {eval_str:>8}"
        )
    print("=" * (col_w + 78))
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
