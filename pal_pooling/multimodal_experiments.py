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
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

from pal_pooling.config import PETFINDER_DATASET_PATH, RefinementConfig
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_petfinder(
    dataset_path: Path,
    n_train: Optional[int],
    seed: int,
) -> tuple[
    np.ndarray, np.ndarray,   # train_patches, train_labels
    np.ndarray, np.ndarray,   # test_patches, test_labels
    np.ndarray, np.ndarray,   # cls_train, cls_test
    np.ndarray, np.ndarray,   # tab_train, tab_test
    dict,                      # idx_to_class
]:
    """Load PetFinder patch, CLS, and tabular features.

    Merges train + val splits as the support set (50% / 10% / 40% PetFinder split).
    Returns arrays in [N, P, D], [N], [N, D], [N, F] shapes.
    """
    petfinder_dir = Path(dataset_path)
    if str(petfinder_dir) not in sys.path:
        sys.path.insert(0, str(petfinder_dir))

    from petfinder_dataset_with_dinov3 import load_petfinder_dataset  # type: ignore

    train_loader, val_loader, test_loader, metadata = load_petfinder_dataset(
        feature_source="dinov3_local",
        use_patches=True,
        use_images=True,
        num_workers=0,
    )

    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    test_ds  = test_loader.dataset

    # Merge train + val → support set.
    train_patches = np.concatenate([
        train_ds.patch_embeddings.float().numpy(),
        val_ds.patch_embeddings.float().numpy(),
    ], axis=0)
    train_labels = np.concatenate([
        train_ds.targets.numpy(),
        val_ds.targets.numpy(),
    ], axis=0).astype(np.int64)
    cls_train = np.concatenate([
        train_ds.image_embeddings.float().numpy(),
        val_ds.image_embeddings.float().numpy(),
    ], axis=0)
    tab_train = np.concatenate([
        train_ds.tabular.float().numpy(),
        val_ds.tabular.float().numpy(),
    ], axis=0)

    test_patches = test_ds.patch_embeddings.float().numpy()
    test_labels  = test_ds.targets.numpy().astype(np.int64)
    cls_test     = test_ds.image_embeddings.float().numpy()
    tab_test     = test_ds.tabular.float().numpy()

    target_encoder = metadata["target_encoder"]
    idx_to_class = {i: str(cls) for i, cls in enumerate(target_encoder.classes_)}

    # Optional n_train subsampling.
    if n_train is not None and n_train < len(train_labels):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_labels), size=n_train, replace=False)
        idx.sort()
        train_patches = train_patches[idx]
        train_labels  = train_labels[idx]
        cls_train     = cls_train[idx]
        tab_train     = tab_train[idx]

    print(
        f"[info] PetFinder (train+val): N={len(train_labels)}  "
        f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}  "
        f"tab_dim={tab_train.shape[1]}"
    )
    print(f"[info] PetFinder (test):  N={len(test_labels)}")
    return (
        train_patches, train_labels,
        test_patches,  test_labels,
        cls_train, cls_test,
        tab_train, tab_test,
        idx_to_class,
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_multimodal_experiment(args: argparse.Namespace) -> None:
    _set_global_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    # ── Load data ────────────────────────────────────────────────────────────
    (train_patches, train_labels,
     test_patches,  test_labels,
     cls_train, cls_test,
     tab_train, tab_test,
     idx_to_class) = _load_petfinder(
        dataset_path=args.dataset_path,
        n_train=args.n_train,
        seed=args.seed,
    )

    N_train, P, D = train_patches.shape
    n_classes = int(train_labels.max()) + 1
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
            seed=args.seed,
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

    # ── Baseline: mean-pool image only ───────────────────────────────────────
    print("\n--- mean_pool_img ---")
    mean_train_raw = train_patches.astype(np.float32).mean(axis=1)  # [N, D]
    mean_test_raw  = test_patches.astype(np.float32).mean(axis=1)
    mean_train, mean_test, mean_pca = _pca_project(mean_train_raw, mean_test_raw, pca_dim, args.seed)
    _eval("mean_pool_img", mean_train, mean_test)

    # ── Baseline: CLS token image only ──────────────────────────────────────
    print("\n--- cls_img ---")
    cls_train_proj, cls_test_proj, cls_pca = _pca_project(cls_train, cls_test, pca_dim, args.seed)
    _eval("cls_img", cls_train_proj, cls_test_proj)

    # ── Baseline: mean-pool image + tabular ──────────────────────────────────
    print("\n--- mean_pool_img+tab ---")
    _eval("mean_pool_img+tab",
          _concat_tabular(mean_train, tab_train),
          _concat_tabular(mean_test,  tab_test))

    # ── Baseline: CLS + tabular ──────────────────────────────────────────────
    print("\n--- cls_img+tab ---")
    _eval("cls_img+tab",
          _concat_tabular(cls_train_proj, tab_train),
          _concat_tabular(cls_test_proj,  tab_test))

    # ── PALPool: fit on image patches only ───────────────────────────────────
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
    )

    pooler = pooler_factory(refinement_cfg=refinement_cfg, seed=args.seed)
    t_fit = time.perf_counter()
    pooler.fit(train_patches, train_labels)
    fit_time_s = time.perf_counter() - t_fit
    print(f"[pal] Pooler fit in {fit_time_s:.1f}s")

    # Pool train and test patches → raw D-dimensional image embeddings.
    pal_train_raw = pooler.transform(train_patches)   # [N_train, D]
    pal_test_raw  = pooler.transform(test_patches)    # [N_test, D]

    # Project to PCA space (same pca_dim used internally by the pooler).
    # The pooler's internal _pca_ was fit on pooled train embeddings; reuse it
    # so image features are in the same space used by the pooler's TabICL scorer.
    final_stage = pooler.stages_[-1]
    pal_pca = final_stage._pca_

    if pal_pca is not None:
        pal_train_proj = pal_pca.transform(pal_train_raw).astype(np.float32)
        pal_test_proj  = pal_pca.transform(pal_test_raw).astype(np.float32)
    else:
        pal_train_proj = pal_train_raw.astype(np.float32)
        pal_test_proj  = pal_test_raw.astype(np.float32)

    # ── PALPool image only ───────────────────────────────────────────────────
    print("\n--- pal_img ---")
    _eval("pal_img", pal_train_proj, pal_test_proj)

    # ── PALPool image + tabular ──────────────────────────────────────────────
    print("\n--- pal_img+tab ---")
    _eval("pal_img+tab",
          _concat_tabular(pal_train_proj, tab_train),
          _concat_tabular(pal_test_proj,  tab_test))

    # ── Context-aware PALPool: fit with tabular as scoring context ───────────
    # The pooler sees tabular features during TabICL scoring (quality targets are
    # tabular-informed), but the Ridge model and pooling weights are still DINO-only.
    print("\n--- Fitting IterativePALPooler (tabular context) ---")
    pooler_ctx = pooler_factory(refinement_cfg=refinement_cfg, seed=args.seed)
    t_fit_ctx = time.perf_counter()
    pooler_ctx.fit(train_patches, train_labels, context_features=tab_train)
    fit_time_ctx_s = time.perf_counter() - t_fit_ctx
    print(f"[pal_ctx] Pooler fit in {fit_time_ctx_s:.1f}s")

    pal_ctx_train_raw = pooler_ctx.transform(train_patches)
    pal_ctx_test_raw  = pooler_ctx.transform(test_patches)

    final_stage_ctx = pooler_ctx.stages_[-1]
    pal_ctx_pca = final_stage_ctx._pca_

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

    # ── Save results ─────────────────────────────────────────────────────────
    total_time_s = time.perf_counter() - t_start
    record = {
        "run_timestamp": run_ts,
        "total_time_s":  round(total_time_s, 2),
        "args": {
            "dataset":            "petfinder",
            "dataset_path":       str(args.dataset_path),
            "n_train":            args.n_train,
            "n_estimators":       args.n_estimators,
            "pca_dim":            pca_dim,
            "patch_group_sizes":  args.patch_group_sizes,
            "temperature":        args.temperature,
            "ridge_alpha":        args.ridge_alpha,
            "weight_method":      args.weight_method,
            "seed":               args.seed,
        },
        "dataset_info": {
            "n_train":   int(N_train),
            "n_test":    int(len(test_labels)),
            "n_patches": int(P),
            "embed_dim": int(D),
            "tab_dim":   int(tab_train.shape[1]),
            "n_classes": int(n_classes),
            "pca_dim":   int(pal_pca.n_components_) if pal_pca is not None else None,
        },
        "results": results,
    }
    results_path = output_dir / "multimodal_results.json"
    with results_path.open("w") as f:
        json.dump(record, f, indent=2)

    print(f"\n[done] Total time: {total_time_s:.1f}s")
    print(f"[done] Results saved → {results_path}")
    _print_summary(results)


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
        description="Multimodal experiment: PALPool image features + tabular (PetFinder)"
    )
    p.add_argument("--dataset-path",   type=Path,  default=PETFINDER_DATASET_PATH,
                   help="Root directory of the PetFinder dataset")
    p.add_argument("--n-train",        type=int,   default=None,
                   help="Subsample this many training images (default: use all)")
    p.add_argument("--n-estimators",   type=int,   default=1,
                   help="TabICL ensemble size (default: 1)")
    p.add_argument("--pca-dim",        type=int,   default=128,
                   help="PCA components for image features (default: 128)")
    p.add_argument("--no-pca",         action="store_true",
                   help="Disable PCA; use full 768-D image embeddings")
    p.add_argument("--patch-group-sizes", type=int, nargs="+", default=[1],
                   help="Patch group sizes for IterativePALPooler (default: [1])")
    p.add_argument("--temperature",    type=float, nargs="+", default=[1.0],
                   help="Softmax temperature(s) for patch pooling (default: [1.0])")
    p.add_argument("--ridge-alpha",    type=float, nargs="+", default=[1.0],
                   help="Ridge regularisation strength(s) (default: [1.0])")
    p.add_argument("--weight-method",  type=str,   default="correct_class_prob",
                   choices=["correct_class_prob", "entropy", "kl_div",
                            "wasserstein", "js_div", "tvd"],
                   help="Patch quality weight method (default: correct_class_prob)")
    p.add_argument("--normalize-features", action="store_true",
                   help="Fit a StandardScaler on patches before Ridge fitting")
    p.add_argument("--batch-size",     type=int,   default=1000,
                   help="Images per TabICL forward pass during refinement (default: 1000)")
    p.add_argument("--max-query-rows", type=int,   default=None,
                   help="Cap on patch-group rows forwarded through TabICL")
    p.add_argument("--gpu-ridge",      action="store_true",
                   help="Solve Ridge on the GPU (requires PyTorch + CUDA)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output-dir",     type=Path,  default=Path("results/multimodal"))
    return p.parse_args()


if __name__ == "__main__":
    run_multimodal_experiment(_parse_args())
