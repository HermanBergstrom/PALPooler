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

from pal_pooling.config import DVM_DATASET_PATH, PAD_UFES_DATASET_PATH, PETFINDER_DATASET_PATH, RefinementConfig
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
]:
    """Load dataset patch, CLS, and tabular features.

    Merges train + val splits as the support set.
    Returns arrays in [N, P, D], [N], [N, D], [N, F] shapes.
    """
    dataset_dir = Path(dataset_path)
    if str(dataset_dir) not in sys.path:
        sys.path.insert(0, str(dataset_dir))

    if dataset_name == "petfinder":
        from petfinder_dataset_with_dinov3 import load_petfinder_dataset  # type: ignore

        train_loader, val_loader, test_loader, metadata = load_petfinder_dataset(
            feature_source="dinov3_local",
            use_patches=True,
            use_images=True,
            num_workers=0,
        )
    elif dataset_name == "dvm":
        from dvm_dataset_with_dinov3 import load_dvm_dataset  # type: ignore

        train_loader, val_loader, test_loader, metadata = load_dvm_dataset(
            feature_source="dinov3_local",
            feature_dir="/scratch/hermanb/temp_datasets/extracted_features/dvm/dvm_dinov3_local_features",
            use_patches=True,
            use_images=True,
            num_workers=0,
            data_dir=dataset_dir
        )
    elif dataset_name == "pad-ufes":
        from pad_ufes_dataset import PADUFESDataset, load_metadata  # type: ignore
        from sklearn.model_selection import train_test_split
        from tqdm import tqdm

        metadata = load_metadata(data_dir=str(dataset_dir))
        df_full  = metadata["df"]
        df_tr, df_te = train_test_split(
            df_full, test_size=0.2, random_state=seed, stratify=df_full["diagnostic"]
        )
        train_ds_pad = PADUFESDataset(df_tr, metadata, use_patches=True, use_images=True)
        test_ds_pad  = PADUFESDataset(df_te, metadata, use_patches=True, use_images=True)

        def _collect_pad(ds, n_limit, desc):
            total_n = len(ds)
            if n_limit is not None and n_limit < total_n:
                rng  = np.random.RandomState(seed)
                idxs = rng.choice(total_n, size=n_limit, replace=False)
                idxs.sort()
            else:
                idxs = np.arange(total_n)
            patches, cls_emb, labels, tab_feat = [], [], [], []
            for i in tqdm(idxs, desc=desc):
                s = ds[int(i)]
                patches.append(s["patch_embedding"].float().numpy())
                cls_emb.append(s["image_embedding"].float().numpy())
                labels.append(s["target"].item())
                tab_feat.append(s["tabular"].float().numpy())
            return (
                np.stack(patches,   axis=0),
                np.stack(cls_emb,   axis=0),
                np.array(labels,    dtype=np.int64),
                np.stack(tab_feat,  axis=0),
            )

        train_patches, cls_train, train_labels, tab_train = _collect_pad(
            train_ds_pad, max_train, f"Loading pad-ufes (train)"
        )
        test_patches, cls_test, test_labels, tab_test = _collect_pad(
            test_ds_pad, max_test, f"Loading pad-ufes (test)"
        )
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
            f"[info] pad-ufes (train): N={len(train_labels)}  "
            f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}  "
            f"tab_dim={tab_train.shape[1]}"
        )
        print(f"[info] pad-ufes (test):  N={len(test_labels)}")
        return (
            train_patches, train_labels,
            test_patches,  test_labels,
            cls_train, cls_test,
            tab_train, tab_test,
            idx_to_class,
        )

    else:
        raise ValueError(f"Unknown multimodal dataset: {dataset_name}")

    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    test_ds  = test_loader.dataset

    def get_subset(ds_list, n_limit=None, desc="Loading Data"):
        lengths = [len(ds) for ds in ds_list]
        total_n = sum(lengths)
        
        if n_limit is not None and n_limit < total_n:
            rng = np.random.RandomState(seed)
            sub_idx = rng.choice(total_n, size=n_limit, replace=False)
            sub_idx.sort()
        else:
            sub_idx = np.arange(total_n)
            
        from tqdm import tqdm
        patches, cls_emb, labels, tab_feat = [], [], [], []
        if dataset_name == "petfinder":
            for ds in ds_list:
                patches.append(ds.patch_embeddings.float().numpy())
                cls_emb.append(ds.image_embeddings.float().numpy())
                labels.append(ds.targets.numpy().astype(np.int64))
                tab_feat.append(ds.tabular.float().numpy())
            
            # Concat all arrays
            all_patches = np.concatenate(patches, axis=0)
            all_cls_emb = np.concatenate(cls_emb, axis=0)
            all_labels = np.concatenate(labels, axis=0)
            all_tab_feat = np.concatenate(tab_feat, axis=0)
            
            # Apply limit if needed
            if n_limit is not None and n_limit < total_n:
                all_patches = all_patches[sub_idx]
                all_cls_emb = all_cls_emb[sub_idx]
                all_labels = all_labels[sub_idx]
                all_tab_feat = all_tab_feat[sub_idx]
                
            return all_patches, all_cls_emb, all_labels, all_tab_feat
        else:
            for i in tqdm(sub_idx, desc=desc):
                offset = 0
                ds_idx = i
                for l, ds in zip(lengths, ds_list):
                    if ds_idx < l:
                        sample = ds[ds_idx]
                        break
                    ds_idx -= l
                patches.append(sample["patch_embedding"].float().numpy())
                cls_emb.append(sample["image_embedding"].float().numpy())
                labels.append(sample["target"])
                tab_feat.append(sample["tabular"].float().numpy())
            return np.stack(patches, axis=0), np.stack(cls_emb, axis=0), np.array(labels, dtype=np.int64), np.stack(tab_feat, axis=0)

    # Merge train + val → support set.
    train_patches, cls_train, train_labels, tab_train = get_subset([train_ds, val_ds], n_limit=max_train, desc=f"Loading {dataset_name} (train+val)")
    test_patches, cls_test, test_labels, tab_test = get_subset([test_ds], n_limit=max_test, desc=f"Loading {dataset_name} (test)")

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
        f"[info] {dataset_name} (train+val): N={len(train_labels)}  "
        f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}  "
        f"tab_dim={tab_train.shape[1]}"
    )
    print(f"[info] {dataset_name} (test):  N={len(test_labels)}")
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
     idx_to_class) = _load_dataset(
        dataset_name=args.dataset,
        dataset_path=dataset_path,
        n_train=args.n_train,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=seed,
    )

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

    # ── Baseline: mean-pool image only ───────────────────────────────────────
    print("\n--- mean_pool_img ---")
    mean_train_raw = train_patches.astype(np.float32).mean(axis=1)  # [N, D]
    mean_test_raw  = test_patches.astype(np.float32).mean(axis=1)
    mean_train, mean_test, mean_pca = _pca_project(mean_train_raw, mean_test_raw, pca_dim, seed)
    _eval("mean_pool_img", mean_train, mean_test)

    # ── Baseline: CLS token image only ──────────────────────────────────────
    print("\n--- cls_img ---")
    cls_train_proj, cls_test_proj, cls_pca = _pca_project(cls_train, cls_test, pca_dim, seed)
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
        use_global_prior=args.use_global_prior,
        use_attn_masking=args.use_attn_masking,
        use_marginal_prior=args.use_marginal_prior,
        model_selection=args.model_selection,
    )

    pooler = pooler_factory(refinement_cfg=refinement_cfg, seed=seed)
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
    best_stage = pooler.stages_[pooler.best_stage_idx_]
    pal_pca = best_stage._pca_

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
    pooler_ctx = pooler_factory(refinement_cfg=refinement_cfg, seed=seed)
    t_fit_ctx = time.perf_counter()
    pooler_ctx.fit(train_patches, train_labels, context_features=tab_train)
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
            "temperature":        args.temperature,
            "ridge_alpha":        args.ridge_alpha,
            "weight_method":      args.weight_method,
            "seed":               seed,
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

    print(f"\n[seed {seed}] Total time: {total_time_s:.1f}s")
    _print_summary(results)
    return record


def run_multimodal_experiment(args: argparse.Namespace) -> None:
    seeds = args.seeds

    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = {
            "dvm":       DVM_DATASET_PATH,
            "petfinder": PETFINDER_DATASET_PATH,
            "pad-ufes":  PAD_UFES_DATASET_PATH,
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
                   choices=["petfinder", "dvm", "pad-ufes"], help="Dataset to run multimodal experiment on")
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
    p.add_argument("--use-global-prior", action="store_true",
                   help="Use the global class prior P(Y) as the divergence reference even when "
                        "context features are provided, instead of the per-image P(Y|X_tab).")
    p.add_argument("--use-attn-masking", action=argparse.BooleanOptionalAction, default=True,
                   help="Use attention masking in PALPooler (default: True)")
    p.add_argument("--use-marginal-prior", action=argparse.BooleanOptionalAction, default=True,
                   help="Use marginal patch prior in PALPooler (default: True)")
    p.add_argument("--model-selection", type=str, default="last_iteration",
                   choices=["last_iteration", "masked_train_accuracy"],
                   help="Which stage to use at inference after iterative refinement. "
                        "'last_iteration' (default) always uses the final stage. "
                        "'masked_train_accuracy' evaluates every stage on the training set "
                        "with a diagonal attention mask and selects the best-performing one.")
    p.add_argument("--seeds",          type=int,   nargs="+", default=[42],
                   help="One or more random seeds, e.g. --seeds 42 123 456. "
                        "Results are saved after every seed. (default: 42)")
    p.add_argument("--output-dir",     type=Path,  default=None,
                   help="Directory to save results (default: results/multimodal/<dataset>)")
    return p.parse_args()


if __name__ == "__main__":
    run_multimodal_experiment(_parse_args())
