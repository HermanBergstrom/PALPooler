"""PCA ablation — raw embeddings vs. PCA before TabICL, one modality at a time.

Single-modality only: no tabular columns, no cross-modal fusion. For the chosen
non-tabular modality the features are pooled two ways (CLS token, mean pooling)
and each pooling is fed to TabICL under three projections:

    raw       — embeddings as-is, no projection
    pca_full  — PCA keeping every component, min(N_train, D) (rotation only,
                no truncation) — isolates the effect of decorrelating/ordering
                the axes from the effect of dimensionality reduction
    pca<k>    — PCA truncated to k components, one condition per --pca-dim
                value (default: 256 and 512)

Poolings (``--pooling``)
------------------------
    image: cls  = CLS token embedding (default);  mean = mean over all patch tokens
    text:  mean = mean over valid tokens, excluding the CLS at pos 0 (default);
           cls  = CLS token embedding

Each modality defaults to its conventional single-vector representation — CLS
for image, mean-pooled tokens for text. Pass ``--pooling cls|mean|both`` to
override.

Usage
-----
    python pal_pooling/pca_ablation.py --dataset petfinder --modality image
    python pal_pooling/pca_ablation.py --dataset mm-imdb --modality text \\
        --pooling both --pca-dim 256 512 --seeds 42 123 --n-estimators 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

from pal_pooling.config import (
    AIRBNB_DATASET_PATH, CBIS_DDSM_DATASET_PATH, CLOTHING_DATASET_PATH,
    DVM_DATASET_PATH, FAKE_JOBS_DATASET_PATH, FEATURES_DIR, JIGSAW_DATASET_PATH,
    MM_IMDB_DATASET_PATH, PAD_UFES_DATASET_PATH, PETFINDER_DATASET_PATH,
    PRODUCT_SENTIMENT_DATASET_PATH, SALARY_INDIA_DATASET_PATH, WINE_REVIEWS_DATASET_PATH,
    WIKIART_DATASET_PATH, DatasetConfig, get_modality,
)
from pal_pooling.crossmodal_ablation import _BACKBONE, _SUPPORTS_TEXT, _acc_auroc, _mean_pool_text
from pal_pooling.data_loading import _load_features
from pal_pooling.multimodal_experiments import _set_global_seeds

_DATASET_PATHS = {
    "petfinder":         PETFINDER_DATASET_PATH,
    "dvm":               DVM_DATASET_PATH,
    "pad-ufes":          PAD_UFES_DATASET_PATH,
    "cbis-ddsm-mass":    CBIS_DDSM_DATASET_PATH,
    "cbis-ddsm-calc":    CBIS_DDSM_DATASET_PATH,
    "clothing":          CLOTHING_DATASET_PATH,
    "salary":            SALARY_INDIA_DATASET_PATH,
    "airbnb":            AIRBNB_DATASET_PATH,
    "fake-jobs":         FAKE_JOBS_DATASET_PATH,
    "jigsaw":            JIGSAW_DATASET_PATH,
    "product-sentiment": PRODUCT_SENTIMENT_DATASET_PATH,
    "wine-reviews":      WINE_REVIEWS_DATASET_PATH,
    "wikiart":           WIKIART_DATASET_PATH,
    "mm-imdb":           MM_IMDB_DATASET_PATH,
}


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _available_modalities(dataset: str) -> list[str]:
    """Which non-tabular modalities *dataset* actually provides.

    A text-primary dataset carries no images; an image-primary one carries text
    only if it ships an auxiliary text stream (``_SUPPORTS_TEXT``, e.g. petfinder
    and mm-imdb). Image-only datasets such as wikiart or dvm therefore return
    ``['image']``.
    """
    primary = get_modality(dataset)
    mods = [primary]
    if primary != "text" and dataset in _SUPPORTS_TEXT:
        mods.append("text")
    return mods


def _pool_features(
    modality: str,
    poolings: list[str],
    train_patches: np.ndarray,
    test_patches: np.ndarray,
    cls_train: np.ndarray,
    cls_test: np.ndarray,
    extra_data: dict,
    dataset: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {pooling: (train, test)} for *modality*, for each requested pooling.

    Text lives either in ``train_patches`` (text-primary datasets, where the
    patch axis holds tokens) or in ``extra_data['text_*']`` (datasets carrying
    an auxiliary text stream, e.g. petfinder / mm-imdb). Only the requested
    poolings are materialised — mean pooling reduces the full [N, P, D] token
    tensor, so it is skipped entirely when unused.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if modality == "image":
        if "cls" in poolings:
            out["cls"] = (cls_train.astype(np.float32), cls_test.astype(np.float32))
        if "mean" in poolings:
            out["mean"] = (train_patches.astype(np.float32).mean(axis=1),
                           test_patches.astype(np.float32).mean(axis=1))
        return out

    is_text_primary = get_modality(dataset) == "text" and extra_data.get("text_train") is None
    if is_text_primary:
        tok_tr, tok_te = train_patches.astype(np.float32), test_patches.astype(np.float32)
        attn_tr, attn_te = extra_data["train_attention_mask"], extra_data["test_attention_mask"]
        cls_tr, cls_te = cls_train, cls_test
    elif extra_data.get("text_train") is None:
        raise ValueError(
            f"{dataset} has no text stream to pool "
            f"(available modalities: {', '.join(_available_modalities(dataset))})"
        )
    else:
        tok_tr, tok_te = extra_data["text_train"], extra_data["text_test"]
        attn_tr, attn_te = extra_data["text_train_attn_mask"], extra_data["text_test_attn_mask"]
        cls_tr, cls_te = extra_data.get("text_cls_train"), extra_data.get("text_cls_test")

    if "cls" in poolings:
        if cls_tr is None:
            raise ValueError(f"{dataset} provides no text CLS embeddings; use --pooling mean")
        out["cls"] = (np.asarray(cls_tr, dtype=np.float32), np.asarray(cls_te, dtype=np.float32))
    if "mean" in poolings:
        out["mean"] = (_mean_pool_text(tok_tr, attn_tr), _mean_pool_text(tok_te, attn_te))
    return out


def _project(
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    n_components: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA-project both splits (fit on train). ``n_components=None`` keeps all
    min(N_train, D) components — a pure rotation, no truncation."""
    n_comp = None if n_components is None else min(n_components, *train_raw.shape)
    pca = PCA(n_components=n_comp, random_state=seed)
    return (pca.fit_transform(train_raw).astype(np.float32),
            pca.transform(test_raw).astype(np.float32))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(args: argparse.Namespace, dataset_path: Path, seed: int) -> tuple:
    """Load features for a single modality (tabular is never loaded here)."""
    dataset_cfg = DatasetConfig(
        dataset=args.dataset,
        backbone=_BACKBONE.get(args.dataset, "dinov3_local"),
        features_dir=FEATURES_DIR,
        dataset_path=Path(dataset_path),
        n_train=args.max_train,
        n_test=args.max_test,
        n_val=None,
        n_sample=0,
        balance_train=False,
        balance_test=False,
    )

    (train_patches, train_labels, test_patches, test_labels,
     cls_train, cls_test, idx_to_class, _, _, extra_data) = _load_features(
        dataset_cfg, seed=seed, load_tabular=False,
        load_text=(args.dataset in _SUPPORTS_TEXT),
    )

    # Optional post-hoc subsampling of the training split.
    if args.n_train is not None and args.n_train < len(train_labels):
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(len(train_labels), size=args.n_train, replace=False))
        train_patches = train_patches[idx]
        train_labels  = train_labels[idx]
        cls_train     = cls_train[idx]
        for k in ["text_train", "text_train_token_ids", "text_train_attn_mask",
                  "text_cls_train", "train_attention_mask", "train_token_ids"]:
            v = extra_data.get(k)
            if isinstance(v, np.ndarray):
                extra_data[k] = v[idx]
            elif isinstance(v, list):
                extra_data[k] = [v[i] for i in idx]

    return (train_patches, train_labels, test_patches, test_labels,
            cls_train, cls_test, idx_to_class, extra_data)


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def _run_single_seed(args: argparse.Namespace, seed: int, dataset_path: Path) -> dict:
    _set_global_seeds(seed)
    run_ts = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    (train_patches, train_labels, test_patches, test_labels,
     cls_train, cls_test, idx_to_class, extra_data) = _load_dataset(args, dataset_path, seed)

    poolings = ["cls", "mean"] if args.pooling == "both" else [args.pooling]
    pooled = _pool_features(args.modality, poolings, train_patches, test_patches,
                            cls_train, cls_test, extra_data, args.dataset)

    embed_dim = int(pooled[poolings[0]][0].shape[1])
    print(f"\n[dataset] {args.dataset}  modality={args.modality}  "
          f"n_train={len(train_labels)}  n_test={len(test_labels)}  "
          f"n_classes={len(idx_to_class)}  embed_dim={embed_dim}")

    results: dict[str, dict] = {}
    for pool_name in poolings:
        tr_raw, te_raw = pooled[pool_name]
        # (tag suffix, PCA components — "none" = no PCA, None = keep all components)
        conditions: list[tuple[str, Optional[int] | str]] = [
            ("raw", "none"),
            ("pca_full", None),
            *[(f"pca{k}", k) for k in args.pca_dim],
        ]
        for suffix, n_comp in conditions:
            tag = f"{args.modality}_{pool_name}_{suffix}"
            t0 = time.perf_counter()
            if n_comp == "none":
                tr, te = tr_raw.astype(np.float32), te_raw.astype(np.float32)
            else:
                tr, te = _project(tr_raw, te_raw, n_comp, seed)

            clf = TabICLClassifier(n_estimators=args.n_estimators, random_state=seed)
            clf.fit(tr, train_labels)
            acc, auroc = _acc_auroc(clf.predict_proba(te), test_labels)
            elapsed = time.perf_counter() - t0

            auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "nan"
            print(f"[{tag}]  acc={acc:.4f}  auroc={auroc_str}  "
                  f"({elapsed:.1f}s)  feat_dim={tr.shape[1]}")
            results[tag] = {
                "acc":      round(acc, 6),
                "auroc":    round(auroc, 6) if not np.isnan(auroc) else None,
                "time_s":   round(elapsed, 2),
                "feat_dim": int(tr.shape[1]),
                "pooling":  pool_name,
                "projection": suffix,
            }

    total_time_s = time.perf_counter() - t_start
    _print_summary(results)
    print(f"[seed {seed}] Total time: {total_time_s:.1f}s")

    return {
        "run_timestamp": run_ts,
        "seed":          seed,
        "total_time_s":  round(total_time_s, 2),
        "args": {
            "dataset":      args.dataset,
            "dataset_path": str(dataset_path),
            "modality":     args.modality,
            "pooling":      args.pooling,
            "pca_dim":      list(args.pca_dim),
            "n_train":      args.n_train,
            "n_estimators": args.n_estimators,
            "seed":         seed,
        },
        "dataset_info": {
            "n_train":   int(len(train_labels)),
            "n_test":    int(len(test_labels)),
            "n_classes": int(len(idx_to_class)),
            "embed_dim": embed_dim,
        },
        "results": results,
    }


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 52)
    print(f"  {'Condition':<26} {'Acc':>7} {'AUROC':>7} {'dim':>6}")
    print("-" * 52)
    for tag, r in results.items():
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "nan"
        print(f"  {tag:<26} {r['acc']:>7.4f} {auroc_str:>7} {r['feat_dim']:>6d}")
    print("=" * 52)


def run(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path or _DATASET_PATHS[args.dataset]

    output_dir = Path(args.output_dir) if args.output_dir is not None \
        else Path("results/pca_ablation") / f"{args.dataset}_{args.modality}"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "pca_ablation_results.json"

    if combined_path.exists():
        with combined_path.open() as f:
            all_records = json.load(f)
        if not isinstance(all_records, list):
            all_records = [all_records]
    else:
        all_records = []
    completed_seeds = {r.get("seed") for r in all_records}

    for i, seed in enumerate(args.seeds):
        print(f"\n{'='*60}\n  Seed {seed}  ({i + 1}/{len(args.seeds)})\n{'='*60}")
        if seed in completed_seeds:
            print(f"[skip] seed {seed} already present in {combined_path}")
            continue
        all_records.append(_run_single_seed(args, seed, dataset_path))
        with combined_path.open("w") as f:
            json.dump(all_records, f, indent=2)
        print(f"[saved] {combined_path}  ({len(all_records)} record(s) total)")

    print(f"\n[done] Results → {combined_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA ablation — raw vs. PCA features into TabICL")
    p.add_argument("--dataset", type=str, default="petfinder", choices=sorted(_DATASET_PATHS),
                   help="Dataset to run (default: petfinder)")
    p.add_argument("--dataset-path", type=Path, default=None,
                   help="Root directory of the dataset (defaults to config value)")
    p.add_argument("--modality", type=str, default=None, choices=["image", "text"],
                   help="Which modality to feed TabICL (default: the dataset's primary modality)")
    p.add_argument("--pooling", type=str, default=None, choices=["cls", "mean", "both"],
                   help="Pooling(s) to evaluate (default: cls for image, mean for text)")
    p.add_argument("--pca-dim", type=int, nargs="+", default=[256, 512],
                   help="Truncated-PCA dimension(s); one condition each (default: 256 512)")
    p.add_argument("--n-train", type=int, default=None,
                   help="Subsample this many training samples AFTER loading (default: use all)")
    p.add_argument("--max-train", type=int, default=None,
                   help="Load at most this many train samples")
    p.add_argument("--max-test", type=int, default=None,
                   help="Load at most this many test samples")
    p.add_argument("--n-estimators", type=int, default=1,
                   help="TabICL ensemble size (default: 1)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="One or more random seeds (default: 42)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to save results (default: results/pca_ablation/<dataset>_<modality>)")
    args = p.parse_args()
    if args.modality is None:
        args.modality = get_modality(args.dataset)
    # Fail before loading features rather than deep inside pooling — the load
    # is minutes of work for the larger datasets.
    available = _available_modalities(args.dataset)
    if args.modality not in available:
        raise SystemExit(
            f"[error] {args.dataset} provides no {args.modality} modality "
            f"(available: {', '.join(available)}). Re-run with "
            f"--modality {available[0]}."
        )
    if args.pooling is None:
        # Each modality's conventional single-vector representation: the CLS
        # token for image backbones, mean-pooled tokens for text encoders.
        args.pooling = "cls" if args.modality == "image" else "mean"
    return args


if __name__ == "__main__":
    run(_parse_args())
