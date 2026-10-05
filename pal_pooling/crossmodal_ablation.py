"""Cross-modal interaction ablation — baseline (full fusion).

Measures the impact of cross-modal interactions *inside* TabICL. This is the
baseline condition: load a dataset, PCA-reduce the non-tabular modality(ies)
(mean-pooled text, CLS-token image — no PALPooling), concatenate every modality
into a single feature matrix, and classify with TabICL. Because all modalities
share one feature matrix, TabICL is free to model cross-modal interactions
across columns; later ablations will constrain that.

Conditions evaluated (only those whose modalities are present are run)
----------------------------------------------------------------------
    tabular_only     — raw tabular features → TabICL
    image_only       — PCA(CLS image)       → TabICL
    text_only        — PCA(mean-pool text)  → TabICL
    fused            — every available modality concatenated → TabICL (full fusion)
    fused_ablated    — Method 1: same matrix, cross-modal feature interaction masked
                       via TabICL's ``modality_sizes`` (see MODALITY_ABLATION_USAGE.md)
    ensemble_mean    — Method 2: per-modality TabICL classifiers, probabilities
                       averaged (late/decision-level fusion)
    ensemble_geomean — Method 2 variant: per-modality probabilities combined by
                       geometric mean (naive-Bayes-style)

Two ways to disable cross-modal interaction, both compared against ``fused``:
Method 1 keeps one model but masks cross-modal feature attention; Method 2 uses
independent per-modality models and only fuses their outputs. The gap from
``fused`` estimates the value of cross-modal interaction.

Only the modalities a dataset actually provides are used, so the fused matrix
adapts per dataset — e.g. petfinder is image+text+tabular, dvm/wikiart are
image+tabular, and mm-imdb is image+text (no tabular). Datasets with a single
modality skip the cross-modal conditions.

Requires the TabICL fork whose ``TabICLClassifier`` accepts ``modality_sizes``
(editable install in the ``aditya_tabicl`` environment):
    source /project/aip-rahulgk/hermanb/environments/aditya_tabicl/bin/activate

Usage
-----
    python pal_pooling/crossmodal_ablation.py \\
        --dataset petfinder [--modalities all|image|text] [--pca-dim 128] \\
        [--n-estimators 1] [--max-train N] [--max-test N] [--seeds 42 123] \\
        [--output-dir results/crossmodal_ablation]

    # Run petfinder tab+img and tab+text as two separate experiments:
    python pal_pooling/crossmodal_ablation.py --dataset petfinder --modalities image
    python pal_pooling/crossmodal_ablation.py --dataset petfinder --modalities text
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
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

from pal_pooling.config import (
    AIRBNB_DATASET_PATH, CBIS_DDSM_DATASET_PATH, CLOTHING_DATASET_PATH,
    DVM_DATASET_PATH, FAKE_JOBS_DATASET_PATH, FEATURES_DIR, JIGSAW_DATASET_PATH,
    MM_IMDB_DATASET_PATH, PAD_UFES_DATASET_PATH, PETFINDER_DATASET_PATH,
    PRODUCT_SENTIMENT_DATASET_PATH, SALARY_INDIA_DATASET_PATH, WINE_REVIEWS_DATASET_PATH,
    WIKIART_DATASET_PATH, DatasetConfig, get_modality,
)
from pal_pooling.data_loading import _load_features
from pal_pooling.multimodal_experiments import (
    _pca_project,
    _set_global_seeds,
    _vectorize_tabular,
)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _mean_pool_text(
    text: np.ndarray,           # [N, T_max, D]
    attn_mask: np.ndarray,      # [N, T_max] bool/int
) -> np.ndarray:
    """Mean-pool token embeddings over valid positions, excluding the CLS at pos 0."""
    attn = attn_mask.astype(text.dtype)
    counts = (attn.sum(axis=1, keepdims=True) - attn[:, 0:1]).clip(min=1)
    full_sum = np.matmul(attn[:, None, :], text).squeeze(1)          # [N, D]
    cls_contrib = text[:, 0, :] * attn[:, 0:1]                       # remove CLS token
    return (full_sum - cls_contrib) / counts


def _resolve_text(
    dataset: str,
    train_patches: np.ndarray,
    test_patches: np.ndarray,
    extra_data: dict,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (text_train, text_test, attn_train, attn_test) or None if no text.

    Handles both text-primary datasets (tokens live in ``train_patches``) and
    datasets carrying an auxiliary text stream in ``extra_data['text_train']``
    (e.g. petfinder).
    """
    is_text_primary = get_modality(dataset) == "text" and extra_data.get("text_train") is None
    if is_text_primary:
        return (
            train_patches.astype(np.float32),
            test_patches.astype(np.float32),
            extra_data["train_attention_mask"],
            extra_data["test_attention_mask"],
        )
    if extra_data.get("text_train") is not None:
        return (
            extra_data["text_train"],
            extra_data["text_test"],
            extra_data["text_train_attn_mask"],
            extra_data["text_test_attn_mask"],
        )
    return None


# Backbone per dataset (mirrors multimodal_experiments). Datasets whose loader
# path ignores it — petfinder, wikiart, mm-imdb — fall through to the default.
_BACKBONE: dict[str, str] = {
    "petfinder":      "dinov3",
    "dvm":            "dinov3_local",
    "pad-ufes":       "dinov3_local",
    "cbis-ddsm-mass": "dinov3_local",
    "cbis-ddsm-calc": "dinov3_local",
    "clothing":          "electra",
    "salary":            "electra",
    "airbnb":            "electra",
    "fake-jobs":         "electra",
    "jigsaw":            "electra",
    "product-sentiment": "electra",
    "wine-reviews":      "electra",
}
# Datasets carrying an auxiliary text stream alongside a non-text primary modality.
_SUPPORTS_TEXT = {"petfinder", "mm-imdb"}


def _load_dataset(
    dataset_name: str,
    dataset_path: Path,
    n_train: Optional[int],
    max_train: Optional[int],
    max_test: Optional[int],
    seed: int,
) -> tuple:
    """Load image/text/tabular features via ``_load_features``.

    Unlike ``multimodal_experiments._load_dataset``, tabular is **optional**:
    datasets without a tabular stream (e.g. mm-imdb, an image+text dataset)
    return ``tab_train = tab_test = None`` instead of raising.
    """
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

    tab_train = extra_data.get("tab_train")
    tab_test  = extra_data.get("tab_test")
    if tab_train is not None:
        tab_train, tab_test = _vectorize_tabular(tab_train, tab_test)
        extra_data["tab_train"] = tab_train
        extra_data["tab_test"]  = tab_test

    # Optional post-hoc n_train subsampling (separate from the max_train load limit).
    if n_train is not None:
        _n = int(round(n_train * len(train_labels))) if isinstance(n_train, float) else n_train
        if _n < len(train_labels):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(train_labels), size=_n, replace=False)
            idx.sort()
            train_patches = train_patches[idx]
            train_labels  = train_labels[idx]
            cls_train     = cls_train[idx]
            if tab_train is not None:
                tab_train = tab_train[idx]
                extra_data["tab_train"] = tab_train
            for k in ["text_train", "text_train_token_ids", "text_train_attn_mask", "text_cls_train",
                      "train_attention_mask", "train_token_ids"]:
                if extra_data.get(k) is not None:
                    if isinstance(extra_data[k], np.ndarray):
                        extra_data[k] = extra_data[k][idx]
                    elif isinstance(extra_data[k], list):
                        extra_data[k] = [extra_data[k][i] for i in idx]

    return (train_patches, train_labels, test_patches, test_labels,
            cls_train, cls_test, tab_train, tab_test, idx_to_class, extra_data)


def _tabicl_proba(
    train_feat: np.ndarray,
    train_labels: np.ndarray,
    test_feat: np.ndarray,
    n_estimators: int,
    seed: int,
    modality_sizes: Optional[list[int]] = None,
) -> np.ndarray:
    """Fit TabICL on train_feat and return predicted probabilities on test_feat.

    When ``modality_sizes`` is given (contiguous per-modality column-block sizes
    summing to ``train_feat.shape[1]``), TabICL masks cross-modal feature
    interaction — the ablation from MODALITY_ABLATION_USAGE.md. ``None`` runs
    stock TabICL with full cross-modal fusion.
    """
    kwargs = dict(n_estimators=n_estimators, random_state=seed)
    if modality_sizes is not None:
        kwargs["modality_sizes"] = modality_sizes
    clf = TabICLClassifier(**kwargs)
    clf.fit(train_feat, train_labels)
    return clf.predict_proba(test_feat)


def _acc_auroc(proba: np.ndarray, test_labels: np.ndarray) -> tuple[float, float]:
    """Accuracy and (binary/OvR-macro) AUROC from a predicted-probability matrix."""
    acc = float((np.argmax(proba, axis=1) == test_labels).mean())
    try:
        if proba.shape[1] == 2:
            auroc = float(roc_auc_score(test_labels, proba[:, 1]))
        else:
            auroc = float(roc_auc_score(test_labels, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = float("nan")
    return acc, auroc


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def _run_single_seed(args: argparse.Namespace, seed: int, dataset_path: Path) -> dict:
    _set_global_seeds(seed)
    run_ts = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

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

    pca_dim = None if args.no_pca else args.pca_dim
    is_text_primary = get_modality(args.dataset) == "text" and extra_data.get("text_train") is None

    # ── Build per-modality feature blocks (train, test), in fused column order:
    #    image, text, tabular. Any modality the dataset lacks is simply absent. ──
    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ``--modalities`` restricts which non-tabular modality is included; tabular
    # (when present) is always kept. E.g. petfinder --modalities image → tab+img.
    want_image = args.modalities in ("all", "image")
    want_text  = args.modalities in ("all", "text")

    # Image (CLS token) — only for datasets whose primary modality is image.
    if want_image and not is_text_primary and get_modality(args.dataset) == "image":
        img_tr, img_te, _ = _pca_project(cls_train, cls_test, pca_dim, seed)
        blocks["image"] = (img_tr, img_te)

    # Text (mean-pooled tokens, excluding CLS).
    text = _resolve_text(args.dataset, train_patches, test_patches, extra_data)
    if want_text and text is not None:
        text_tr_raw = _mean_pool_text(text[0], text[2])
        text_te_raw = _mean_pool_text(text[1], text[3])
        text_tr, text_te, _ = _pca_project(text_tr_raw, text_te_raw, pca_dim, seed)
        blocks["text"] = (text_tr, text_te)

    # Tabular (vectorised, no PCA) — optional; absent for e.g. mm-imdb.
    if tab_train is not None:
        blocks["tabular"] = (tab_train.astype(np.float32), tab_test.astype(np.float32))

    n_classes = len(idx_to_class)
    tab_dim = int(tab_train.shape[1]) if tab_train is not None else 0
    print(f"\n[dataset] {args.dataset}  n_train={len(train_labels)}  n_test={len(test_labels)}  "
          f"n_classes={n_classes}  tab_dim={tab_dim}  "
          f"modality_blocks={list(blocks)}  pca_dim={pca_dim}")
    if len(blocks) < 2:
        print("[warn] Fewer than two modalities present — cross-modal conditions "
              "(fused / fused_ablated / ensemble) will be skipped.")

    results: dict[str, dict] = {}
    proba_store: dict[str, np.ndarray] = {}

    def _record(tag: str, acc: float, auroc: float, elapsed: float,
                feat_dim: Optional[int], extra: Optional[dict] = None) -> None:
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "nan"
        dim_str = f"feat_dim={feat_dim}" if feat_dim is not None else "feat_dim=—"
        extra_str = f"  {extra}" if extra else ""
        print(f"[{tag}]  acc={acc:.4f}  auroc={auroc_str}  ({elapsed:.1f}s)  {dim_str}{extra_str}")
        results[tag] = {
            "acc":      round(acc, 6),
            "auroc":    round(auroc, 6) if not np.isnan(auroc) else None,
            "time_s":   round(elapsed, 2),
            "feat_dim": feat_dim,
            **(extra or {}),
        }

    def _eval(tag: str, train_feat: np.ndarray, test_feat: np.ndarray,
              modality_sizes: Optional[list[int]] = None) -> np.ndarray:
        t0 = time.perf_counter()
        proba = _tabicl_proba(
            train_feat, train_labels, test_feat,
            n_estimators=args.n_estimators, seed=seed, modality_sizes=modality_sizes,
        )
        acc, auroc = _acc_auroc(proba, test_labels)
        elapsed = time.perf_counter() - t0
        _record(tag, acc, auroc, elapsed, int(train_feat.shape[1]),
                extra={"modality_sizes": modality_sizes} if modality_sizes is not None else None)
        proba_store[tag] = proba
        return proba

    # ── Per-modality classifiers (one per present modality, incl. tabular) ────
    for name, (b_tr, b_te) in blocks.items():
        _eval(f"{name}_only", b_tr, b_te)

    # Modality blocks are laid out contiguously in insertion order, so
    # modality_sizes matches the fused concatenation order exactly.
    modality_sizes = [int(b_tr.shape[1]) for b_tr, _ in blocks.values()]

    # Cross-modal conditions require at least two modalities to compare.
    if len(blocks) >= 2:
        # ── Fused: every modality concatenated into one matrix ───────────────
        fused_tr = np.concatenate([b_tr for b_tr, _ in blocks.values()], axis=1)
        fused_te = np.concatenate([b_te for _, b_te in blocks.values()], axis=1)

        # Method 1 — full cross-modal fusion (stock TabICL) vs. cross-modal
        # feature interaction masked via modality_sizes. The gap estimates the
        # value of that interaction.
        _eval("fused", fused_tr, fused_te)
        _eval("fused_ablated", fused_tr, fused_te, modality_sizes=modality_sizes)

        # ── Method 2 — late-fusion ensemble ──────────────────────────────────
        # Each modality is classified by its own independent TabICL; only the
        # output distributions are combined. No modality ever sees another's
        # features, so cross-modal interaction is disabled at the decision level
        # instead of inside a single model. Reuses the per-modality probabilities.
        member_tags = [f"{name}_only" for name in blocks]
        stacked = np.stack([proba_store[t] for t in member_tags], axis=0)  # [M, N_test, C]
        member_time = float(sum(results[t]["time_s"] for t in member_tags))

        # Arithmetic mean of probabilities.
        mean_proba = stacked.mean(axis=0)
        acc, auroc = _acc_auroc(mean_proba, test_labels)
        _record("ensemble_mean", acc, auroc, member_time, None, extra={"members": member_tags})

        # Geometric mean (mean of log-probs, renormalised) — naive-Bayes-style fusion.
        geo = np.exp(np.log(np.clip(stacked, 1e-12, None)).mean(axis=0))
        geo /= geo.sum(axis=1, keepdims=True)
        acc, auroc = _acc_auroc(geo, test_labels)
        _record("ensemble_geomean", acc, auroc, member_time, None, extra={"members": member_tags})

    total_time_s = time.perf_counter() - t_start
    record = {
        "run_timestamp": run_ts,
        "seed":          seed,
        "total_time_s":  round(total_time_s, 2),
        "args": {
            "dataset":      args.dataset,
            "dataset_path": str(dataset_path),
            "modalities":   args.modalities,
            "n_train":      args.n_train,
            "n_estimators": args.n_estimators,
            "pca_dim":      pca_dim,
            "seed":         seed,
        },
        "dataset_info": {
            "n_train":   int(len(train_labels)),
            "n_test":    int(len(test_labels)),
            "tab_dim":   tab_dim,
            "n_classes": int(n_classes),
            # Column layout of the fused matrix, in block insertion order.
            "modality_order": list(blocks),
            "modality_sizes": modality_sizes,
        },
        "results": results,
    }
    _print_summary(results)
    print(f"[seed {seed}] Total time: {total_time_s:.1f}s")
    return record


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 52)
    print(f"{'Condition':<20} {'Acc':>8} {'AUROC':>8} {'dim':>8}")
    print("-" * 52)
    for tag, r in results.items():
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "  nan"
        dim_str = f"{r['feat_dim']:>8d}" if r["feat_dim"] is not None else f"{'—':>8}"
        print(f"  {tag:<18} {r['acc']:>8.4f} {auroc_str:>8} {dim_str}")
    print("=" * 52)


def run(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = {
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
        }[args.dataset]

    # Restricted-modality runs go to their own subdir (e.g. petfinder_image) so
    # they are treated as separate experiments rather than colliding with — and
    # being seed-skipped against — the full run.
    subdir = args.dataset if args.modalities == "all" else f"{args.dataset}_{args.modalities}"
    output_dir = Path(args.output_dir) if args.output_dir is not None \
        else Path("results/crossmodal_ablation") / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "baseline_results.json"

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
        record = _run_single_seed(args, seed, dataset_path)
        all_records.append(record)
        with combined_path.open("w") as f:
            json.dump(all_records, f, indent=2)
        print(f"[saved] {combined_path}  ({len(all_records)} record(s) total)")

    print(f"\n[done] Results → {combined_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-modal interaction ablation — full-fusion baseline")
    p.add_argument("--dataset", type=str, default="petfinder",
                   choices=["petfinder", "dvm", "pad-ufes", "cbis-ddsm-mass", "cbis-ddsm-calc",
                            "clothing", "salary", "airbnb", "fake-jobs", "jigsaw",
                            "product-sentiment", "wine-reviews", "wikiart", "mm-imdb"],
                   help="Dataset to run (default: petfinder)")
    p.add_argument("--dataset-path", type=Path, default=None,
                   help="Root directory of the dataset (defaults to config value)")
    p.add_argument("--modalities", type=str, default="all", choices=["all", "image", "text"],
                   help="Which non-tabular modality to include alongside tabular: "
                        "'all' (default), 'image' (tab+img), or 'text' (tab+text). "
                        "Restricted runs save to results/crossmodal_ablation/<dataset>_<modalities>/.")
    p.add_argument("--n-train", type=int, default=None,
                   help="Subsample this many training samples AFTER loading (default: use all)")
    p.add_argument("--max-train", type=int, default=None,
                   help="Load at most this many train+val samples")
    p.add_argument("--max-test", type=int, default=None,
                   help="Load at most this many test samples")
    p.add_argument("--n-estimators", type=int, default=1,
                   help="TabICL ensemble size (default: 1)")
    p.add_argument("--pca-dim", type=int, default=128,
                   help="PCA components for each non-tabular modality (default: 128)")
    p.add_argument("--no-pca", action="store_true",
                   help="Disable PCA; use full-dimensional embeddings")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="One or more random seeds (default: 42)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to save results (default: results/crossmodal_ablation/<dataset>)")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
