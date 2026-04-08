"""Minimal demo: PAL patch pooling vs. mean-pool baseline on the butterfly dataset.

Pipeline:
  1. Load pre-extracted DINOv3 patch features          [N, P, D]
  2. Baseline: mean-pool → PCA(128) → TabICL
  3. PAL pool: IterativePALPooler at group sizes [16, 4, 1] → same eval

Run with:
    python pal_pooling/demo.py
"""

from __future__ import annotations
import sys
from pathlib import Path

if __package__ in (None, ""):
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
	
import numpy as np
from sklearn.decomposition import PCA
from tabicl import TabICLClassifier

from pal_pooling.config import FEATURES_DIR, RefinementConfig
from pal_pooling.data_loading import ButterflyPatchDataset
from pal_pooling.pal_pooler import IterativePALPooler

SEED    = 42
PCA_DIM = 128

# ---------------------------------------------------------------------------
# 1. Load pre-extracted DINOv3 patch features
# ---------------------------------------------------------------------------
train_ds = ButterflyPatchDataset(FEATURES_DIR, split="train")
test_ds  = ButterflyPatchDataset(FEATURES_DIR, split="test")

train_patches = train_ds.features.numpy()   # [N_train, P, D]  e.g. [~4800, 196, 768]
train_labels  = train_ds.labels.numpy()
test_patches  = test_ds.features.numpy()    # [N_test,  P, D]
test_labels   = test_ds.labels.numpy()

# ---------------------------------------------------------------------------
# 2. Baseline: mean-pool all patches → PCA → TabICL
# ---------------------------------------------------------------------------
pca = PCA(n_components=min(PCA_DIM, len(train_labels), train_patches.shape[-1]), random_state=SEED)
train_mean = pca.fit_transform(train_patches.mean(axis=1)).astype(np.float32)
test_mean  = pca.transform(test_patches.mean(axis=1)).astype(np.float32)

clf = TabICLClassifier(n_estimators=1, random_state=SEED)
clf.fit(train_mean, train_labels)
proba        = clf.predict_proba(test_mean)
baseline_acc = float((np.argmax(proba, axis=1) == test_labels).mean())

print(f"\n[baseline]  mean-pool → PCA({PCA_DIM}) → TabICL  acc={baseline_acc:.4f}")

# ---------------------------------------------------------------------------
# 3. PAL pooling: iterative refinement at group sizes [16, 4, 1]
#
#    Each stage fits a Ridge quality predictor that scores every patch group,
#    then re-pools the training embeddings as a quality-weighted average.
#    The refined support is passed to the next finer-grained stage.
#    ridge_alpha=1e3 applies strong regularisation to the quality predictor.
# ---------------------------------------------------------------------------
refinement_cfg = RefinementConfig(
    refine=True,
    patch_size=16,
    patch_group_sizes=[16, 4, 1],  # coarse → medium → individual patches
    temperature=[0.5],             # softmax temperature; broadcast to all stages
    weight_method="correct_class_prob",
    ridge_alpha=[1e3],             # Ridge regularisation; broadcast to all stages
    normalize_features=False,
    batch_size=1000,
    max_query_rows=int(3e5),
    use_random_subsampling=True,
    aoe_class=None,
    aoe_handling="filter",
    gpu_ridge=False,
    tabicl_n_estimators=1,
    tabicl_pca_dim=PCA_DIM,
)

tabicl = TabICLClassifier(n_estimators=1, random_state=SEED)
pooler = IterativePALPooler(tabicl=tabicl, refinement_cfg=refinement_cfg, seed=SEED)
pooler.fit(train_patches, train_labels)

# Evaluate: transform test patches with final stage → internal PCA → TabICL
pal_acc, pal_auroc = pooler.score_tabicl(test_patches, test_labels)

print(f"[PAL pool]  [16 → 4 → 1] → PCA({PCA_DIM}) → TabICL  acc={pal_acc:.4f}  auroc={pal_auroc:.4f}")
print(f"\nΔ acc (PAL - baseline) = {pal_acc - baseline_acc:+.4f}")
