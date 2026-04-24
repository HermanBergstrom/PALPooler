"""
Minimal example: configure and run IterativePALPooler for image and text data.

Prerequisites
-------------
  pip install tabicl
  source /project/aip-rahulgk/hermanb/environments/aditya_tabicl/bin/activate

Inputs expected
---------------
  train_patches  : np.ndarray, shape (N_train, n_patches, embed_dim)
  train_labels   : np.ndarray, shape (N_train,)
  test_patches   : np.ndarray, shape (N_test,  n_patches, embed_dim)
  test_labels    : np.ndarray, shape (N_test,)

  For text, additionally:
    train_token_ids      : np.ndarray, shape (N_train, seq_len)  — integer token IDs
    train_attention_mask : np.ndarray, shape (N_train, seq_len)  — 1 for real tokens, 0 for padding
    test_token_ids       : np.ndarray, shape (N_test,  seq_len)
    test_attention_mask  : np.ndarray, shape (N_test,  seq_len)
"""

import numpy as np
import torch
import h5py
from tabicl import TabICLClassifier

from pal_pooling.config import ImagePALConfig, TextPALConfig
from pal_pooling.pal_pooler import IterativePALPooler

# ---------------------------------------------------------------------------
# Image config
# ---------------------------------------------------------------------------
image_cfg = ImagePALConfig(
    # -- grouping & weighting
    patch_size=16,
    patch_group_sizes=[16, 4, 1],       # coarse → medium → individual patches
    temperature=[0.5],
    weight_method="js_div",
    ridge_alpha=[1e4],
    # -- data flow
    batch_size=1000, #Unused due to use_random_subsampling=True 
    max_query_rows=600_000,
    use_random_subsampling=True,
    normalize_features=False,
    # -- model selection
    train_val_fraction=0.5,
    model_selection="validation_accuracy",
    # -- TabICL
    tabicl_n_estimators=1,
    tabicl_pca_dim=128,
    gpu_ridge=False,
)

# ---------------------------------------------------------------------------
# Text config
# ---------------------------------------------------------------------------
text_cfg = TextPALConfig(
    # -- grouping & weighting
    text_group_modes=["none", "none", "none"],  # three stages, individual tokens
    temperature=[1.0],
    weight_method="js_div",
    ridge_alpha=[1e4],
    # -- data flow
    batch_size=1000, #Unused due to use_random_subsampling=True 
    max_query_rows=600_000,
    use_random_subsampling=True,
    normalize_features=False,
    length_importance_weight_basis="full_length",
    # -- model selection
    train_val_fraction=0.5,
    model_selection="validation_accuracy",
    # -- TabICL
    tabicl_n_estimators=1,
    tabicl_pca_dim=128,
    gpu_ridge=False,
)

# ---------------------------------------------------------------------------
# Image usage
# ---------------------------------------------------------------------------
SEED = 42

# Load butterfly dataset (small, ~5k train / ~1.3k test)
print("[image] Loading butterfly features...")

butterfly_train = torch.load(
    "/scratch/hermanb/temp_datasets/extracted_features/butterfly_train_dinov3_patch_features.pt",
    map_location="cpu"
)
butterfly_test = torch.load(
    "/scratch/hermanb/temp_datasets/extracted_features/butterfly_test_dinov3_patch_features.pt",
    map_location="cpu"
)

train_patches = butterfly_train["features"].numpy()  # (5200, 256, 768)
train_labels = butterfly_train["labels"].numpy()      # (5200,)
test_patches = butterfly_test["features"].numpy()     # (1299, 256, 768)
test_labels = butterfly_test["labels"].numpy()        # (1299,)

print(f"  train: {train_patches.shape}, labels: {train_labels.shape}")
print(f"  test:  {test_patches.shape},  labels: {test_labels.shape}")

tabicl = TabICLClassifier(n_estimators=image_cfg.tabicl_n_estimators, random_state=SEED)
image_pooler = IterativePALPooler(tabicl=tabicl, refinement_cfg=image_cfg, seed=SEED, verbose=False)

image_pooler.fit(train_patches, train_labels)

# transform returns the quality-weighted pooled embeddings for the best stage
train_embeddings = image_pooler.transform(train_patches)
test_embeddings  = image_pooler.transform(test_patches)

acc, auroc = image_pooler.score_tabicl(test_patches, test_labels)
print(f"[image] acc={acc:.4f}  auroc={auroc:.4f}")

# ---------------------------------------------------------------------------
# Text usage
# ---------------------------------------------------------------------------
# Load IMDB dataset (50k samples, we'll use first 1000 for quick testing)
print("[text] Loading IMDB features...")

imdb_h5_path = "/scratch/hermanb/temp_datasets/extracted_features/imdb/electra/imdb_electra_features.h5"
with h5py.File(imdb_h5_path, "r") as f:
    # Load subset for faster testing (first 1000 samples)
    n_samples = 1000
    n_train = 800
    n_test = n_samples - n_train

    # Load and convert from float16 to float32
    train_patches = np.array(f["token_embeddings"][:n_train], dtype=np.float32)
    train_labels = np.array(f["targets"][:n_train], dtype=np.int64)
    train_token_ids = np.array(f["token_ids"][:n_train], dtype=np.int32)
    train_attention_mask = np.array(f["attention_masks"][:n_train], dtype=np.bool_)

    test_patches = np.array(f["token_embeddings"][n_train:n_samples], dtype=np.float32)
    test_labels = np.array(f["targets"][n_train:n_samples], dtype=np.int64)
    test_token_ids = np.array(f["token_ids"][n_train:n_samples], dtype=np.int32)
    test_attention_mask = np.array(f["attention_masks"][n_train:n_samples], dtype=np.bool_)

print(f"  train: {train_patches.shape}, labels: {train_labels.shape}")
print(f"  test:  {test_patches.shape},  labels: {test_labels.shape}")

tabicl = TabICLClassifier(n_estimators=text_cfg.tabicl_n_estimators, random_state=SEED)
text_pooler = IterativePALPooler(tabicl=tabicl, refinement_cfg=text_cfg, seed=SEED,
                                  modality="text", verbose=False)

text_pooler.fit(
    train_patches, train_labels,
    token_ids=train_token_ids,
    attention_mask=train_attention_mask,
)

train_embeddings = text_pooler.transform(
    train_patches, token_ids=train_token_ids, attention_mask=train_attention_mask
)
test_embeddings = text_pooler.transform(
    test_patches, token_ids=test_token_ids, attention_mask=test_attention_mask
)

acc, auroc = text_pooler.score_tabicl(
    test_patches, test_labels,
    query_token_ids=test_token_ids, query_attention_mask=test_attention_mask,
)
print(f"[text]  acc={acc:.4f}  auroc={auroc:.4f}")
