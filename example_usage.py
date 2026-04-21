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

# load your data here, e.g.:
#   train_patches, train_labels = ...
#   test_patches,  test_labels  = ...

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
# load your data here, e.g.:
#   train_patches, train_labels   = ...   (shape: N x T x D, where T = seq_len)
#   train_token_ids               = ...   (shape: N x T, integer token IDs)
#   train_attention_mask          = ...   (shape: N x T, 1=real token, 0=padding)
#   test_patches, test_labels     = ...
#   test_token_ids, test_attn_mask = ...

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
