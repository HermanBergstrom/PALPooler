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

  For contextual (side-feature) variants:
    train_context_features : np.ndarray, shape (N_train, D_ctx)  — image only; auto-computes P(Y|X_tab)
    train_tabular_probs    : np.ndarray, shape (N_train, n_cls)  — text only; pre-computed P(Y|X_tab)
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
    max_query_rows=500_000,
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
    max_query_rows=500_000,
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

# ---------------------------------------------------------------------------
# Contextual image usage  (tabular/side features passed at fit time)
# ---------------------------------------------------------------------------
# context_features : np.ndarray, shape (N, D_ctx)
#   Any per-sample tabular features (e.g. metadata, handcrafted descriptors).
#   The pooler uses them to compute P(Y|X_tab), which is used as a prior for
#   the divergence-based patch weighting.  When train_val_fraction > 0 (the
#   default), a fresh TabICL is fit once per stage on that stage's training
#   fold only, keeping the val-fold out-of-support.  Without a train/val
#   split, a single TabICL is fit on the full dataset before all stages.
#   Only needed at fit time — transform / score_tabicl are unchanged.

# load your data here, e.g.:
#   train_patches, train_labels     = ...
#   test_patches,  test_labels      = ...
#   train_context_features          = ...   (shape: N_train x D_ctx)
#   test_context_features           = ...   (shape: N_test  x D_ctx)  # not used — shown for clarity

tabicl = TabICLClassifier(n_estimators=image_cfg.tabicl_n_estimators, random_state=SEED)
ctx_image_pooler = IterativePALPooler(tabicl=tabicl, refinement_cfg=image_cfg, seed=SEED, verbose=False)

ctx_image_pooler.fit(train_patches, train_labels, context_features=train_context_features)

train_embeddings = ctx_image_pooler.transform(train_patches)
test_embeddings  = ctx_image_pooler.transform(test_patches)

acc, auroc = ctx_image_pooler.score_tabicl(test_patches, test_labels)
print(f"[image+context] acc={acc:.4f}  auroc={auroc:.4f}")

# ---------------------------------------------------------------------------
# Contextual text usage  (pre-computed tabular probs passed at fit time)
# ---------------------------------------------------------------------------
# For text, auto-computation from context_features is not supported.
# Instead, supply tabular_probs directly: a (N, n_classes) array of
# P(Y|X_tab) probabilities, one row per sample, columns ordered by class.
# These can come from any external classifier trained on your tabular features.
#
# load your data here, e.g.:
#   train_patches, train_labels        = ...
#   train_token_ids, train_attn_mask   = ...
#   test_patches,  test_labels         = ...
#   test_token_ids,  test_attn_mask    = ...
#   train_tabular_probs                = ...   (shape: N_train x n_classes)

tabicl = TabICLClassifier(n_estimators=text_cfg.tabicl_n_estimators, random_state=SEED)
ctx_text_pooler = IterativePALPooler(tabicl=tabicl, refinement_cfg=text_cfg, seed=SEED,
                                     modality="text", verbose=False)

ctx_text_pooler.fit(
    train_patches, train_labels,
    token_ids=train_token_ids,
    attention_mask=train_attention_mask,
    tabular_probs=train_tabular_probs,
)

train_embeddings = ctx_text_pooler.transform(
    train_patches, token_ids=train_token_ids, attention_mask=train_attention_mask
)
test_embeddings = ctx_text_pooler.transform(
    test_patches, token_ids=test_token_ids, attention_mask=test_attention_mask
)

acc, auroc = ctx_text_pooler.score_tabicl(
    test_patches, test_labels,
    query_token_ids=test_token_ids, query_attention_mask=test_attention_mask,
)
print(f"[text+context]  acc={acc:.4f}  auroc={auroc:.4f}")
