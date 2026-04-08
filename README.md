# PALPooling — Pseudo Attention Label Pooling

Runs PAL (Pseudo Attention Label) experiments: iterative multi-scale quality-weighted
patch pooling evaluated against mean-pool and CLS-token baselines, with optional
attention-pooling upper bound and patch-quality visualisations.

## File structure

```
pal_pooling/
    pal_experiment.py        ← CLI entry point + experiment orchestration
    demo.py                  ← minimal demo (PAL vs. mean-pool on butterfly)
    config.py                ← dataclasses + CLI arg parser (parse_args)
    data_loading.py          ← dataset loaders: butterfly, RSNA
    patch_pooling.py         ← core NumPy/sklearn pooling algorithms
    pal_pooler.py            ← PALPooler / IterativePALPooler (sklearn-style API)
    attention_pooling.py     ← learnable attention pooling head + training loop
    frozen_tabicl.py         ← frozen TabICL backbone for episodic training
    patch_visualisation.py   ← matplotlib heatmap generation
    plot_n_train_sweep.py    ← utility: plot sweep_results.json
    __init__.py              ← re-exports public API
```

---

## Prerequisites

### Python dependencies

- `tabicl` — `TabICLClassifier`
- `sklearn` — `PCA`, `Ridge`, `StandardScaler`
- `torch`, `PIL`, `matplotlib`, `numpy`, `tqdm`, `joblib`

### Required files

| Path | Description |
|------|-------------|
| `<features-dir>/butterfly_train_dinov3_patch_features.pt` | Pre-extracted DINOv2 patch features — `{"features": float16 [N, P, D], "labels": int64 [N]}` |
| `<features-dir>/butterfly_test_dinov3_patch_features.pt` | Same for the test split |
| `<features-dir>/butterfly_train_dinov3_features.pt` | CLS token embeddings (optional; used for CLS-token baseline) |
| `<dataset-path>/Training_set.csv` | CSV with `filename` + `label` columns (needed for visualisation only) |
| `<dataset-path>/train/<filename>` | Raw training images (needed for visualisation only) |

Default paths (set in `config.py`):
- `FEATURES_DIR = /scratch/hermanb/temp_datasets/extracted_features`
- `BUTTERFLY_DATASET_PATH = /project/aip-rahulgk/hermanb/datasets/butterfly-image-classification`

Typical feature dimensions: `N ≈ 4800 (train) / 1200 (test)`, `P = 196` patches per image
(224 px image, 16 px patches), `D = 768` (DINOv2 ViT-B/16).

---

## Execution flow

### 1. Baselines

Mean-pool all patches per image → optional PCA(128) → TabICL accuracy (**mean-pool baseline**).
If CLS-token `.pt` files are present, a separate PCA + TabICL pass gives the **CLS-token baseline**.

### 2. Iterative PAL refinement (`--refine`)

`IterativePALPooler` chains one `PALPooler` per entry in `--patch-group-sizes` (e.g. `16 4 1`).
Each stage:

1. **Group** original DINO patches at `group_size` (must be a perfect square): `[N, P, D]` → `[N, P', D]`.
2. **Score** patch groups via TabICL fitted on the current support — produces per-group quality logits.
3. **Repool**: softmax-weighted average of raw (pre-PCA) grouped features → `[N, D]`.
4. **Re-fit PCA + Ridge**: a Ridge model is trained to predict quality logits from grouped features, then used to repool again. The Ridge model is saved as `ridge_quality_model_iter_{k}_g{group_size}.joblib`.
5. The refined projected support is passed to the next stage.

After all stages, accuracy is reported for each stage and a comparison table is printed.

### 3. Attention pooling (`--attn-pool` / `--attn-pool-only`)

Trains a learnable `AttentionPoolingHead` on the patch tensor with a frozen TabICL episodic objective.
Post-training evaluation: pool → optional PCA → TabICL (matching the other baselines).
Results saved to `attn_pool_results.json`.

### 4. Visualisation (`--n-sample N`)

When `--n-sample > 0`, per-image figures are saved at each stage showing:

| Panel | Content |
|-------|---------|
| 1 | Original image |
| 2 | P(true class) per patch — RdYlGn heatmap |
| 3–5 | Pooling weights for `correct_class_prob`, `entropy`, `kl_div` |
| 6 | Ridge pooling weights (post-refinement figures only) |

The active `--weight-method` panel is marked ★.

---

## Pooling weight methods (`--weight-method`)

Input: `dist [P, n_classes]` from TabICL `predict_proba`, temperature `T`, optional `class_prior`.

```
correct_class_prob  (default, supervised)
  l_i = ln(dist[i, true_label])
  w   = softmax(l / T)

entropy  (label-agnostic)
  H_i = -Σ_c dist[i,c] log dist[i,c]
  l_i = ln(1 - H_i / ln(C))
  w   = softmax(l / T)

kl_div  (label-agnostic, imbalance-aware; requires class_prior)
  KL_i = Σ_c dist[i,c] ln(dist[i,c] / prior_c)
  l_i  = ln(KL_i / max_KL)
  w    = softmax(l / T)
```

`T → ∞` → uniform (mean pool). `T → 0` → all weight on the best patch.

`entropy` and `kl_div` are label-agnostic and work for the AoE class (see below).

---

## Absence-of-evidence (AoE) class (`--aoe-class`)

For datasets with a "no finding" class, true-class confidence is a poor quality signal.

- **`--aoe-handling filter`** (default): AoE images excluded from TabICL scoring and Ridge fitting. Still included in the support and pooled at inference.
- **`--aoe-handling entropy`**: AoE images included but scored with `entropy` regardless of `--weight-method`.

---

## CLI reference

```
python pal_pooling/pal_experiment.py [OPTIONS]
```

**Dataset**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `butterfly` | Dataset: `butterfly` or `rsna` |
| `--backbone` | `rad-dino` | Backbone features to load for RSNA (`rad-dino` or `dinov3`); ignored for butterfly |
| `--features-dir` | see `config.py` | Directory with `.pt` feature files |
| `--dataset-path` | see `config.py` | Dataset root (CSV + images; needed for visualisation) |
| `--n-train` | `None` | Limit support set to this many training images |
| `--balance-train` | `False` | Undersample majority classes in the training set |
| `--balance-test` | `False` | Undersample majority classes in the test set |

**Evaluation**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-estimators` | `1` | TabICL ensemble size |
| `--pca-dim` | `128` | PCA output dimension |
| `--no-pca` | `False` | Disable PCA; use full 768-D embeddings |
| `--seed` | `42` | RNG seed for splits, PCA, TabICL |
| `--output-dir` | `patch_quality_results/` | Root output directory |

**PAL refinement**

| Argument | Default | Description |
|----------|---------|-------------|
| `--refine` | `False` | Run iterative multi-scale PAL refinement |
| `--patch-group-sizes` | `1` | Ordered group sizes per stage (perfect squares, e.g. `16 4 1`) |
| `--patch-size` | `16` | Base patch size in pixels |
| `--weight-method` | `correct_class_prob` | Pooling weight method: `correct_class_prob`, `entropy`, `kl_div` |
| `--temperature` | `1.0` | Softmax temperature (one value broadcast to all stages, or one per stage) |
| `--ridge-alpha` | `1.0` | Ridge regularisation strength (broadcast or per-stage) |
| `--normalize-features` | `False` | StandardScaler on patches before Ridge fitting |
| `--batch-size` | `1000` | Images per TabICL call during refinement |
| `--max-query-rows` | `None` | Cap on patch-group rows forwarded through TabICL |
| `--use-random-subsampling` | `False` | Random subsampling of patch-group rows for Ridge fitting |
| `--gpu-ridge` | `False` | Solve Ridge regression on GPU (requires CUDA) |
| `--aoe-class` | `None` | Absence-of-evidence class (index or name) |
| `--aoe-handling` | `filter` | AoE handling: `filter` or `entropy` |

**Attention pooling**

| Argument | Default | Description |
|----------|---------|-------------|
| `--attn-pool` | `False` | Train attention pooling head alongside PAL stages |
| `--attn-pool-only` | `False` | Skip PAL; only train and evaluate the attention head |
| `--attn-steps` | `500` | Training steps |
| `--attn-lr` | `1e-3` | AdamW learning rate |
| `--attn-max-step-samples` | `512` | Max training rows per step |
| `--attn-num-queries` | `1` | Learnable query vectors (1 = CLS-like) |
| `--attn-num-heads` | `8` | Attention heads (must divide embed_dim=768) |
| `--device` | `auto` | Torch device: `auto`, `cuda`, or `cpu` |

**Visualisation & sweeps**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-sample` | `0` | Images to visualise per split; `0` skips visualisation |
| `--post-refinement-viz` | `False` | Only produce post-refinement figures (with Ridge weight panel) |
| `--n-train-sweep` | `None` | Run one experiment per value and write `sweep_results.json` (mutually exclusive with `--n-train`) |

### Example: three-stage PAL refinement

```bash
python pal_pooling/pal_experiment.py \
    --refine \
    --patch-group-sizes 16 4 1 \
    --temperature 1.0 \
    --ridge-alpha 1e3 \
    --max-query-rows 300000 \
    --use-random-subsampling \
    --weight-method kl_div \
    --output-dir results/pal_3stage
```

### Example: n-train sweep

```bash
python pal_pooling/pal_experiment.py \
    --refine --patch-group-sizes 16 4 1 \
    --ridge-alpha 1e3 --use-random-subsampling \
    --n-train-sweep 500 1000 2000 4000 \
    --output-dir results/sweep
```

---

## Key functions

**`patch_pooling.py`**

| Function | Purpose |
|----------|---------|
| `group_patches(patches, group_size)` | `[N, P, D]` → `[N, P', D]` by mean-pooling spatially adjacent patches |
| `compute_patch_pooling_weights(dist, true_label, temperature, weight_method, class_prior)` | Softmax pooling weights `[P]` from TabICL probabilities |
| `compute_patch_quality_logits(...)` | Pre-softmax quality logits used as Ridge regression targets |
| `refine_dataset_features(train_patches, train_labels, support, pca, ...)` | Full single-stage refinement; returns `(refined_support, new_pca, ridge_model, ...)` |

**`pal_pooler.py`**

| Class/Function | Purpose |
|----------------|---------|
| `PALPooler` | Single-stage sklearn-style pooler: `fit(patches, labels)` → `transform(patches)` |
| `IterativePALPooler` | Chains multiple `PALPooler` stages; `fit` + `score_tabicl(query_patches, query_labels)` |

**`data_loading.py`**

| Function | Purpose |
|----------|---------|
| `ButterflyPatchDataset` | Loads pre-extracted DINOv3 patch features from `.pt` files |
| `_load_features(dataset_cfg, seed)` | Dispatcher: loads patch features + optional CLS features for butterfly or RSNA |

**`pal_experiment.py`**

| Function | Purpose |
|----------|---------|
| `_compute_accuracy_from_features(support, labels, query_features, query_labels, ...)` | TabICL accuracy + AUROC from pre-projected features |
| `_run_attn_only(train_patches, ..., attn_cfg, seed, cfg)` | Train attention head, evaluate post-hoc, save `attn_pool_results.json`; returns result dict |
| `_make_stage_callback(cfg, ...)` | Builds the per-stage callback for `IterativePALPooler.fit` (viz, eval, Ridge save) |
| `run_pal_experiment(cfg)` | Top-level orchestrator: baselines → PAL stages → summary table → `results.json` |
| `run_n_train_sweep(cfg)` | Loops `run_pal_experiment` over `cfg.run.n_train_sweep`; writes `sweep_results.json` |
