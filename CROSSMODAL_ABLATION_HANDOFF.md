# Cross-Modal Interaction Ablation — Reproduction Handoff

This document describes how to rebuild our cross-modal interaction ablation on **your**
datasets, using the same baselines and preprocessing we used. It is written so that an
agent can implement it directly. Nothing here is dataset-specific — you supply the feature
extraction for your data; the conditions and the fusion/ablation logic stay identical.

**Goal.** For each dataset, produce one number per *condition* (accuracy and AUROC), where
the conditions measure how much **cross-modal interaction inside TabICL** contributes. The
`fused` condition is the full multimodal model; two ablations disable cross-modal interaction
in two different ways. The gap `fused − ablation` is the quantity of interest.

> **Sanity check (do this first).** The `fused` condition is exactly our fully-fused
> multimodal setup. **Its numbers must line up with your Table 3 fully-fused results**
> (within seed-to-seed noise). If they don't, a preprocessing/hyperparameter mismatch is the
> cause — fix that before trusting the ablation columns. See *Matching Table 3* below.

---

## 1. The pipeline (per dataset)

For each sample you need, per modality present:

- **Image** → a single CLS-token embedding, `[N, D]`. (Use the *same* image backbone you used
  for Table 3 — for us that was DINOv3.)
- **Text** → token embeddings `[N, T, D]` plus an attention mask `[N, T]`. (Same text backbone
  as Table 3 — for us ELECTRA.)
- **Tabular** → the raw tabular columns `[N, F_raw]`.

Then build the per-modality **feature blocks**:

1. **Image block** = CLS embedding, then **PCA** to `pca_dim`. → `[N, pca_dim]`
2. **Text block** = **mean-pool the token embeddings, excluding the CLS token (position 0) and
   padding** (masked mean over valid, non-CLS positions), then **PCA** to `pca_dim`.
   → `[N, pca_dim]`
3. **Tabular block** = vectorize mixed/missing columns (we used skrub `TableVectorizer`, then
   `np.nan_to_num`). **No PCA.** → `[N, F]`

Fit PCA and the tabular vectorizer **on train only**; transform test with the fitted objects.
Use a fixed `random_state`/seed for PCA so runs are reproducible.

**Concatenate the blocks into one matrix in a fixed, contiguous order: `image, text, tabular`**
(omit any modality the dataset lacks). This contiguous layout is what makes the ablation in §4
work, so keep the order consistent everywhere.

Downstream classifier is `TabICLClassifier` for every condition.

---

## 2. Conditions to produce

Run only the conditions whose modalities exist for the dataset.

| Condition          | Input to TabICL                                   | What it measures |
|--------------------|---------------------------------------------------|------------------|
| `tabular_only`     | tabular block                                     | unimodal reference |
| `image_only`       | image block                                       | unimodal reference |
| `text_only`        | text block                                        | unimodal reference |
| `fused`            | **all blocks concatenated** → stock TabICL        | full cross-modal fusion (**= Table 3**) |
| `fused_ablated`    | same concatenated matrix, **cross-modal feature attention masked** (§4, Method 1) | interaction removed inside one model |
| `ensemble_mean`    | one TabICL per modality, **probabilities averaged** (§5, Method 2) | interaction removed via independent models |
| `ensemble_geomean` | *(optional)* per-modality probabilities combined by geometric mean | naive-Bayes-style variant |

Notes:
- `fused` / `fused_ablated` / the ensemble only make sense with **≥2 modalities**. With one
  modality, just report the unimodal number.
- If asked to pick one ensemble rule, use **`ensemble_mean`** (arithmetic mean). Geometric mean
  assumes conditional independence across modalities, which is violated here and reintroduces an
  implicit interaction prior — it's a poorer "no-interaction" baseline. Keep `ensemble_geomean`
  only as a diagnostic.

---

## 3. Hyperparameters — set these to match Table 3

These are the knobs. **Every one that affects the model must match the value you used to
produce Table 3**, or the `fused` sanity check will fail. Our script defaults are listed only
so you know what to override — they are *not* necessarily the paper values.

| Knob | What it controls | Our script default | What to set it to |
|------|------------------|--------------------|-------------------|
| **image backbone** | CLS-embedding extractor | DINOv3 | **Exactly the image backbone used for your Table 3 embeddings.** |
| **text backbone** | token-embedding extractor | ELECTRA | **Exactly the text backbone used for your Table 3 embeddings.** |
| **`pca_dim`** | PCA components per *non-tabular* modality | `128` | **Your Table 3 value.** Same value applied to image and text blocks. Tabular is never PCA'd. |
| **`n_estimators`** | TabICL ensemble size | `1` | **Your Table 3 value.** (Our script default of 1 is for speed — override it.) Use the same value for *every* condition, including each per-modality model in the ensemble. |
| **`seeds`** | seeds averaged over | `[42]` | **The same seed set you averaged in Table 3.** Report mean ± SE over these seeds. |
| **train/test split & sizes** | which rows are support vs. query | full set | **Identical split and sizes to Table 3.** Do not subsample (no `n_train`/`max_*` limits) unless Table 3 did. |
| **tabular preprocessing** | vectorizing mixed/missing columns | skrub `TableVectorizer` + `nan_to_num` | Whatever you used for Table 3's tabular features. |
| **`modalities`** | which non-tabular modality to include | `all` | `all` for the full experiment; `image` (tab+img) or `text` (tab+text) if you also want those restricted runs as *separate* experiments. |

Other fixed choices we made (keep them unless Table 3 differs): PCA is fit on train only; text
mean-pool **excludes CLS and padding**; no `StandardScaler` before PCA; features cast to
`float32`; `TabICLClassifier(random_state=seed)`.

**If `fused` doesn't match Table 3**, check in this order: (1) backbones/embeddings identical,
(2) `pca_dim` identical, (3) `n_estimators` identical, (4) same split/sizes, (5) same seeds,
(6) same tabular vectorization, (7) block concatenation order and PCA-fit-on-train-only.

---

## 4. Method 1 — blocking cross-modal interaction inside TabICL (`fused_ablated`)

This uses our TabICL fork that adds a `modality_sizes` parameter to `TabICLClassifier`. It
restricts feature-column attention so columns attend only within their own modality; modalities
still combine at the final CLS pooling. Full details live in `MODALITY_ABLATION_USAGE.md` in the
fork. The essentials for your agent:

**Requirement.** You must be running the TabICL fork whose `TabICLClassifier` accepts
`modality_sizes`. Stock/upstream TabICL will reject the kwarg. (For us it's the editable install
in the `aditya_tabicl` environment.) All other conditions run on any TabICL build.

**Usage.** Pass the contiguous per-modality block sizes, in the **same order as your fused
concatenation** (`image, text, tabular`), summing to the number of columns **before** any
internal filtering:

```python
from tabicl import TabICLClassifier

# fused matrix columns laid out as [image | text | tabular]
modality_sizes = [image_dim, text_dim, tab_dim]   # e.g. [128, 128, F]; must sum to X.shape[1]

# ablated: cross-modal feature interaction masked
clf = TabICLClassifier(n_estimators=N_EST, random_state=seed, modality_sizes=modality_sizes)
clf.fit(X_fused_train, y_train)
proba_ablated = clf.predict_proba(X_fused_test)

# baseline `fused`: identical call but modality_sizes=None (the default)
clf_full = TabICLClassifier(n_estimators=N_EST, random_state=seed)
clf_full.fit(X_fused_train, y_train)
proba_fused = clf_full.predict_proba(X_fused_test)
```

**Rules and gotchas (tell the agent explicitly):**
- `modality_sizes` is a list of **contiguous** blocks; a modality is one block. Sizes must sum to
  `X_fused_train.shape[1]` **before** TabICL's internal unique-feature filter. A mismatch raises
  `ValueError`.
- The list order **must** match the physical column order of your fused matrix. If you drop a
  modality (e.g. tab+text only), drop its entry too — e.g. `[text_dim, tab_dim]`.
- **A trained checkpoint is required for the ablation to have any effect.** On an untrained model
  the attention path is zero-initialized and `fused_ablated == fused`. `TabICLClassifier` loads
  the pretrained checkpoint by default, so this is satisfied — just don't swap in random weights.
- CLS tokens are the fusion point: the *prediction* still uses all modalities; only cross-modal
  **feature–feature** interaction is removed.
- The masked path disables Flash-Attention, so `fused_ablated` runs somewhat slower than `fused`.
  Expected, not a bug.
- After `fit`, the effective (post-filter) sizes are on
  `clf.ensemble_generator_.modality_sizes_` — useful for verifying the split survived filtering.

---

## 5. Method 2 — late-fusion ensemble (`ensemble_mean`)

No fork needed; stock TabICL. Fit **one independent `TabICLClassifier` per modality block** on
that block's features alone, get `predict_proba` for each, then combine the probability matrices.
The modalities never share a forward pass, so cross-modal interaction is removed at the decision
level.

```python
import numpy as np

# per-modality probabilities on the test set (reuse the *_only classifiers you already fit)
probas = [proba_image, proba_text, proba_tabular]   # each [N_test, n_classes]
stacked = np.stack(probas, axis=0)                  # [M, N_test, n_classes]

# arithmetic mean (recommended single rule)
ens_mean = stacked.mean(axis=0)
pred = ens_mean.argmax(axis=1)

# geometric mean (optional diagnostic): mean of log-probs, renormalized
geo = np.exp(np.log(np.clip(stacked, 1e-12, None)).mean(axis=0))
geo /= geo.sum(axis=1, keepdims=True)
```

Use the **same `n_estimators` and `seed`** for each per-modality model as everywhere else. The
per-modality models are literally the `image_only` / `text_only` / `tabular_only` conditions, so
you can reuse their probabilities rather than refitting.

---

## 6. Output schema (so our plotting/table tooling can read it)

If you want to reuse our plotting/table script, emit results as a **JSON list of records**, one
record per seed. Each record:

```json
{
  "seed": 42,
  "args": {"dataset": "<name>", "modalities": "all", "n_estimators": N, "pca_dim": 128, "seed": 42},
  "dataset_info": {
    "n_train": 0, "n_test": 0, "tab_dim": 0, "n_classes": 0,
    "modality_order": ["image", "text", "tabular"],
    "modality_sizes": [128, 128, 0]
  },
  "results": {
    "tabular_only":     {"acc": 0.0, "auroc": 0.0, "feat_dim": 0},
    "image_only":       {"acc": 0.0, "auroc": 0.0, "feat_dim": 0},
    "text_only":        {"acc": 0.0, "auroc": 0.0, "feat_dim": 0},
    "fused":            {"acc": 0.0, "auroc": 0.0, "feat_dim": 0},
    "fused_ablated":    {"acc": 0.0, "auroc": 0.0, "feat_dim": 0, "modality_sizes": [128,128,0]},
    "ensemble_mean":    {"acc": 0.0, "auroc": 0.0, "feat_dim": null, "members": ["image_only","text_only","tabular_only"]},
    "ensemble_geomean": {"acc": 0.0, "auroc": 0.0, "feat_dim": null, "members": ["image_only","text_only","tabular_only"]}
  }
}
```

AUROC: binary → `roc_auc_score(y, proba[:, 1])`; multiclass → `roc_auc_score(y, proba,
multi_class="ovr", average="macro")`.

The table view collapses this to datasets × conditions with the metric in each cell, which is
what to send back for comparison against Table 3.

---

## 7. Checklist

- [ ] Same image/text backbones and embeddings as Table 3.
- [ ] `pca_dim`, `n_estimators`, `seeds`, train/test split all match Table 3.
- [ ] Blocks concatenated as `image, text, tabular`; PCA fit on train only; text mean-pool
      excludes CLS + padding.
- [ ] `fused` ≈ Table 3 fully-fused (per-dataset, within seed noise). **Do not proceed until this holds.**
- [ ] `fused_ablated` uses the `modality_sizes` fork; sizes sum to the fused column count and
      match block order; trained checkpoint in use.
- [ ] `ensemble_mean` fuses independent per-modality TabICL probabilities by arithmetic mean.
- [ ] Results emitted per seed in the schema above; report mean ± SE across seeds.
