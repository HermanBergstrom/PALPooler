# Plan: Add Text Modality to multimodal_experiments.py

## Goal
Extend `multimodal_experiments.py` to support petfinder's 3 modalities (image, text, tabular).
Keep code modular for future text+tabular datasets (no images).

## Changes to `data_loading.py`

### 1. Add `load_text: bool = False` parameter to `_load_features`

```python
def _load_features(
    dataset_cfg: DatasetConfig,
    seed: int,
    dtype: torch.dtype = torch.float32,
    load_tabular: bool = False,
    load_text: bool = False,
) -> tuple:
```

### 2. In the petfinder branch, load text when `load_text=True`

Change the `load_petfinder_dataset` call to pass `use_local_text=load_text`:

```python
train_loader, val_loader, test_loader, metadata = load_petfinder_dataset(
    feature_source=backbone,
    use_patches=True,
    use_images=True,
    use_local_text=load_text,
    num_workers=0,
)
```

Then after existing code that builds train_patches/cls_train etc., add:

```python
if load_text:
    def _to_text_arrays(embs_t, first_pad_t):
        embs = embs_t.float().numpy()      # [N, T, D]
        first_pad = first_pad_t.numpy()    # [N]
        N_, T = embs.shape[:2]
        # Synthetic token_ids: pos 0 = 101 ([CLS]), pos 1..fp-1 = 1, pos fp.. = 0
        # This lets group_text_tokens correctly exclude [CLS] and padding.
        tok_ids = np.ones((N_, T), dtype=np.int32)
        tok_ids[:, 0] = 101  # [CLS]
        for i in range(N_):
            tok_ids[i, first_pad[i]:] = 0  # padding
        attn_mask = (np.arange(T)[None, :] < first_pad[:, None])
        return embs, tok_ids, attn_mask

    tr_embs, tr_tok, tr_mask = _to_text_arrays(train_ds.local_text_embeddings, train_ds.local_text_first_pad)
    va_embs, va_tok, va_mask = _to_text_arrays(val_ds.local_text_embeddings, val_ds.local_text_first_pad)
    te_embs, te_tok, te_mask = _to_text_arrays(test_ds.local_text_embeddings, test_ds.local_text_first_pad)

    text_train_arr = np.concatenate([tr_embs, va_embs], axis=0)
    extra_data["text_train"]          = text_train_arr
    extra_data["text_train_token_ids"]= np.concatenate([tr_tok, va_tok], axis=0)
    extra_data["text_train_attn_mask"]= np.concatenate([tr_mask, va_mask], axis=0)
    extra_data["text_cls_train"]      = text_train_arr[:, 0, :].copy()  # [CLS] embedding

    extra_data["text_test"]           = te_embs
    extra_data["text_test_token_ids"] = te_tok
    extra_data["text_test_attn_mask"] = te_mask
    extra_data["text_cls_test"]       = te_embs[:, 0, :].copy()
```

### 3. Extend subsampling blocks to handle text keys

In the `n_train` subsampling block (near end of `_load_features`), extend:
```python
for k in ["tab_train", "text_train", "text_train_token_ids", "text_train_attn_mask", "text_cls_train"]:
    if k in extra_data and extra_data[k] is not None:
        extra_data[k] = extra_data[k][sub_idx]
```

In the `n_test` subsampling block:
```python
for k in ["tab_test", "text_test", "text_test_token_ids", "text_test_attn_mask", "text_cls_test"]:
    if k in extra_data and extra_data[k] is not None:
        extra_data[k] = extra_data[k][test_sub]
```

---

## Changes to `multimodal_experiments.py`

### 1. Add import for `TextRefinementConfig`

```python
from pal_pooling.config import (
    CBIS_DDSM_DATASET_PATH, DatasetConfig, DVM_DATASET_PATH, FEATURES_DIR,
    PAD_UFES_DATASET_PATH, PETFINDER_DATASET_PATH, RefinementConfig, TextRefinementConfig,
)
```

### 2. Update `_load_dataset` return type to include `extra_data`

Change signature and body:
```python
def _load_dataset(...) -> tuple[..., dict]:
    _SUPPORTS_TEXT = {"petfinder"}
    dataset_cfg = DatasetConfig(...)
    (..., extra_data) = _load_features(
        dataset_cfg, seed=seed,
        load_tabular=True,
        load_text=(dataset_name in _SUPPORTS_TEXT),
    )
    tab_train = extra_data["tab_train"]
    tab_test  = extra_data["tab_test"]
    # n_train post-hoc subsampling (also subsample text keys in extra_data)
    if n_train is not None and n_train < len(train_labels):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(train_labels), size=n_train, replace=False)
        idx.sort()
        train_patches = train_patches[idx]
        train_labels  = train_labels[idx]
        cls_train     = cls_train[idx]
        tab_train     = tab_train[idx]
        for k in ["text_train", "text_train_token_ids", "text_train_attn_mask", "text_cls_train"]:
            if extra_data.get(k) is not None:
                extra_data[k] = extra_data[k][idx]
    return (
        train_patches, train_labels,
        test_patches,  test_labels,
        cls_train, cls_test,
        tab_train, tab_test,
        idx_to_class,
        extra_data,   # NEW: contains text_train, text_test, etc. when available
    )
```

### 3. Add `--text-group-modes` CLI arg

```python
p.add_argument("--text-group-modes", type=str, nargs="+", default=["none"],
               choices=["none", "sentence"],
               help="Token grouping modes for the text PAL pooler (default: ['none'])")
```

### 4. Update `_run_single_seed`

Unpack extra_data:
```python
(train_patches, train_labels,
 test_patches,  test_labels,
 cls_train, cls_test,
 tab_train, tab_test,
 idx_to_class,
 extra_data) = _load_dataset(...)

has_text = extra_data.get("text_train") is not None
text_train          = extra_data.get("text_train")
text_test           = extra_data.get("text_test")
text_train_tok_ids  = extra_data.get("text_train_token_ids")
text_test_tok_ids   = extra_data.get("text_test_token_ids")
text_train_attn     = extra_data.get("text_train_attn_mask")
text_test_attn      = extra_data.get("text_test_attn_mask")
text_cls_train      = extra_data.get("text_cls_train")
text_cls_test       = extra_data.get("text_cls_test")
```

### 5. Add text conditions after existing image conditions (gated on `has_text`)

```python
if has_text:
    # ── Baseline: text mean-pool (exclude CLS at pos 0) ─────────────────
    print("\n--- mean_pool_text ---")
    valid_mask_tr = text_train_attn.copy(); valid_mask_tr[:, 0] = False
    valid_mask_te = text_test_attn.copy();  valid_mask_te[:, 0] = False
    counts_tr = valid_mask_tr.sum(axis=1, keepdims=True).clip(min=1)
    counts_te = valid_mask_te.sum(axis=1, keepdims=True).clip(min=1)
    mean_text_raw_tr = (text_train * valid_mask_tr[:, :, None]).sum(axis=1) / counts_tr
    mean_text_raw_te = (text_test  * valid_mask_te[:, :, None]).sum(axis=1) / counts_te
    mean_text_tr, mean_text_te, _ = _pca_project(mean_text_raw_tr, mean_text_raw_te, pca_dim, seed)
    _eval("mean_pool_text", mean_text_tr, mean_text_te)

    # ── Baseline: text CLS ───────────────────────────────────────────────
    print("\n--- cls_text ---")
    cls_text_tr, cls_text_te, _ = _pca_project(text_cls_train, text_cls_test, pca_dim, seed)
    _eval("cls_text", cls_text_tr, cls_text_te)

    # ── Baselines with tabular ────────────────────────────────────────────
    _eval("mean_pool_text+tab", _concat_tabular(mean_text_tr, tab_train), _concat_tabular(mean_text_te, tab_test))
    _eval("cls_text+tab",       _concat_tabular(cls_text_tr, tab_train),  _concat_tabular(cls_text_te, tab_test))

    # ── Build TextRefinementConfig ────────────────────────────────────────
    text_refinement_cfg = TextRefinementConfig(
        refine=True,
        text_group_modes=args.text_group_modes,
        temperature=args.temperature,
        weight_method=args.weight_method,
        ridge_alpha=args.ridge_alpha,
        normalize_features=args.normalize_features,
        batch_size=args.batch_size,
        max_query_rows=args.max_query_rows,
        use_random_subsampling=True,
        gpu_ridge=args.gpu_ridge,
        tabicl_n_estimators=args.n_estimators,
        tabicl_pca_dim=pca_dim,
        append_cls=False,
        use_global_prior=args.use_global_prior,
        use_attn_masking=args.use_attn_masking,
        prior=args.prior,
        model_selection=args.model_selection,
    )

    # ── Fit text PAL pooler (no tabular context) ──────────────────────────
    print("\n--- Fitting text IterativePALPooler (text-only) ---")
    pooler_text = pooler_factory(refinement_cfg=text_refinement_cfg, seed=seed, modality="text")
    pooler_text.fit(text_train, train_labels,
                    token_ids=text_train_tok_ids,
                    attention_mask=text_train_attn)
    pal_text_tr_raw = pooler_text.transform(text_train, token_ids=text_train_tok_ids, attention_mask=text_train_attn)
    pal_text_te_raw = pooler_text.transform(text_test,  token_ids=text_test_tok_ids,  attention_mask=text_test_attn)
    best_stage_txt  = pooler_text.stages_[pooler_text.best_stage_idx_]
    pal_text_pca    = best_stage_txt._pca_
    if pal_text_pca is not None:
        pal_text_tr = pal_text_pca.transform(pal_text_tr_raw).astype(np.float32)
        pal_text_te = pal_text_pca.transform(pal_text_te_raw).astype(np.float32)
    else:
        pal_text_tr = pal_text_tr_raw.astype(np.float32)
        pal_text_te = pal_text_te_raw.astype(np.float32)
    _eval("pal_text",     pal_text_tr, pal_text_te)
    _eval("pal_text+tab", _concat_tabular(pal_text_tr, tab_train), _concat_tabular(pal_text_te, tab_test))

    # ── Fit text PAL pooler (tabular context) ────────────────────────────
    print("\n--- Fitting text IterativePALPooler (tabular context) ---")
    pooler_text_ctx = pooler_factory(refinement_cfg=text_refinement_cfg, seed=seed, modality="text")
    pooler_text_ctx.fit(text_train, train_labels,
                        token_ids=text_train_tok_ids,
                        attention_mask=text_train_attn,
                        context_features=tab_train)
    pal_ctx_text_tr_raw = pooler_text_ctx.transform(text_train, token_ids=text_train_tok_ids, attention_mask=text_train_attn)
    pal_ctx_text_te_raw = pooler_text_ctx.transform(text_test,  token_ids=text_test_tok_ids,  attention_mask=text_test_attn)
    best_stage_txt_ctx  = pooler_text_ctx.stages_[pooler_text_ctx.best_stage_idx_]
    pal_ctx_text_pca    = best_stage_txt_ctx._pca_
    if pal_ctx_text_pca is not None:
        pal_ctx_text_tr = pal_ctx_text_pca.transform(pal_ctx_text_tr_raw).astype(np.float32)
        pal_ctx_text_te = pal_ctx_text_pca.transform(pal_ctx_text_te_raw).astype(np.float32)
    else:
        pal_ctx_text_tr = pal_ctx_text_tr_raw.astype(np.float32)
        pal_ctx_text_te = pal_ctx_text_te_raw.astype(np.float32)
    _eval("pal_context_text",     pal_ctx_text_tr, pal_ctx_text_te)
    _eval("pal_context_text+tab", _concat_tabular(pal_ctx_text_tr, tab_train), _concat_tabular(pal_ctx_text_te, tab_test))

    # ── Combined image + text conditions (4 methods × ±tab) ─────────────
    print("\n--- Combined image + text conditions ---")
    # mean+mean
    _eval("mean_pool_img+mean_pool_text",
          np.concatenate([mean_train, mean_text_tr], axis=1),
          np.concatenate([mean_test,  mean_text_te], axis=1))
    _eval("mean_pool_img+mean_pool_text+tab",
          _concat_tabular(np.concatenate([mean_train, mean_text_tr], axis=1), tab_train),
          _concat_tabular(np.concatenate([mean_test,  mean_text_te], axis=1), tab_test))
    # cls+cls
    _eval("cls_img+cls_text",
          np.concatenate([cls_train_proj, cls_text_tr], axis=1),
          np.concatenate([cls_test_proj,  cls_text_te], axis=1))
    _eval("cls_img+cls_text+tab",
          _concat_tabular(np.concatenate([cls_train_proj, cls_text_tr], axis=1), tab_train),
          _concat_tabular(np.concatenate([cls_test_proj,  cls_text_te], axis=1), tab_test))
    # pal+pal
    _eval("pal_img+pal_text",
          np.concatenate([pal_train_proj, pal_text_tr], axis=1),
          np.concatenate([pal_test_proj,  pal_text_te], axis=1))
    _eval("pal_img+pal_text+tab",
          _concat_tabular(np.concatenate([pal_train_proj, pal_text_tr], axis=1), tab_train),
          _concat_tabular(np.concatenate([pal_test_proj,  pal_text_te], axis=1), tab_test))
    # pal_context+pal_context
    _eval("pal_context_img+pal_context_text",
          np.concatenate([pal_ctx_train_proj, pal_ctx_text_tr], axis=1),
          np.concatenate([pal_ctx_test_proj,  pal_ctx_text_te], axis=1))
    _eval("pal_context_img+pal_context_text+tab",
          _concat_tabular(np.concatenate([pal_ctx_train_proj, pal_ctx_text_tr], axis=1), tab_train),
          _concat_tabular(np.concatenate([pal_ctx_test_proj,  pal_ctx_text_te], axis=1), tab_test))
```

### 6. Update `record["args"]` to include `text_group_modes`

```python
"args": {
    ...
    "text_group_modes": args.text_group_modes,
},
```

---

## Key Design Notes

- `token_ids` are synthetic (derived from `local_text_first_pad`): CLS=101, valid=1, pad=0.
  This is sufficient for `group_text_tokens` with `mode="none"` to correctly exclude [CLS] and padding.
  For `mode="sentence"`, actual [SEP] ids (102) would be needed — not supported with synthetic ids.
- `pca_dim` is the same for both image and text (shared `--pca-dim` arg, default 128).
- Text pooler hyperparams reuse image pooler args: `--temperature`, `--ridge-alpha`, `--weight-method`, etc.
  Only new arg needed: `--text-group-modes` (default `["none"]`).
- Structure is modular: all text/combined conditions are gated on `has_text`, making it easy to add
  future text+tabular datasets (just set text features in extra_data and leave train_patches as tokens).
- For future text+tabular-only datasets: add them to `_SUPPORTS_TEXT` in `_load_dataset` and extend
  `_load_features` for those datasets similarly.
