"""Feature-loading utilities for patch-quality experiments.

Dispatches to the appropriate dataset class (HDF5-backed or custom) for each
supported dataset and provides shared helpers for collecting embeddings,
balancing classes, and ELECTRA token features.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from pal_pooling.config import (
    AG_NEWS_DATASET_PATH,
    AIRBNB_DATASET_PATH,
    AIRCRAFTS_DATASET_PATH,
    COCO_DATASET_PATH,
    OPEN_IMAGES_DATASET_PATH,
    DTD_DATASET_PATH,
    CBIS_DDSM_DATASET_PATH,
    CLOTHING_DATASET_PATH,
    FAKE_JOBS_DATASET_PATH,
    HAM10000_DATASET_PATH,
    IMAGENET_EMBEDDINGS_PATH,
    IMAGENET_IMAGES_PATH,
    IMAGENET_SUBSETS,
    JIGSAW_DATASET_PATH,
    OXFORD_FLOWERS_DATASET_PATH,
    PAD_UFES_DATASET_PATH,
    PETFINDER_DATASET_PATH,
    PRODUCT_SENTIMENT_DATASET_PATH,
    SALARY_INDIA_DATASET_PATH,
    MM_IMDB_DATASET_PATH,
    WINE_REVIEWS_DATASET_PATH,
    WIKIART_DATASET_PATH,
    YELP_DATASET_PATH,
    DatasetConfig,
)


# Cache: images_dir → {pet_id: Path}.  Populated once per directory per process.
_PETFINDER_IMAGE_INDEX: dict[Path, dict[str, Path]] = {}


def _build_petfinder_image_index(images_dir: Path) -> dict[str, Path]:
    """Scan *images_dir* once and return a {pet_id: best_path} dict.

    For each pet, the image with the lowest numeric suffix is chosen
    (i.e. ``{pet_id}-1.jpg`` when it exists).  The result is cached at
    module level so subsequent calls for the same directory are O(1).
    """
    if images_dir in _PETFINDER_IMAGE_INDEX:
        return _PETFINDER_IMAGE_INDEX[images_dir]

    print(f"[info] Building petfinder image index from {images_dir} ...")
    index: dict[str, Path] = {}
    for p in images_dir.glob("*.jpg"):
        # Filename format: {pet_id}-{img_num}.jpg
        stem = p.stem                          # e.g. "000aa306a-3"
        dash  = stem.rfind("-")
        if dash == -1:
            continue
        pet_id  = stem[:dash]
        try:
            img_num = int(stem[dash + 1:])
        except ValueError:
            continue
        # Keep the entry with the smallest image number (prefer -1).
        if pet_id not in index or img_num < int(index[pet_id].stem.rsplit("-", 1)[1]):
            index[pet_id] = p

    _PETFINDER_IMAGE_INDEX[images_dir] = index
    print(f"[info] Petfinder image index built: {len(index)} pets.")
    return index


def _get_petfinder_image_paths(
    dataset_path: Path,
) -> tuple[list[Path], list[Path]]:
    """Return (train_paths, test_paths) aligned with _load_features petfinder output.

    Train paths correspond to the merged train+val splits (in that order), test paths
    to the test split — exactly matching the arrays returned by ``_load_features``.
    Each path points to the lowest-numbered image for the pet (typically ``-1.jpg``).
    """
    petfinder_dir = Path(dataset_path)
    processed_dir = petfinder_dir / "extracted_features" / "preprocessed_dinov3_local"
    images_dir    = petfinder_dir / "petfinder-adoption-prediction" / "train_images"

    index = _build_petfinder_image_index(images_dir)

    def _pet_to_path(pet_id: str) -> Path:
        if pet_id in index:
            return index[pet_id]
        return images_dir / f"{pet_id}-1.jpg"   # missing image; path kept for alignment

    train_data = torch.load(processed_dir / "train.pt", map_location="cpu", weights_only=False)
    val_data   = torch.load(processed_dir / "val.pt",   map_location="cpu", weights_only=False)
    test_data  = torch.load(processed_dir / "test.pt",  map_location="cpu", weights_only=False)

    train_paths = [_pet_to_path(pid) for pid in train_data["pet_ids"] + val_data["pet_ids"]]
    test_paths  = [_pet_to_path(pid) for pid in test_data["pet_ids"]]
    return train_paths, test_paths


def _get_dvm_image_paths(
    dataset_path: Path,
) -> tuple[list[Path], list[Path]]:
    """Return (train_paths, test_paths) for DVM dataset."""
    dvm_dir = Path(dataset_path)
    if str(dvm_dir) not in sys.path:
        sys.path.insert(0, str(dvm_dir))
    from dvm_dataset_with_dinov3 import load_dvm_dataset

    train_loader, val_loader, test_loader, _ = load_dvm_dataset(
        feature_source="dinov3_local",
        feature_dir="/scratch/hermanb/temp_datasets/extracted_features/dvm/dvm_dinov3_local_features",
        use_patches=False,
        use_images=False,
        num_workers=0,
        data_dir=dvm_dir
    )

    # Note: Like petfinder, we merge train and val
    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    test_ds  = test_loader.dataset

    train_paths = [train_ds.get_image_path(i) for i in range(len(train_ds))]
    train_paths += [val_ds.get_image_path(i) for i in range(len(val_ds))]
    test_paths = [test_ds.get_image_path(i) for i in range(len(test_ds))]
    
    return train_paths, test_paths


def _get_cbis_ddsm_image_paths(
    dataset_path: Path,
    kind: str,  # 'mass' or 'calc'
) -> tuple[list[Path], list[Path]]:
    """Return (train_paths, test_paths) for CBIS-DDSM crop JPEGs.

    Delegates to the upstream get_image_paths helper which correctly resolves
    each row's JPEG via the DICOM index embedded in the CSV path string.
    """
    cbis_module_dir = "/home/hermanb/projects/aip-rahulgk/image_icl_project/cbis-ddsm"
    if cbis_module_dir not in sys.path:
        sys.path.insert(0, cbis_module_dir)
    from cbis_ddsm import get_image_paths  # type: ignore

    return get_image_paths(kind, image_type="full", data_dir=str(dataset_path))


def _get_pad_ufes_image_paths(
    dataset_path: Path,
    seed: int = 42,
    test_fraction: float = 0.2,
) -> tuple[list[Path], list[Path]]:
    """Return (train_paths, test_paths) aligned with _load_features pad-ufes output."""
    pad_ufes_dir = Path(dataset_path)
    if str(pad_ufes_dir) not in sys.path:
        sys.path.insert(0, str(pad_ufes_dir))
    from pad_ufes_dataset import load_metadata  # type: ignore
    from sklearn.model_selection import train_test_split

    meta = load_metadata(data_dir=str(pad_ufes_dir))
    df   = meta["df"]
    df_train, df_test = train_test_split(
        df, test_size=test_fraction, random_state=seed, stratify=df["diagnostic"]
    )

    def _stem_to_path(stem: str) -> Path:
        return meta["img_index"].get(stem, pad_ufes_dir / "images" / f"{stem}.png")

    train_paths = [_stem_to_path(Path(r["img_id"]).stem) for _, r in df_train.iterrows()]
    test_paths  = [_stem_to_path(Path(r["img_id"]).stem) for _, r in df_test.iterrows()]
    return train_paths, test_paths


def _get_imagenet_image_paths(
    dataset_name: str,
    embeddings_root: Path = IMAGENET_EMBEDDINGS_PATH,
    images_root: Path = IMAGENET_IMAGES_PATH,
) -> tuple[list[Path], list[Path]]:
    """Return (train_paths, test_paths) for an ImageNet subset.

    Iteration order mirrors _load_features exactly: synsets in IMAGENET_SUBSETS
    order, files sorted lexicographically within each synset/split, matching the
    feature array row order.
    """
    synsets = IMAGENET_SUBSETS[dataset_name]
    train_paths: list[Path] = []
    test_paths:  list[Path] = []

    for synset in synsets:
        for split, paths_list in [("train", train_paths), ("val", test_paths)]:
            emb_dir = embeddings_root / synset / split
            img_dir = images_root / split / synset
            for npy_path in sorted(emb_dir.glob("*.npy")):
                paths_list.append(img_dir / (npy_path.stem + ".JPEG"))

    return train_paths, test_paths


# ---------------------------------------------------------------------------
# Generic helper for tabular + ELECTRA text datasets
# ---------------------------------------------------------------------------

def _collect_electra_tabular(ds, indices: np.ndarray):
    """Collect padded ELECTRA token embeddings and tabular features for a split.

    Args:
        ds:      A dataset with ``._electra_embs`` (list of variable-length tensors),
                 ``._electra_first_pad`` (list[int]), and ``.__getitem__`` returning
                 a dict with ``"tabular"`` and ``"target"`` keys.
        indices: 1-D int array of dataset indices to collect.

    Returns:
        patches   : [N, T_max, D]  float32  — padded token embeddings
        labels    : [N]            int64
        cls_emb   : [N, D]         float32  — CLS token (position 0)
        tab       : [N, F]         float32  — tabular features
        token_ids : [N, T_max]     int32    — synthetic: CLS=101, real=1, pad=0
        attn_mask : [N, T_max]     bool
    """
    if ds._electra_embs is None:
        raise FileNotFoundError(
            "ELECTRA embeddings not found for this dataset. "
            "Run the feature extraction script first."
        )

    N = len(indices)
    labels = ds.targets[indices].astype(np.int64)
    tab = np.stack([ds[int(i)]["tabular"].float().numpy() for i in indices], axis=0)

    first_pads = [ds._electra_first_pad[int(i)] for i in indices]
    max_len = max(first_pads)
    D = ds._electra_embs[0].shape[-1]

    patches = np.zeros((N, max_len, D), dtype=np.float32)
    for j, (orig_i, fp) in enumerate(zip(indices.tolist(), first_pads)):
        emb = ds._electra_embs[orig_i].float().numpy()
        patches[j, :len(emb)] = emb

    first_pads_arr = np.array(first_pads, dtype=np.int32)
    token_ids = np.ones((N, max_len), dtype=np.int32)
    token_ids[:, 0] = 101  # [CLS]
    for j, fp in enumerate(first_pads):
        token_ids[j, fp:] = 0  # padding
    attn_mask = np.arange(max_len)[None, :] < first_pads_arr[:, None]
    cls_emb = patches[:, 0, :].copy()

    return patches, labels, cls_emb, tab, token_ids, attn_mask


# ---------------------------------------------------------------------------
# Generic helper for HDF5-backed image datasets (aircrafts / ham10000 / oxford-flowers)
# ---------------------------------------------------------------------------

def _collect_h5_dataset(
    ds,
    indices: np.ndarray,
    with_tabular: bool = False,
    desc: str = "",
) -> tuple:
    """Collect patch/CLS embeddings for *indices* from an HDF5-backed dataset.

    The dataset's ``__getitem__`` must return a dict with keys:
      ``"image"``   : (P, D) float32 — patch embeddings
      ``"cls"``     : (D,)   float32 — CLS embedding
      ``"target"``  : scalar int
      ``"tabular"`` : (F,)   float32  (optional, only read when *with_tabular*)

    Returns (patches, cls_embs, labels, tab_arr) where tab_arr is None unless
    *with_tabular* is True and the dataset provides a ``"tabular"`` key.
    """
    from tqdm import tqdm
    patches_list, cls_list, labels_list, tab_list = [], [], [], []
    for i in tqdm(indices, desc=desc):
        sample = ds[int(i)]
        patches_list.append(sample["image"].numpy())
        cls_list.append(sample["cls"].numpy())
        labels_list.append(sample["target"].item())
        if with_tabular and "tabular" in sample:
            tab_list.append(sample["tabular"].float().numpy())
    tab_arr = np.stack(tab_list, axis=0) if tab_list else None
    return (
        np.stack(patches_list, axis=0),
        np.stack(cls_list,     axis=0),
        np.array(labels_list,  dtype=np.int64),
        tab_arr,
    )


# ---------------------------------------------------------------------------
# Feature loading dispatcher
# ---------------------------------------------------------------------------

def _load_features(
    dataset_cfg: DatasetConfig,
    seed: int,
    dtype:        torch.dtype = torch.float32,
    load_tabular: bool = False,
    load_text: bool = False,
) -> tuple:
    """Load patch/token and CLS features for the requested dataset.

    Returns:
        train_patches:   [N_train, P, D]  float32 numpy
        train_labels:    [N_train]         int64   numpy
        test_patches:    [N_test,  P, D]  float32 numpy
        test_labels:     [N_test]          int64   numpy
        cls_train:       [N_train, D] or None
        cls_test:        [N_test,  D] or None
        idx_to_class:    {int → class_name}
        train_sub_idx:   [N_train] int64 or None
        test_sub_idx:    [N_test]  int64 or None
        extra_data:      dict — modality-specific arrays.  For text datasets
                         (e.g. 'imdb') this contains:
                           'train_token_ids'      [N_train, T_max] int32
                           'train_attention_mask' [N_train, T_max] bool
                           'test_token_ids'       [N_test,  T_max] int32
                           'test_attention_mask'  [N_test,  T_max] bool
                         For image datasets this is an empty dict.
    """
    features_dir = Path(dataset_cfg.features_dir)
    train_sub_idx: Optional[np.ndarray] = None
    test_sub_idx:  Optional[np.ndarray] = None
    n_train = dataset_cfg.n_train
    extra_data: dict = {}

    if dataset_cfg.dataset == "butterfly":
        butterfly_module_dir = "/project/aip-rahulgk/image_icl_project/butterfly"
        if butterfly_module_dir not in sys.path:
            sys.path.insert(0, butterfly_module_dir)
        from butterfly_dataset import ButterflyDataset  # type: ignore

        ds = ButterflyDataset()
        train_idx, test_idx = ds.default_split()

        train_patches, cls_train, train_labels, _ = _collect_h5_dataset(
            ds, train_idx, desc="Loading Butterfly (train)")
        test_patches, cls_test, test_labels, _ = _collect_h5_dataset(
            ds, test_idx, desc="Loading Butterfly (test)")

        idx_to_class: dict[int, str] = {i: name for i, name in enumerate(ds.class_names)}
        extra_data["butterfly_train_idx"] = train_idx
        extra_data["butterfly_test_idx"]  = test_idx

        print(f"[info] Butterfly (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] Butterfly (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "rsna":
        rsna_module_dir = "/project/aip-rahulgk/image_icl_project/rsna"
        if rsna_module_dir not in sys.path:
            sys.path.insert(0, rsna_module_dir)
        from rsna_dataset import RSNADataset  # type: ignore

        ds = RSNADataset()
        train_idx, test_idx = ds.default_split()

        train_patches, cls_train, train_labels, _ = _collect_h5_dataset(
            ds, train_idx, desc="Loading RSNA (train)")
        test_patches, cls_test, test_labels, _ = _collect_h5_dataset(
            ds, test_idx, desc="Loading RSNA (test)")

        idx_to_class = {i: name for i, name in enumerate(ds.class_names)}
        extra_data["rsna_train_idx"] = train_idx
        extra_data["rsna_test_idx"]  = test_idx

        print(f"[info] RSNA (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] RSNA (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "petfinder":
        petfinder_module_dir = "/home/hermanb/projects/aip-rahulgk/image_icl_project/petfinder"
        if petfinder_module_dir not in sys.path:
            sys.path.insert(0, petfinder_module_dir)
        from petfinder_dataset import PetFinderDataset  # type: ignore
        from tqdm import tqdm

        #ds = PetFinderDataset(data_dir=Path(dataset_cfg.dataset_path), image_resize=256)
        #ds = PetFinderDataset(image_resize=256)
        ds = PetFinderDataset()
        train_idx, test_idx = ds.default_split()

        def _collect_petfinder(ds, indices, desc: str, with_tabular: bool = False, with_text: bool = False):
            patches_list, cls_list, labels_list, tab_list, text_list = [], [], [], [], []
            _warned_registers = False
            for i in tqdm(indices, desc=desc):
                sample = ds[int(i)]
                emb = sample["image"]   # (201, 768)
                if not _warned_registers:
                    print(
                        f"[info] PetFinder image embeddings have shape {tuple(emb.shape)} — "
                        "removing 4 register tokens (rows 1-4); these are likely DINOv2 features."
                    )
                    _warned_registers = True
                cls_list.append(emb[0].float().numpy())
                patches_list.append(emb[5:].float().numpy())
                labels_list.append(sample["target"].item())
                if with_tabular:
                    tab_list.append(sample["tabular"].float().numpy())
                if with_text and "electra_text" in sample:
                    text_list.append(sample["electra_text"])

            patches = np.stack(patches_list, axis=0)
            cls_emb = np.stack(cls_list,     axis=0)
            labels  = np.array(labels_list,  dtype=np.int64)
            tab_arr = np.stack(tab_list, axis=0) if tab_list else None

            text_arr = tok_ids = attn_mask = text_cls = None
            if text_list:
                first_pads = [ds._electra_first_pad[int(i)] for i in indices]
                max_len = max(first_pads)
                D_t = text_list[0].shape[-1]
                N = len(indices)
                text_arr = np.zeros((N, max_len, D_t), dtype=np.float32)
                for j, (e, fp) in enumerate(zip(text_list, first_pads)):
                    e_np = e.float().numpy()
                    text_arr[j, :len(e_np)] = e_np
                fp_arr   = np.array(first_pads, dtype=np.int32)
                tok_ids  = np.ones((N, max_len), dtype=np.int32)
                tok_ids[:, 0] = 101  # [CLS]
                for j, fp in enumerate(first_pads):
                    tok_ids[j, fp:] = 0
                attn_mask = np.arange(max_len)[None, :] < fp_arr[:, None]
                text_cls  = text_arr[:, 0, :].copy()

            return patches, cls_emb, labels, tab_arr, text_arr, tok_ids, attn_mask, text_cls

        (train_patches, cls_train, train_labels, tab_train_pf,
         text_train, train_tok_ids, train_attn_mask, text_cls_train) = _collect_petfinder(
            ds, train_idx, "Loading PetFinder (train+val)", with_tabular=load_tabular, with_text=load_text)
        (test_patches, cls_test, test_labels, tab_test_pf,
         text_test, test_tok_ids, test_attn_mask, text_cls_test) = _collect_petfinder(
            ds, test_idx, "Loading PetFinder (test)", with_tabular=load_tabular, with_text=load_text)

        if load_tabular and tab_train_pf is not None:
            extra_data["tab_train"] = tab_train_pf
            extra_data["tab_test"]  = tab_test_pf

        if text_train is not None:
            extra_data["text_train"]           = text_train
            extra_data["text_train_token_ids"] = train_tok_ids
            extra_data["text_train_attn_mask"] = train_attn_mask
            extra_data["text_cls_train"]       = text_cls_train
            extra_data["text_test"]            = text_test
            extra_data["text_test_token_ids"]  = test_tok_ids
            extra_data["text_test_attn_mask"]  = test_attn_mask
            extra_data["text_cls_test"]        = text_cls_test

        idx_to_class = {i: str(i) for i in range(ds.n_classes)}

        print(
            f"[info] PetFinder (train+val): N={len(train_labels)}  "
            f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}"
        )
        print(f"[info] PetFinder (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "dvm":
        dvm_dir = Path(dataset_cfg.dataset_path)
        if str(dvm_dir) not in sys.path:
            sys.path.insert(0, str(dvm_dir))
        from dvm_dataset_with_dinov3 import load_dvm_dataset

        train_loader, val_loader, test_loader, metadata = load_dvm_dataset(
            feature_dir="/scratch/hermanb/temp_datasets/extracted_features/dvm/dvm_dinov3_local_features",
            feature_source="dinov3_local",
            use_patches=True,
            use_images=True,
            num_workers=0,
            data_dir=dvm_dir
        )

        train_ds = train_loader.dataset
        val_ds   = val_loader.dataset
        test_ds  = test_loader.dataset

        def get_dvm_subset(ds_list, n_limit, desc="Loading DVM", with_tabular=False):
            # Combine multiple datasets (like train + val)
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
            for i in tqdm(sub_idx, desc=desc):
                # Find which dataset this index belongs to
                ds_idx = i
                for l, ds in zip(lengths, ds_list):
                    if ds_idx < l:
                        sample = ds[ds_idx]
                        break
                    ds_idx -= l
                patches.append(sample["patch_embedding"].numpy())
                cls_emb.append(sample["image_embedding"].numpy())
                labels.append(sample["target"])
                if with_tabular:
                    tab_feat.append(sample["tabular"].float().numpy())

            tab_arr = np.stack(tab_feat, axis=0) if with_tabular else None
            return np.stack(patches, axis=0), np.stack(cls_emb, axis=0), np.array(labels, dtype=np.int64), sub_idx, tab_arr

        # We merge train + val for support set, similar to Petfinder.
        # For float fractions, pass None so the generic block below resolves the count.
        _dvm_n_train = n_train if isinstance(n_train, int) else None
        train_patches, cls_train, train_labels, train_sub_idx, tab_train_dvm = get_dvm_subset(
            [train_ds, val_ds], _dvm_n_train, desc="Loading DVM (train+val)", with_tabular=load_tabular
        )

        # Test set
        test_patches, cls_test, test_labels, test_sub_idx, tab_test_dvm = get_dvm_subset(
            [test_ds], dataset_cfg.n_test, desc="Loading DVM (test)", with_tabular=load_tabular
        )
        if load_tabular:
            extra_data["tab_train"] = tab_train_dvm
            extra_data["tab_test"]  = tab_test_dvm

        target_encoder = metadata["target_encoder"]
        idx_to_class = {i: str(cls) for i, cls in enumerate(target_encoder.classes_)}

        print(f"[info] DVM (train+val): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] DVM (test): N={len(test_labels)}")
        
        # Mark n_train as None so the generic block is skipped (subsampling already done above).
        # For float fractions the generic block handles it, so only skip for int counts.
        if isinstance(n_train, int):
            n_train = None

    elif dataset_cfg.dataset == "pad-ufes":
        pad_ufes_dir = Path(dataset_cfg.dataset_path)
        if str(pad_ufes_dir) not in sys.path:
            sys.path.insert(0, str(pad_ufes_dir))
        from pad_ufes_dataset import PADUFESDataset, load_metadata  # type: ignore
        from sklearn.model_selection import train_test_split
        from tqdm import tqdm

        metadata = load_metadata(data_dir=str(pad_ufes_dir))
        df = metadata["df"]

        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df["diagnostic"]
        )

        train_ds = PADUFESDataset(df_train, metadata, use_patches=True, use_images=True)
        test_ds  = PADUFESDataset(df_test,  metadata, use_patches=True, use_images=True)

        def _collect_pad_ufes(ds, desc: str, with_tabular: bool = False):
            patches_list, cls_list, labels_list, tab_list = [], [], [], []
            for i in tqdm(range(len(ds)), desc=desc):
                sample = ds[i]
                patches_list.append(sample["patch_embedding"].numpy())
                cls_list.append(sample["image_embedding"].numpy())
                labels_list.append(sample["target"].item())
                if with_tabular:
                    tab_list.append(sample["tabular"].float().numpy())
            tab_arr = np.stack(tab_list, axis=0) if with_tabular else None
            return (
                np.stack(patches_list, axis=0),
                np.stack(cls_list,     axis=0),
                np.array(labels_list,  dtype=np.int64),
                tab_arr,
            )

        train_patches, cls_train, train_labels, tab_train_pad = _collect_pad_ufes(
            train_ds, "Loading PAD-UFES (train)", with_tabular=load_tabular)
        test_patches,  cls_test,  test_labels, tab_test_pad  = _collect_pad_ufes(
            test_ds,  "Loading PAD-UFES (test)",  with_tabular=load_tabular)
        if load_tabular:
            extra_data["tab_train"] = tab_train_pad
            extra_data["tab_test"]  = tab_test_pad

        target_encoder = metadata["target_encoder"]
        idx_to_class = {i: str(cls) for i, cls in enumerate(target_encoder.classes_)}

        print(f"[info] PAD-UFES (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] PAD-UFES (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset in ("cbis-ddsm-mass", "cbis-ddsm-calc"):
        cbis_dir = Path(dataset_cfg.dataset_path)
        cbis_module_dir = "/home/hermanb/projects/aip-rahulgk/image_icl_project/cbis-ddsm"
        if cbis_module_dir not in sys.path:
            sys.path.insert(0, cbis_module_dir)
        from cbis_ddsm import CBISDDSMDataset, load_metadata, get_image_paths  # type: ignore
        from tqdm import tqdm

        kind = "mass" if dataset_cfg.dataset == "cbis-ddsm-mass" else "calc"
        metadata = load_metadata(kind, data_dir=str(cbis_dir))
        df = metadata["df"]

        df_train = df[df["split"] == "train"].reset_index(drop=True)
        df_test  = df[df["split"] == "test"].reset_index(drop=True)

        train_ds = CBISDDSMDataset(df_train, metadata, image_type="crop",
                                   use_images=True, use_patches=True)
        test_ds  = CBISDDSMDataset(df_test,  metadata, image_type="crop",
                                   use_images=True, use_patches=True)

        # CBISDDSMDataset drops rows with missing embeddings. We need to align
        # image_paths with the rows that the dataset kept.
        # Get all image paths (before filtering), then keep only those for kept rows.
        all_train_paths, all_test_paths = get_image_paths(kind, image_type="full", data_dir=str(cbis_dir))

        # For train: match train_ds.df rows back to original df_train
        # The dataset's df is a filtered version of the input df_train
        train_image_paths = []
        test_image_paths = []

        # Build mapping: for each row in train_ds.df, find its corresponding image path.
        # Since CBISDDSMDataset already filtered df, we need to find which rows were kept
        # by matching on a unique identifier (the crop image path).

        train_image_paths = []
        for _, row in train_ds.df.iterrows():
            crop_path_str = row['cropped image file path']
            # Find this row in all_train_paths by finding its position in the original df_train
            matching_mask = df_train['cropped image file path'] == crop_path_str
            matching_pos = matching_mask.idxmax() if matching_mask.any() else -1
            if matching_pos >= 0:
                train_image_paths.append(all_train_paths[matching_pos])
            else:
                print(f"[warn] Could not find image path for crop: {crop_path_str}")
                train_image_paths.append(Path(''))

        test_image_paths = []
        for _, row in test_ds.df.iterrows():
            crop_path_str = row['cropped image file path']
            matching_mask = df_test['cropped image file path'] == crop_path_str
            matching_pos = matching_mask.idxmax() if matching_mask.any() else -1
            if matching_pos >= 0:
                test_image_paths.append(all_test_paths[matching_pos])
            else:
                print(f"[warn] Could not find image path for crop: {crop_path_str}")
                test_image_paths.append(Path(''))

        def _collect_cbis(ds, desc: str, with_tabular: bool = False):
            patches_list, cls_list, labels_list, tab_list = [], [], [], []
            for i in tqdm(range(len(ds)), desc=desc):
                sample = ds[i]
                patches_list.append(sample["patch_embedding"].numpy())
                cls_list.append(sample["image_embedding"].numpy())
                labels_list.append(sample["target"].item())
                if with_tabular:
                    tab_list.append(sample["tabular"].float().numpy())
            tab_arr = np.stack(tab_list, axis=0) if with_tabular else None
            return (
                np.stack(patches_list, axis=0),
                np.stack(cls_list,     axis=0),
                np.array(labels_list,  dtype=np.int64),
                tab_arr,
            )

        train_patches, cls_train, train_labels, tab_train_cbis = _collect_cbis(
            train_ds, f"Loading CBIS-DDSM {kind} (train)", with_tabular=load_tabular)
        test_patches,  cls_test,  test_labels, tab_test_cbis  = _collect_cbis(
            test_ds,  f"Loading CBIS-DDSM {kind} (test)",  with_tabular=load_tabular)
        if load_tabular:
            extra_data["tab_train"] = tab_train_cbis
            extra_data["tab_test"]  = tab_test_cbis

        target_encoder = metadata["target_encoder"]
        idx_to_class = {i: str(cls) for i, cls in enumerate(target_encoder.classes_)}

        print(f"[info] CBIS-DDSM {kind} (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] CBIS-DDSM {kind} (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "imdb":
        imdb_module_dir = "/project/aip-rahulgk/image_icl_project/imdb"
        if imdb_module_dir not in sys.path:
            sys.path.insert(0, imdb_module_dir)
        from imdb_dataset import load_imdb_dataset  # type: ignore

        _max_train = dataset_cfg.n_train if isinstance(dataset_cfg.n_train, int) else None
        train_loader, _val_loader, test_loader, metadata = load_imdb_dataset(
            use_token_embeddings=True,
            max_train=_max_train,
            max_val=0,       # val split is unused; skip to avoid loading ~3.9 GB of token embeddings
            max_test=dataset_cfg.n_test,
            num_workers=0,
            batch_size=1,  # we access Dataset objects directly below
        )
        train_ds = train_loader.dataset
        test_ds  = test_loader.dataset

        train_patches = train_ds.token_embeddings.float().numpy()      # [N, T_max, D]
        train_labels  = train_ds.targets.numpy().astype(np.int64)
        cls_train     = train_ds.cls_embeddings.float().numpy()        # [N, D]
        train_token_ids      = train_ds.token_ids.numpy().astype(np.int32)   # [N, T_max]
        train_attention_mask = train_ds.attention_masks.bool().numpy()       # [N, T_max]

        test_patches         = test_ds.token_embeddings.float().numpy()
        test_labels          = test_ds.targets.numpy().astype(np.int64)
        cls_test             = test_ds.cls_embeddings.float().numpy()
        test_token_ids       = test_ds.token_ids.numpy().astype(np.int32)
        test_attention_mask  = test_ds.attention_masks.bool().numpy()

        label_map    = metadata["label_map"]   # e.g. {"negative": 0, "positive": 1}
        idx_to_class = {v: k for k, v in label_map.items()}

        # Reconstruct display text from tokenization so that text.split() aligns
        # exactly with word_weights (indexed by word_ids() from the original text).
        # tokenizer.decode() on the full sequence adds spaces around punctuation,
        # producing more "words" than word_ids() has entries — causing misalignment.
        # Instead, group tokens by word_id and decode each group separately.
        from transformers import AutoTokenizer as _AutoTokenizer
        _HF_CACHE = "/scratch/hermanb/.cache/huggingface/transformers"
        _tokenizer = _AutoTokenizer.from_pretrained(
            metadata.get("model_name", "google/electra-base-discriminator"),
            cache_dir=_HF_CACHE,
        )

        def _reconstruct_words(token_ids_row: np.ndarray, token_to_word_row: np.ndarray) -> str:
            """Decode tokens grouped by word_id so text.split() aligns with word_weights."""
            from collections import defaultdict
            word_tokens: dict = defaultdict(list)
            for tok_id, wid in zip(token_ids_row.tolist(), token_to_word_row.tolist()):
                if wid >= 0:
                    word_tokens[wid].append(tok_id)
            if not word_tokens:
                return ""
            n_words = max(word_tokens) + 1
            parts = []
            for w in range(n_words):
                toks = word_tokens.get(w, [])
                if toks:
                    tok_strs = _tokenizer.convert_ids_to_tokens(toks)
                    parts.append(_tokenizer.convert_tokens_to_string(tok_strs).strip())
                else:
                    parts.append(f"[w{w}]")
            return " ".join(parts)

        print("[info] Reconstructing IMDB word-aligned texts ...", flush=True)
        train_token_to_word = train_ds.token_to_word.numpy().astype(np.int32)
        test_token_to_word  = test_ds.token_to_word.numpy().astype(np.int32)
        train_texts = [
            _reconstruct_words(train_token_ids[i], train_token_to_word[i])
            for i in range(len(train_token_ids))
        ]
        test_texts = [
            _reconstruct_words(test_token_ids[i], test_token_to_word[i])
            for i in range(len(test_token_ids))
        ]

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "train_texts":          train_texts,
            "train_token_to_word":  train_token_to_word,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
            "test_texts":           test_texts,
            "test_token_to_word":   test_token_to_word,
        }

        print(f"[info] IMDB (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] IMDB (test):  N={len(test_labels)}")

        # n_train subsampling already handled by load_imdb_dataset(max_train=...) for int counts.
        # For float fractions, defer to the generic block below.
        if isinstance(dataset_cfg.n_train, int):
            n_train = None

    elif dataset_cfg.dataset == "20news":
        news_module_dir = "/project/aip-rahulgk/image_icl_project/20news"
        if news_module_dir not in sys.path:
            sys.path.insert(0, news_module_dir)
        import importlib
        news_module = importlib.import_module("20news_dataset")  # type: ignore
        load_20news_dataset = news_module.load_20news_dataset

        _max_train = dataset_cfg.n_train if isinstance(dataset_cfg.n_train, int) else None
        train_loader, _val_loader, test_loader, metadata = load_20news_dataset(
            use_token_embeddings=True,
            max_train=_max_train,
            max_val=0,       # val split is unused; skip to avoid loading ~3.9 GB of token embeddings
            max_test=dataset_cfg.n_test,
            num_workers=0,
            batch_size=1,  # we access Dataset objects directly below
        )
        train_ds = train_loader.dataset
        test_ds  = test_loader.dataset

        train_patches = train_ds.token_embeddings.float().numpy()      # [N, T_max, D]
        train_labels  = train_ds.targets.numpy().astype(np.int64)
        cls_train     = train_ds.cls_embeddings.float().numpy()        # [N, D]
        train_token_ids      = train_ds.token_ids.numpy().astype(np.int32)   # [N, T_max]
        train_attention_mask = train_ds.attention_masks.bool().numpy()       # [N, T_max]

        test_patches         = test_ds.token_embeddings.float().numpy()
        test_labels          = test_ds.targets.numpy().astype(np.int64)
        cls_test             = test_ds.cls_embeddings.float().numpy()
        test_token_ids       = test_ds.token_ids.numpy().astype(np.int32)
        test_attention_mask  = test_ds.attention_masks.bool().numpy()

        categories = metadata.get("categories", [f"class_{i}" for i in range(metadata["num_classes"])])
        idx_to_class = {i: cat for i, cat in enumerate(categories)}

        # Reconstruct display text per word_id group (same fix as IMDB above).
        from transformers import AutoTokenizer as _AutoTokenizer
        _HF_CACHE = "/scratch/hermanb/.cache/huggingface/transformers"
        _tokenizer = _AutoTokenizer.from_pretrained(
            metadata.get("model_name", "google/electra-base-discriminator"),
            cache_dir=_HF_CACHE,
        )

        def _reconstruct_words(token_ids_row: np.ndarray, token_to_word_row: np.ndarray) -> str:
            from collections import defaultdict
            word_tokens: dict = defaultdict(list)
            for tok_id, wid in zip(token_ids_row.tolist(), token_to_word_row.tolist()):
                if wid >= 0:
                    word_tokens[wid].append(tok_id)
            if not word_tokens:
                return ""
            n_words = max(word_tokens) + 1
            parts = []
            for w in range(n_words):
                toks = word_tokens.get(w, [])
                if toks:
                    tok_strs = _tokenizer.convert_ids_to_tokens(toks)
                    parts.append(_tokenizer.convert_tokens_to_string(tok_strs).strip())
                else:
                    parts.append(f"[w{w}]")
            return " ".join(parts)

        print("[info] Reconstructing 20NEWS word-aligned texts ...", flush=True)
        train_token_to_word = train_ds.token_to_word.numpy().astype(np.int32)
        test_token_to_word  = test_ds.token_to_word.numpy().astype(np.int32)
        train_texts = [
            _reconstruct_words(train_token_ids[i], train_token_to_word[i])
            for i in range(len(train_token_ids))
        ]
        test_texts = [
            _reconstruct_words(test_token_ids[i], test_token_to_word[i])
            for i in range(len(test_token_ids))
        ]

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "train_texts":          train_texts,
            "train_token_to_word":  train_token_to_word,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
            "test_texts":           test_texts,
            "test_token_to_word":   test_token_to_word,
        }

        print(f"[info] 20NEWS (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] 20NEWS (test):  N={len(test_labels)}")

        # n_train subsampling already handled by load_20news_dataset(max_train=...) for int counts.
        # For float fractions, defer to the generic block below.
        if isinstance(dataset_cfg.n_train, int):
            n_train = None

    elif dataset_cfg.dataset == "ag_news":
        ag_news_module_dir = "/project/aip-rahulgk/image_icl_project/ag_news"
        if ag_news_module_dir not in sys.path:
            sys.path.insert(0, ag_news_module_dir)
        from ag_news_dataset import load_ag_news_dataset  # type: ignore

        _max_train = dataset_cfg.n_train if isinstance(dataset_cfg.n_train, int) else None
        train_loader, _val_loader, test_loader, metadata = load_ag_news_dataset(
            use_token_embeddings=True,
            max_train=_max_train,
            max_val=0,
            max_test=dataset_cfg.n_test,
            num_workers=0,
            batch_size=1,
        )
        train_ds = train_loader.dataset
        test_ds  = test_loader.dataset

        train_patches = train_ds.token_embeddings.float().numpy()
        train_labels  = train_ds.targets.numpy().astype(np.int64)
        cls_train     = train_ds.cls_embeddings.float().numpy()
        train_token_ids      = train_ds.token_ids.numpy().astype(np.int32)
        train_attention_mask = train_ds.attention_masks.bool().numpy()

        test_patches         = test_ds.token_embeddings.float().numpy()
        test_labels          = test_ds.targets.numpy().astype(np.int64)
        cls_test             = test_ds.cls_embeddings.float().numpy()
        test_token_ids       = test_ds.token_ids.numpy().astype(np.int32)
        test_attention_mask  = test_ds.attention_masks.bool().numpy()

        label_map    = metadata["label_map"]
        idx_to_class = {v: k for k, v in label_map.items()}

        from transformers import AutoTokenizer as _AutoTokenizer
        _HF_CACHE = "/scratch/hermanb/.cache/huggingface/transformers"
        _tokenizer = _AutoTokenizer.from_pretrained(
            metadata.get("model_name", "google/electra-base-discriminator"),
            cache_dir=_HF_CACHE,
        )

        def _reconstruct_words(token_ids_row: np.ndarray, token_to_word_row: np.ndarray) -> str:
            from collections import defaultdict
            word_tokens: dict = defaultdict(list)
            for tok_id, wid in zip(token_ids_row.tolist(), token_to_word_row.tolist()):
                if wid >= 0:
                    word_tokens[wid].append(tok_id)
            if not word_tokens:
                return ""
            n_words = max(word_tokens) + 1
            parts = []
            for w in range(n_words):
                toks = word_tokens.get(w, [])
                if toks:
                    tok_strs = _tokenizer.convert_ids_to_tokens(toks)
                    parts.append(_tokenizer.convert_tokens_to_string(tok_strs).strip())
                else:
                    parts.append(f"[w{w}]")
            return " ".join(parts)

        print("[info] Reconstructing AG News word-aligned texts ...", flush=True)
        train_token_to_word = train_ds.token_to_word.numpy().astype(np.int32)
        test_token_to_word  = test_ds.token_to_word.numpy().astype(np.int32)
        train_texts = [
            _reconstruct_words(train_token_ids[i], train_token_to_word[i])
            for i in range(len(train_token_ids))
        ]
        test_texts = [
            _reconstruct_words(test_token_ids[i], test_token_to_word[i])
            for i in range(len(test_token_ids))
        ]

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "train_texts":          train_texts,
            "train_token_to_word":  train_token_to_word,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
            "test_texts":           test_texts,
            "test_token_to_word":   test_token_to_word,
        }

        print(f"[info] AG News (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] AG News (test):  N={len(test_labels)}")

        if isinstance(dataset_cfg.n_train, int):
            n_train = None

    elif dataset_cfg.dataset == "yelp":
        yelp_module_dir = "/project/aip-rahulgk/image_icl_project/yelp"
        if yelp_module_dir not in sys.path:
            sys.path.insert(0, yelp_module_dir)
        from yelp_dataset import load_yelp_dataset  # type: ignore

        _max_train = dataset_cfg.n_train if isinstance(dataset_cfg.n_train, int) else None
        train_loader, _val_loader, test_loader, metadata = load_yelp_dataset(
            use_token_embeddings=True,
            max_train=_max_train,
            max_val=0,
            max_test=dataset_cfg.n_test,
            num_workers=0,
            batch_size=1,
        )
        train_ds = train_loader.dataset
        test_ds  = test_loader.dataset

        train_patches = train_ds.token_embeddings.float().numpy()
        train_labels  = train_ds.targets.numpy().astype(np.int64)
        cls_train     = train_ds.cls_embeddings.float().numpy()
        train_token_ids      = train_ds.token_ids.numpy().astype(np.int32)
        train_attention_mask = train_ds.attention_masks.bool().numpy()

        test_patches         = test_ds.token_embeddings.float().numpy()
        test_labels          = test_ds.targets.numpy().astype(np.int64)
        cls_test             = test_ds.cls_embeddings.float().numpy()
        test_token_ids       = test_ds.token_ids.numpy().astype(np.int32)
        test_attention_mask  = test_ds.attention_masks.bool().numpy()

        label_map    = metadata["label_map"]
        idx_to_class = {v: k for k, v in label_map.items()}

        from transformers import AutoTokenizer as _AutoTokenizer
        _HF_CACHE = "/scratch/hermanb/.cache/huggingface/transformers"
        _tokenizer = _AutoTokenizer.from_pretrained(
            metadata.get("model_name", "google/electra-base-discriminator"),
            cache_dir=_HF_CACHE,
        )

        def _reconstruct_words(token_ids_row: np.ndarray, token_to_word_row: np.ndarray) -> str:
            from collections import defaultdict
            word_tokens: dict = defaultdict(list)
            for tok_id, wid in zip(token_ids_row.tolist(), token_to_word_row.tolist()):
                if wid >= 0:
                    word_tokens[wid].append(tok_id)
            if not word_tokens:
                return ""
            n_words = max(word_tokens) + 1
            parts = []
            for w in range(n_words):
                toks = word_tokens.get(w, [])
                if toks:
                    tok_strs = _tokenizer.convert_ids_to_tokens(toks)
                    parts.append(_tokenizer.convert_tokens_to_string(tok_strs).strip())
                else:
                    parts.append(f"[w{w}]")
            return " ".join(parts)

        print("[info] Reconstructing Yelp word-aligned texts ...", flush=True)
        train_token_to_word = train_ds.token_to_word.numpy().astype(np.int32)
        test_token_to_word  = test_ds.token_to_word.numpy().astype(np.int32)
        train_texts = [
            _reconstruct_words(train_token_ids[i], train_token_to_word[i])
            for i in range(len(train_token_ids))
        ]
        test_texts = [
            _reconstruct_words(test_token_ids[i], test_token_to_word[i])
            for i in range(len(test_token_ids))
        ]

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "train_texts":          train_texts,
            "train_token_to_word":  train_token_to_word,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
            "test_texts":           test_texts,
            "test_token_to_word":   test_token_to_word,
        }

        print(f"[info] Yelp (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] Yelp (test):  N={len(test_labels)}")

        if isinstance(dataset_cfg.n_train, int):
            n_train = None

    elif dataset_cfg.dataset == "airbnb":
        airbnb_dir = str(Path(dataset_cfg.dataset_path))
        if airbnb_dir not in sys.path:
            sys.path.insert(0, airbnb_dir)
        from airbnb_melbourne_dataset import AirbnbMelbourneDataset  # type: ignore
        ds = AirbnbMelbourneDataset(Path(dataset_cfg.dataset_path))
        idx_to_class = {i: str(c) for i, c in enumerate(ds.classes)}
        train_idx, test_idx = ds.default_split()

        (train_patches, train_labels, cls_train,
         tab_tr, train_token_ids, train_attention_mask) = _collect_electra_tabular(ds, train_idx)
        (test_patches, test_labels, cls_test,
         tab_te, test_token_ids, test_attention_mask) = _collect_electra_tabular(ds, test_idx)

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
        }
        if load_tabular:
            extra_data["tab_train"] = tab_tr
            extra_data["tab_test"]  = tab_te

        print(f"[info] airbnb (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] airbnb (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "product-sentiment":
        prod_dir = str(Path(dataset_cfg.dataset_path))
        if prod_dir not in sys.path:
            sys.path.insert(0, prod_dir)
        from product_sentiment_dataset import ProductSentimentDataset  # type: ignore
        ds = ProductSentimentDataset(Path(dataset_cfg.dataset_path))
        idx_to_class = {i: str(c) for i, c in enumerate(ds.classes)}
        train_idx, test_idx = ds.default_split(random_state=seed)

        (train_patches, train_labels, cls_train,
         tab_tr, train_token_ids, train_attention_mask) = _collect_electra_tabular(ds, train_idx)
        (test_patches, test_labels, cls_test,
         tab_te, test_token_ids, test_attention_mask) = _collect_electra_tabular(ds, test_idx)

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
        }
        if load_tabular:
            extra_data["tab_train"] = tab_tr
            extra_data["tab_test"]  = tab_te

        print(f"[info] product-sentiment (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] product-sentiment (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "wine-reviews":
        wine_dir = str(Path(dataset_cfg.dataset_path))
        if wine_dir not in sys.path:
            sys.path.insert(0, wine_dir)
        from wine_reviews_dataset import WineReviewsDataset  # type: ignore
        ds = WineReviewsDataset(Path(dataset_cfg.dataset_path))
        idx_to_class = {i: str(c) for i, c in enumerate(ds.classes)}
        train_idx, test_idx = ds.default_split(random_state=seed)

        (train_patches, train_labels, cls_train,
         tab_tr, train_token_ids, train_attention_mask) = _collect_electra_tabular(ds, train_idx)
        (test_patches, test_labels, cls_test,
         tab_te, test_token_ids, test_attention_mask) = _collect_electra_tabular(ds, test_idx)

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
        }
        if load_tabular:
            extra_data["tab_train"] = tab_tr
            extra_data["tab_test"]  = tab_te

        print(f"[info] wine-reviews (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] wine-reviews (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "fake-jobs":
        fakejobs_dir = str(Path(dataset_cfg.dataset_path))
        if fakejobs_dir not in sys.path:
            sys.path.insert(0, fakejobs_dir)
        from fake_jobs_dataset import FakeJobsDataset  # type: ignore
        ds = FakeJobsDataset(Path(dataset_cfg.dataset_path))
        idx_to_class = {i: str(c) for i, c in enumerate(ds.classes)}
        train_idx, test_idx = ds.default_split(random_state=seed)

        (train_patches, train_labels, cls_train,
         tab_tr, train_token_ids, train_attention_mask) = _collect_electra_tabular(ds, train_idx)
        (test_patches, test_labels, cls_test,
         tab_te, test_token_ids, test_attention_mask) = _collect_electra_tabular(ds, test_idx)

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
        }
        if load_tabular:
            extra_data["tab_train"] = tab_tr
            extra_data["tab_test"]  = tab_te

        print(f"[info] fake-jobs (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] fake-jobs (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "jigsaw":
        jigsaw_dir = str(Path(dataset_cfg.dataset_path))
        if jigsaw_dir not in sys.path:
            sys.path.insert(0, jigsaw_dir)
        from jigsaw_dataset import JigsawDataset  # type: ignore
        ds = JigsawDataset(Path(dataset_cfg.dataset_path))
        idx_to_class = {i: str(c) for i, c in enumerate(ds.classes)}
        train_idx, test_idx = ds.default_split()

        (train_patches, train_labels, cls_train,
         tab_tr, train_token_ids, train_attention_mask) = _collect_electra_tabular(ds, train_idx)
        (test_patches, test_labels, cls_test,
         tab_te, test_token_ids, test_attention_mask) = _collect_electra_tabular(ds, test_idx)

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
        }
        if load_tabular:
            extra_data["tab_train"] = tab_tr
            extra_data["tab_test"]  = tab_te

        print(f"[info] jigsaw (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] jigsaw (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset in ("clothing", "salary"):
        from sklearn.model_selection import train_test_split as _tts

        if dataset_cfg.dataset == "clothing":
            clothing_dir = str(Path(dataset_cfg.dataset_path))
            if clothing_dir not in sys.path:
                sys.path.insert(0, clothing_dir)
            from womens_clothing_dataset import WomensClothingDataset  # type: ignore
            ds = WomensClothingDataset(Path(dataset_cfg.dataset_path))
            idx_to_class = {0: "not_recommended", 1: "recommended"}
        else:  # salary
            salary_dir = str(Path(dataset_cfg.dataset_path))
            if salary_dir not in sys.path:
                sys.path.insert(0, salary_dir)
            from salary_india_dataset import SalaryIndiaDataset  # type: ignore
            ds = SalaryIndiaDataset(Path(dataset_cfg.dataset_path))
            idx_to_class = {i: str(c) for i, c in enumerate(ds.classes)}

        all_idx = np.arange(len(ds))
        train_idx, test_idx = _tts(
            all_idx, test_size=0.2, random_state=seed, stratify=ds.targets
        )

        (train_patches, train_labels, cls_train,
         tab_tr, train_token_ids, train_attention_mask) = _collect_electra_tabular(ds, train_idx)
        (test_patches, test_labels, cls_test,
         tab_te, test_token_ids, test_attention_mask) = _collect_electra_tabular(ds, test_idx)

        extra_data = {
            "train_token_ids":      train_token_ids,
            "train_attention_mask": train_attention_mask,
            "test_token_ids":       test_token_ids,
            "test_attention_mask":  test_attention_mask,
        }
        if load_tabular:
            extra_data["tab_train"] = tab_tr
            extra_data["tab_test"]  = tab_te

        print(f"[info] {dataset_cfg.dataset} (train): N={len(train_labels)}  "
              f"max_length={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] {dataset_cfg.dataset} (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "aircrafts":
        aircrafts_module_dir = "/project/aip-rahulgk/image_icl_project/aircrafts"
        if aircrafts_module_dir not in sys.path:
            sys.path.insert(0, aircrafts_module_dir)
        from aircrafts_dataset import AircraftsDataset  # type: ignore

        ds = AircraftsDataset(raw_data_dir=Path(dataset_cfg.dataset_path))

        # Merge train + val into the support set; keep test as the query set.
        flags = np.array(ds._split_flags)
        train_idx = np.where((flags == "train") | (flags == "val"))[0]
        test_idx  = np.where(flags == "test")[0]

        train_patches, cls_train, train_labels, _ = _collect_h5_dataset(
            ds, train_idx, desc="Loading Aircrafts (train+val)")
        test_patches, cls_test, test_labels, _ = _collect_h5_dataset(
            ds, test_idx, desc="Loading Aircrafts (test)")

        idx_to_class = {i: name for i, name in enumerate(ds.class_names)}

        print(f"[info] Aircrafts (train+val): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] Aircrafts (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "ham10000":
        ham_module_dir = "/project/aip-rahulgk/image_icl_project/HAM10000"
        if ham_module_dir not in sys.path:
            sys.path.insert(0, ham_module_dir)
        from ham10000_dataset import HAM10000Dataset  # type: ignore

        ds = HAM10000Dataset(raw_data_dir=Path(dataset_cfg.dataset_path))
        train_idx, test_idx = ds.default_split()

        train_patches, cls_train, train_labels, tab_train_ham = _collect_h5_dataset(
            ds, train_idx, with_tabular=load_tabular, desc="Loading HAM10000 (train)")
        test_patches, cls_test, test_labels, tab_test_ham = _collect_h5_dataset(
            ds, test_idx, with_tabular=load_tabular, desc="Loading HAM10000 (test)")

        if load_tabular:
            extra_data["tab_train"] = tab_train_ham
            extra_data["tab_test"]  = tab_test_ham

        idx_to_class = {i: name for i, name in enumerate(ds.class_names)}

        print(f"[info] HAM10000 (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] HAM10000 (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "oxford-flowers":
        flowers_module_dir = "/project/aip-rahulgk/image_icl_project/oxford_flowers"
        if flowers_module_dir not in sys.path:
            sys.path.insert(0, flowers_module_dir)
        from oxford_flowers_dataset import OxfordFlowersDataset  # type: ignore

        ds = OxfordFlowersDataset(raw_data_dir=Path(dataset_cfg.dataset_path))
        # train/ → support set; valid/ → query set (test/ has no labels)
        train_idx, test_idx = ds.default_split()

        train_patches, cls_train, train_labels, _ = _collect_h5_dataset(
            ds, train_idx, desc="Loading Oxford Flowers (train)")
        test_patches, cls_test, test_labels, _ = _collect_h5_dataset(
            ds, test_idx, desc="Loading Oxford Flowers (valid)")

        idx_to_class = {i: name for i, name in enumerate(ds.class_names)}

        print(f"[info] Oxford Flowers (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] Oxford Flowers (valid/test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "dtd":
        dtd_module_dir = "/project/aip-rahulgk/image_icl_project/DTD"
        if dtd_module_dir not in sys.path:
            sys.path.insert(0, dtd_module_dir)
        from dtd_dataset import DTDDataset  # type: ignore

        ds = DTDDataset(raw_data_dir=Path(dataset_cfg.dataset_path))
        # default_split() already merges train+val into the support set
        train_idx, test_idx = ds.default_split()

        train_patches, cls_train, train_labels, _ = _collect_h5_dataset(
            ds, train_idx, desc="Loading DTD (train+val)")
        test_patches, cls_test, test_labels, _ = _collect_h5_dataset(
            ds, test_idx, desc="Loading DTD (test)")

        idx_to_class = {i: name for i, name in enumerate(ds.class_names)}

        print(f"[info] DTD (train+val): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] DTD (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "coco":
        coco_module_dir = "/project/aip-rahulgk/image_icl_project/coco"
        if coco_module_dir not in sys.path:
            sys.path.insert(0, coco_module_dir)
        from coco_dataset import COCODataset  # type: ignore

        ds = COCODataset(coco_dir=Path(dataset_cfg.dataset_path))
        # default_split() returns (train, val, test); use train as support, val as test
        train_idx, val_idx, test_idx = ds.default_split()

        train_patches, cls_train, train_labels, _ = _collect_h5_dataset(
            ds, train_idx, desc="Loading COCO (train)")
        test_patches, cls_test, test_labels, _ = _collect_h5_dataset(
            ds, val_idx, desc="Loading COCO (val)")

        idx_to_class = {i: name for i, name in enumerate(ds.class_names)}
        # Store original dataset indices so visualization can call ds.get_image(idx)
        extra_data["coco_train_idx"] = train_idx
        extra_data["coco_test_idx"]  = val_idx

        print(f"[info] COCO (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] COCO (val/test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "open-images":
        oi_module_dir = "/home/hermanb/projects/aip-rahulgk/image_icl_project/open_images"
        if oi_module_dir not in sys.path:
            sys.path.insert(0, oi_module_dir)
        from open_images_dataset import OpenImagesDataset  # type: ignore

        ds = OpenImagesDataset(img_dir=Path(dataset_cfg.dataset_path) / "images")
        train_idx, test_idx = ds.default_split()

        train_patches, cls_train, train_labels, _ = _collect_h5_dataset(
            ds, train_idx, desc="Loading Open Images (train)")
        test_patches, cls_test, test_labels, _ = _collect_h5_dataset(
            ds, test_idx, desc="Loading Open Images (test)")

        idx_to_class = {i: name for i, name in enumerate(ds.class_names)}
        # Store original dataset indices so visualization can call ds.get_image(idx)
        extra_data["open_images_train_idx"] = train_idx
        extra_data["open_images_test_idx"]  = test_idx

        print(f"[info] Open Images (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] Open Images (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "mm-imdb":
        mmimdb_module_dir = "/home/hermanb/projects/aip-rahulgk/image_icl_project/mm-imdb"
        if mmimdb_module_dir not in sys.path:
            sys.path.insert(0, mmimdb_module_dir)
        from mmimdb_dataset import MMIMDbDataset  # type: ignore
        from tqdm import tqdm

        ds = MMIMDbDataset(data_dir=Path(dataset_cfg.dataset_path))
        train_idx, test_idx = ds.default_split()

        def _collect_mmimdb(ds, indices, desc: str):
            patches_list, cls_list, labels_list = [], [], []
            text_list, first_pad_list = [], []
            _warned_registers = False
            for i in tqdm(indices, desc=desc):
                sample = ds[int(i)]
                emb = sample["image"]   # (201, 768)
                if not _warned_registers:
                    print(
                        f"[info] MM-IMDb image embeddings have shape {tuple(emb.shape)} — "
                        "removing 4 register tokens (rows 1-4); these are likely DINOv2 features."
                    )
                    _warned_registers = True
                cls_list.append(emb[0].float().numpy())
                patches_list.append(emb[5:].float().numpy())
                labels_list.append(sample["target"].item())
                if "electra_text" in sample:
                    text_list.append(sample["electra_text"])
            patches = np.stack(patches_list, axis=0)
            cls_emb = np.stack(cls_list,     axis=0)
            labels  = np.array(labels_list,  dtype=np.int64)

            text_arr = tok_ids = attn_mask = text_cls = None
            if text_list:
                first_pads = [ds._electra_first_pad[int(i)] for i in indices]
                max_len = max(first_pads)
                D_t = text_list[0].shape[-1]
                N = len(indices)
                text_arr = np.zeros((N, max_len, D_t), dtype=np.float32)
                for j, (e, fp) in enumerate(zip(text_list, first_pads)):
                    e_np = e.float().numpy()
                    text_arr[j, :len(e_np)] = e_np
                fp_arr  = np.array(first_pads, dtype=np.int32)
                tok_ids = np.ones((N, max_len), dtype=np.int32)
                tok_ids[:, 0] = 101  # [CLS]
                for j, fp in enumerate(first_pads):
                    tok_ids[j, fp:] = 0
                attn_mask = np.arange(max_len)[None, :] < fp_arr[:, None]
                text_cls  = text_arr[:, 0, :].copy()

            return patches, cls_emb, labels, text_arr, tok_ids, attn_mask, text_cls

        (train_patches, cls_train, train_labels,
         text_train, train_tok_ids, train_attn_mask, text_cls_train) = _collect_mmimdb(
            ds, train_idx, "Loading MM-IMDb (train)")
        (test_patches, cls_test, test_labels,
         text_test, test_tok_ids, test_attn_mask, text_cls_test) = _collect_mmimdb(
            ds, test_idx, "Loading MM-IMDb (test)")

        if text_train is not None:
            extra_data["text_train"]           = text_train
            extra_data["text_train_token_ids"] = train_tok_ids
            extra_data["text_train_attn_mask"] = train_attn_mask
            extra_data["text_cls_train"]       = text_cls_train
            extra_data["text_test"]            = text_test
            extra_data["text_test_token_ids"]  = test_tok_ids
            extra_data["text_test_attn_mask"]  = test_attn_mask
            extra_data["text_cls_test"]        = text_cls_test

        idx_to_class = {i: str(i) for i in range(ds.n_classes)}

        print(f"[info] MM-IMDb (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] MM-IMDb (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset == "wikiart":
        wikiart_module_dir = "/home/hermanb/projects/aip-rahulgk/image_icl_project/wikiart"
        if wikiart_module_dir not in sys.path:
            sys.path.insert(0, wikiart_module_dir)
        from wikiart_dataset import WikiArtDataset  # type: ignore
        from tqdm import tqdm

        ds = WikiArtDataset(data_dir=Path(dataset_cfg.dataset_path))
        train_idx, test_idx = ds.default_split()

        def _collect_wikiart(ds, indices, desc: str, with_tabular: bool = False):
            patches_list, cls_list, labels_list, tab_list = [], [], [], []
            _warned_registers = False
            for i in tqdm(indices, desc=desc):
                sample = ds[int(i)]
                emb = sample["image"]   # (201, 768) or (768,)
                if emb.ndim == 2:
                    if not _warned_registers:
                        print(
                            f"[info] WikiArt embeddings have shape {tuple(emb.shape)} — "
                            "removing 4 register tokens (rows 1-4); these are likely DINOv2 features."
                        )
                        _warned_registers = True
                    # row 0 = CLS, rows 1-4 = register tokens (dropped), rows 5: = spatial patches
                    cls_list.append(emb[0].float().numpy())
                    patches_list.append(emb[5:].float().numpy())
                else:
                    # CLS-only fallback — expand to (1, D) so shape is consistent
                    cls_list.append(emb.float().numpy())
                    patches_list.append(emb.float().numpy()[None, :])
                labels_list.append(sample["target"].item())
                if with_tabular:
                    tab_list.append(sample["tabular"].float().numpy())
            tab_arr = np.stack(tab_list, axis=0) if with_tabular else None
            return (
                np.stack(patches_list, axis=0),
                np.stack(cls_list,     axis=0),
                np.array(labels_list,  dtype=np.int64),
                tab_arr,
            )

        train_patches, cls_train, train_labels, tab_train_wa = _collect_wikiart(
            ds, train_idx, "Loading WikiArt (train)", with_tabular=load_tabular)
        test_patches,  cls_test,  test_labels,  tab_test_wa  = _collect_wikiart(
            ds, test_idx,  "Loading WikiArt (test)",  with_tabular=load_tabular)

        if load_tabular:
            extra_data["tab_train"] = tab_train_wa
            extra_data["tab_test"]  = tab_test_wa

        idx_to_class = {i: str(i) for i in range(ds.n_classes)}

        print(f"[info] WikiArt (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] WikiArt (test):  N={len(test_labels)}")

    elif dataset_cfg.dataset in IMAGENET_SUBSETS:
        synsets = IMAGENET_SUBSETS[dataset_cfg.dataset]
        embeddings_root = Path(dataset_cfg.dataset_path)

        train_patches_list, train_labels_list = [], []
        test_patches_list,  test_labels_list  = [], []
        cls_train_list, cls_test_list = [], []

        for label_idx, synset in enumerate(synsets):
            synset_dir = embeddings_root / synset
            for split_dir, patches_list, labels_list, cls_list in [
                (synset_dir / "train", train_patches_list, train_labels_list, cls_train_list),
                (synset_dir / "val",   test_patches_list,  test_labels_list,  cls_test_list),
            ]:
                for npy_path in sorted(split_dir.glob("*.npy")):
                    emb = np.load(npy_path)          # (201, 768): row 0 = CLS, rows 1-4 = register tokens, rows 5: = 196 spatial patches
                    cls_list.append(emb[0])
                    patches_list.append(emb[5:])
                    labels_list.append(label_idx)

        train_patches = np.stack(train_patches_list, axis=0).astype(np.float32)
        train_labels  = np.array(train_labels_list, dtype=np.int64)
        cls_train     = np.stack(cls_train_list, axis=0).astype(np.float32)
        test_patches  = np.stack(test_patches_list, axis=0).astype(np.float32)
        test_labels   = np.array(test_labels_list, dtype=np.int64)
        cls_test      = np.stack(cls_test_list, axis=0).astype(np.float32)

        idx_to_class = {i: s for i, s in enumerate(synsets)}

        print(f"[info] {dataset_cfg.dataset} (train): N={len(train_labels)}  "
              f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        print(f"[info] {dataset_cfg.dataset} (test):  N={len(test_labels)}")

    else:
        raise ValueError(f"Unknown dataset '{dataset_cfg.dataset}'. "
                         f"Choices: butterfly, rsna, petfinder, dvm, pad-ufes, "
                         f"cbis-ddsm-mass, cbis-ddsm-calc, imdb, 20news, ag_news, yelp, "
                         f"clothing, salary, airbnb, fake-jobs, jigsaw, "
                         f"product-sentiment, wine-reviews, "
                         f"aircrafts, ham10000, oxford-flowers, dtd, coco, open-images, wikiart, mm-imdb, " +
                         ", ".join(sorted(IMAGENET_SUBSETS)))

    # --- Optional n_train subsampling ---
    if dataset_cfg.n_train is not None:
        n_orig = len(train_labels)
        _spec  = dataset_cfg.n_train
        n_train_count = int(round(_spec * n_orig)) if isinstance(_spec, float) else _spec
        if n_train_count < n_orig:
            rng     = np.random.RandomState(seed)
            sub_idx = rng.choice(n_orig, size=n_train_count, replace=False)
            sub_idx.sort()
            train_sub_idx = sub_idx
            train_patches = train_patches[sub_idx]
            train_labels  = train_labels[sub_idx]
            if cls_train is not None:
                cls_train = cls_train[sub_idx]
            if "tab_train" in extra_data and extra_data["tab_train"] is not None:
                extra_data["tab_train"] = extra_data["tab_train"][sub_idx]
            for k in ["text_train", "text_train_token_ids", "text_train_attn_mask", "text_cls_train",
                      "train_token_ids", "train_attention_mask", "train_texts", "train_token_to_word"]:
                if k in extra_data and extra_data[k] is not None:
                    if isinstance(extra_data[k], np.ndarray):
                        extra_data[k] = extra_data[k][sub_idx]
                    elif isinstance(extra_data[k], list):
                        extra_data[k] = [extra_data[k][i] for i in sub_idx]
            print(f"[info] Training set subsampled: {n_train_count} / {n_orig} images")

    # --- Optional n_test subsampling ---
    if dataset_cfg.n_test is not None and dataset_cfg.n_test < len(test_labels):
        n_orig_test = len(test_labels)
        rng_test    = np.random.RandomState(seed + 1)
        test_sub    = rng_test.choice(n_orig_test, size=dataset_cfg.n_test, replace=False)
        test_sub.sort()
        test_sub_idx  = test_sub
        test_patches  = test_patches[test_sub]
        test_labels   = test_labels[test_sub]
        if cls_test is not None:
            cls_test = cls_test[test_sub]
        if "tab_test" in extra_data and extra_data["tab_test"] is not None:
            extra_data["tab_test"] = extra_data["tab_test"][test_sub]
        for k in ["text_test", "text_test_token_ids", "text_test_attn_mask", "text_cls_test",
                  "test_token_ids", "test_attention_mask", "test_texts", "test_token_to_word"]:
            if k in extra_data and extra_data[k] is not None:
                if isinstance(extra_data[k], np.ndarray):
                    extra_data[k] = extra_data[k][test_sub]
                elif isinstance(extra_data[k], list):
                    extra_data[k] = [extra_data[k][i] for i in test_sub]
        print(f"[info] Test set subsampled: {dataset_cfg.n_test} / {n_orig_test} images")

    return train_patches, train_labels, test_patches, test_labels, cls_train, cls_test, idx_to_class, train_sub_idx, test_sub_idx, extra_data


def _balance_classes(
    patches:   np.ndarray,           # [N, P, D]
    labels:    np.ndarray,           # [N]
    cls_feats: Optional[np.ndarray], # [N, D] or None
    rng:       np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Undersample majority classes so every class has the same number of examples.

    Samples are drawn without replacement.  The returned arrays are in a
    random (shuffled) order so the caller does not need to shuffle again.

    Returns (patches, labels, cls_feats, keep_idx) where keep_idx are the
    indices into the input arrays that were selected, in the returned order.
    Callers that maintain parallel lists (e.g. image paths) must apply the
    same keep_idx to stay aligned.
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_min = int(counts.min())
    keep: list[np.ndarray] = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        keep.append(rng.choice(idx, size=n_min, replace=False))
    keep_idx = np.concatenate(keep)
    rng.shuffle(keep_idx)   # mix classes so support set isn't class-sorted

    bal_patches = patches[keep_idx]
    bal_labels  = labels[keep_idx]
    bal_cls     = cls_feats[keep_idx] if cls_feats is not None else None

    orig_counts_str = "  ".join(f"cls{c}:{n}" for c, n in zip(classes, counts))
    print(f"[balance] {len(labels)} → {len(bal_labels)} samples  "
          f"({n_min} per class)  was: {orig_counts_str}")
    return bal_patches, bal_labels, bal_cls, keep_idx
