"""Dataset classes and feature-loading utilities for patch-quality experiments.

Covers both the Butterfly image-classification dataset and the RSNA Pneumonia
X-ray dataset.  All public symbols are re-exported through this module so that
other scripts only need a single import site.
"""

from __future__ import annotations

import csv as _csv
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pal_pooling.config import (
    BUTTERFLY_DATASET_PATH,
    FEATURES_DIR,
    RSNA_DATASET_PATH,
    DatasetConfig,
)

DATASET_PATH = BUTTERFLY_DATASET_PATH   # kept for backward compat


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ButterflyPatchDataset(Dataset):
    """Loads pre-extracted DINOv3 patch features for the butterfly dataset.

    Each sample is ``(patches, label)`` where:
        patches : float tensor of shape [num_patches, embed_dim]  (e.g. [196, 768])
        label   : int class index

    Args:
        features_dir: Directory containing the .pt files produced by the
            feature extraction script.
        split: ``"train"`` or ``"test"``.
        dtype: Cast features to this dtype on load (default: float32 for model compat).
    """

    def __init__(
        self,
        features_dir: Path = FEATURES_DIR,
        split: str = "train",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        features_dir = Path(features_dir)
        pt_path = features_dir / f"butterfly_{split}_dinov3_patch_features.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Patch features not found at {pt_path}. "
                "Run the feature extraction script first."
            )

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        self.features: torch.Tensor = ckpt["features"].to(dtype)   # [N, P, D]
        self.labels:   torch.Tensor = ckpt["labels"].long()        # [N]

        N, P, D = self.features.shape
        print(
            f"[info] ButterflyPatchDataset ({split}): "
            f"N={N}  num_patches={P}  embed_dim={D}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Reconstruct training-split file paths (must match extraction order)
# ---------------------------------------------------------------------------

def _get_image_paths(
    dataset_path: Path = DATASET_PATH,
    split: str = "train",
    seed: int = 42,
    test_fraction: float = 0.2,
) -> tuple[list[Path], list[int], dict[int, str]]:
    """Return (paths, labels, idx_to_class) for the requested split.

    Uses the exact same shuffle/split logic as ButterflyPatchDataset so that
    index i here corresponds to row i in the .pt feature file.
    """
    csv_path = Path(dataset_path) / "Training_set.csv"
    img_dir  = Path(dataset_path) / "train"

    rows: list[tuple[str, str]] = []
    all_labels: list[str] = []
    with csv_path.open(newline="") as f:
        for row in _csv.DictReader(f):
            rows.append((row["filename"], row["label"]))
            all_labels.append(row["label"])

    class_to_idx: dict[str, int] = {
        lbl: i for i, lbl in enumerate(sorted(set(all_labels)))
    }
    idx_to_class: dict[int, str] = {v: k for k, v in class_to_idx.items()}

    n_total = len(rows)
    rng     = np.random.RandomState(seed)
    idx     = rng.permutation(n_total)
    n_test  = int(n_total * test_fraction)
    selected = [rows[i] for i in (idx[:n_test] if split == "test" else idx[n_test:])]

    paths  = [img_dir / fname   for fname, _   in selected]
    labels = [class_to_idx[lbl] for _,    lbl  in selected]
    return paths, labels, idx_to_class


# ---------------------------------------------------------------------------
# RSNA helpers
# ---------------------------------------------------------------------------

def _dicom_to_pil(path: Path) -> Image.Image:
    """Load a DICOM file and return an 8-bit grayscale-as-RGB PIL Image."""
    import pydicom
    dcm    = pydicom.dcmread(str(path))
    pixels = dcm.pixel_array.astype(np.float32)
    pi     = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if pi.strip() == "MONOCHROME1":
        pixels = pixels.max() - pixels
    pmin, pmax = pixels.min(), pixels.max()
    if pmax > pmin:
        pixels = (pixels - pmin) / (pmax - pmin) * 255.0
    return Image.fromarray(pixels.astype(np.uint8)).convert("RGB")


def _get_rsna_image_paths(
    dataset_path: Path,
    features_dir: Path,
    split:        str,
    backbone:     str = "rad-dino",
) -> tuple[list[Path], list[int], dict[int, str]]:
    """Return (paths, labels, idx_to_class) for the RSNA split.

    Paths and labels are read directly from the saved .pt metadata so that
    index i here corresponds exactly to row i in the patch-features tensor.
    """
    backbone_tag = backbone.replace("-", "_")
    pt_path = Path(features_dir) / f"rsna_{split}_{backbone_tag}_features.pt"
    ckpt    = torch.load(pt_path, map_location="cpu", weights_only=False)

    patient_ids  = ckpt["patient_ids"]          # list of strings
    labels_t     = ckpt["labels"]               # [N] int tensor
    class_to_idx = ckpt["class_to_idx"]         # {"Normal": 0, "Pneumonia": 1}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    image_dir = Path(dataset_path) / "stage_2_train_images"
    paths  = [image_dir / f"{pid}.dcm" for pid in patient_ids]
    labels = labels_t.tolist()
    return paths, labels, idx_to_class


# ---------------------------------------------------------------------------
# Feature loading dispatcher
# ---------------------------------------------------------------------------

def _load_features(
    dataset_cfg: DatasetConfig,
    seed: int,
    dtype:        torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           Optional[np.ndarray], Optional[np.ndarray], dict[int, str],
           Optional[np.ndarray]]:
    """Load patch and CLS features for the requested dataset.

    Returns:
        train_patches:   [N_train, P, D]  float32 numpy
        train_labels:    [N_train]         int64   numpy
        test_patches:    [N_test,  P, D]  float32 numpy
        test_labels:     [N_test]          int64   numpy
        cls_train:       [N_train, D] or None
        cls_test:        [N_test,  D] or None
        idx_to_class:    {int → class_name}
        train_sub_idx:   [N_train] int64 or None — indices into the original full training set
                         that were selected; None when no subsampling was applied.  Callers
                         that maintain a parallel list of training image paths must apply
                         the same selection to keep alignment with train_patches.
    """
    features_dir = Path(dataset_cfg.features_dir)
    train_sub_idx: Optional[np.ndarray] = None
    n_train = dataset_cfg.n_train

    if dataset_cfg.dataset == "butterfly":
        train_ds = ButterflyPatchDataset(features_dir, split="train", dtype=dtype)
        test_ds  = ButterflyPatchDataset(features_dir, split="test",  dtype=dtype)
        train_patches = train_ds.features.numpy()
        train_labels  = train_ds.labels.numpy()
        test_patches  = test_ds.features.numpy()
        test_labels   = test_ds.labels.numpy()

        # idx_to_class from feature labels (class indices 0..C-1)
        n_cls = int(train_labels.max()) + 1
        idx_to_class: dict[int, str] = {i: str(i) for i in range(n_cls)}

        # CLS features (optional)
        cls_train: Optional[np.ndarray] = None
        cls_test:  Optional[np.ndarray] = None
        cls_pt_train = features_dir / "butterfly_train_dinov3_features.pt"
        cls_pt_test  = features_dir / "butterfly_test_dinov3_features.pt"
        if cls_pt_train.exists() and cls_pt_test.exists():
            _ct = torch.load(cls_pt_train, map_location="cpu", weights_only=False)
            _cv = torch.load(cls_pt_test,  map_location="cpu", weights_only=False)
            cls_train = _ct["features"].float().numpy()
            cls_test  = _cv["features"].float().numpy()
        else:
            print(f"[warn] CLS feature files not found; skipping CLS baseline "
                  f"(expected {cls_pt_train} and {cls_pt_test})")

    elif dataset_cfg.dataset == "rsna":
        backbone_tag = dataset_cfg.backbone.replace("-", "_")
        pt_train = features_dir / f"rsna_train_{backbone_tag}_features.pt"
        pt_test  = features_dir / f"rsna_test_{backbone_tag}_features.pt"
        for p in (pt_train, pt_test):
            if not p.exists():
                raise FileNotFoundError(
                    f"RSNA feature file not found: {p}. "
                    "Run feature_extraction/extract_rsna_features.py first."
                )

        # Use memory-mapped loading (PyTorch >= 2.1) so only accessed pages are
        # paged in from disk — critical when n_train << N since we slice before
        # materialising the full patch tensor.
        _load_kwargs: dict = dict(map_location="cpu", weights_only=False)
        try:
            ck_train = torch.load(pt_train, mmap=True,  **_load_kwargs)
            ck_test  = torch.load(pt_test,  mmap=True,  **_load_kwargs)
        except TypeError:
            ck_train = torch.load(pt_train, **_load_kwargs)
            ck_test  = torch.load(pt_test,  **_load_kwargs)

        # Read lightweight tensors (labels, CLS) first — these are always small.
        train_labels  = ck_train["labels"].numpy().astype(np.int64)
        test_labels   = ck_test ["labels"].numpy().astype(np.int64)
        class_to_idx  = ck_train["class_to_idx"]
        idx_to_class  = {v: k for k, v in class_to_idx.items()}

        # Determine n_train subsample indices BEFORE touching the patch tensor so
        # that with mmap loading we only page in the selected rows.
        if n_train is not None and n_train < len(train_labels):
            n_orig  = len(train_labels)
            rng     = np.random.RandomState(seed)
            sub_idx = rng.choice(n_orig, size=n_train, replace=False)
            sub_idx.sort()
            train_sub_idx = sub_idx
            train_labels  = train_labels[sub_idx]
            # Slice the torch tensor before .numpy() — avoids materialising the
            # full [N, P, D] array in RAM (critical for large datasets).
            train_patches = ck_train["patch_features"][sub_idx].numpy()   # float16
            cls_train     = ck_train["cls_features"][sub_idx].float().numpy()
            print(f"[info] RSNA (train): N={n_train}/{n_orig} (subsampled)  "
                  f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")
        else:
            # Load full training set — keep as float16 to halve RAM vs float32.
            train_patches = ck_train["patch_features"].numpy()            # float16
            cls_train     = ck_train["cls_features"].float().numpy()
            print(f"[info] RSNA (train): N={len(train_labels)}  "
                  f"num_patches={train_patches.shape[1]}  embed_dim={train_patches.shape[2]}")

        del ck_train   # release checkpoint; patch tensor is now a numpy array

        test_patches = ck_test["patch_features"].numpy()                  # float16
        cls_test     = ck_test["cls_features"].float().numpy()
        del ck_test
        print(f"[info] RSNA (test):  N={len(test_labels)}")

        # n_train subsampling already applied above — skip the general block below.
        n_train = None

    else:
        raise ValueError(f"Unknown dataset '{dataset_cfg.dataset}'. Choices: butterfly, rsna")

    # --- Optional n_train subsampling for non-RSNA datasets ---
    if dataset_cfg.n_train is not None and dataset_cfg.n_train < len(train_labels):
        n_orig  = len(train_labels)
        rng     = np.random.RandomState(seed)
        sub_idx = rng.choice(n_orig, size=n_train, replace=False)
        sub_idx.sort()
        train_sub_idx = sub_idx
        train_patches = train_patches[sub_idx]
        train_labels  = train_labels[sub_idx]
        if cls_train is not None:
            cls_train = cls_train[sub_idx]
        print(f"[info] Training set subsampled: {n_train} / {n_orig} images")

    return train_patches, train_labels, test_patches, test_labels, cls_train, cls_test, idx_to_class, train_sub_idx


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
