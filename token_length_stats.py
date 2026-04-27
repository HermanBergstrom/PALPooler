"""Token-length statistics for all text datasets.

For ELECTRA .pt datasets (airbnb, product-sentiment, wine-reviews, clothing,
salary) we load with mmap=True so only the small first_pad list is deserialised
and the large embeddings tensor stays on disk.

For HDF5 datasets (imdb, 20news, ag_news, yelp) we open each split file with
h5py and read only the attention_masks dataset — embeddings are never touched.

Usage:
    source /project/aip-rahulgk/hermanb/environments/aditya_tabicl/bin/activate
    python token_length_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# ELECTRA .pt datasets
# Format: {"embeddings": Tensor[N,T,D], "first_pad": list[int]}
# first_pad[i] = number of real tokens (incl. CLS) for sample i.
# ---------------------------------------------------------------------------
ELECTRA_PT: dict[str, Path] = {
    "fake-jobs":         Path("/project/6101781/image_icl_project/fake-jobs/fake-jobs_electra_text_features.pt"),
    "jigsaw":            Path("/project/6101781/image_icl_project/jigsaw/jigsaw_electra_text_features.pt"),
    "clothing":          Path("/project/aip-rahulgk/image_icl_project/womens-ecommerce-clothing-reviews/clothing_electra_text_features.pt"),
    "salary":            Path("/project/aip-rahulgk/image_icl_project/predict-the-data-scientists-salary-in-india/salary_electra_text_features.pt"),
    "airbnb":            Path("/project/6101781/image_icl_project/airbnb-melbourne/airbnb_melbourne_electra_text_features.pt"),
    "product-sentiment": Path("/project/6101781/image_icl_project/product-sentiment/product-sentiment_electra_text_features.pt"),
    "wine-reviews":      Path("/project/6101781/image_icl_project/wine-reviews/wine-reviews_electra_text_features.pt"),
    "petfinder":         Path("/project/6101781/image_icl_project/petfinder/petfinder_electra_text_features.pt"),
}

# ---------------------------------------------------------------------------
# HDF5 datasets
# Format: h5 files at preprocessed/{train,test}.h5 with key "attention_masks"
# Each row is a bool vector; sum = number of real tokens (incl. CLS).
# ---------------------------------------------------------------------------
_SCRATCH = Path("/scratch/hermanb/temp_datasets/extracted_features")
HDF5_DIRS: dict[str, Path] = {
    "imdb":    _SCRATCH / "imdb"   / "electra" / "preprocessed",
    "20news":  _SCRATCH / "20news" / "electra" / "preprocessed",
    "ag_news": _SCRATCH / "ag_news"/ "electra" / "preprocessed",
    "yelp":    _SCRATCH / "yelp"   / "electra" / "preprocessed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_row(name: str, lengths: np.ndarray) -> None:
    a = lengths.astype(np.float32)
    print(
        f"  {name:<22}  n={len(a):>7,}  "
        f"mean={a.mean():>7.1f}  median={np.median(a):>6.0f}  "
        f"p25={np.percentile(a, 25):>5.0f}  p75={np.percentile(a, 75):>5.0f}  "
        f"max={int(a.max()):>5}"
    )


def _load_electra_pt(path: Path) -> np.ndarray:
    """Return per-sample real-token lengths from a .pt ELECTRA feature file."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        # mmap not supported in older PyTorch
        raw = torch.load(path, map_location="cpu", weights_only=False)
    fp = raw.get("first_pad")
    if fp is not None:
        return np.array(fp, dtype=np.int32)
    # Fallback: infer max-length from the embeddings shape (no padding info).
    emb = raw["embeddings"]
    n, t = emb.shape[0], emb.shape[1]
    print(f"    [warn] no first_pad key — assuming all {n} samples have length {t}")
    return np.full(n, t, dtype=np.int32)


def _load_hdf5_lengths(processed_dir: Path) -> np.ndarray:
    """Concatenate attention_mask lengths from train + test split .h5 files."""
    all_lengths: list[np.ndarray] = []
    for split in ("train", "test"):
        h5_path = processed_dir / f"{split}.h5"
        if not h5_path.exists():
            print(f"    [warn] missing {h5_path}")
            continue
        with h5py.File(h5_path, "r") as hf:
            if "attention_masks" not in hf:
                print(f"    [warn] no attention_masks key in {h5_path}")
                continue
            masks = hf["attention_masks"][:]          # bool [N, T]
            all_lengths.append(masks.sum(axis=1).astype(np.int32))
    return np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    header = f"  {'Dataset':<22}  {'n':>8}  {'mean':>8}  {'median':>7}  {'p25':>6}  {'p75':>6}  {'max':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    print("\n  -- ELECTRA .pt datasets --")
    for name, pt_path in ELECTRA_PT.items():
        if not pt_path.exists():
            print(f"  {name:<22}  [file not found: {pt_path}]")
            continue
        lengths = _load_electra_pt(pt_path)
        _print_row(name, lengths)

    print("\n  -- HDF5 datasets --")
    for name, proc_dir in HDF5_DIRS.items():
        if not proc_dir.exists():
            print(f"  {name:<22}  [dir not found: {proc_dir}]")
            continue
        lengths = _load_hdf5_lengths(proc_dir)
        if len(lengths) == 0:
            print(f"  {name:<22}  [no data loaded]")
            continue
        _print_row(name, lengths)

    print("=" * len(header) + "\n")


if __name__ == "__main__":
    main()
