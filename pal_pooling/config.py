import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

BUTTERFLY_DATASET_PATH = Path("/project/aip-rahulgk/hermanb/datasets/butterfly-image-classification")
RSNA_DATASET_PATH      = Path("/project/aip-rahulgk/hermanb/datasets/rsna-pneumonia")
FEATURES_DIR           = Path("/scratch/hermanb/temp_datasets/extracted_features")

@dataclass
class DatasetConfig:
    dataset: str
    backbone: str
    features_dir: Path
    dataset_path: Path
    n_train: Optional[int]
    n_sample: int
    balance_train: bool
    balance_test: bool
@dataclass
class RefinementConfig:
    refine: bool
    patch_size: int
    patch_group_sizes: List[int]
    temperature: List[float]
    weight_method: str
    ridge_alpha: List[float]
    normalize_features: bool
    batch_size: int
    max_query_rows: Optional[int]
    use_random_subsampling: bool
    aoe_class: Optional[str]
    aoe_handling: str
    gpu_ridge: bool
    tabicl_n_estimators: int
    tabicl_pca_dim: Optional[int]

@dataclass
class AttentionPoolConfig:
    attn_pool: bool
    attn_pool_only: bool
    attn_steps: int
    attn_lr: float
    attn_max_step_samples: int
    attn_num_queries: int
    attn_num_heads: int
    device: str
    tabicl_n_estimators: int
    tabicl_pca_dim: Optional[int]

@dataclass
class RunConfig:
    output_dir: Path
    post_refinement_viz: bool
    n_train_sweep: Optional[List[int]]

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    refinement: RefinementConfig
    attention: AttentionPoolConfig
    run: RunConfig
    seed: int
    cli_args: Optional[Dict[str, Any]] = field(default=None)

def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Patch quality evaluation with TabICL")
    p.add_argument("--dataset",       type=str,   default="butterfly",
                   choices=["butterfly", "rsna"],
                   help="Which dataset to run on (default: butterfly)")
    p.add_argument("--backbone",      type=str,   default="rad-dino",
                   choices=["rad-dino", "dinov3"],
                   help="Which backbone's features to load for RSNA "
                        "(default: rad-dino; ignored for butterfly)")
    p.add_argument("--features-dir",  type=Path,  default=FEATURES_DIR)
    p.add_argument("--dataset-path",  type=Path,  default=None,
                   help="Root path of the raw dataset (images + labels). "
                        "Defaults: butterfly → butterfly-image-classification, "
                        "rsna → rsna-pneumonia")
    p.add_argument("--n-sample",      type=int,   default=0)
    p.add_argument("--n-train",       type=int,   default=None,
                   help="Limit the support set to this many training images (random subsample)")
    p.add_argument("--n-estimators",  type=int,   default=1)
    p.add_argument("--pca-dim",       type=int,   default=128)
    p.add_argument("--no-pca",        action="store_true",
                   help="Disable PCA (use full 768-D embeddings)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output-dir",    type=Path,  default=Path("patch_quality_results"))
    p.add_argument("--patch-size",    type=int,   default=16)
    p.add_argument("--patch-group-sizes", type=int,   nargs="+",  default=[1],
                   help="Ordered list of patch group sizes for iterative refinement "
                        "(must each be a perfect square: 1, 4, 9, 16, …). "
                        "1 = no grouping (individual patches).")
    p.add_argument("--refine",        action="store_true",
                   help="Refine support features with patch-quality weighting before eval")
    p.add_argument("--temperature",    type=float, nargs="+",  default=[1.0],
                   help="Softmax temperature for patch pooling weights.")
    p.add_argument("--batch-size",     type=int,   default=1000,
                   help="Number of images per TabICL call during refinement")
    p.add_argument("--weight-method",  type=str,   default="correct_class_prob",
                   choices=["correct_class_prob", "entropy", "kl_div"],
                   help="How to derive patch pooling weights from TabICL probabilities.")
    p.add_argument("--ridge-alpha",  type=float, nargs="+",  default=[1.0],
                   help="Regularisation strength for the Ridge quality model.")
    p.add_argument("--normalize-features", action="store_true",
                   help="Fit a StandardScaler on training patches before Ridge fitting.")
    p.add_argument("--max-query-rows", type=int, default=None,
                   help="Cap on the total number of patch-group rows forwarded through TabICL.")
    p.add_argument("--use-random-subsampling", action="store_true",
                   help="Enable random subsampling of patch-group rows for Ridge fitting.")
    p.add_argument("--balance-train", action="store_true",
                   help="Undersample majority classes in the training set.")
    p.add_argument("--balance-test", action="store_true",
                   help="Undersample majority classes in the test set.")
    
    # Attention pooling upper-bound baseline
    p.add_argument("--attn-pool", action="store_true",
                   help="Train an attention pooling head (upper-bound baseline).")
    p.add_argument("--attn-pool-only", action="store_true",
                   help="Skip all feature-refinement stages and only train the attention pooling head.")
    p.add_argument("--attn-steps",            type=int,   default=500,
                   help="Training steps for the attention pooling head (default: 500)")
    p.add_argument("--attn-lr",               type=float, default=1e-3,
                   help="AdamW learning rate for attention pooling (default: 1e-3)")
    p.add_argument("--attn-max-step-samples", type=int,   default=512,
                   help="Max training rows forwarded per step (default: 512)")
    p.add_argument("--attn-num-queries",      type=int,   default=1,
                   help="Learnable query vectors (1 = CLS-like; default: 1)")
    p.add_argument("--attn-num-heads",        type=int,   default=8,
                   help="Attention heads (must divide embed_dim=768; default: 8)")
    p.add_argument("--device", type=str, default="auto",
                   help="Torch device for attention pooling training: 'auto', 'cuda', 'cpu' (default: auto)")
    
    p.add_argument("--aoe-class", type=str, default=None,
                   help="Absence-of-evidence class.")
    p.add_argument("--aoe-handling", type=str, default="filter",
                   choices=["filter", "entropy"],
                   help="How to handle the AoE class during Ridge fitting.")
    p.add_argument("--gpu-ridge", action="store_true",
                   help="Solve Ridge regression on the GPU (requires PyTorch + CUDA).")
    p.add_argument("--post-refinement-viz", action="store_true",
                   help="Skip pre-refinement visualisations; only produce post-refinement figures.")
    p.add_argument("--n-train-sweep", type=int, nargs="+", default=None,
                   metavar="N",
                   help="Run one experiment per value and collect results into a single sweep_results.json. Mutually exclusive with --n-train.")

    args = p.parse_args()

    if args.n_train_sweep is not None and args.n_train is not None:
        p.error("--n-train and --n-train-sweep are mutually exclusive.")

    _dataset_defaults = {"butterfly": BUTTERFLY_DATASET_PATH, "rsna": RSNA_DATASET_PATH}
    dataset_path = args.dataset_path or _dataset_defaults[args.dataset]
    
    pca_dim = None if args.no_pca else args.pca_dim
    
    dataset_cfg = DatasetConfig(
        dataset=args.dataset,
        backbone=args.backbone,
        features_dir=args.features_dir,
        dataset_path=dataset_path,
        n_train=args.n_train,
        n_sample=args.n_sample,
        balance_train=args.balance_train,
        balance_test=args.balance_test
    )
    
    refinement_cfg = RefinementConfig(
        refine=args.refine,
        patch_size=args.patch_size,
        patch_group_sizes=args.patch_group_sizes,
        temperature=args.temperature,
        weight_method=args.weight_method,
        ridge_alpha=args.ridge_alpha,
        normalize_features=args.normalize_features,
        batch_size=args.batch_size,
        max_query_rows=args.max_query_rows,
        use_random_subsampling=args.use_random_subsampling,
        aoe_class=args.aoe_class,
        aoe_handling=args.aoe_handling,
        gpu_ridge=args.gpu_ridge,
        tabicl_n_estimators=args.n_estimators,
        tabicl_pca_dim=pca_dim,
    )
    
    attention_cfg = AttentionPoolConfig(
        attn_pool=args.attn_pool or args.attn_pool_only,
        attn_pool_only=args.attn_pool_only,
        attn_steps=args.attn_steps,
        attn_lr=args.attn_lr,
        attn_max_step_samples=args.attn_max_step_samples,
        attn_num_queries=args.attn_num_queries,
        attn_num_heads=args.attn_num_heads,
        device=args.device,
        tabicl_n_estimators=args.n_estimators,
        tabicl_pca_dim=pca_dim,
    )
    
    run_cfg = RunConfig(
        output_dir=args.output_dir,
        post_refinement_viz=args.post_refinement_viz,
        n_train_sweep=args.n_train_sweep
    )
    
    return ExperimentConfig(
        dataset=dataset_cfg,
        refinement=refinement_cfg,
        attention=attention_cfg,
        run=run_cfg,
        seed=args.seed,
    )
