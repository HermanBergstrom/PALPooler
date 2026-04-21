import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

BUTTERFLY_DATASET_PATH  = Path("/project/aip-rahulgk/hermanb/datasets/butterfly-image-classification")
RSNA_DATASET_PATH       = Path("/project/aip-rahulgk/hermanb/datasets/rsna-pneumonia")
PETFINDER_DATASET_PATH  = Path("/project/aip-rahulgk/image_icl_project/petfinder")
DVM_DATASET_PATH        = Path("/project/6101781/image_icl_project/DVM_Dataset")
PAD_UFES_DATASET_PATH   = Path("/project/6101781/image_icl_project/pad-ufes-20-copy")
CBIS_DDSM_DATASET_PATH  = Path("/project/6101781/image_icl_project/cbis-ddsm/cbis-ddsm-breast-cancer-image-dataset")
IMDB_DATASET_PATH        = Path("/project/aip-rahulgk/image_icl_project/imdb")
TWENTY_NEWS_DATASET_PATH = Path("/project/aip-rahulgk/image_icl_project/20news")
AG_NEWS_DATASET_PATH     = Path("/project/aip-rahulgk/image_icl_project/ag_news")
YELP_DATASET_PATH        = Path("/project/aip-rahulgk/image_icl_project/yelp")
CLOTHING_DATASET_PATH    = Path("/project/aip-rahulgk/image_icl_project/womens-ecommerce-clothing-reviews")
SALARY_INDIA_DATASET_PATH = Path("/project/aip-rahulgk/image_icl_project/predict-the-data-scientists-salary-in-india")
FEATURES_DIR            = Path("/scratch/hermanb/temp_datasets/extracted_features")

# Maps each dataset name to its modality.  Image datasets default to DINOv3;
# text datasets default to ELECTRA.
MODALITY_MAP: Dict[str, str] = {
    "butterfly":      "image",
    "rsna":           "image",
    "petfinder":      "image",
    "dvm":            "image",
    "pad-ufes":       "image",
    "cbis-ddsm-mass": "image",
    "cbis-ddsm-calc": "image",
    "imdb":           "text",
    "20news":         "text",
    "ag_news":        "text",
    "yelp":           "text",
    "clothing":       "text",
    "salary":         "text",
}

# Default backbone per modality (used when --backbone is not set by the user).
DEFAULT_BACKBONE: Dict[str, str] = {
    "image": "dinov3",
    "text":  "electra",
}


def get_modality(dataset: str) -> str:
    """Return ``'image'`` or ``'text'`` for *dataset*."""
    return MODALITY_MAP.get(dataset, "image")

@dataclass
class DatasetConfig:
    dataset: str
    backbone: str
    features_dir: Path
    dataset_path: Path
    n_train: Optional[int]
    n_test: Optional[int]
    n_val: Optional[int]
    n_sample: int
    balance_train: bool
    balance_test: bool

    @property
    def modality(self) -> str:
        """Return ``'image'`` or ``'text'`` based on the dataset name."""
        return get_modality(self.dataset)
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
    append_cls: bool = False
    use_global_prior: bool = False
    prior: str = "label_frequency"
    use_attn_masking: bool = False
    model_selection: str = "last_iteration"
    binary_dist: bool = False
    train_val_fraction: Optional[float] = None
    cross_validation_cap: Optional[int] = None  # use 4-fold CV when N ≤ this value

@dataclass
class TextRefinementConfig:
    """Hyperparameters for text token PAL pooling.

    Mirrors :class:`RefinementConfig` for the text modality.  The key
    difference is ``text_group_modes`` (a per-stage list of grouping strategies)
    replacing ``patch_group_sizes``.

    Grouping modes
    --------------
    ``"none"``
        Individual non-padding tokens are forwarded as-is (identity grouping).
    ``"sentence"``
        Tokens are mean-pooled within each sentence, where sentence boundaries
        are defined by the ``[SEP]`` token (``sep_token_id``).

    The ``[CLS]`` token is always excluded from pooling by default.  Set
    ``append_cls=True`` to append its embedding as an extra group (matching
    the image CLS-token behaviour).
    """
    refine: bool
    text_group_modes: List[str]       # e.g. ["sentence", "none"]; one entry per stage
    temperature: List[float]
    ridge_alpha: List[float]
    weight_method: str
    normalize_features: bool
    batch_size: int
    max_query_rows: Optional[int]
    use_random_subsampling: bool
    gpu_ridge: bool
    tabicl_n_estimators: int
    tabicl_pca_dim: Optional[int]
    sep_token_id: int = 102           # BERT [SEP]
    cls_token_id: int = 101           # BERT [CLS] — excluded from pooling by default
    append_cls: bool = False          # if True, [CLS] embedding appended as extra group
    use_global_prior: bool = False
    use_attn_masking: bool = False
    model_selection: str = "last_iteration"
    binary_dist: bool = False
    prior: str = "label_frequency"    # "label_frequency", "token_marginal", or "current_pool_marginal"
    # "none" | "full_length" | "full_length_clip" | "sampled_count"
    length_importance_weight_basis: str = "none"
    length_importance_floor: int = 25  # floor for full_length_clip
    train_val_fraction: Optional[float] = None   # if set, split internally per stage instead of in the experiment script
    cross_validation_cap: Optional[int] = None   # use 4-fold CV when N ≤ this value


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
    viz_loo_train: bool
    show_pred_label: bool
    show_minority_prob: bool
    show_per_class_probs: bool
    n_train_sweep: Optional[List[int]]
    seeds: Optional[List[int]] = None

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    refinement: Union[RefinementConfig, TextRefinementConfig]
    attention: AttentionPoolConfig
    run: RunConfig
    seed: int
    device: str = "auto"
    cli_args: Optional[Dict[str, Any]] = field(default=None)

def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Patch quality evaluation with TabICL")
    p.add_argument("--dataset",       type=str,   default="butterfly",
                   choices=["butterfly", "rsna", "petfinder", "dvm", "pad-ufes",
                            "cbis-ddsm-mass", "cbis-ddsm-calc", "imdb", "20news",
                            "ag_news", "yelp", "clothing", "salary"],
                   help="Which dataset to run on")
    p.add_argument("--backbone",      type=str,   default=None,
                   choices=["rad-dino", "dinov3", "mae", "ijepa", "electra"],
                   help="Which backbone's features to load. "
                        "Defaults to 'dinov3' for image datasets and 'electra' for text datasets. "
                        "For RSNA the default is 'rad-dino'.")
    p.add_argument("--features-dir",  type=Path,  default=FEATURES_DIR)
    p.add_argument("--dataset-path",  type=Path,  default=None,
                   help="Root path of the raw dataset (images + labels). "
                        "Defaults: butterfly → butterfly-image-classification, "
                        "rsna → rsna-pneumonia, petfinder → petfinder, dvm → DVM_Dataset")
    p.add_argument("--n-sample",      type=int,   default=0)
    p.add_argument("--n-train",       type=int,   default=None,
                   help="Limit the support set to this many training images (random subsample)")
    p.add_argument("--n-test",        type=int,   default=None,
                   help="Limit the test set to this many testing images (random subsample, only DVM)")
    p.add_argument("--n-val",         type=int,   default=None,
                   help="Limit the val set to this many validation images (random subsample, only DVM)")
    p.add_argument("--train-val-fraction", type=float, default=None,
                   help="Fraction of the training set to hold out as a validation set for Ridge fitting (e.g. 0.2)")
    p.add_argument("--cross-validation-cap", type=int, default=None,
                   help="When the training set size N is at or below this cap, use 4-fold cross-validation "
                        "for Ridge pseudo-label collection instead of the standard single split. "
                        "Overrides --train-val-fraction when N ≤ cap (e.g. 2000).")
    p.add_argument("--n-estimators",  type=int,   default=1)
    p.add_argument("--pca-dim",       type=int,   default=128)
    p.add_argument("--no-pca",        action="store_true",
                   help="Disable PCA (use full 768-D embeddings)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output-dir",    type=Path,  default=Path("results/pal_pooling"))
    p.add_argument("--patch-size",    type=int,   default=16)
    p.add_argument("--patch-group-sizes", type=int,   nargs="+",  default=[1],
                   help="Ordered list of patch group sizes for iterative refinement "
                        "(must each be a perfect square: 1, 4, 9, 16, …). "
                        "1 = no grouping (individual patches). Image datasets only.")
    p.add_argument("--text-group-modes", type=str,   nargs="+",  default=["none"],
                   choices=["none", "sentence"],
                   help="Ordered list of text token grouping modes for iterative refinement. "
                        "'none' = individual non-padding tokens; "
                        "'sentence' = mean-pool within sentence spans delimited by [SEP]. "
                        "Text datasets only.")
    p.add_argument("--refine",        action="store_true",
                   help="Refine support features with patch-quality weighting before eval")
    p.add_argument("--temperature",    type=float, nargs="+",  default=[1.0],
                   help="Softmax temperature for patch pooling weights.")
    p.add_argument("--batch-size",     type=int,   default=1000,
                   help="Number of images per TabICL call during refinement")
    p.add_argument("--weight-method",  type=str,   default="correct_class_prob",
                   choices=["correct_class_prob", "entropy", "kl_div", "wasserstein", "js_div", "tvd"],
                   help="How to derive patch pooling weights from TabICL probabilities.")
    p.add_argument("--ridge-alpha",  type=float, nargs="+",  default=[1.0],
                   help="Regularisation strength for the Ridge quality model.")
    p.add_argument("--normalize-features", action="store_true",
                   help="Fit a StandardScaler on training patches before Ridge fitting.")
    p.add_argument("--max-query-rows", type=int, default=None,
                   help="Cap on the total number of patch-group rows forwarded through TabICL.")
    p.add_argument("--use-random-subsampling", action="store_true",
                   help="Enable random subsampling of patch-group rows for Ridge fitting.")
    p.add_argument("--length-importance-weight-basis", type=str, default="none",
                   choices=["none", "full_length", "full_length_clip", "sampled_count"],
                   help="Importance-weight basis for Ridge fitting on text tokens. "
                        "'none': no reweighting (default). "
                        "'full_length': weight ∝ 1/sqrt(L_full). "
                        "'full_length_clip': weight ∝ 1/sqrt(max(L_full, floor)) — same as full_length but "
                        "clips the denominator to avoid overweighting very short sequences "
                        "(see --length-importance-floor). "
                        "'sampled_count': weight ∝ 1/sqrt(n_sampled) — downweights tokens from sequences "
                        "that contribute more rows to the actual fit batch.")
    p.add_argument("--length-importance-floor", type=int, default=25,
                   help="Floor value for 'full_length_clip' basis: lengths below this are treated as "
                        "this value when computing importance weights (default: 25).")
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
    p.add_argument("--attn-num-heads",        type=int,   default=1,
                   help="Attention heads (must divide embed_dim=768; default: 1)")
    p.add_argument("--device", type=str, default="auto",
                   help="Torch device for attention pooling training: 'auto', 'cuda', 'cpu' (default: auto)")
    
    p.add_argument("--append-cls", action="store_true",
                   help="Append the CLS token as an extra (ungrouped) patch for Ridge fitting and pooling.")
    p.add_argument("--use-global-prior", action="store_true",
                   help="Use the global empirical class prior P(Y) as the divergence reference even when "
                        "context (tabular) features are provided, instead of the per-image P(Y|X_tab).")
    p.add_argument("--prior", type=str, default="label_frequency",
                   choices=["label_frequency", "token_marginal", "current_pool_marginal"],
                   help="Prior to use as the divergence reference: 'label_frequency' (default) uses the "
                        "empirical class prior; 'token_marginal' uses the marginal token/patch prediction distribution; "
                        "'current_pool_marginal' uses the current pooled marginal distribution.")
    p.add_argument("--use-attn-masking", action="store_true",
                   help="During refinement, prevent each image's patches from attending to that image's "
                        "own support row in the TabICL ICL transformer.")
    p.add_argument("--model-selection", type=str, default="last_iteration",
                   choices=["last_iteration", "masked_train_accuracy", "validation_accuracy"],
                   help="Which stage to use at inference after iterative refinement. "
                        "'last_iteration' (default) always uses the final stage. "
                        "'masked_train_accuracy' evaluates every stage on the training set with a "
                        "diagonal attention mask and selects the best-performing one. "
                        "'validation_accuracy' selects the stage with the highest held-out val "
                        "accuracy (requires --train-val-fraction or --cross-validation-cap).")
    p.add_argument("--binary-dist", action="store_true",
                   help="For divergence-based and entropy weight methods, collapse all non-correct "
                        "classes into one before measuring the distributional distance. The comparison "
                        "becomes P(correct) vs P(non-correct) rather than the full class distribution, "
                        "which avoids upweighting patches whose spurious class probabilities happen to "
                        "shift away from the prior.")
    p.add_argument("--aoe-class", type=str, default=None,
                   help="Absence-of-evidence class.")
    p.add_argument("--aoe-handling", type=str, default="filter",
                   choices=["filter", "entropy"],
                   help="How to handle the AoE class during Ridge fitting.")
    p.add_argument("--no-gpu-ridge", dest="gpu_ridge", action="store_false",
                   help="Solve Ridge regression on the CPU (sklearn) instead of the GPU. "
                        "Useful when GPU memory is tight. Default: GPU Ridge when CUDA is available.")
    p.set_defaults(gpu_ridge=True)
    p.add_argument("--post-refinement-viz", action="store_true",
                   help="Skip pre-refinement visualisations; only produce post-refinement figures.")
    p.add_argument("--viz-loo-train", action="store_true",
                   help="Add a leave-one-out visual evaluation of train images without themselves in the support set.")
    p.add_argument("--pred-label-viz", action="store_true",
                   help="Add a discrete per-patch predicted-label panel to visualisation figures.")
    p.add_argument("--minority-prob-viz", action="store_true",
                   help="Add a P(minority class) panel with image-local colour scale to visualisation figures.")
    p.add_argument("--per-class-probs-viz", action="store_true",
                   help="Add one P(class k) heatmap panel per class (only when n_classes <= 10).")
    p.add_argument("--n-train-sweep", type=int, nargs="+", default=None,
                   metavar="N",
                   help="Run one experiment per value and collect results into a single sweep_results.json. Mutually exclusive with --n-train.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   metavar="S",
                   help="Run the experiment once per seed and save results continuously. Mutually exclusive with --seed.")

    args = p.parse_args()

    if args.n_train_sweep is not None and args.n_train is not None:
        p.error("--n-train and --n-train-sweep are mutually exclusive.")
    if args.seeds is not None and args.seed != 42:
        p.error("--seeds and --seed are mutually exclusive.")

    _dataset_defaults = {
        "butterfly":      BUTTERFLY_DATASET_PATH,
        "rsna":           RSNA_DATASET_PATH,
        "petfinder":      PETFINDER_DATASET_PATH,
        "dvm":            DVM_DATASET_PATH,
        "pad-ufes":       PAD_UFES_DATASET_PATH,
        "cbis-ddsm-mass": CBIS_DDSM_DATASET_PATH,
        "cbis-ddsm-calc": CBIS_DDSM_DATASET_PATH,
        "imdb":           IMDB_DATASET_PATH,
        "20news":         TWENTY_NEWS_DATASET_PATH,
        "ag_news":        AG_NEWS_DATASET_PATH,
        "yelp":           YELP_DATASET_PATH,
        "clothing":       CLOTHING_DATASET_PATH,
        "salary":         SALARY_INDIA_DATASET_PATH,
    }
    dataset_path = args.dataset_path or _dataset_defaults[args.dataset]

    # Resolve backbone default based on modality when not explicitly supplied.
    modality = get_modality(args.dataset)
    if args.backbone is not None:
        backbone = args.backbone
    elif args.dataset == "rsna":
        backbone = "rad-dino"
    else:
        backbone = DEFAULT_BACKBONE[modality]

    pca_dim = None if args.no_pca else args.pca_dim

    dataset_cfg = DatasetConfig(
        dataset=args.dataset,
        backbone=backbone,
        features_dir=args.features_dir,
        dataset_path=dataset_path,
        n_train=args.n_train,
        n_test=args.n_test,
        n_val=args.n_val,
        n_sample=args.n_sample,
        balance_train=args.balance_train,
        balance_test=args.balance_test
    )
    
    if modality == "text":
        refinement_cfg: Union[RefinementConfig, TextRefinementConfig] = TextRefinementConfig(
            refine=args.refine,
            text_group_modes=args.text_group_modes,
            temperature=args.temperature,
            ridge_alpha=args.ridge_alpha,
            weight_method=args.weight_method,
            normalize_features=args.normalize_features,
            batch_size=args.batch_size,
            max_query_rows=args.max_query_rows,
            use_random_subsampling=args.use_random_subsampling,
            gpu_ridge=args.gpu_ridge,
            tabicl_n_estimators=args.n_estimators,
            tabicl_pca_dim=pca_dim,
            append_cls=args.append_cls,
            use_global_prior=args.use_global_prior,
            use_attn_masking=args.use_attn_masking,
            model_selection=args.model_selection,
            binary_dist=args.binary_dist,
            prior=args.prior,
            length_importance_weight_basis=args.length_importance_weight_basis,
            length_importance_floor=args.length_importance_floor,
            train_val_fraction=args.train_val_fraction,
            cross_validation_cap=args.cross_validation_cap,
        )
    else:
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
            append_cls=args.append_cls,
            use_global_prior=args.use_global_prior,
            prior=args.prior,
            use_attn_masking=args.use_attn_masking,
            model_selection=args.model_selection,
            binary_dist=args.binary_dist,
            train_val_fraction=args.train_val_fraction,
            cross_validation_cap=args.cross_validation_cap,
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
        viz_loo_train=args.viz_loo_train,
        show_pred_label=args.pred_label_viz,
        show_minority_prob=args.minority_prob_viz,
        show_per_class_probs=args.per_class_probs_viz,
        n_train_sweep=args.n_train_sweep,
        seeds=args.seeds,
    )
    
    return ExperimentConfig(
        dataset=dataset_cfg,
        refinement=refinement_cfg,
        attention=attention_cfg,
        run=run_cfg,
        seed=args.seed,
        device=args.device,
    )
