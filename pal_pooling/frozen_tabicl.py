"""Frozen TabICL backbone and episodic training utilities.

Provides the frozen TabICL backbone (``FrozenTabICLBackbone``) used as a
training signal in attention-pooling experiments, together with the shared
helpers for episodic support/query splits used in the training loop
(``_class_safe_support_query_indices``, ``_remap_labels_from_support``,
``_sample_step_subset``, ``_validate_config``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tabicl.model.inference_config import InferenceConfig
from tabicl.model.tabicl import TabICL

try:
	from huggingface_hub import hf_hub_download
	from huggingface_hub.utils import LocalEntryNotFoundError
except Exception:  # pragma: no cover - optional dependency at import time
	hf_hub_download = None
	LocalEntryNotFoundError = Exception


@dataclass
class FrozenTabICLConfig:
	"""Configuration for frozen TabICL model + feature-shuffle pool."""

	n_models: int = 1
	n_feature_shuffles: int | None = None
	model_seeds: list[int] | None = None
	feature_shuffle_method: str = "random"
	shuffle_seed: int = 0
	checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt"
	model_path: str | Path | None = None
	allow_auto_download: bool = True
	softmax_temperature: float = 0.9


@dataclass
class EpisodicTrainingConfig:
	"""Hyperparameters for episodic support/query training."""

	num_steps: int = 1000
	learning_rate: float = 1e-3
	weight_decay: float = 0.0
	query_fraction_min: float = 0.1
	query_fraction_max: float = 0.4
	min_query_size: int = 8
	query_minority_weight: float = 1.0
	max_step_samples: int | None = None
	grad_clip_norm: float | None = None
	seed: int = 0
	log_every: int = 50


def freeze_module(module: nn.Module) -> None:
	"""Freeze a torch module in-place and switch to eval mode."""
	module.eval()
	for param in module.parameters():
		param.requires_grad_(False)


class FrozenTabICLBackbone(nn.Module):
	"""Frozen single TabICL model with per-step feature-shuffle selection."""

	def __init__(
		self,
		models: list[TabICL],
		feature_shuffles: list[list[int] | None],
		softmax_temperature: float = 0.9,
		inference_config: InferenceConfig | None = None,
	) -> None:
		super().__init__()
		if len(models) != 1:
			raise ValueError(
				"FrozenTabICLBackbone expects exactly one TabICL model. "
				"Use feature shuffles across steps instead of model ensembling."
			)
		if len(feature_shuffles) == 0:
			raise ValueError("feature_shuffles must contain at least one pattern")

		self.models = nn.ModuleList(models)
		self.feature_shuffles = feature_shuffles
		self.softmax_temperature = float(softmax_temperature)
		if inference_config is None:
			# Start from full defaults so required fields (e.g. offload/device) exist,
			# then enable autograd for projection-head training.
			cfg = InferenceConfig()
			cfg.update_from_dict(
				{
					"COL_CONFIG": {"enable_grad": True},
					"ROW_CONFIG": {"enable_grad": True},
					"ICL_CONFIG": {"enable_grad": True},
				}
			)
			self.inference_config = cfg
		else:
			self.inference_config = inference_config

		for model in self.models:
			freeze_module(model)

	def forward(
		self,
		X_support: torch.Tensor,
		y_support: torch.Tensor,
		X_query: torch.Tensor,
		step_index: int | None = None,
	) -> torch.Tensor:
		"""Return logits for one selected shuffle pattern.

		If ``step_index`` is provided, a deterministic shuffle pattern is selected
		via ``step_index % n_feature_shuffles``.
		"""
		if X_support.ndim != 2 or X_query.ndim != 2:
			raise ValueError("X_support and X_query must be rank-2 tensors")
		if X_support.shape[1] != X_query.shape[1]:
			raise ValueError("X_support and X_query must have matching feature dimensions")
		if y_support.ndim != 1:
			y_support = y_support.reshape(-1)
		if y_support.shape[0] != X_support.shape[0]:
			raise ValueError("y_support size must match X_support rows")

		X_concat = torch.cat([X_support, X_query], dim=0)
		y_train = y_support.float()

		shuffle_idx = 0 if step_index is None else int(step_index) % len(self.feature_shuffles)
		shuffle = self.feature_shuffles[shuffle_idx]
		shuffle_arg = None if shuffle is None else [shuffle]

		model = self.models[0]
		out = model(
			X=X_concat.unsqueeze(0),
			y_train=y_train.unsqueeze(0),
			feature_shuffles=shuffle_arg,
			return_logits=True,
			softmax_temperature=self.softmax_temperature,
			inference_config=self.inference_config,
		)
		return out[:, -X_query.shape[0] :, :].squeeze(0)


def _resolve_checkpoint_path(
	model_path: str | Path | None,
	checkpoint_version: str,
	allow_auto_download: bool,
) -> Path:
	"""Resolve checkpoint path from local filesystem or Hugging Face Hub."""
	repo_id = "jingang/TabICL"

	if model_path is not None:
		p = Path(model_path)
		if p.exists():
			return p
		if not allow_auto_download:
			raise FileNotFoundError(
				f"Checkpoint not found at {p} and allow_auto_download=False"
			)
		if hf_hub_download is None:
			raise ImportError(
				"huggingface_hub is required for auto-download but is not installed"
			)
		p.parent.mkdir(parents=True, exist_ok=True)
		cache_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_version, local_dir=p.parent)
		cache_file = Path(cache_path)
		if cache_file != p:
			cache_file.rename(p)
		return p

	if hf_hub_download is None:
		raise ImportError(
			"huggingface_hub is required when model_path is None"
		)

	try:
		cached = hf_hub_download(repo_id=repo_id, filename=checkpoint_version, local_files_only=True)
		return Path(cached)
	except LocalEntryNotFoundError:
		if not allow_auto_download:
			raise FileNotFoundError(
				f"Checkpoint {checkpoint_version} not cached and allow_auto_download=False"
			)
		return Path(hf_hub_download(repo_id=repo_id, filename=checkpoint_version))


def _load_frozen_tabicl_model(
	*,
	checkpoint_path: Path,
	device: torch.device,
	model_seed: int | None,
) -> TabICL:
	"""Instantiate and load a frozen PyTorch TabICL model from checkpoint."""
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
	if "config" not in checkpoint or "state_dict" not in checkpoint:
		raise ValueError(
			f"Invalid checkpoint format at {checkpoint_path}; expected config and state_dict"
		)

	if model_seed is not None:
		torch.manual_seed(int(model_seed))

	model = TabICL(**checkpoint["config"])  # type: ignore[arg-type]
	model.load_state_dict(checkpoint["state_dict"])
	model = model.to(device)
	freeze_module(model)
	return model


def _generate_feature_shuffle(
	n_features: int,
	method: str,
	rng: np.random.Generator,
	index: int,
) -> list[int] | None:
	"""Create one feature-shuffle pattern for one training-step variant."""
	if method == "none":
		return None
	if method == "random":
		return np.asarray(rng.permutation(n_features), dtype=np.int64).tolist()
	if method == "shift":
		shift = int(index % max(1, n_features))
		base = np.arange(n_features, dtype=np.int64)
		return np.roll(base, shift).tolist()
	raise ValueError("feature_shuffle_method must be one of: 'none', 'random', 'shift'")


def build_frozen_tabicl_backbone(
	input_dim: int,
	*,
	config: FrozenTabICLConfig,
	device: str | torch.device = "cpu",
) -> FrozenTabICLBackbone:
	"""Build a frozen single TabICL backbone with a shuffle pool."""
	if input_dim <= 0:
		raise ValueError("input_dim must be > 0")
	if config.n_models <= 0:
		raise ValueError("n_models must be > 0")

	model_seeds = config.model_seeds
	if model_seeds is None:
		model_seeds = [config.shuffle_seed + i for i in range(config.n_models)]
	if len(model_seeds) != config.n_models:
		raise ValueError("len(model_seeds) must match n_models")

	device_t = torch.device(device)
	ckpt_path = _resolve_checkpoint_path(
		model_path=config.model_path,
		checkpoint_version=config.checkpoint_version,
		allow_auto_download=config.allow_auto_download,
	)

	shuffle_rng = np.random.default_rng(config.shuffle_seed)
	models: list[TabICL] = []
	feature_shuffles: list[list[int] | None] = []

	# Use one TabICL model; vary only feature shuffles across training steps.
	model = _load_frozen_tabicl_model(
		checkpoint_path=ckpt_path,
		device=device_t,
		model_seed=model_seeds[0],
	)
	models.append(model)

	n_shuffles = int(config.n_feature_shuffles or config.n_models)
	n_shuffles = max(1, n_shuffles)
	for idx in range(n_shuffles):
		feature_shuffles.append(
			_generate_feature_shuffle(
				n_features=input_dim,
				method=config.feature_shuffle_method,
				rng=shuffle_rng,
				index=idx,
			)
		)

	return FrozenTabICLBackbone(
		models=models,
		feature_shuffles=feature_shuffles,
		softmax_temperature=config.softmax_temperature,
	)


def _validate_config(cfg: EpisodicTrainingConfig) -> None:
	if cfg.num_steps <= 0:
		raise ValueError("num_steps must be > 0")
	if cfg.learning_rate <= 0:
		raise ValueError("learning_rate must be > 0")
	if not (0.0 < cfg.query_fraction_min < 1.0):
		raise ValueError("query_fraction_min must be in (0, 1)")
	if not (0.0 < cfg.query_fraction_max < 1.0):
		raise ValueError("query_fraction_max must be in (0, 1)")
	if cfg.query_fraction_min > cfg.query_fraction_max:
		raise ValueError("query_fraction_min cannot exceed query_fraction_max")
	if cfg.min_query_size <= 0:
		raise ValueError("min_query_size must be > 0")
	if cfg.query_minority_weight < 1.0:
		raise ValueError("query_minority_weight must be >= 1.0")
	if cfg.max_step_samples is not None and cfg.max_step_samples <= 1:
		raise ValueError("max_step_samples must be > 1 when set")


def _sample_support_query_indices(
	n_rows: int,
	rng: np.random.Generator,
	frac_min: float,
	frac_max: float,
	min_query_size: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Split rows into support and query indices for one optimization step."""
	if n_rows < 3:
		raise ValueError("Need at least 3 rows to create support/query splits")

	q_frac = float(rng.uniform(frac_min, frac_max))
	q_size = int(round(q_frac * n_rows))
	q_size = max(min_query_size, q_size)
	q_size = min(q_size, n_rows - 1)

	perm = rng.permutation(n_rows)
	query_idx = np.asarray(perm[:q_size], dtype=np.int64)
	support_idx = np.asarray(perm[q_size:], dtype=np.int64)
	if support_idx.size == 0:
		support_idx = query_idx[-1:]
		query_idx = query_idx[:-1]
	return support_idx, query_idx


def _class_safe_support_query_indices(
	y: torch.Tensor,
	rng: np.random.Generator,
	frac_min: float,
	frac_max: float,
	min_query_size: int,
	query_minority_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
	"""Split rows so every class in query is present in support.

	For each class we reserve at least one support sample, then sample query from
	remaining rows. This avoids unseen query classes and keeps TabICL class ids
	well-defined for each step.
	"""
	y_np = np.asarray(y.detach().cpu().numpy()).reshape(-1)
	n_rows = int(y_np.shape[0])
	if n_rows < 3:
		raise ValueError("Need at least 3 rows to create support/query splits")

	classes, inverse = np.unique(y_np, return_inverse=True)
	if classes.size <= 1:
		return _sample_support_query_indices(n_rows, rng, frac_min, frac_max, min_query_size)

	# Keep one support exemplar per class.
	support_keep: list[int] = []
	remaining: list[int] = []
	for cls_id in range(classes.size):
		cls_idx = np.flatnonzero(inverse == cls_id)
		rng.shuffle(cls_idx)
		support_keep.append(int(cls_idx[0]))
		if cls_idx.size > 1:
			remaining.extend(int(i) for i in cls_idx[1:])

	remaining_np = np.asarray(remaining, dtype=np.int64)
	rng.shuffle(remaining_np)

	q_frac = float(rng.uniform(frac_min, frac_max))
	q_size = int(round(q_frac * n_rows))
	max_query = int(remaining_np.size)
	q_size = min(max_query, max(1, q_size))
	if max_query > 0:
		q_size = max(min_query_size, q_size)
		q_size = min(max_query, q_size)

	if q_size > 0 and query_minority_weight > 1.0:
		# Upweight least frequent class(es) when drawing query rows.
		class_counts = np.bincount(inverse, minlength=classes.size)
		minority_cls = np.flatnonzero(class_counts == class_counts.min())
		row_class_ids = inverse[remaining_np]
		weights = np.ones(remaining_np.shape[0], dtype=np.float64)
		weights[np.isin(row_class_ids, minority_cls)] *= float(query_minority_weight)
		probs = weights / weights.sum()
		query_idx = np.asarray(
			rng.choice(remaining_np, size=q_size, replace=False, p=probs),
			dtype=np.int64,
		)
		remaining_after_query = np.setdiff1d(remaining_np, query_idx, assume_unique=True)
	else:
		query_idx = remaining_np[:q_size] if q_size > 0 else np.empty((0,), dtype=np.int64)
		remaining_after_query = remaining_np[q_size:]

	support_idx = np.concatenate(
		[
			np.asarray(support_keep, dtype=np.int64),
			remaining_after_query,
		]
	)
	rng.shuffle(support_idx)

	if query_idx.size == 0:
		# Fallback: move one non-reserved support item into query when possible.
		if support_idx.size <= classes.size:
			return _sample_support_query_indices(n_rows, rng, frac_min, frac_max, min_query_size)
		move_idx = int(support_idx[-1])
		support_idx = support_idx[:-1]
		query_idx = np.asarray([move_idx], dtype=np.int64)

	return support_idx, query_idx


def _remap_labels_from_support(
	y_support: torch.Tensor,
	y_query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Map labels to contiguous ids based on support-set classes.

	Returns remapped support/query labels in ``[0, n_classes_step)``.
	"""
	support_classes = torch.unique(y_support)
	if support_classes.numel() <= 0:
		raise ValueError("Support labels are empty")

	min_cls = int(torch.min(support_classes).item())
	max_cls = int(torch.max(support_classes).item())
	mapping = torch.full(
		(max_cls - min_cls + 1,),
		-1,
		dtype=torch.long,
		device=y_support.device,
	)
	mapping[support_classes - min_cls] = torch.arange(
		support_classes.numel(), dtype=torch.long, device=y_support.device
	)

	y_support_mapped = mapping[y_support - min_cls]
	y_query_mapped = mapping[y_query - min_cls]
	if torch.any(y_query_mapped < 0):
		raise ValueError(
			"Query contains classes absent from support after split; "
			"cannot compute class-consistent loss"
		)
	return y_support_mapped, y_query_mapped


def _sample_step_subset(
	X: torch.Tensor,
	y: torch.Tensor,
	rng: np.random.Generator,
	max_step_samples: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Optionally downsample rows used in the current training step."""
	if max_step_samples is None or X.shape[0] <= max_step_samples:
		return X, y
	idx = rng.choice(X.shape[0], size=max_step_samples, replace=False)
	idx_t = torch.as_tensor(idx, device=X.device, dtype=torch.long)
	return X.index_select(0, idx_t), y.index_select(0, idx_t)
