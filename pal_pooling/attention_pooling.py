"""Learnable attention pooling over patch tokens with a frozen TabICL objective.

``AttentionPoolingHead`` takes ``[N, P, D]`` patch features and outputs
``[N, out_dim]``:

  1. LayerNorm on input patches (for stable Q/K computation only).
  2. K learnable query vectors project into Q; patch tokens project into K.
     **No value (V) projection** — the values are always the original,
     un-normalized patch embeddings.
  3. Multi-head attention weights computed from Q and K, averaged across heads
     → ``[N, K, P]``.
  4. Pooled output = ``attn_weights @ patches`` — a pure weighted average of
     the original patch tokens in the original D-dimensional space.
  5. Flatten K pooled vectors → optional linear projection to ``out_dim``
     (separate post-pooling step for TabICL input-size compatibility).

Steps 1-4 guarantee that the pooled representation is a convex combination of
the original patch embeddings; only the attention weights are learned, not the
embedding contents.  Setting ``num_queries=1`` gives a CLS-like aggregation;
larger values pool at K different attention patterns, all projected to
``out_dim`` in step 5.

Training is driven by an episodic support/query cross-entropy loss through the
frozen TabICL backbone (see ``frozen_tabicl.py``).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from pal_pooling.frozen_tabicl import (
    FrozenTabICLConfig,
    FrozenTabICLBackbone,
    EpisodicTrainingConfig,
    build_frozen_tabicl_backbone,
    _class_safe_support_query_indices,
    _remap_labels_from_support,
    _sample_step_subset,
    _validate_config,
)


# ---------------------------------------------------------------------------
# Attention pooling module
# ---------------------------------------------------------------------------

class AttentionPoolingHead(nn.Module):
    """Attention pooling where the output is a weighted average of original patches.

    Q and K are projected (multi-head), but V is always the raw patch embeddings
    — no value projection.  This ensures the pooled vector lives in the original
    embedding space and represents a convex combination of the input patches.

    Forward pass:
        1. LayerNorm patches → used only for Q/K projection (stabilises training).
        2. Project Q (learnable queries) and K (normed patches) per head.
        3. Compute softmax attention weights ``[N, H, K, P]``, average over heads
           → ``[N, K, P]``.
        4. ``pooled = attn_weights @ patches``  (original, un-normalized patches)
           → ``[N, K, D]``.  This is the pure weighted-average step.
        5. Flatten K pooled vectors ``[N, K*D]`` → linear projection to
           ``out_dim`` (post-pooling step; separate from the attention mechanism).

    Args:
        embed_dim:   Dimension D of input patch tokens.
        out_dim:     Output feature dimension fed to TabICL.  If ``None``,
                     defaults to ``embed_dim`` (no projection in step 5).
        num_queries: Number of learnable query vectors.
        num_heads:   Attention heads (must divide ``embed_dim``).
        dropout:     Dropout on attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        out_dim: Optional[int] = None,
        num_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self.out_dim = embed_dim if out_dim is None else out_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learnable query tokens
        self.queries = nn.Parameter(torch.empty(num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # LayerNorm for Q/K computation only — values always use raw patches
        self.input_norm = nn.LayerNorm(embed_dim)

        # Q and K projections — NO value projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # Post-pooling projection (step 5, separate from attention)
        pooled_dim = num_queries * embed_dim
        self.proj = (
            nn.Linear(pooled_dim, self.out_dim)
            if pooled_dim != self.out_dim
            else nn.Identity()
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: float tensor ``[N, P, D]``

        Returns:
            float tensor ``[N, out_dim]``
        """
        N, P, D = patches.shape
        K, H, d_h = self.num_queries, self.num_heads, self.head_dim

        # Normalize patches for Q/K only; raw patches remain unchanged for V
        patches_norm = self.input_norm(patches)                          # [N, P, D]

        # Expand learnable queries and project Q, K
        Q = self.q_proj(self.queries.unsqueeze(0).expand(N, -1, -1))    # [N, K, D]
        K_proj = self.k_proj(patches_norm)                               # [N, P, D]

        # Reshape to multi-head: [N, H, K/P, d_h]
        Q      = Q.view(N, K, H, d_h).transpose(1, 2)                   # [N, H, K, d_h]
        K_proj = K_proj.view(N, P, H, d_h).transpose(1, 2)              # [N, H, P, d_h]

        # Attention weights
        attn = (Q @ K_proj.transpose(-2, -1)) * (d_h ** -0.5)           # [N, H, K, P]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Average attention weights across heads → [N, K, P]
        attn_avg = attn.mean(dim=1)

        # Step 4: weighted average of ORIGINAL (un-normalized) patch tokens
        pooled = torch.bmm(attn_avg, patches)                            # [N, K, D]

        # Step 5: flatten and project (post-pooling, not part of attention)
        return self.proj(pooled.reshape(N, K * D))                       # [N, out_dim]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_attention_pooling_head(
    train_patches: torch.Tensor,
    y_train: np.ndarray,
    *,
    val_patches: Optional[torch.Tensor] = None,
    y_val: Optional[np.ndarray] = None,
    embed_dim: int,
    out_dim: Optional[int] = None,
    num_queries: int = 1,
    num_heads: int = 8,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
    tabicl_config: Optional[FrozenTabICLConfig] = None,
    tabicl_backbone: Optional[FrozenTabICLBackbone] = None,
    config: Optional[EpisodicTrainingConfig] = None,
    verbose: bool = True,
) -> tuple[AttentionPoolingHead, dict]:
    """Train an attention pooling head on patch features with frozen TabICL.

    The training objective is identical to ``train_projection_head``: episodic
    support/query cross-entropy through the frozen TabICL backbone.  The key
    difference is that the input is a 3D patch tensor ``[N, P, D]`` and the
    head performs learnable pooling as its first step.

    If ``val_patches`` / ``y_val`` are provided, validation accuracy is tracked
    every ``config.log_every`` steps and the best checkpoint is restored.

    Args:
        train_patches: float tensor ``[N_train, P, D]``.
        y_train:       integer class labels ``[N_train]``.
        val_patches:   optional float tensor ``[N_val, P, D]``.
        y_val:         optional integer class labels ``[N_val]``.
        embed_dim:     patch token dimension D.
        out_dim:       output feature dimension (fed to TabICL backbone).
                       Defaults to ``embed_dim`` (no post-pooling projection).
        num_queries:   number of learnable attention query vectors.
        num_heads:     number of attention heads.
        config:        ``EpisodicTrainingConfig`` (uses defaults if None).

    Returns:
        ``(head, history)`` — head restored to best val checkpoint if available.
    """
    if config is None:
        config = EpisodicTrainingConfig()
    _validate_config(config)

    device = torch.device(device)
    X_t = train_patches.to(dtype=torch.float32, device=device)   # [N, P, D]
    y_t = torch.as_tensor(y_train, dtype=torch.long, device=device)

    use_validation = val_patches is not None and y_val is not None
    if use_validation:
        X_val_t = val_patches.to(dtype=torch.float32, device=device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)
    else:
        X_val_t = y_val_t = None

    # Resolve effective output dim for backbone input_dim
    effective_out_dim = embed_dim if out_dim is None else out_dim

    # Build frozen TabICL backbone
    if tabicl_backbone is None:
        tabicl_cfg = tabicl_config or FrozenTabICLConfig()
        tabicl_backbone = build_frozen_tabicl_backbone(
            input_dim=effective_out_dim,
            config=tabicl_cfg,
            device=device,
        )
    tabicl_backbone = tabicl_backbone.to(device)
    tabicl_backbone.eval()

    head = AttentionPoolingHead(
        embed_dim=embed_dim,
        out_dim=out_dim,
        num_queries=num_queries,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[attention pooling] Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed)

    history: dict[str, list[float]] = {
        "loss": [],
        "query_fraction": [],
        "query_size": [],
        "val_step": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_state: Optional[dict] = None
    best_val_step: int = 0
    best_time_s: float = 0.0
    t_train_start = time.perf_counter()

    def _run_validation(step_for_eval: int) -> tuple[float, float]:
        assert X_val_t is not None and y_val_t is not None
        head.eval()
        with torch.no_grad():
            X_support_proj = head(X_t)            # [N_train, out_dim]
            X_query_proj = head(X_val_t)          # [N_val,   out_dim]
            y_sup_local, y_q_local = _remap_labels_from_support(y_t, y_val_t)
            logits_val = tabicl_backbone(
                X_support_proj, y_sup_local, X_query_proj,
                step_index=step_for_eval - 1,
            )
            val_loss = float(criterion(logits_val, y_q_local).item())
            val_acc = float((logits_val.argmax(dim=1) == y_q_local).float().mean().item())
        return val_loss, val_acc

    for step in range(1, config.num_steps + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)

        # Optionally downsample rows for this step
        X_step, y_step = _sample_step_subset(X_t, y_t, rng, config.max_step_samples)
        n_step = int(X_step.shape[0])

        support_idx_np, query_idx_np = _class_safe_support_query_indices(
            y=y_step,
            rng=rng,
            frac_min=config.query_fraction_min,
            frac_max=config.query_fraction_max,
            min_query_size=config.min_query_size,
            query_minority_weight=config.query_minority_weight,
        )
        support_idx = torch.as_tensor(support_idx_np, device=device, dtype=torch.long)
        query_idx   = torch.as_tensor(query_idx_np,   device=device, dtype=torch.long)

        # Pool patches → feature vectors, then split support/query
        X_proj    = head(X_step)                              # [n_step, out_dim]
        X_support = X_proj.index_select(0, support_idx)
        y_support = y_step.index_select(0, support_idx)
        X_query   = X_proj.index_select(0, query_idx)
        y_query   = y_step.index_select(0, query_idx)

        y_support_local, y_query_local = _remap_labels_from_support(y_support, y_query)
        logits = tabicl_backbone(X_support, y_support_local, X_query, step_index=step - 1)
        loss = criterion(logits, y_query_local)
        loss.backward()
        if config.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(head.parameters(), config.grad_clip_norm)
        optimizer.step()

        query_frac = float(y_query.shape[0] / max(1, n_step))
        history["loss"].append(float(loss.detach().cpu().item()))
        history["query_fraction"].append(query_frac)
        history["query_size"].append(float(y_query.shape[0]))

        if verbose and (step == 1 or step % config.log_every == 0 or step == config.num_steps):
            print(
                f"[step {step:04d}/{config.num_steps}] "
                f"loss={history['loss'][-1]:.4f}  "
                f"query={int(y_query.shape[0])}/{n_step} ({query_frac:.3f})"
            )

        if use_validation and (step % config.log_every == 0 or step == config.num_steps):
            val_loss, val_acc = _run_validation(step)
            history["val_step"].append(float(step))
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            if verbose:
                print(
                    f"[val  step {step:04d}/{config.num_steps}] "
                    f"loss={val_loss:.4f}  acc={val_acc:.4f}"
                )
            is_better = (val_acc > best_val_acc) or (
                val_acc == best_val_acc and val_loss < best_val_loss
            )
            if is_better:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                best_val_step = step
                best_time_s = time.perf_counter() - t_train_start

    if use_validation and best_state is not None:
        print(f"[attention pooling] Restoring best checkpoint (val_acc={best_val_acc:.4f}  "
              f"step={best_val_step}  time_to_best={best_time_s:.1f}s)")
        head.load_state_dict(best_state)

    history["best_val_step"] = best_val_step
    history["time_to_best_s"] = round(best_time_s, 2)

    return head, history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _pool_with_head(
    head: AttentionPoolingHead,
    patches: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Apply a trained attention pooling head in mini-batches."""
    head.eval()
    head = head.to(device)
    results = []
    for start in range(0, len(patches), batch_size):
        batch = patches[start : start + batch_size].to(dtype=torch.float32, device=device)
        results.append(head(batch).cpu())
    return torch.cat(results, dim=0).numpy()
