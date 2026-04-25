"""Core patch quality scoring and pooling algorithms.

Functions here are pure NumPy/sklearn — no I/O, no visualisation.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pal_pooling.config import ImagePALConfig
from pal_pooling.tabicl_gpu_adapter import TabICLGPUAdapter
from tabicl import TabICLClassifier
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Safe TabICL wrapper for compatibility
# ---------------------------------------------------------------------------

def _safe_predict_proba(clf, X, blocked_indices=None):
    """Call predict_proba with optional blocked_indices for leave-one-out masking.

    Falls back to standard predict_proba if blocked_indices is not supported.
    """
    if blocked_indices is None:
        return clf.predict_proba(X)

    try:
        return clf.predict_proba(X, blocked_indices=blocked_indices)
    except TypeError as e:
        if 'blocked_indices' in str(e):
            return clf.predict_proba(X)
        raise


def _safe_predict_proba_tensor(clf, X_t, blocked_indices=None):
    """Call predict_proba_tensor with optional blocked_indices for leave-one-out masking.

    Falls back to standard predict_proba_tensor if blocked_indices is not supported.
    """
    if blocked_indices is None:
        return clf.predict_proba_tensor(X_t)

    try:
        return clf.predict_proba_tensor(X_t, blocked_indices=blocked_indices)
    except TypeError as e:
        if 'blocked_indices' in str(e):
            return clf.predict_proba_tensor(X_t)
        raise


# ---------------------------------------------------------------------------
# GPU-accelerated Ridge (optional drop-in for sklearn Ridge)
# ---------------------------------------------------------------------------

class RidgeGPU:
    """Ridge regression solved on the GPU via normal equations.

    Solves  (Xc^T Xc + alpha * I) w = Xc^T yc  where Xc/yc are mean-centred,
    then recovers the intercept.  This mirrors sklearn Ridge(fit_intercept=True)
    and is a drop-in replacement with the same ``fit / predict / score`` API.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength (same semantics as sklearn Ridge).
    device : str
        Torch device string, e.g. ``"cuda"``, ``"cuda:1"``, or ``"cpu"``.
        If the requested device is not available, falls back to CPU with a warning.
    """

    def __init__(self, alpha: float = 1.0, device: str = "cuda") -> None:
        self.alpha = alpha
        self.device = device
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self._w = None   # torch.Tensor on device
        self._b = None   # torch.Tensor on device

    def _get_device(self):
        import torch
        dev = torch.device(self.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            import warnings
            warnings.warn(
                f"[RidgeGPU] CUDA requested but not available; falling back to CPU.",
                RuntimeWarning, stacklevel=3,
            )
            dev = torch.device("cpu")
        return dev

    def fit(
        self,
        X: "Union[np.ndarray, torch.Tensor]",
        y: "Union[np.ndarray, torch.Tensor]",
        sample_weight: "Optional[Union[np.ndarray, torch.Tensor]]" = None,
    ) -> "RidgeGPU":
        """Fit Ridge on GPU.  X: [N, D], y: [N].

        Accepts either numpy arrays or torch tensors.  Tensors already on the
        target device are used in-place (no copy); tensors on a different device
        are moved with ``.to()``.

        sample_weight : array [N], optional
            Per-sample weights.  Should sum to N (mean = 1) so that the
            effective regularisation strength is comparable to the unweighted
            case.  When None, uniform weights are used.
        """
        import torch
        dev = self._get_device()
        if isinstance(X, torch.Tensor):
            Xt = X.to(dev).float()
        else:
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(dev)
        if isinstance(y, torch.Tensor):
            yt = y.to(dev).float()
        else:
            yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(dev)

        if sample_weight is not None:
            if isinstance(sample_weight, torch.Tensor):
                wt = sample_weight.to(dev).float()
            else:
                wt = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)).to(dev)
            w_sum = wt.sum()
            X_mean = (wt[:, None] * Xt).sum(0) / w_sum
            y_mean = (wt * yt).sum() / w_sum
            Xc = Xt - X_mean
            yc = yt - y_mean
            # Weighted normal equations: (Xc^T W Xc + alpha I) w = Xc^T W yc
            WXc = wt[:, None] * Xc
            A     = Xc.T @ WXc
            b_vec = Xc.T @ (wt * yc)
        else:
            # Mean-centre to absorb the intercept
            X_mean = Xt.mean(0)   # [D]
            y_mean = yt.mean()    # scalar
            Xc = Xt - X_mean
            yc = yt - y_mean
            # Normal equations: (Xc^T Xc + alpha * I) w = Xc^T yc
            A     = Xc.T @ Xc                     # [D, D]
            b_vec = Xc.T @ yc                     # [D]

        A.diagonal().add_(self.alpha)     # in-place ridge penalty
        w = torch.linalg.solve(A, b_vec)  # [D]
        intercept = y_mean - X_mean @ w

        self._w = w
        self._b = intercept
        self.coef_      = w.cpu().float().numpy()
        self.intercept_ = intercept.item()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on GPU; returns CPU numpy array."""
        import torch
        dev = self._w.device
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(dev)
        out = Xt @ self._w + self._b
        return out.cpu().numpy()

    def score(self, X: "Union[np.ndarray, torch.Tensor]", y: "Union[np.ndarray, torch.Tensor]") -> float:
        """R² on CPU (matches sklearn Ridge.score).  Accepts tensors."""
        import torch
        y_pred = self.predict(X)
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().float().numpy()
        else:
            y_np = np.asarray(y, dtype=np.float32)
        ss_res = float(((y_np - y_pred) ** 2).sum())
        ss_tot = float(((y_np - y_np.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Patch entropy and pooling weights
# ---------------------------------------------------------------------------

def compute_patch_entropy(patch_probs: np.ndarray) -> np.ndarray:
    """Per-patch Shannon entropy in nats.

    Parameters
    ----------
    patch_probs : np.ndarray, shape [P, n_classes]

    Returns
    -------
    np.ndarray, shape [P]
        Raw entropy values in [0, ln(n_classes)].
    """
    eps = 1e-9
    return -(patch_probs * np.log(patch_probs + eps)).sum(axis=1)


def compute_patch_pooling_weights(
    patch_probs:   np.ndarray,             # [P, n_classes]
    true_label:    int,
    temperature:   float = 1.0,
    weight_method: str   = "correct_class_prob",
    class_prior:   Optional[np.ndarray] = None,  # [n_classes]  empirical class frequencies
    binary_dist:   bool  = False,
) -> np.ndarray:                           # [P]  sums to 1
    """Derive per-patch pooling weights from TabICL softmax predictions.

    Four methods are supported, selected by *weight_method*:

    ``"correct_class_prob"`` (default)
        1. Extract the true-class probability for each patch  →  p_i in (0, 1).
        2. Take the log-probability                           →  ln(p_i).
        3. Apply a temperature-scaled softmax across patches  →  weights w_i.

        temperature → ∞  : log-probs collapse to zero  →  uniform (mean) pooling
        temperature = 1  : weights ∝ p_true
        temperature → 0  : all weight on the single most-confident patch

    ``"entropy"``
        1. Compute Shannon entropy H_i = -Σ p_c log p_c for each patch.
        2. Normalise to [0, 1]: score_i = 1 - H_i / ln(C)  (0 = max entropy, 1 = zero entropy).
        3. Apply log to get a logit: logit_i = ln(score_i)  (in (-∞, 0]).
        4. Apply temperature-scaled softmax across patches  →  weights w_i.

        temperature → ∞  : logits collapse to zero  →  uniform (mean) pooling
        temperature → 0  : all weight on the single lowest-entropy patch

        Note: *true_label* is not used by this method unless *binary_dist* is True.

    ``"kl_div"``
        1. Compute KL(Q_i || P_prior) for each patch, where Q_i is the predicted
           distribution and P_prior is the empirical class frequency vector.
           KL measures how much the patch prediction deviates from the base rates;
           high KL → patch is discriminative.
        2. Normalise by the maximum achievable KL: max_KL = -ln(min_c P_prior_c),
           attained when Q is a point mass on the rarest class  →  score_i in [0, 1].
        3. Apply log to get a logit: logit_i = ln(score_i)  (in (-∞, 0]).
        4. Apply temperature-scaled softmax across patches  →  weights w_i.

        Requires *class_prior* (empirical class frequencies, shape [n_classes]).
        Unlike ``"entropy"``, this method rewards predictions that deviate from the
        base rates, making it sensitive to class imbalance: a patch that confidently
        predicts a rare class receives a higher score than one equally confident about
        a common class.  Note: *true_label* is not used by this method unless
        *binary_dist* is True.

    ``"wasserstein"``
        1. Compute the Wasserstein-1 (Earth Mover's) distance between Q_i and
           P_prior for each patch: W_i = Σ_k |CDF_Q_i(k) - CDF_P(k)|, where the
           CDF is taken over classes ordered 0 … C-1.  High W → patch prediction
           is far from the base-rate distribution → discriminative.
        2. Normalise by C - 1 (the maximum W1 distance for C classes)
           →  score_i in [0, 1].
        3. Apply log to get a logit: logit_i = ln(score_i)  (in (-∞, 0]).
        4. Apply temperature-scaled softmax across patches  →  weights w_i.

        Requires *class_prior* (empirical class frequencies, shape [n_classes]).
        Note: *true_label* is not used by this method unless *binary_dist* is True.

    ``"tvd"``
        1. Compute the Total Variation Distance (TVD) between Q_i and P_prior
           for each patch: TVD = 0.5 * Σ_k |Q_i(k) - P_prior(k)|.
           High TVD → discriminative.
        2. TVD is naturally bounded in [0, 1]. score_i = TVD_i.
        3. Apply log to get a logit: logit_i = ln(score_i)  (in (-∞, 0]).
        4. Apply temperature-scaled softmax across patches  →  weights w_i.

        Requires *class_prior* (empirical class frequencies, shape [n_classes]).
        Note: *true_label* is not used by this method unless *binary_dist* is True.

    ``"js_div"``
        1. Compute the Jensen-Shannon divergence JSD(Q_i || P_prior) for each
           patch: JSD = 0.5·KL(Q||M) + 0.5·KL(P||M) where M = (Q + P) / 2.
           JSD is symmetric and bounded in [0, ln 2].
        2. Normalise by ln 2  →  score_i in [0, 1].
        3. Apply log to get a logit: logit_i = ln(score_i)  (in (-∞, 0]).
        4. Apply temperature-scaled softmax across patches  →  weights w_i.

        Requires *class_prior* (empirical class frequencies, shape [n_classes]).
        Note: *true_label* is not used by this method unless *binary_dist* is True.

    When *binary_dist* is True (and *weight_method* is not ``"correct_class_prob"``),
    the predicted distribution and prior are collapsed to a 2-class representation
    ``[P(correct), P(non-correct)]`` before any distance is computed.  This avoids
    upweighting patches whose spurious shifts among non-correct classes happen to look
    discriminative, while still rewarding patches whose correct-class probability
    diverges from the prior.
    """
    if binary_dist and weight_method != "correct_class_prob":
        p_correct = patch_probs[:, true_label]                                  # [P]
        patch_probs = np.stack([p_correct, 1.0 - p_correct], axis=1)           # [P, 2]
        if class_prior is not None:
            pc = float(np.asarray(class_prior)[true_label])
            class_prior = np.array([pc, 1.0 - pc], dtype=np.float64)

    if weight_method == "entropy":
        n_classes = patch_probs.shape[1]
        raw_entropy = compute_patch_entropy(patch_probs)          # [P] in [0, ln(C)]
        scores = (1.0 - raw_entropy / np.log(n_classes)).clip(1e-7, 1.0)  # [P] in (0, 1]
        logits = np.log(scores)                                    # [P] in (-inf, 0]
        logits_scaled = logits / temperature
        logits_scaled -= logits_scaled.max()                       # numerical stability
        weights = np.exp(logits_scaled)
        weights /= weights.sum()
        return weights

    if weight_method == "kl_div":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='kl_div'")
        prior = np.asarray(class_prior, dtype=np.float64).clip(1e-9, 1.0)
        prior /= prior.sum()
        q = patch_probs.astype(np.float64).clip(1e-9, 1.0)        # [P, n_classes]
        kl = (q * np.log(q / prior[None, :])).sum(axis=1)         # [P] KL(Q_i || P_prior)
        max_kl = -np.log(prior.min())                              # theoretical maximum
        scores = (kl / max_kl).clip(1e-7, 1.0)                    # [P] in (0, 1]
        logits = np.log(scores)                                    # [P] in (-inf, 0]
        logits_scaled = logits / temperature
        logits_scaled -= logits_scaled.max()                       # numerical stability
        weights = np.exp(logits_scaled)
        weights /= weights.sum()
        return weights.astype(np.float32)

    if weight_method == "wasserstein":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='wasserstein'")
        prior = np.asarray(class_prior, dtype=np.float64).clip(1e-9, 1.0)
        prior /= prior.sum()
        q = patch_probs.astype(np.float64)                         # [P, n_classes]
        cdf_q = np.cumsum(q, axis=1)                               # [P, n_classes]
        cdf_p = np.cumsum(prior)                                   # [n_classes]
        w1 = np.abs(cdf_q - cdf_p[None, :]).sum(axis=1)           # [P] W1 distance
        n_classes = patch_probs.shape[1]
        scores = (w1 / (n_classes - 1)).clip(1e-7, 1.0)           # [P] normalised to (0, 1]
        logits = np.log(scores)                                    # [P] in (-inf, 0]
        logits_scaled = logits / temperature
        logits_scaled -= logits_scaled.max()                       # numerical stability
        weights = np.exp(logits_scaled)
        weights /= weights.sum()
        return weights.astype(np.float32)

    if weight_method == "tvd":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='tvd'")
        prior = np.asarray(class_prior, dtype=np.float64).clip(1e-9, 1.0)
        prior /= prior.sum()
        q = patch_probs.astype(np.float64)                         # [P, n_classes]
        tvd = 0.5 * np.abs(q - prior[None, :]).sum(axis=1)         # [P] TVD
        scores = tvd.clip(1e-7, 1.0)                               # [P] bounds
        logits = np.log(scores)                                    # [P] in (-inf, 0]
        logits_scaled = logits / temperature
        logits_scaled -= logits_scaled.max()                       # numerical stability
        weights = np.exp(logits_scaled)
        weights /= weights.sum()
        return weights.astype(np.float32)

    if weight_method == "js_div":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='js_div'")
        prior = np.asarray(class_prior, dtype=np.float64).clip(1e-9, 1.0)
        prior /= prior.sum()
        q = patch_probs.astype(np.float64).clip(1e-9, 1.0)        # [P, n_classes]
        m = 0.5 * (q + prior[None, :])                            # [P, n_classes] mixture
        jsd = 0.5 * (q * np.log(q / m)).sum(axis=1) \
            + 0.5 * (prior * np.log(prior / m)).sum(axis=1)       # [P] in [0, ln2]
        scores = (jsd / np.log(2)).clip(1e-7, 1.0)                # [P] normalised to (0, 1]
        logits = np.log(scores)                                    # [P] in (-inf, 0]
        logits_scaled = logits / temperature
        logits_scaled -= logits_scaled.max()                       # numerical stability
        weights = np.exp(logits_scaled)
        weights /= weights.sum()
        return weights.astype(np.float32)

    # --- default: correct_class_prob method ---
    true_class_probs = patch_probs[:, true_label].clip(1e-7, 1.0 - 1e-7)  # [P]
    logits = np.log(true_class_probs)
    logits_scaled = logits / temperature
    logits_scaled -= logits_scaled.max()                               # numerical stability
    weights = np.exp(logits_scaled)
    weights /= weights.sum()
    return weights                                                      # [P]


def _class_normalize_scores(
    y: np.ndarray,
    labels_per_element: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Standardize quality logits within each class.

    For class c: y~[i] = (y[i] - mu_c) / sigma_c, where mu_c and sigma_c are
    computed over all elements with label c.  Falls back to zero-centering when
    std < 1e-8 (degenerate class with constant scores).
    """
    y_out = y.copy().astype(np.float32)
    for c in np.unique(labels_per_element):
        mask = labels_per_element == c
        mu = float(y[mask].mean())
        sigma = float(y[mask].std())
        if sigma > 1e-8:
            y_out[mask] = (y[mask] - mu) / sigma
        else:
            y_out[mask] = y[mask] - mu
    if verbose:
        n_cls = len(np.unique(labels_per_element))
        print(f"[class_norm] Within-class score normalization applied ({n_cls} classes)")
    return y_out


def compute_patch_quality_logits(
    patch_probs:   np.ndarray,             # [P, n_classes]
    true_label:    int,
    temperature:   float = 1.0,
    weight_method: str   = "correct_class_prob",
    class_prior:   Optional[np.ndarray] = None,  # [n_classes]  empirical class frequencies
    binary_dist:   bool  = False,
) -> np.ndarray:                           # [P]  pre-normalization scaled logits
    """Return the pre-normalization score for each patch, matching the
    intermediate value computed inside compute_patch_pooling_weights.

    These values are suitable as Ridge regression targets: fitting a Ridge
    model to predict them from raw DINO features transfers the patch-quality
    signal into a lightweight, label-free scorer.

    Method correspondence
    ---------------------
    ``"correct_class_prob"`` → log(p_true) / temperature
    ``"entropy"``            → log(1 - H/ln(C)) / temperature
    ``"kl_div"``             → log(KL(Q||P_prior) / max_KL) / temperature
                               Requires *class_prior* [n_classes].
    ``"wasserstein"``        → log(W1(Q, P_prior) / (C-1)) / temperature
                               Requires *class_prior* [n_classes].
    ``"tvd"``                → log(TVD(Q, P_prior)) / temperature
                               Requires *class_prior* [n_classes].
    ``"js_div"``             → log(JSD(Q, P_prior) / ln2) / temperature
                               Requires *class_prior* [n_classes].

    When *binary_dist* is True (and *weight_method* is not ``"correct_class_prob"``),
    distributions are collapsed to ``[P(correct), P(non-correct)]`` before the
    distance is computed.  See ``compute_patch_pooling_weights`` for details.
    """
    if binary_dist and weight_method != "correct_class_prob":
        p_correct = patch_probs[:, true_label]                                  # [P]
        patch_probs = np.stack([p_correct, 1.0 - p_correct], axis=1)           # [P, 2]
        if class_prior is not None:
            pc = float(np.asarray(class_prior)[true_label])
            class_prior = np.array([pc, 1.0 - pc], dtype=np.float64)

    if weight_method == "entropy":
        n_classes = patch_probs.shape[1]
        raw_entropy = compute_patch_entropy(patch_probs)               # [P]
        scores = (1.0 - raw_entropy / np.log(n_classes)).clip(1e-7, 1.0)
        return (np.log(scores) / temperature).astype(np.float32)

    if weight_method == "kl_div":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='kl_div'")
        prior = np.asarray(class_prior, dtype=np.float64).clip(1e-9, 1.0)
        prior /= prior.sum()
        q = patch_probs.astype(np.float64).clip(1e-9, 1.0)
        kl = (q * np.log(q / prior[None, :])).sum(axis=1)             # [P]
        max_kl = -np.log(prior.min())
        scores = (kl / max_kl).clip(1e-7, 1.0)
        return (np.log(scores) / temperature).astype(np.float32)

    if weight_method == "wasserstein":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='wasserstein'")
        prior = np.asarray(class_prior, dtype=np.float64).clip(1e-9, 1.0)
        prior /= prior.sum()
        q = patch_probs.astype(np.float64)                             # [P, n_classes]
        cdf_q = np.cumsum(q, axis=1)                                   # [P, n_classes]
        cdf_p = np.cumsum(prior)                                       # [n_classes]
        w1 = np.abs(cdf_q - cdf_p[None, :]).sum(axis=1)               # [P] W1 distance
        n_classes = patch_probs.shape[1]
        scores = (w1 / (n_classes - 1)).clip(1e-7, 1.0)               # [P] normalised to (0, 1]
        return (np.log(scores) / temperature).astype(np.float32)

    if weight_method == "tvd":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='tvd'")
        prior = np.asarray(class_prior, dtype=np.float64).clip(1e-9, 1.0)
        prior /= prior.sum()
        q = patch_probs.astype(np.float64)                             # [P, n_classes]
        tvd = 0.5 * np.abs(q - prior[None, :]).sum(axis=1)             # [P] TVD
        scores = tvd.clip(1e-7, 1.0)                                   # [P] bounds
        return (np.log(scores) / temperature).astype(np.float32)

    if weight_method == "js_div":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='js_div'")
        prior = np.asarray(class_prior, dtype=np.float64).clip(1e-9, 1.0)
        prior /= prior.sum()
        q = patch_probs.astype(np.float64).clip(1e-9, 1.0)            # [P, n_classes]
        m = 0.5 * (q + prior[None, :])                                # [P, n_classes] mixture
        jsd = 0.5 * (q * np.log(q / m)).sum(axis=1) \
            + 0.5 * (prior * np.log(prior / m)).sum(axis=1)           # [P] in [0, ln2]
        scores = (jsd / np.log(2)).clip(1e-7, 1.0)                    # [P] normalised to (0, 1]
        return (np.log(scores) / temperature).astype(np.float32)

    # --- default: correct_class_prob method ---
    p = patch_probs[:, true_label].clip(1e-7, 1.0 - 1e-7)
    return (np.log(p) / temperature).astype(np.float32)


def compute_patch_quality_logits_gpu(
    patch_probs:   "torch.Tensor",             # [P, n_classes]  on GPU
    true_label:    int,
    temperature:   float = 1.0,
    weight_method: str   = "correct_class_prob",
    class_prior:   Optional["torch.Tensor"] = None,  # [n_classes]  on GPU
    binary_dist:   bool  = False,
) -> "torch.Tensor":                           # [P]  on GPU
    """GPU-native analogue of ``compute_patch_quality_logits``.

    Accepts and returns ``torch.Tensor`` objects on the same device as
    *patch_probs*.  All operations mirror the numpy version exactly.
    Requires *class_prior* to be a tensor on the same device for the
    ``kl_div``, ``wasserstein``, ``js_div``, and ``tvd`` methods.

    When *binary_dist* is True (and *weight_method* is not ``"correct_class_prob"``),
    distributions are collapsed to ``[P(correct), P(non-correct)]`` before the
    distance is computed.  See ``compute_patch_pooling_weights`` for details.
    """
    import torch

    if binary_dist and weight_method != "correct_class_prob":
        p_correct = patch_probs[:, true_label]                                  # [P]
        patch_probs = torch.stack([p_correct, 1.0 - p_correct], dim=1)         # [P, 2]
        if class_prior is not None:
            pc = class_prior[true_label]
            class_prior = torch.stack([pc, 1.0 - pc])

    eps_prob = 1e-9
    eps_clip = 1e-7

    if weight_method == "entropy":
        C = patch_probs.shape[1]
        q = patch_probs.clamp(eps_prob, 1.0)
        raw_entropy = -(q * q.log()).sum(dim=1)                         # [P]
        scores = (1.0 - raw_entropy / math.log(C)).clamp(eps_clip, 1.0)
        return (scores.log() / temperature).float()

    if weight_method == "kl_div":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='kl_div'")
        prior = class_prior.double().clamp(eps_prob, 1.0)
        prior = prior / prior.sum()
        q = patch_probs.double().clamp(eps_prob, 1.0)
        kl = (q * (q / prior.unsqueeze(0)).log()).sum(dim=1)            # [P]
        max_kl = -prior.min().log()
        scores = (kl / max_kl).clamp(eps_clip, 1.0)
        return (scores.log() / temperature).float()

    if weight_method == "wasserstein":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='wasserstein'")
        prior = class_prior.double().clamp(eps_prob, 1.0)
        prior = prior / prior.sum()
        q = patch_probs.double()
        cdf_q = q.cumsum(dim=1)
        cdf_p = prior.cumsum(dim=0)
        w1 = (cdf_q - cdf_p.unsqueeze(0)).abs().sum(dim=1)             # [P]
        C = patch_probs.shape[1]
        scores = (w1 / (C - 1)).clamp(eps_clip, 1.0)
        return (scores.log() / temperature).float()

    if weight_method == "tvd":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='tvd'")
        prior = class_prior.double().clamp(eps_prob, 1.0)
        prior = prior / prior.sum()
        q = patch_probs.double()
        tvd = 0.5 * (q - prior.unsqueeze(0)).abs().sum(dim=1)          # [P]
        scores = tvd.clamp(eps_clip, 1.0)
        return (scores.log() / temperature).float()

    if weight_method == "js_div":
        if class_prior is None:
            raise ValueError("class_prior must be provided for weight_method='js_div'")
        prior = class_prior.double().clamp(eps_prob, 1.0)
        prior = prior / prior.sum()
        q = patch_probs.double().clamp(eps_prob, 1.0)
        m = 0.5 * (q + prior.unsqueeze(0))
        jsd = (0.5 * (q * (q / m).log()).sum(dim=1)
               + 0.5 * (prior * (prior / m).log()).sum(dim=1))          # [P]
        scores = (jsd / math.log(2)).clamp(eps_clip, 1.0)
        return (scores.log() / temperature).float()

    # --- default: correct_class_prob ---
    p = patch_probs[:, true_label].clamp(eps_clip, 1.0 - eps_clip)
    return (p.log() / temperature).float()


# ---------------------------------------------------------------------------
# Patch grouping
# ---------------------------------------------------------------------------

def group_patches(patches: np.ndarray, patch_group_size: int) -> np.ndarray:
    """Mean-pool spatially neighbouring patches into groups.

    Parameters
    ----------
    patches : np.ndarray
        Shape ``[N, P, D]`` (batch) or ``[P, D]`` (single image).
    patch_group_size : int
        Number of original patches per group.  Must be a perfect square
        (1, 4, 9, 16, …).  ``1`` is the identity (no grouping).

    Returns
    -------
    np.ndarray
        Shape ``[N, P', D]`` or ``[P', D]`` where
        ``P' = ceil(sqrt(P) / group_side) ** 2``.
        When the grid side is not divisible by ``group_side``, boundary groups
        cover fewer patches (equivalent to ``ceil_mode=True`` avg-pooling).
        Dtype is preserved from *patches*.
    """
    if patch_group_size == 1:
        return patches

    group_side = int(round(patch_group_size ** 0.5))
    if group_side * group_side != patch_group_size:
        raise ValueError(
            f"patch_group_size must be a perfect square, got {patch_group_size}"
        )

    single = patches.ndim == 2
    if single:
        patches = patches[None]   # [1, P, D]

    N, P, D = patches.shape
    n_side = int(round(P ** 0.5))
    if n_side * n_side != P:
        raise ValueError(f"P={P} is not a perfect square; cannot form a spatial grid")

    new_n_side = math.ceil(n_side / group_side)

    if n_side % group_side == 0:
        # Fast path: exact divisibility — no padding needed
        grouped = (
            patches
            .reshape(N, new_n_side, group_side, new_n_side, group_side, D)
            .mean(axis=(2, 4))          # [N, new_n_side, new_n_side, D]
        ).reshape(N, new_n_side * new_n_side, D)
    else:
        # Handle interior and boundary groups separately — avoids np.pad on the
        # full [N, n_side, n_side, D] array, which is the dominant memory cost.
        x = patches.astype(np.float32).reshape(N, n_side, n_side, D)
        int_side = (n_side // group_side) * group_side  # largest multiple ≤ n_side
        n_int = int_side // group_side                  # complete groups per dim

        grouped = np.empty((N, new_n_side, new_n_side, D), dtype=np.float32)

        # Interior block: exact-divisibility fast path, no allocation overhead
        if n_int > 0:
            grouped[:, :n_int, :n_int, :] = (
                x[:, :int_side, :int_side, :]
                .reshape(N, n_int, group_side, n_int, group_side, D)
                .mean(axis=(2, 4))
            )

        # Right boundary column (interior rows, partial col)
        if n_int > 0:
            grouped[:, :n_int, n_int, :] = (
                x[:, :int_side, int_side:, :]           # [N, int_side, rem, D]
                .reshape(N, n_int, group_side, -1, D)
                .mean(axis=(2, 3))
            )

        # Bottom boundary row (partial row, interior cols)
        if n_int > 0:
            grouped[:, n_int, :n_int, :] = (
                x[:, int_side:, :int_side, :]           # [N, rem, int_side, D]
                .reshape(N, -1, n_int, group_side, D)
                .mean(axis=(1, 3))
            )

        # Corner (partial row × partial col)
        grouped[:, n_int, n_int, :] = x[:, int_side:, int_side:, :].mean(axis=(1, 2))

        grouped = grouped.reshape(N, new_n_side * new_n_side, D)

    if single:
        grouped = grouped[0]
    return grouped.astype(patches.dtype)


# ---------------------------------------------------------------------------
# Ridge softmax pooling helper
# ---------------------------------------------------------------------------

def _ridge_pool_weights(
    patches:        np.ndarray,               # [N, P, D]
    ridge_model:    Ridge,
    feature_scaler: Optional[StandardScaler],
) -> np.ndarray:                              # [N, P]  softmax weights summing to 1
    """Compute per-patch softmax pooling weights from a fitted Ridge model."""
    N, P, D = patches.shape
    flat = patches.reshape(N * P, D)
    if feature_scaler is not None:
        flat = feature_scaler.transform(flat)
    logits = ridge_model.predict(flat).reshape(N, P).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)   # numerical stability
    exp_l   = np.exp(logits)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Full refinement pass
# ---------------------------------------------------------------------------

def refine_dataset_features(
    train_patches:      np.ndarray,      # [N, P, D]  raw DINO patch features
    train_labels:       np.ndarray,      # [N]
    support_features:   np.ndarray,      # [N_train, d]  initial mean-pooled (post-PCA) features
    refinement_cfg: ImagePALConfig,
    pca:                Optional[PCA],   # PCA fitted on the baseline support set
    seed:               int   = 42,
    aoe_mask:              Optional[np.ndarray] = None,  # [N] bool; True = absence-of-evidence class
    gpu_ridge_device:      str                 = "cuda",
    #Optional tabicl classifier
    tabicl: Optional[TabICLClassifier] = None,
    context_features:   Optional[np.ndarray] = None,  # [N, D_context]  per-image side features
    tabular_probs:      Optional[np.ndarray] = None,  # [N, n_classes]  pre-computed P(Y|X_tab); replaces global prior for divergence methods
    val_patches:        Optional[np.ndarray] = None,  # [N_val, P, D] raw DINO patch features for validation
    val_labels:         Optional[np.ndarray] = None,  # [N_val] validation labels
    val_context_features: Optional[np.ndarray] = None, # [N_val, D_context]
    val_tabular_probs:  Optional[np.ndarray] = None,  # [N_val, n_classes]
    val_aoe_mask:       Optional[np.ndarray] = None,  # [N_val] bool
    verbose:            bool                = True,
) -> tuple[np.ndarray, Optional[PCA], np.ndarray, Union[Ridge, "RidgeGPU"], Optional[StandardScaler], TabICLClassifier, float, float, Optional[np.ndarray]]:
    """Refine mean-pooled support features with Ridge-predicted quality-weighted pooling.

    A TabICL classifier fitted on the *input* support is used solely to generate
    quality-logit training targets for a Ridge regressor.  The Ridge model then
    predicts per-patch quality logits from raw DINO features and drives the final
    pooling — making the pooling label-free and fast at inference time.

    If *aoe_mask* is provided, its effect on steps 2–4 depends on *aoe_handling*:

    ``"filter"``
        AoE-class images are excluded from the TabICL forward pass and Ridge
        fitting entirely.  The Ridge model sees only non-AoE patches.

    ``"entropy"``
        AoE-class images are included, but their per-patch quality logits are
        computed with the ``"entropy"`` method regardless of *weight_method*.
        This lets the Ridge model learn AoE-patch quality without relying on labels.

    In both cases, AoE-class images are still included in the Ridge-based support
    pooling (step 5) and in the returned refined support.

    Flow
    ----
    1. Fit a ``TabICLClassifier`` on *support_features* (fixed scorer for this stage).
    2. Collect ``(patch_feature, quality_logit)`` pairs for Ridge fitting:

       - **Default** (``max_query_rows`` is ``None`` or active rows ≤ ``max_query_rows``):
         forward all active patch rows in batches of *batch_size*.
       - **Subsampled** (active rows > ``max_query_rows``): draw *max_query_rows*
         ``(image, patch)`` pairs uniformly at random from active images and
         forward them in a single pass.  The fraction of rows forwarded is printed.

    3. Optionally fit a ``StandardScaler`` on the collected patch features
       (``normalize_features=True``).
    4. Fit ``Ridge(alpha=ridge_alpha)`` on the collected pairs.
    5. Predict quality logits for **all** patches of **all** images with Ridge
       (full-image pooling including AoE-class images, regardless of step 2).
    6. Apply softmax weights derived from Ridge logits → ``repooled_raw [N, D]``.
    7. Re-fit PCA on the Ridge-pooled raw features.

    Returns
    -------
    refined : np.ndarray, shape [N, d]
    new_pca : PCA or None
    weights_ridge : np.ndarray, shape [N, P]   (Ridge softmax weights, full images)
    ridge_model : Ridge
    feature_scaler : StandardScaler or None
    clf : TabICLClassifier  (scorer fitted on the *input* support_features)
    fit_time_s : float  — seconds from start of function until Ridge.fit() completes
                          (TabICL forward pass + Ridge fitting; the "learning" phase)
    pool_time_s : float — seconds from after Ridge.fit() until return
                          (Ridge prediction over all images + repooling + PCA refit;
                          scales with N×P' and is the dominant cost for group_size=1)

    Notes
    -----
    When *use_gpu_ridge* is ``True``, a :class:`RidgeGPU` is used instead of
    sklearn ``Ridge``.  It solves the same normal equations on the GPU, so results
    should be numerically equivalent.  The biggest speedup is in the pooling phase
    (step 5), where Ridge has to predict logits for the full ``[N×P', D]`` matrix;
    for ``group_size=1`` (P'=196) this is the bottleneck that the GPU eliminates.
    Requires PyTorch with a working CUDA installation.
    """
    #Assert that refinement_cfg.temperature and refinement_cfg.ridge_alpha are either floats or ints
    if not isinstance(refinement_cfg.temperature, (int, float)):
        raise TypeError("refinement_cfg.temperature must be a float or int when forwarded to refine_dataset_features")
    if not isinstance(refinement_cfg.ridge_alpha, (int, float)):
        raise TypeError("refinement_cfg.ridge_alpha must be a float or int when forwarded to refine_dataset_features")

    t_start = time.perf_counter()
    N, P, D = train_patches.shape

    # Precompute empirical class prior
    n_cls_local = int(train_labels.max()) + 1
    counts = np.bincount(train_labels.astype(np.int64), minlength=n_cls_local)
    empirical_prior = (counts / counts.sum()).astype(np.float32)

    _divergence_methods = ("kl_div", "wasserstein", "js_div", "tvd")
    if refinement_cfg.weight_method in _divergence_methods:
        class_prior: Optional[np.ndarray] = empirical_prior
    else:
        class_prior = None

    # When context features are provided with a divergence method, compute tabular-only
    # P(Y|X_tab) as the per-image reference prior (replaces global class_prior).
    # If pre-computed externally (e.g. by IterativePALPooler), use it directly.
    if (context_features is not None
            and refinement_cfg.weight_method in _divergence_methods
            and tabular_probs is None
            and not getattr(refinement_cfg, "use_global_prior", False)):
        if verbose: print("[multimodal] Computing tabular-only P(Y|X_tab) for per-image prior ...")
        _clf_tab = TabICLClassifier(n_estimators=refinement_cfg.tabicl_n_estimators, random_state=seed)
        _clf_tab.fit(context_features, train_labels)
        raw_tab_probs = _clf_tab.predict_proba(context_features).astype(np.float32)
        n_cls_local = int(train_labels.max()) + 1
        if raw_tab_probs.shape[1] != n_cls_local:
            tabular_probs = np.zeros((raw_tab_probs.shape[0], n_cls_local), dtype=np.float32)
            tabular_probs[:, _clf_tab.classes_] = raw_tab_probs
        else:
            tabular_probs = raw_tab_probs

    # Determine which images contribute to Ridge fitting and their per-image method.
    # If val blocks are present, fit Ridge strictly on validation patches.
    # Otherwise, fallback to train patches.
    # "filter": only non-AoE images; all use weight_method.
    # "entropy": all images; AoE images use "entropy" instead of weight_method.
    
    if val_patches is not None and val_labels is not None:
        source_patches = val_patches
        source_labels = val_labels
        source_aoe = val_aoe_mask
        source_tabular = val_tabular_probs
        source_context = val_context_features
    else:
        source_patches = train_patches
        source_labels = train_labels
        source_aoe = aoe_mask
        source_tabular = tabular_probs
        source_context = context_features

    M, P_dim, _D_dim = source_patches.shape

    if source_aoe is not None and refinement_cfg.aoe_handling == "filter":
        active_indices = np.where(~source_aoe)[0]
    else:
        active_indices = np.arange(M)
    
    N_active          = len(active_indices)
    active_patches    = source_patches[active_indices]   # [N_active, P, D]
    active_labels     = source_labels[active_indices]    # [N_active]
    active_total_rows = N_active * P_dim

    # Per-image reference prior: slice source_tabular to active images.
    active_tabular_probs: Optional[np.ndarray] = (
        source_tabular[active_indices].astype(np.float32) if source_tabular is not None else None
    )  # [N_active, n_classes] or None

    # Per-image effective weight method (None = all images use weight_method)
    if source_aoe is not None and refinement_cfg.aoe_handling == "entropy":
        active_is_aoe = source_aoe[active_indices]   # [N_active] bool
    else:
        active_is_aoe = None

    def _eff_method(local_idx: int) -> str:
        if active_is_aoe is not None and active_is_aoe[local_idx]:
            return "entropy"
        return refinement_cfg.weight_method

    # Fit one shared classifier (support set is fixed for all queries this stage).
    # When context_features are provided, append them to support and to every query
    # so the scoring is context-aware.  The Ridge model still only sees raw DINO.
    if tabicl is not None:
        clf = tabicl
    else:
        clf = TabICLClassifier(n_estimators=refinement_cfg.tabicl_n_estimators, random_state=seed)

    support_for_clf = support_features
    if context_features is not None:
        support_for_clf = np.concatenate(
            [support_features, context_features.astype(np.float32)], axis=1
        )
    clf.fit(support_for_clf, train_labels)

    # For current_pool_marginal, compute the prior by running the just-fitted classifier
    # on the current pooled support with a diagonal attention mask (each image cannot
    # attend to its own support row).  This gives an image-level marginal that reflects
    # the actual discriminability of the pooled representation at this stage.
    if refinement_cfg.prior == "current_pool_marginal" and refinement_cfg.weight_method in _divergence_methods:
        if verbose: print("[calibration] Computing current_pool_marginal prior from current pooled features ...")
        N_supp = len(support_for_clf)
        n_cls_prior = int(train_labels.max()) + 1
        blocked_diag_np = np.arange(N_supp)
        if isinstance(clf, TabICLGPUAdapter):
            import torch as _torch
            blocked_diag_t = _torch.from_numpy(blocked_diag_np).long()
            probs_supp_t = _safe_predict_proba_tensor(clf, support_for_clf, blocked_indices=blocked_diag_t)
            if probs_supp_t.shape[1] != n_cls_prior:
                cls_t = _torch.tensor(clf.classes_, device=probs_supp_t.device, dtype=_torch.long)
                full_t = _torch.zeros((N_supp, n_cls_prior), dtype=probs_supp_t.dtype, device=probs_supp_t.device)
                full_t[:, cls_t] = probs_supp_t
                probs_supp_t = full_t
            class_prior = probs_supp_t.mean(dim=0).cpu().numpy().astype(np.float32)
        else:
            probs_supp = _safe_predict_proba(clf, support_for_clf, blocked_indices=blocked_diag_np)
            if probs_supp.shape[1] != n_cls_prior:
                full = np.zeros((N_supp, n_cls_prior), dtype=probs_supp.dtype)
                full[:, clf.classes_] = probs_supp
                probs_supp = full
            class_prior = probs_supp.mean(axis=0).astype(np.float32)
        if verbose: print(f"[calibration] current_pool_marginal prior: {np.round(class_prior, 4)}")

    # GPU path: skip D2H/H2D round-trip for probs and quality-logit targets.
    use_gpu = isinstance(clf, TabICLGPUAdapter)
    if use_gpu:
        import torch as _torch
        _dev = clf.device_
        class_prior_t = (
            _torch.from_numpy(class_prior).to(_dev) if class_prior is not None else None
        )
        active_tabular_probs_t = (
            _torch.from_numpy(active_tabular_probs).to(_dev)
            if active_tabular_probs is not None else None
        )  # [N_active, n_classes] on GPU, or None

    # Decide forward-pass strategy:
    #   one_pass=True  → forward a contiguous block of rows in a single predict_proba call
    #                    (either all active rows, or a random subset when subsampling)
    #   one_pass=False → forward in batches of batch_size images (fallback / default)
    exceeded = refinement_cfg.max_query_rows is not None and active_total_rows > refinement_cfg.max_query_rows
    one_pass = refinement_cfg.max_query_rows is not None and (not exceeded or refinement_cfg.use_random_subsampling)

    # Attention masking is only meaningful when the query images are the same as the support
    # images (i.e. when no separate validation set is used for Ridge fitting).  When val_patches
    # are supplied, the query images are out-of-support val images, so masking doesn't apply.
    _apply_attn_mask = refinement_cfg.use_attn_masking and (val_patches is None)

    if one_pass:
        if exceeded:
            # Subsampled: draw max_query_rows (image, patch) pairs from active images
            n_fwd    = refinement_cfg.max_query_rows
            aoe_note = f" (aoe_handling={refinement_cfg.aoe_handling})" if aoe_mask is not None else ""
            if verbose: print(f"[sampling] Subsampling {n_fwd:,} / {active_total_rows:,} patch-group rows "
                  f"({100 * n_fwd / active_total_rows:.1f}%) for Ridge fitting{aoe_note}")
            rng           = np.random.RandomState(seed)
            sampled_flat  = rng.choice(active_total_rows, size=n_fwd, replace=False)
            sampled_flat.sort()                          # sort → group by image for searchsorted
            local_img_idx = sampled_flat // P_dim        # index into active_patches
            patch_idx_all = sampled_flat % P_dim
            query_raw     = active_patches[local_img_idx, patch_idx_all].astype(np.float32)
            img_boundaries = np.searchsorted(local_img_idx, np.arange(N_active + 1))
            # Each sampled row belongs to active image local_img_idx[i].
            ctx_for_queries = (
                source_context[active_indices][local_img_idx].astype(np.float32)
                if source_context is not None else None
            )
            # blocked_indices: query row k must not attend to support row active_indices[local_img_idx[k]].
            # Uses O(n_fwd) memory instead of O(n_fwd × N) for the dense mask.
            blocked_indices_np = active_indices[local_img_idx] if _apply_attn_mask else None
        else:
            # All active rows in one pass; sequential layout matches reshape order
            n_fwd          = active_total_rows
            query_raw      = active_patches.reshape(active_total_rows, D).astype(np.float32)
            img_boundaries = np.arange(N_active + 1) * P_dim     # [0, P, 2P, ..., N_active*P]
            # Repeat each active image's context P times to align with the flat patch layout.
            ctx_for_queries = (
                np.repeat(source_context[active_indices].astype(np.float32), P_dim, axis=0)
                if source_context is not None else None
            )
            # blocked_indices: patches for image i must not attend to support row active_indices[i].
            # np.repeat gives shape (N_active * P_dim,) = (n_fwd,) with O(n_fwd) memory.
            blocked_indices_np = np.repeat(active_indices, P_dim) if _apply_attn_mask else None

        query_features = pca.transform(query_raw) if pca is not None else query_raw
        if ctx_for_queries is not None:
            query_features = np.concatenate([query_features, ctx_for_queries], axis=1)
        all_features   = query_raw

        n_cls_expected = int(train_labels.max()) + 1

        if use_gpu:
            _blocked_indices_t = (
                _torch.from_numpy(blocked_indices_np).long() if blocked_indices_np is not None else None
            )
            probs_flat_t = _safe_predict_proba_tensor(clf, query_features, blocked_indices=_blocked_indices_t)  # [n_fwd, n_classes] GPU
            if n_cls_expected is not None and probs_flat_t.shape[1] != n_cls_expected:
                cls_tensor = _torch.tensor(clf.classes_, device=probs_flat_t.device, dtype=_torch.long)
                full_t = _torch.zeros((probs_flat_t.shape[0], n_cls_expected), dtype=probs_flat_t.dtype, device=probs_flat_t.device)
                full_t[:, cls_tensor] = probs_flat_t
                probs_flat_t = full_t

            mean_probs = probs_flat_t.mean(dim=0).cpu().numpy()

            all_targets_t = _torch.empty(n_fwd, dtype=_torch.float32, device=_dev)
        else:
            probs_flat  = _safe_predict_proba(clf, query_features, blocked_indices=blocked_indices_np)  # [n_fwd, n_classes] CPU
            if n_cls_expected is not None and probs_flat.shape[1] != n_cls_expected:
                full = np.zeros((probs_flat.shape[0], n_cls_expected), dtype=probs_flat.dtype)
                full[:, clf.classes_] = probs_flat
                probs_flat = full

            mean_probs = probs_flat.mean(axis=0)

            all_targets = np.empty(n_fwd, dtype=np.float32)

    else:
        # Batched loop: used when max_query_rows is None, or when cap is exceeded
        # but --use-random-subsampling was not requested.
        all_features = np.empty((active_total_rows, D), dtype=np.float32)
        if use_gpu:
            all_targets = _torch.empty(active_total_rows, dtype=_torch.float32, device=_dev)
        else:
            all_targets = np.empty(active_total_rows, dtype=np.float32)
        row_ptr      = 0

        sum_probs = None
        count_probs = 0
        saved_probs = []

        for batch_start in tqdm(range(0, M, refinement_cfg.batch_size),
                                desc="Computing patch quality scores", unit="batch"):
            batch_end = min(batch_start + refinement_cfg.batch_size, M)
            # For "filter" mode, exclude AoE images from the batch.
            # For "entropy" mode (and no AoE), include all images.
            if source_aoe is not None and refinement_cfg.aoe_handling == "filter":
                active_in_batch = [j for j in range(batch_start, batch_end) if not source_aoe[j]]
                if not active_in_batch:
                    continue
                batch_patches_arr = source_patches[active_in_batch]   # [B_a, P, D]
                batch_labels_arr  = source_labels[active_in_batch]
                batch_is_aoe      = None
            else:
                batch_patches_arr = source_patches[batch_start:batch_end]   # [B, P, D]
                batch_labels_arr  = source_labels[batch_start:batch_end]
                batch_is_aoe      = (source_aoe[batch_start:batch_end]
                                     if source_aoe is not None else None)

            B_a = len(batch_patches_arr)
            query_raw      = batch_patches_arr.reshape(B_a * P_dim, D)
            query_features = pca.transform(query_raw) if pca is not None else query_raw
            if source_context is not None:
                # Gather the context for this batch's images and repeat P times per image.
                if source_aoe is not None and refinement_cfg.aoe_handling == "filter":
                    batch_ctx = source_context[active_in_batch].astype(np.float32)
                else:
                    batch_ctx = source_context[batch_start:batch_end].astype(np.float32)
                query_features = np.concatenate(
                    [query_features, np.repeat(batch_ctx, P_dim, axis=0)], axis=1
                )
            base           = row_ptr * P_dim

            # Build blocked_indices for this batch when use_attn_masking is enabled.
            # Patches for image j must not attend to its own support row orig_idx[j].
            # np.repeat gives shape (B_a * P,) with O(B_a * P) memory instead of O(B_a * P * N).
            if _apply_attn_mask:
                if source_aoe is not None and refinement_cfg.aoe_handling == "filter":
                    batch_orig_indices = active_in_batch        # already original indices
                else:
                    batch_orig_indices = list(range(batch_start, batch_end))
                _blocked_indices_np = np.repeat(batch_orig_indices, P_dim)
            else:
                _blocked_indices_np = None

            n_cls_expected = int(train_labels.max()) + 1

            if use_gpu:
                _blocked_indices_t = (
                    _torch.from_numpy(_blocked_indices_np).long() if _blocked_indices_np is not None else None
                )
                probs_t = _safe_predict_proba_tensor(clf, query_features, blocked_indices=_blocked_indices_t)
                if n_cls_expected is not None and probs_t.shape[1] != n_cls_expected:
                    cls_tensor = _torch.tensor(clf.classes_, device=probs_t.device, dtype=_torch.long)
                    full_t = _torch.zeros((probs_t.shape[0], n_cls_expected), dtype=probs_t.dtype, device=probs_t.device)
                    full_t[:, cls_tensor] = probs_t
                    probs_t = full_t

                batch_mean_sum = probs_t.sum(dim=0).cpu().numpy()
                count_probs += probs_t.shape[0]
                if sum_probs is None:
                    sum_probs = batch_mean_sum
                else:
                    sum_probs += batch_mean_sum

                probs_t = probs_t.reshape(B_a, P, -1)  # GPU
                saved_probs.append((probs_t, batch_labels_arr, batch_is_aoe, B_a))
                for j in range(B_a):
                    s, e = base + j * P, base + (j + 1) * P
                    all_features[s:e] = batch_patches_arr[j]
            else:
                probs = _safe_predict_proba(clf, query_features, blocked_indices=_blocked_indices_np)
                if n_cls_expected is not None and probs.shape[1] != n_cls_expected:
                    full = np.zeros((probs.shape[0], n_cls_expected), dtype=probs.dtype)
                    full[:, clf.classes_] = probs
                    probs = full

                batch_mean_sum = probs.sum(axis=0)
                count_probs += probs.shape[0]
                if sum_probs is None:
                    sum_probs = batch_mean_sum
                else:
                    sum_probs += batch_mean_sum

                probs = probs.reshape(B_a, P, -1)  # CPU
                saved_probs.append((probs, batch_labels_arr, batch_is_aoe, B_a))
                for j in range(B_a):
                    s, e = base + j * P, base + (j + 1) * P
                    all_features[s:e] = batch_patches_arr[j]
            row_ptr += B_a

        if count_probs > 0:
            mean_probs = sum_probs / count_probs

    if verbose: print(f"[calibration] Empirical label prior:    {np.round(empirical_prior, 4)}")
    if verbose: print(f"[calibration] Marginal patch predicted: {np.round(mean_probs, 4)}")

    if refinement_cfg.prior == "token_marginal":
        class_prior = mean_probs.astype(np.float32)

    if use_gpu:
        import torch as _torch
        class_prior_t = (
            _torch.from_numpy(class_prior).to(_dev) if class_prior is not None else None
        )
        
    # Now we assign quality-logit targets using the optionally updated class_prior.
    n_cls_expected = int(train_labels.max()) + 1

    _binary_dist = getattr(refinement_cfg, "binary_dist", False)

    if one_pass:
        if use_gpu:
            for idx in range(N_active):
                start, end = int(img_boundaries[idx]), int(img_boundaries[idx + 1])
                if start == end:
                    continue
                all_targets_t[start:end] = compute_patch_quality_logits_gpu(
                    probs_flat_t[start:end], int(active_labels[idx]),
                    refinement_cfg.temperature, _eff_method(idx),
                    active_tabular_probs_t[idx] if active_tabular_probs_t is not None else class_prior_t,
                    binary_dist=_binary_dist,
                )
            all_targets = all_targets_t
        else:
            for idx in range(N_active):
                start, end = int(img_boundaries[idx]), int(img_boundaries[idx + 1])
                if start == end:
                    continue
                all_targets[start:end] = compute_patch_quality_logits(
                    probs_flat[start:end], int(active_labels[idx]),
                    refinement_cfg.temperature, _eff_method(idx),
                    active_tabular_probs[idx] if active_tabular_probs is not None else class_prior,
                    binary_dist=_binary_dist,
                )

    else:
        # Re-iterate over saved probabilities to compute targets
        row_ptr = 0
        for probs_batch, batch_labels, batch_is_aoe, B_a in saved_probs:
            base = row_ptr * P
            if use_gpu:
                for j in range(B_a):
                    eff = ("entropy"
                           if batch_is_aoe is not None and batch_is_aoe[j]
                           else refinement_cfg.weight_method)
                    s, e = base + j * P, base + (j + 1) * P
                    all_targets[s:e]  = compute_patch_quality_logits_gpu(
                        probs_batch[j], int(batch_labels[j]),
                        refinement_cfg.temperature, eff,
                        active_tabular_probs_t[row_ptr + j] if active_tabular_probs_t is not None else class_prior_t,
                        binary_dist=_binary_dist,
                    )
            else:
                for j in range(B_a):
                    eff = ("entropy"
                           if batch_is_aoe is not None and batch_is_aoe[j]
                           else refinement_cfg.weight_method)
                    s, e = base + j * P, base + (j + 1) * P
                    all_targets[s:e]  = compute_patch_quality_logits(
                        probs_batch[j], int(batch_labels[j]),
                        refinement_cfg.temperature, eff,
                        active_tabular_probs[row_ptr + j] if active_tabular_probs is not None else class_prior,
                        binary_dist=_binary_dist,
                    )
            row_ptr += B_a


    # --- Fit Ridge on collected (patch_feature, quality_logit) pairs ---
    feature_scaler: Optional[StandardScaler] = None
    if refinement_cfg.normalize_features:
        if verbose: print("[ridge] Fitting StandardScaler on training patches ...")
        feature_scaler = StandardScaler()
        all_features = feature_scaler.fit_transform(all_features)

    backend = "GPU" if gpu_ridge_device else "CPU"
    if verbose: print(f"[ridge] Fitting Ridge(alpha={refinement_cfg.ridge_alpha}) on {len(all_features):,} patch samples "
          f"(D={D}, method={refinement_cfg.weight_method}, backend={backend}) ...")
    if getattr(refinement_cfg, "class_normalized_scores", False):
        import torch as _torch_cn
        _tgt_np = (
            all_targets.cpu().numpy() if isinstance(all_targets, _torch_cn.Tensor)
            else np.asarray(all_targets, dtype=np.float32)
        )
        _cn_labels = np.empty(len(_tgt_np), dtype=np.int64)
        if one_pass:
            for _idx in range(N_active):
                _s, _e = int(img_boundaries[_idx]), int(img_boundaries[_idx + 1])
                if _s < _e:
                    _cn_labels[_s:_e] = int(active_labels[_idx])
        else:
            _rptr = 0
            for _pb, _bl, _ia, _Ba in saved_probs:
                for _j in range(_Ba):
                    _s = (_rptr + _j) * P_dim
                    _e = (_rptr + _j + 1) * P_dim
                    _cn_labels[_s:_e] = int(_bl[_j])
                _rptr += _Ba
        all_targets = _class_normalize_scores(_tgt_np, _cn_labels, verbose=verbose)

    if gpu_ridge_device:
        ridge_model: Union[Ridge, RidgeGPU] = RidgeGPU(alpha=refinement_cfg.ridge_alpha, device=gpu_ridge_device)
    else:
        ridge_model = Ridge(alpha=refinement_cfg.ridge_alpha)
        import torch as _torch
        if isinstance(all_targets, _torch.Tensor):
            all_targets = all_targets.cpu().numpy()
    ridge_model.fit(all_features, all_targets)
    if verbose: print(f"[ridge] Train R²: {ridge_model.score(all_features, all_targets):.4f}")

    # Split: everything up to here is the "learning" phase (TabICL forward + Ridge fit).
    # Everything below is the "pooling" phase (Ridge predict + repooling + PCA refit),
    # which scales with N×P' and is the dominant cost difference between group sizes.
    t_fit_done = time.perf_counter()
    fit_time_s = t_fit_done - t_start

    # --- Pool all training images with Ridge weights (always full images) ---
    if verbose: print("[ridge] Pooling support set with Ridge-predicted weights ...")
    weights_ridge = _ridge_pool_weights(train_patches, ridge_model, feature_scaler)   # [N, P]
    repooled_raw  = (weights_ridge[:, :, None] * train_patches).sum(axis=1)           # [N, D]

    if pca is not None:
        new_pca = PCA(n_components=pca.n_components_, random_state=seed)
        refined = new_pca.fit_transform(repooled_raw.astype(np.float32)).astype(np.float32)
    else:
        refined = repooled_raw.astype(np.float32)
        new_pca = None

    pool_time_s = time.perf_counter() - t_fit_done
    if verbose: print(f"[timing] fit={fit_time_s:.1f}s  pool={pool_time_s:.1f}s  "
          f"total={fit_time_s + pool_time_s:.1f}s")

    return refined, new_pca, weights_ridge, ridge_model, feature_scaler, clf, fit_time_s, pool_time_s, class_prior


# ---------------------------------------------------------------------------
# Cross-validation helpers (used by IterativePALPooler CV path)
# ---------------------------------------------------------------------------

def collect_pseudo_labels_image(
    query_grouped: np.ndarray,                         # [N_q, P', D] already grouped
    query_labels: np.ndarray,                          # [N_q]
    train_labels: np.ndarray,                          # [N_tr] — for empirical prior
    support_features: np.ndarray,                      # [N_tr, d] — PCA-projected support
    pca: Optional[PCA],                                # to project query features for TabICL
    tabicl: "TabICLClassifier",
    refinement_cfg: ImagePALConfig,
    tabular_probs: Optional[np.ndarray] = None,        # [N_q, n_cls] for query set
    query_context_features: Optional[np.ndarray] = None,   # [N_q, D_ctx]
    train_context_features: Optional[np.ndarray] = None,   # [N_tr, D_ctx]
    seed: int = 42,
) -> tuple:
    """Collect (patch_features, quality_logit) pairs for one CV fold (image path).

    Fits TabICL on the fold's 75% support and forwards the 25% query patches
    to produce quality-logit targets for downstream Ridge fitting.

    Returns
    -------
    X_flat : np.ndarray [N_q * P', D]  — flat query patch features (Ridge inputs)
    y_flat : np.ndarray [N_q * P']     — quality logit targets (Ridge outputs)
    """
    query_grouped = np.asarray(query_grouped, dtype=np.float32)
    N_q, P, D = query_grouped.shape

    n_cls = int(train_labels.max()) + 1
    counts = np.bincount(train_labels.astype(np.int64), minlength=n_cls)
    empirical_prior = (counts / counts.sum()).astype(np.float32)

    _divergence_methods = ("kl_div", "wasserstein", "js_div", "tvd")
    class_prior: Optional[np.ndarray] = (
        empirical_prior if refinement_cfg.weight_method in _divergence_methods else None
    )

    # Fit TabICL on this fold's support (optionally context-augmented).
    support_for_clf = support_features
    if train_context_features is not None:
        support_for_clf = np.concatenate(
            [support_features, train_context_features.astype(np.float32)], axis=1
        )
    clf = copy.deepcopy(tabicl)
    clf.fit(support_for_clf, train_labels)

    use_gpu = isinstance(clf, TabICLGPUAdapter)
    _binary_dist = getattr(refinement_cfg, "binary_dist", False)

    # Flatten: [N_q * P', D]
    X_flat = query_grouped.reshape(N_q * P, D).astype(np.float32)

    # Project via PCA + optional context for TabICL query features.
    query_features = pca.transform(X_flat) if pca is not None else X_flat
    if query_context_features is not None:
        ctx_rep = np.repeat(query_context_features.astype(np.float32), P, axis=0)
        query_features = np.concatenate([query_features, ctx_rep], axis=1)

    # Repeat tabular probs to align with flat patch layout.
    active_tab: Optional[np.ndarray] = (
        np.repeat(tabular_probs.astype(np.float32), P, axis=0)
        if tabular_probs is not None else None
    )

    n_cls_expected = n_cls

    if use_gpu:
        import torch as _torch
        _dev = clf.device_
        class_prior_t = (
            _torch.from_numpy(class_prior).to(_dev) if class_prior is not None else None
        )
        probs_t = _safe_predict_proba_tensor(clf, query_features, blocked_indices=None)
        if probs_t.shape[1] != n_cls_expected:
            cls_t = _torch.tensor(clf.classes_, device=probs_t.device, dtype=_torch.long)
            full_t = _torch.zeros(
                (probs_t.shape[0], n_cls_expected), dtype=probs_t.dtype, device=probs_t.device
            )
            full_t[:, cls_t] = probs_t
            probs_t = full_t

        # current_pool_marginal: run support through clf with diagonal masking.
        if refinement_cfg.prior == "current_pool_marginal" and class_prior_t is not None:
            N_tr = len(support_features)
            blocked_diag = _torch.arange(N_tr, device=_dev, dtype=_torch.long)
            sup_proj = _torch.from_numpy(support_for_clf.astype(np.float32)).to(_dev)
            marg_t = _safe_predict_proba_tensor(clf, sup_proj, blocked_indices=blocked_diag)
            if marg_t.shape[1] != n_cls_expected:
                cls_t2 = _torch.tensor(clf.classes_, device=_dev, dtype=_torch.long)
                full_m = _torch.zeros((N_tr, n_cls_expected), dtype=marg_t.dtype, device=_dev)
                full_m[:, cls_t2] = marg_t
                marg_t = full_m
            class_prior_t = marg_t.mean(dim=0)
        elif refinement_cfg.prior == "token_marginal" and class_prior_t is not None:
            class_prior_t = probs_t.mean(dim=0)

        active_tab_t = (
            _torch.from_numpy(active_tab).to(_dev) if active_tab is not None else None
        )

        y_flat_t = _torch.empty(N_q * P, dtype=_torch.float32, device=_dev)
        for i in range(N_q):
            s, e = i * P, (i + 1) * P
            prior_i = active_tab_t[s] if active_tab_t is not None else class_prior_t
            y_flat_t[s:e] = compute_patch_quality_logits_gpu(
                probs_t[s:e], int(query_labels[i]),
                refinement_cfg.temperature, refinement_cfg.weight_method,
                prior_i, binary_dist=_binary_dist,
            )
        y_flat = y_flat_t.cpu().numpy()
    else:
        probs = _safe_predict_proba(clf, query_features, blocked_indices=None)
        if probs.shape[1] != n_cls_expected:
            full = np.zeros((probs.shape[0], n_cls_expected), dtype=probs.dtype)
            full[:, clf.classes_] = probs
            probs = full

        if refinement_cfg.prior == "current_pool_marginal" and class_prior is not None:
            N_tr = len(support_features)
            blocked_diag = np.arange(N_tr)
            marg = _safe_predict_proba(clf, support_for_clf, blocked_indices=blocked_diag)
            if marg.shape[1] != n_cls_expected:
                full_m = np.zeros((N_tr, n_cls_expected), dtype=marg.dtype)
                full_m[:, clf.classes_] = marg
                marg = full_m
            class_prior = marg.mean(axis=0).astype(np.float32)
        elif refinement_cfg.prior == "token_marginal" and class_prior is not None:
            class_prior = probs.mean(axis=0).astype(np.float32)

        y_flat = np.empty(N_q * P, dtype=np.float32)
        for i in range(N_q):
            s, e = i * P, (i + 1) * P
            prior_i = active_tab[s] if active_tab is not None else class_prior
            y_flat[s:e] = compute_patch_quality_logits(
                probs[s:e], int(query_labels[i]),
                refinement_cfg.temperature, refinement_cfg.weight_method,
                prior_i, binary_dist=_binary_dist,
            )

    return X_flat, y_flat


def fit_ridge_repool_image(
    X_flat: np.ndarray,                # [M, D] accumulated flat patch features from all folds
    y_flat: np.ndarray,                # [M] quality logit targets
    all_patches_grouped: np.ndarray,   # [N, P', D] all N training grouped patches
    pca: Optional[PCA],
    refinement_cfg: ImagePALConfig,
    seed: int = 42,
    gpu_ridge_device: str = "cuda",
    verbose: bool = True,
) -> tuple:
    """Fit Ridge from CV pseudo-labels and repool all N training patches.

    Called once after accumulating pseudo-labels from all 4 CV folds.

    Returns
    -------
    (refined, new_pca, weights_ridge, ridge_model, feature_scaler, fit_time_s, pool_time_s)
    """
    import time as _time
    t_start = _time.perf_counter()
    N, P, D = all_patches_grouped.shape

    fit_X = X_flat.astype(np.float32)
    fit_y = y_flat
    if not isinstance(fit_y, np.ndarray):
        try:
            fit_y = fit_y.cpu().numpy()
        except Exception:
            fit_y = np.asarray(fit_y, dtype=np.float32)
    fit_y = fit_y.astype(np.float32)

    feature_scaler: Optional[StandardScaler] = None
    if refinement_cfg.normalize_features:
        feature_scaler = StandardScaler()
        fit_X = feature_scaler.fit_transform(fit_X)

    backend = "GPU" if gpu_ridge_device else "CPU"
    if verbose: print(
        f"[cv_ridge_image] Fitting Ridge(alpha={refinement_cfg.ridge_alpha}) on "
        f"{len(fit_X):,} accumulated samples (D={D}, backend={backend}) ..."
    )
    if gpu_ridge_device:
        ridge_model: Union[Ridge, "RidgeGPU"] = RidgeGPU(
            alpha=refinement_cfg.ridge_alpha, device=gpu_ridge_device
        )
    else:
        ridge_model = Ridge(alpha=refinement_cfg.ridge_alpha)
    ridge_model.fit(fit_X, fit_y)
    if verbose: print(f"[cv_ridge_image] Train R²: {ridge_model.score(fit_X, fit_y):.4f}")

    t_fit_done = _time.perf_counter()
    fit_time_s = t_fit_done - t_start

    if verbose: print("[cv_ridge_image] Pooling all training samples with Ridge weights ...")
    weights_ridge = _ridge_pool_weights(all_patches_grouped, ridge_model, feature_scaler)
    repooled_raw = (weights_ridge[:, :, None] * all_patches_grouped).sum(axis=1).astype(np.float32)

    if pca is not None:
        new_pca = PCA(n_components=pca.n_components_, random_state=seed)
        refined = new_pca.fit_transform(repooled_raw).astype(np.float32)
    else:
        refined = repooled_raw
        new_pca = None

    pool_time_s = _time.perf_counter() - t_fit_done
    if verbose: print(
        f"[cv_ridge_image] fit={fit_time_s:.1f}s  pool={pool_time_s:.1f}s  "
        f"total={fit_time_s + pool_time_s:.1f}s"
    )
    return refined, new_pca, weights_ridge, ridge_model, feature_scaler, fit_time_s, pool_time_s
