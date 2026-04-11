"""GPU adapter for TabICLClassifier.

Eliminates repeated host↔device transfers by:
  - Converting the full numpy ensemble arrays to GPU tensors *once* per forward
    call (rather than per mini-batch of estimators).
  - Keeping intermediate outputs on GPU across all mini-batches and through the
    ensemble aggregation step.
  - Exposing ``predict_proba_tensor`` which returns a GPU ``torch.Tensor``
    directly, so callers that have further GPU work (e.g. RidgeGPU) never need
    to round-trip through CPU numpy.

``predict_proba`` is fully overridden to preserve backward compatibility — it
just calls ``predict_proba_tensor(...).cpu().numpy()``.

Usage
-----
    clf = TabICLGPUAdapter(n_estimators=4, random_state=42)
    clf.fit(support_features_np, train_labels_np)   # unchanged: sklearn path
    probs_gpu = clf.predict_proba_tensor(query_features_np)  # torch.Tensor on GPU
"""

from __future__ import annotations

import math
import multiprocessing as mp
from typing import Optional

import numpy as np
import torch

from tabicl import TabICLClassifier


class TabICLGPUAdapter(TabICLClassifier):
    """TabICLClassifier with GPU-persistent tensors during inference.

    All parameters are identical to ``TabICLClassifier``.  The only behavioural
    difference is in ``_batch_forward``, ``_batch_forward_with_cache``, and
    ``predict_proba`` / ``predict_proba_tensor``:

    * Numpy arrays are transferred host→device **once per call** (not once per
      mini-batch of estimators), using ``torch.from_numpy`` (zero-copy) followed
      by a single ``.to(device)`` call.
    * Outputs from every mini-batch are kept on the GPU and concatenated with
      ``torch.cat``.
    * Ensemble aggregation (class-shuffle correction + averaging) is performed
      entirely on the GPU.
    * ``predict_proba_tensor`` returns the final ``[N, C]`` probability tensor
      on the device without ever moving it back to CPU.
    * ``predict_proba`` wraps ``predict_proba_tensor`` and adds ``.cpu().numpy()``
      for drop-in compatibility with sklearn pipelines.
    """

    # ------------------------------------------------------------------
    # Internal forward helpers
    # ------------------------------------------------------------------

    def _batch_forward(
        self,
        Xs: np.ndarray,
        ys: np.ndarray,
        feature_shuffles: Optional[np.ndarray] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """GPU-persistent version of the parent's ``_batch_forward``.

        Parameters
        ----------
        Xs : np.ndarray, shape (n_datasets, n_samples, n_features)
        ys : np.ndarray, shape (n_datasets, train_size)
        feature_shuffles : np.ndarray or None
        attn_mask : torch.Tensor or None
            Attention mask of shape ``(test_size, train_size)`` forwarded to the
            ICL transformer.  Non-zero / True values suppress attention scores.

        Returns
        -------
        torch.Tensor, shape (n_datasets, test_size, n_classes)
            On ``self.device_``.
        """
        # --- single H2D transfer for the full ensemble ---
        Xs_t = torch.from_numpy(np.ascontiguousarray(Xs)).float().to(self.device_)
        ys_t = torch.from_numpy(np.ascontiguousarray(ys)).float().to(self.device_)

        n = Xs_t.shape[0]
        batch_size = self.batch_size or n
        n_batches = math.ceil(n / batch_size)

        # Split on GPU (no additional H2D)
        Xs_batches = Xs_t.split(batch_size, dim=0)
        ys_batches = ys_t.split(batch_size, dim=0)

        if feature_shuffles is None:
            shuffle_batches: list = [None] * n_batches
        else:
            fs_split = np.array_split(feature_shuffles, n_batches)
            shuffle_batches = [
                fs.tolist() if fs is not None and len(fs) > 0 else None
                for fs in fs_split
            ]

        outputs: list[torch.Tensor] = []
        for X_batch, y_batch, shuffle_batch in zip(Xs_batches, ys_batches, shuffle_batches):
            with torch.no_grad():
                out = self.model_(
                    X=X_batch,
                    y_train=y_batch,
                    feature_shuffles=shuffle_batch,
                    return_logits=True if self.average_logits else False,
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                    attn_mask=attn_mask,
                )
            outputs.append(out.float())  # stays on device

        return torch.cat(outputs, dim=0)  # [n_datasets, test_size, n_classes]

    def _batch_forward_with_cache(
        self,
        Xs: np.ndarray,
        kv_cache,
    ) -> torch.Tensor:
        """GPU-persistent version of the parent's ``_batch_forward_with_cache``.

        Parameters
        ----------
        Xs : np.ndarray, shape (n_datasets, test_size, n_features)
        kv_cache : TabICLCache

        Returns
        -------
        torch.Tensor, shape (n_datasets, test_size, n_classes)
            On ``self.device_``.
        """
        n_total = Xs.shape[0]
        batch_size = self.batch_size or n_total

        # --- single H2D transfer ---
        Xs_t = torch.from_numpy(np.ascontiguousarray(Xs)).float().to(self.device_)

        n_batches = math.ceil(n_total / batch_size)
        Xs_batches = Xs_t.split(batch_size, dim=0)

        outputs: list[torch.Tensor] = []
        offset = 0
        for X_batch in Xs_batches:
            bs = X_batch.shape[0]
            cache_subset = kv_cache.slice_batch(offset, offset + bs)
            offset += bs
            with torch.no_grad():
                out = self.model_.forward_with_cache(
                    X_test=X_batch,
                    cache=cache_subset,
                    return_logits=True if self.average_logits else False,
                    softmax_temperature=self.softmax_temperature,
                    inference_config=self.inference_config_,
                )
            outputs.append(out.float())  # stays on device

        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # Public prediction interface
    # ------------------------------------------------------------------

    def predict_proba_tensor(
        self,
        X: np.ndarray,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Like ``predict_proba`` but returns a GPU ``torch.Tensor``.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Test features (same dtype/format as for the parent class).
        attn_mask : torch.Tensor or None, default=None
            Attention mask of shape ``(n_test, n_train)`` forwarded to the ICL
            transformer.  Non-zero / True values suppress attention scores.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_classes)
            Class probabilities on ``self.device_``.  No D2H transfer is
            performed; the caller is responsible for moving to CPU if needed.
        """
        # --- sklearn / n_jobs thread-count bookkeeping (mirrors parent) ---
        old_n_threads = torch.get_num_threads()
        if self.n_jobs is not None:
            n_logical_cores = mp.cpu_count()
            if self.n_jobs > 0:
                n_threads = min(n_logical_cores, self.n_jobs)
            else:
                n_threads = max(1, n_logical_cores + 1 + self.n_jobs)
            torch.set_num_threads(n_threads)

        try:
            # --- numpy preprocessing (can't be avoided: sklearn + ensemble) ---
            from sklearn.utils.validation import validate_data  # type: ignore
            X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)
            X = self.X_encoder_.transform(X)

            if hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None:
                test_data = self.ensemble_generator_.transform(X, mode="test")
                outputs_list: list[torch.Tensor] = []
                for norm_method, (Xs_test,) in test_data.items():
                    kv_cache = self.model_kv_cache_[norm_method]
                    outputs_list.append(self._batch_forward_with_cache(Xs_test, kv_cache))
                outputs = torch.cat(outputs_list, dim=0)
            else:
                data = self.ensemble_generator_.transform(X, mode="both")
                outputs_list = []
                for norm_method, (Xs, ys) in data.items():
                    feature_shuffles = self.ensemble_generator_.feature_shuffles_[norm_method]
                    outputs_list.append(self._batch_forward(Xs, ys, feature_shuffles, attn_mask=attn_mask))
                outputs = torch.cat(outputs_list, dim=0)

            # outputs: [n_estimators, test_size, n_classes]  on GPU

            # --- ensemble aggregation on GPU ---
            class_shuffles = []
            for shuffles in self.ensemble_generator_.class_shuffles_.values():
                class_shuffles.extend(shuffles)
            n_estimators = len(class_shuffles)

            E, T, C = outputs.shape
            shuffles_t = torch.tensor(
                class_shuffles, dtype=torch.long, device=self.device_
            )  # [E, C]
            idx = shuffles_t.unsqueeze(1).expand(-1, T, -1)   # [E, T, C]
            out_shuffled = torch.gather(outputs, 2, idx)       # [E, T, C]
            avg = out_shuffled.sum(dim=0) / n_estimators       # [T, C]

            if self.average_logits:
                avg = torch.softmax(avg / self.softmax_temperature, dim=-1)

            avg = avg / avg.sum(dim=-1, keepdim=True)
            return avg  # [T, C]  on device

        finally:
            if self.n_jobs is not None:
                torch.set_num_threads(old_n_threads)

    def predict_proba(
        self,
        X: np.ndarray,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Drop-in replacement: calls ``predict_proba_tensor`` and converts to numpy."""
        return self.predict_proba_tensor(X, attn_mask=attn_mask).cpu().numpy()
