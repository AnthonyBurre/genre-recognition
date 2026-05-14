"""Base contract every genre classifier must satisfy.

Standard sklearn-style: ``fit(X, y)``, ``predict(X)``, optional
``predict_proba(X)``. Predictions are class indices ordered by the active
dataset's genre list (``Dataset.genres``); probability matrices are shape
``(n_samples, n_classes)`` in the same order.

The ``requires_features`` class attribute tells ``train.py`` whether to run
the (slow) feature pipeline for this model. When ``False`` (e.g. the
random baseline, which only needs label statistics), the driver hands the
model a zero-width ``X`` so the call sites stay uniform - every model
accepts ``(X, y)`` regardless.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class GenreClassifier(ABC):
    name: str = "base"
    requires_features: bool = True

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GenreClassifier":
        """Train on (X, y). For ``requires_features=False`` models, X may be
        zero-width - the model is expected to use only y (and ``X.shape[0]``
        at predict time)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices, shape ``(n_samples,)``."""

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Optional. Default ``None`` - evaluation falls back to one-hots."""
        return None
