"""Base contract every genre classifier must satisfy.

Two execution paths are supported:

  - feature-based: ``requires_features = True`` — model is fitted on
    ``(X, y)`` matrices produced by ``src.features.extract``.
  - manifest-based: ``requires_features = False`` — model receives the
    raw manifest DataFrame (paths + labels) and decides what to do with
    it. Used by the random baseline (which only needs labels) and is the
    seam where future end-to-end audio models will plug in.

Models always return predicted class indices, and optionally a
probability matrix shaped ``(n_samples, n_classes)`` ordered by
``config.GENRES``.
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from ..features import FeatureMatrix


class GenreClassifier(ABC):
    name: str = "base"
    requires_features: bool = True

    @abstractmethod
    def fit(
        self,
        train_features: Optional[FeatureMatrix],
        train_manifest: pd.DataFrame,
    ) -> "GenreClassifier":
        """Train the model. Implementations use whichever input they need."""

    @abstractmethod
    def predict(
        self,
        features: Optional[FeatureMatrix],
        manifest: pd.DataFrame,
    ) -> np.ndarray:
        """Return predicted class indices, shape ``(n_samples,)``."""

    def predict_proba(
        self,
        features: Optional[FeatureMatrix],
        manifest: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Optional. Default ``None`` — evaluation falls back to one-hots."""
        return None
