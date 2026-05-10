"""Histogram-based gradient boosting on the summary feature vector.

Tree-based, so feature scaling isn't needed and feature interactions are
captured cheaply. A useful contrast to the linear/kernel models — different
inductive biases tend to make different errors, which shows up clearly in
the confusion matrices.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import RANDOM_SEED
from ..features import FeatureMatrix
from .base import GenreClassifier


class GBTClassifier(GenreClassifier):
    name = "gbt"
    requires_features = True

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.08,
        max_depth: Optional[int] = None,
        seed: int = RANDOM_SEED,
    ):
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
        )

    def fit(
        self,
        train_features: Optional[FeatureMatrix],
        train_manifest: pd.DataFrame,
    ) -> "GBTClassifier":
        assert train_features is not None
        self.clf.fit(train_features.X, train_features.y)
        return self

    def predict(
        self,
        features: Optional[FeatureMatrix],
        manifest: pd.DataFrame,
    ) -> np.ndarray:
        assert features is not None
        return self.clf.predict(features.X)

    def predict_proba(
        self,
        features: Optional[FeatureMatrix],
        manifest: pd.DataFrame,
    ) -> np.ndarray:
        assert features is not None
        return self.clf.predict_proba(features.X)
