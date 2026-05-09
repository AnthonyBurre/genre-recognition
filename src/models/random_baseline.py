"""Stratified random baseline.

Samples predictions from the empirical class distribution observed during
``fit``. On a balanced dataset like GTZAN this is functionally uniform,
but on imbalanced data it correctly defaults to the prior — and it gives
us a slightly tighter expected-accuracy floor (sum p_i^2) than uniform.

This is the floor every real model must beat.
"""
from typing import Optional

import numpy as np
import pandas as pd

from ..config import GENRES, RANDOM_SEED
from ..features import FeatureMatrix
from .base import GenreClassifier


class StratifiedRandomClassifier(GenreClassifier):
    name = "random"
    requires_features = False

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.class_priors_: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)

    def fit(
        self,
        train_features: Optional[FeatureMatrix],
        train_manifest: pd.DataFrame,
    ) -> "StratifiedRandomClassifier":
        n_classes = len(GENRES)
        counts = np.bincount(train_manifest["label"].to_numpy(), minlength=n_classes)
        self.class_priors_ = counts / counts.sum()
        return self

    def _check_fitted(self) -> None:
        if self.class_priors_ is None:
            raise RuntimeError("Call fit() before predict().")

    def predict(
        self,
        features: Optional[FeatureMatrix],
        manifest: pd.DataFrame,
    ) -> np.ndarray:
        self._check_fitted()
        n = len(manifest)
        return self._rng.choice(len(GENRES), size=n, p=self.class_priors_)

    def predict_proba(
        self,
        features: Optional[FeatureMatrix],
        manifest: pd.DataFrame,
    ) -> np.ndarray:
        """Return the class prior tiled per-sample.

        This is the *expected* probability under the stratified-random
        strategy — useful for honest reliability diagrams later.
        """
        self._check_fitted()
        n = len(manifest)
        return np.tile(self.class_priors_, (n, 1))
