"""Stratified random baseline.

Samples predictions from the empirical class distribution observed during
``fit``. On a balanced dataset like GTZAN this is functionally uniform,
but on imbalanced data it correctly defaults to the prior - and it gives
us a slightly tighter expected-accuracy floor (sum p_i^2) than uniform.

This is the floor every real model must beat.
"""
import numpy as np

from ..config import RANDOM_SEED
from .base import GenreClassifier


class StratifiedRandomClassifier(GenreClassifier):
    name = "random"
    requires_features = False

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.class_priors_: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StratifiedRandomClassifier":
        # Class count comes from the labels themselves, so the baseline adapts
        # to whatever dataset it's handed (GTZAN's 10 genres, FMA's 8, ...).
        n_classes = int(y.astype(np.int64).max()) + 1
        counts = np.bincount(y.astype(np.int64), minlength=n_classes)
        self.class_priors_ = counts / counts.sum()
        return self

    def _check_fitted(self) -> None:
        if self.class_priors_ is None:
            raise RuntimeError("Call fit() before predict().")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        n = X.shape[0]
        return self._rng.choice(len(self.class_priors_), size=n, p=self.class_priors_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the class prior tiled per-sample.

        This is the *expected* probability under the stratified-random
        strategy - useful for honest reliability diagrams later.
        """
        self._check_fitted()
        n = X.shape[0]
        return np.tile(self.class_priors_, (n, 1))
