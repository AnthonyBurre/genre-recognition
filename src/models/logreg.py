"""Multinomial logistic regression on the summary feature vector.

Strong, fast, well-calibrated baseline. The L2 prior keeps us safe with
~360 features on ~700 training tracks — feature-dim is comparable to
sample size, which is fine for ridged linear models but death for an
unregularized one.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_SEED
from ..features import FeatureMatrix
from .base import GenreClassifier


class LogisticRegressionClassifier(GenreClassifier):
    name = "logreg"
    requires_features = True

    def __init__(self, C: float = 1.0, seed: int = RANDOM_SEED):
        # multi_class is omitted — sklearn auto-selects multinomial for lbfgs
        # with >2 classes, which is what we want.
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, max_iter=2000, solver="lbfgs", random_state=seed,
            )),
        ])

    def fit(
        self,
        train_features: Optional[FeatureMatrix],
        train_manifest: pd.DataFrame,
    ) -> "LogisticRegressionClassifier":
        assert train_features is not None
        self.pipe.fit(train_features.X, train_features.y)
        return self

    def predict(
        self,
        features: Optional[FeatureMatrix],
        manifest: pd.DataFrame,
    ) -> np.ndarray:
        assert features is not None
        return self.pipe.predict(features.X)

    def predict_proba(
        self,
        features: Optional[FeatureMatrix],
        manifest: pd.DataFrame,
    ) -> np.ndarray:
        assert features is not None
        return self.pipe.predict_proba(features.X)
