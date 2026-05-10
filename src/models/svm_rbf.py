"""SVM with an RBF kernel on the summary feature vector.

Historically the strongest pre-deep-learning model on GTZAN. Slower than
logreg (O(n^2) at fit time) but n is small here so it's still seconds.
``probability=True`` enables Platt-scaled probabilities at the cost of an
internal CV fit; we want them for the evaluation reliability plots.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import RANDOM_SEED
from ..features import FeatureMatrix
from .base import GenreClassifier


class SVMRBFClassifier(GenreClassifier):
    name = "svm_rbf"
    requires_features = True

    def __init__(self, C: float = 10.0, gamma: str | float = "scale", seed: int = RANDOM_SEED):
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=C, gamma=gamma, kernel="rbf",
                probability=True, random_state=seed,
            )),
        ])

    def fit(
        self,
        train_features: Optional[FeatureMatrix],
        train_manifest: pd.DataFrame,
    ) -> "SVMRBFClassifier":
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
