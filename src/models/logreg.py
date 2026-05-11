"""Multinomial logistic regression on the summary feature vector.

Strong, fast, well-calibrated baseline. The L2 prior keeps us safe with
~360 features on ~700 training tracks - feature-dim is comparable to
sample size, which is fine for ridged linear models but death for an
unregularized one.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_SEED
from .base import GenreClassifier


class LogisticRegressionClassifier(GenreClassifier):
    name = "logreg"
    requires_features = True

    def __init__(self, C: float = 1.0, seed: int = RANDOM_SEED):
        # multi_class is omitted - sklearn auto-selects multinomial for lbfgs
        # with >2 classes, which is what we want.
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, max_iter=2000, solver="lbfgs", random_state=seed,
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        self.pipe.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X)
