"""Model registry - adding a model is one line."""
from typing import Callable, Dict

from .base import GenreClassifier
from .gbt import GBTClassifier
from .logreg import LogisticRegressionClassifier
from .random_baseline import StratifiedRandomClassifier
from .svm_rbf import SVMRBFClassifier


REGISTRY: Dict[str, Callable[[], GenreClassifier]] = {
    "random": StratifiedRandomClassifier,
    "logreg": LogisticRegressionClassifier,
    "svm_rbf": SVMRBFClassifier,
    "gbt": GBTClassifier,
}


def build(name: str) -> GenreClassifier:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["GenreClassifier", "REGISTRY", "build"]
