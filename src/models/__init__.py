"""Model registry — adding a model is one line."""
from typing import Callable, Dict

from .base import GenreClassifier
from .random_baseline import StratifiedRandomClassifier


REGISTRY: Dict[str, Callable[[], GenreClassifier]] = {
    "random": StratifiedRandomClassifier,
}


def build(name: str) -> GenreClassifier:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["GenreClassifier", "REGISTRY", "build"]
