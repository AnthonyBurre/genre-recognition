"""Metrics + visualizations.

Every artifact below is written under ``results/<run_name>/`` so different
models can be compared side by side. The plots are intended to answer
two questions a stakeholder will actually ask:

  1. "How good is this model overall, and where does it fall apart?"
     → confusion matrix (counts + row-normalized) and per-class metrics.
  2. "Which tracks did it get wrong?"
     → predicted-vs-actual jitter scatter + the predictions CSV.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import GENRES


@dataclass
class EvalSummary:
    model: str
    split: str
    n_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float


def evaluate(
    *,
    model_name: str,
    split_name: str,
    manifest: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    out_dir: Path,
) -> EvalSummary:
    """Compute metrics, save plots + CSV, and return a JSON-friendly summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = manifest["label"].to_numpy()

    summary = EvalSummary(
        model=model_name,
        split=split_name,
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )

    _save_predictions_csv(manifest, y_pred, y_proba, out_dir / f"predictions_{split_name}.csv")
    _plot_confusion(y_true, y_pred, out_dir / f"confusion_{split_name}.png", split_name)
    _plot_per_class_metrics(y_true, y_pred, out_dir / f"per_class_{split_name}.png", split_name)
    _plot_predicted_vs_actual(y_true, y_pred, out_dir / f"pred_vs_actual_{split_name}.png", split_name)

    report = classification_report(
        y_true, y_pred, target_names=list(GENRES), zero_division=0, output_dict=True
    )
    (out_dir / f"classification_report_{split_name}.json").write_text(json.dumps(report, indent=2))
    (out_dir / f"summary_{split_name}.json").write_text(json.dumps(asdict(summary), indent=2))
    return summary


def _save_predictions_csv(
    manifest: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    path: Path,
) -> None:
    df = manifest[["track_id", "path", "genre", "label"]].copy()
    df["pred_label"] = y_pred
    df["pred_genre"] = [GENRES[int(i)] for i in y_pred]
    df["correct"] = df["label"] == df["pred_label"]
    if y_proba is not None:
        for i, g in enumerate(GENRES):
            df[f"p_{g}"] = y_proba[:, i]
    df.to_csv(path, index=False)


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path, split_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(GENRES))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=GENRES, yticklabels=GENRES, ax=axes[0],
    )
    axes[0].set_title(f"Confusion ({split_name}) — counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, vmin=0, vmax=1,
        xticklabels=GENRES, yticklabels=GENRES, ax=axes[1],
    )
    axes[1].set_title(f"Confusion ({split_name}) — row-normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path, split_name: str
) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(GENRES))), zero_division=0
    )
    df = pd.DataFrame({
        "genre": list(GENRES),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    long = df.melt(id_vars=["genre", "support"], value_vars=["precision", "recall", "f1"],
                   var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="genre", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class metrics ({split_name})")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_predicted_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path, split_name: str
) -> None:
    """Jittered scatter — every track is one point.

    Diagonal = correct, off-diagonal = error. This view tells you *which*
    classes are bleeding into which, on a per-track basis, in a way the
    confusion-matrix counts can't (you see clusters, not just totals).
    """
    rng = np.random.default_rng(0)
    jitter = 0.18
    x = y_true + rng.uniform(-jitter, jitter, size=y_true.shape)
    y = y_pred + rng.uniform(-jitter, jitter, size=y_pred.shape)
    correct = y_true == y_pred

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x[correct], y[correct], s=22, alpha=0.7, color="#2a9d8f", label="correct")
    ax.scatter(x[~correct], y[~correct], s=22, alpha=0.7, color="#e76f51", label="incorrect")
    lim = (-0.5, len(GENRES) - 0.5)
    ax.plot(lim, lim, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(range(len(GENRES)))
    ax.set_yticks(range(len(GENRES)))
    ax.set_xticklabels(GENRES, rotation=45)
    ax.set_yticklabels(GENRES)
    ax.set_xlabel("Actual genre")
    ax.set_ylabel("Predicted genre")
    ax.set_title(f"Predicted vs. actual ({split_name})")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
