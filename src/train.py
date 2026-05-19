"""End-to-end training + evaluation driver.

Usage:
    python -m src.train --model logreg
    python -m src.train --model svm_rbf --augment default
    python -m src.train --model gbt --split filtered --segments
    python -m src.train --model logreg --rebuild-split
"""
import argparse
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

from . import augment as augment_mod
from . import features as features_mod
from . import models
from .config import DATASETS, DEFAULT_DATASET, RESULTS_DIR, get_dataset
from .data import get_or_build_split
from .evaluate import evaluate
from .features import AudioTransform, FeatureMatrix


def _prepare(
    model,
    manifest: pd.DataFrame,
    desc: str,
    *,
    sr: int,
    use_segments: bool,
    transform: Optional[AudioTransform] = None,
) -> tuple[FeatureMatrix, pd.DataFrame]:
    """Return (features, possibly-expanded manifest) for ``model``.

    For ``requires_features=False`` models we hand back a zero-width
    FeatureMatrix carrying labels and track ids only - that keeps the
    model call sites uniform without paying for real feature extraction.
    With ``use_segments`` set, the manifest is expanded to one row per
    ~3s segment and the FeatureMatrix's row count matches.
    """
    if not model.requires_features:
        return FeatureMatrix.empty(manifest), manifest
    if use_segments:
        return features_mod.extract_segments(
            manifest, sr=sr, desc=desc, transform=transform,
        )
    return features_mod.extract(manifest, sr=sr, desc=desc, transform=transform), manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="random", choices=sorted(models.REGISTRY))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASETS),
                        help="Which dataset to train/evaluate on.")
    parser.add_argument("--run-name", default=None,
                        help="Subdirectory under results/. "
                             "Defaults to <dataset>-<model>-<timestamp>.")
    parser.add_argument("--rebuild-split", action="store_true",
                        help="Regenerate train/val/test split files even if they already exist.")
    parser.add_argument("--eval-test", action="store_true",
                        help="Also evaluate on the held-out test set. Off by default to avoid "
                             "test-set leakage during model iteration.")
    parser.add_argument("--augment", default="none", choices=["none", "default"],
                        help="Waveform augmentation chain applied to the train split only.")
    parser.add_argument("--split", default="naive", choices=["naive", "filtered"],
                        help="Which split to evaluate on. 'filtered' suppresses GTZAN's "
                             "duplicates and artist/album leakage - accuracy will be lower "
                             "but more honest.")
    parser.add_argument("--segments", action="store_true",
                        help="Train per-segment (~3s windows, 1.5s hop) and aggregate "
                             "predictions back to the track level for headline metrics.")
    args = parser.parse_args(argv)

    dataset = get_dataset(args.dataset)
    run_name = args.run_name or (
        f"{dataset.name}-{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")
    print(f"[dataset] {dataset.name} ({len(dataset.genres)} genres)")

    split = get_or_build_split(dataset, rebuild=args.rebuild_split, variant=args.split)
    print(f"[split:{args.split}]")
    print(split.describe(dataset.genres))

    model = models.build(args.model)
    print(f"[model] {model.name} (requires_features={model.requires_features}) "
          f"segments={args.segments}")

    train_transform = augment_mod.default_chain() if args.augment == "default" else None
    if train_transform is not None:
        print(f"[augment] train-only chain: {args.augment}")

    train_feats, _ = _prepare(
        model, split.train, desc=("segments:train" if args.segments else "features:train"),
        sr=dataset.sample_rate, use_segments=args.segments, transform=train_transform,
    )
    val_feats, val_manifest = _prepare(
        model, split.val, desc=("segments:val" if args.segments else "features:val"),
        sr=dataset.sample_rate, use_segments=args.segments,
    )

    model.fit(train_feats.X, train_feats.y)

    val_pred = model.predict(val_feats.X)
    val_proba = model.predict_proba(val_feats.X)
    val_summary = evaluate(
        model_name=model.name,
        split_name="val",
        manifest=val_manifest,
        y_pred=val_pred,
        y_proba=val_proba,
        out_dir=run_dir,
        genres=dataset.genres,
    )
    print(f"[val] acc={val_summary.accuracy:.3f}  macro_f1={val_summary.macro_f1:.3f}")

    if args.eval_test:
        test_feats, test_manifest = _prepare(
            model, split.test, desc=("segments:test" if args.segments else "features:test"),
            sr=dataset.sample_rate, use_segments=args.segments,
        )
        test_pred = model.predict(test_feats.X)
        test_proba = model.predict_proba(test_feats.X)
        test_summary = evaluate(
            model_name=model.name,
            split_name="test",
            manifest=test_manifest,
            y_pred=test_pred,
            y_proba=test_proba,
            out_dir=run_dir,
            genres=dataset.genres,
        )
        print(f"[test] acc={test_summary.accuracy:.3f}  macro_f1={test_summary.macro_f1:.3f}")

    print(f"[done] artifacts in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
