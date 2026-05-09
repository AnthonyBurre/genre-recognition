"""End-to-end training + evaluation driver.

Usage:
    python -m src.train --model random
    python -m src.train --model random --run-name baseline-v1
    python -m src.train --model random --rebuild-split
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import features as features_mod
from . import models
from .config import RESULTS_DIR
from .data import get_or_build_split
from .evaluate import evaluate
from .features import FeatureMatrix


def _maybe_extract(model, manifest, desc: str) -> Optional[FeatureMatrix]:
    if not model.requires_features:
        return None
    return features_mod.extract(manifest, desc=desc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="random", choices=sorted(models.REGISTRY))
    parser.add_argument("--run-name", default=None,
                        help="Subdirectory under results/. Defaults to <model>-<timestamp>.")
    parser.add_argument("--rebuild-split", action="store_true",
                        help="Regenerate train/val/test split files even if they already exist.")
    parser.add_argument("--eval-test", action="store_true",
                        help="Also evaluate on the held-out test set. Off by default to avoid "
                             "test-set leakage during model iteration.")
    args = parser.parse_args(argv)

    run_name = args.run_name or f"{args.model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_dir}")

    split = get_or_build_split(rebuild=args.rebuild_split)
    print("[split]")
    print(split.describe())

    model = models.build(args.model)
    print(f"[model] {model.name} (requires_features={model.requires_features})")

    train_feats = _maybe_extract(model, split.train, desc="features:train")
    val_feats = _maybe_extract(model, split.val, desc="features:val")

    model.fit(train_feats, split.train)

    val_pred = model.predict(val_feats, split.val)
    val_proba = model.predict_proba(val_feats, split.val)
    val_summary = evaluate(
        model_name=model.name,
        split_name="val",
        manifest=split.val,
        y_pred=val_pred,
        y_proba=val_proba,
        out_dir=run_dir,
    )
    print(f"[val] acc={val_summary.accuracy:.3f}  macro_f1={val_summary.macro_f1:.3f}")

    if args.eval_test:
        test_feats = _maybe_extract(model, split.test, desc="features:test")
        test_pred = model.predict(test_feats, split.test)
        test_proba = model.predict_proba(test_feats, split.test)
        test_summary = evaluate(
            model_name=model.name,
            split_name="test",
            manifest=split.test,
            y_pred=test_pred,
            y_proba=test_proba,
            out_dir=run_dir,
        )
        print(f"[test] acc={test_summary.accuracy:.3f}  macro_f1={test_summary.macro_f1:.3f}")

    print(f"[done] artifacts in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
