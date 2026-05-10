# genre-recognition

Music genre classification on the GTZAN dataset. Ships a stratified random
baseline plus three classical models (logistic regression, RBF-SVM, gradient
boosting) on a four-moment summary of MFCC+deltas, chroma, tonnetz, spectral
contrast, spectral shape, ZCR, onset envelope, tempo, and dynamics. Optional
fault-filtered split (dedup + pseudo-artist grouping) and segment-level
training with track-level aggregation are available behind CLI flags.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fetch GTZAN (streams the ~1.2 GB `genres.tar.gz` from the `marsyas/gtzan`
Hugging Face mirror directly via `urllib`. The tarball lands in a temp
file under `data/raw/`, gets extracted, and is deleted):

```bash
python -m src.data --download
```

The resulting layout is `data/raw/genres_original/<genre>/<genre>.NNNNN.wav`
across 10 genres (blues, classical, country, disco, hiphop, jazz, metal,
pop, reggae, rock - 100 tracks each, 30s mono at 22050 Hz). Re-running is
a no-op once all 10×100 wavs are on disk; pass `--force` to redownload.

## Run

```bash
python -m src.train --model logreg                            # classical baseline
python -m src.train --model svm_rbf --augment default         # SVM + train-only augmentation
python -m src.train --model gbt --segments                    # segment-level training, track-aggregated metrics
python -m src.train --model logreg --split filtered           # honest, leakage-controlled split
```

Available models: `random`, `logreg`, `svm_rbf`, `gbt`. Artifacts land in
`results/<run-name>/`:

- `confusion_val.png` - count + row-normalized confusion matrices
- `per_class_val.png` - precision / recall / F1 per genre
- `pred_vs_actual_val.png` - jittered scatter, one point per track
- `predictions_val.csv` - per-track predictions + class probabilities
- `predictions_segments_val.csv` - per-segment predictions, only when `--segments` is on
- `classification_report_val.json`, `summary_val.json`

The held-out test split is only evaluated when you pass `--eval-test`, to
avoid test-set leakage during model iteration.

### Fault-filtered split

GTZAN's known duplicates and artist/album leakage inflate accuracy. Build
the leakage-controlled split once, then opt into it via `--split filtered`:

```bash
python -m src.data --build-filtered-split   # one-time, prints per-genre group counts
python -m src.train --model logreg --split filtered
```

The filtered builder drops near-duplicates and clusters tracks within each
genre into pseudo-artist groups (cosine distance on MFCC means), then assigns
whole groups to train/val/test so no group spans folds. Numbers will drop
5–15 pp; they're more honest.

### Segments + augmentation

`--segments` slides a 3 s window with 1.5 s hop across each clip, trains
per-segment, and averages probabilities per `track_id` for the headline
metrics. Use it with `--augment default` (random gain, polarity inversion,
gaussian noise, train-only) for a stronger model.

## Project layout

```
src/
├── config.py             # genres, sample rate, paths, seed
├── data.py               # GTZAN indexer + naive and fault-filtered splits
├── features.py           # 4-moment feature summary + segment extractor, joblib-cached
├── augment.py            # waveform transforms + Compose + default chain
├── models/
│   ├── base.py           # GenreClassifier ABC
│   ├── random_baseline.py
│   ├── logreg.py         # multinomial LR on standardized features
│   ├── svm_rbf.py        # RBF-kernel SVM
│   ├── gbt.py            # histogram gradient boosting
│   └── __init__.py       # name → constructor registry
├── evaluate.py           # metrics + plots + CSV + segment→track aggregation
└── train.py              # CLI driver
```

## Adding a new model

1. Subclass `src.models.base.GenreClassifier` (set `requires_features` as
   appropriate).
2. Register it in `src/models/__init__.py`.
3. `python -m src.train --model <name>`.

## Adding augmentation

`src/augment.py` defines a transform interface `(audio, sr) -> audio` and a
`Compose`. Build a chain and pass it to `features.extract(..., transform=...)`,
or wire it through `src/train.py`'s `--augment` flag (which already enforces
train-only). The `default_chain` is a conservative `RandomGain →
PolarityInversion → AddGaussianNoise` sequence; `TimeStretch` and
`PitchShift` are available but excluded by default - both can confuse genre
cues (tempo for rhythm-heavy genres, key/timbre for classical/jazz).
**Apply only to the training split** - val/test must stay clean.

## GTZAN caveats

The dataset is known to contain duplicate clips, mislabelings, and
artist/album leakage that inflate naive accuracies (Sturm 2013, Kereliuk
2015). One known-corrupt file (`jazz.00054.wav`) is dropped at the indexer
level; the fault-filtered split (`--split filtered`) handles the duplicates
and pseudo-artist grouping. For headline numbers worth quoting, train and
evaluate on the filtered split.
