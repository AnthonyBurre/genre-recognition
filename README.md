# genre-recognition

Music genre classification on the GTZAN dataset. Currently wired with a
stratified random baseline; designed so swapping in real models and adding
audio augmentation are localized changes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Fetch GTZAN (downloads from the `marsyas/gtzan` Hugging Face mirror — the
first run pulls ~1.2 GB into the HF cache and then writes wavs into
`data/raw/`):

```bash
python -m src.data --download
```

The resulting layout is `data/raw/genres_original/<genre>/<genre>.NNNNN.wav`
across 10 genres (blues, classical, country, disco, hiphop, jazz, metal,
pop, reggae, rock — 100 tracks each, 30s mono at 22050 Hz). Re-running is
a no-op; pass `--force` to redownload.

## Run

```bash
python -m src.train --model random
```

Artifacts land in `results/<run-name>/`:

- `confusion_val.png` — count + row-normalized confusion matrices
- `per_class_val.png` — precision / recall / F1 per genre
- `pred_vs_actual_val.png` — jittered scatter, one point per track
- `predictions_val.csv` — per-track predictions + class probabilities
- `classification_report_val.json`, `summary_val.json`

The held-out test split is only evaluated when you pass `--eval-test`, to
avoid test-set leakage during model iteration.

## Project layout

```
src/
├── config.py             # genres, sample rate, paths, seed
├── data.py               # GTZAN indexer + reproducible stratified split
├── features.py           # MFCC + spectral summary, joblib-cached
├── augment.py            # Compose + AddGaussianNoise (+ stubs)
├── models/
│   ├── base.py           # GenreClassifier ABC
│   ├── random_baseline.py
│   └── __init__.py       # name → constructor registry
├── evaluate.py           # metrics + plots + CSV
└── train.py              # CLI driver
```

## Adding a new model

1. Subclass `src.models.base.GenreClassifier` (set `requires_features` as
   appropriate).
2. Register it in `src/models/__init__.py`.
3. `python -m src.train --model <name>`.

## Adding augmentation

`src/augment.py` defines a transform interface `(audio, sr) -> audio` and a
`Compose`. Build a chain and pass it to `features.extract(..., transform=...)`.
**Apply only to the training split** — val/test must stay clean.

## GTZAN caveats

The dataset is known to contain duplicate clips, mislabelings, and
artist/album leakage that inflate naive accuracies (Sturm 2013). For
serious benchmarking, plug in a fault-filtered split (e.g. Kereliuk et al.
2015) by writing alternative CSVs to `data/splits/`.
