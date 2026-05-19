# Music Genre Recognition

Music genre classification on the **GTZAN** and **FMA small** datasets. Ships a stratified random baseline plus three classical models (logistic
regression, RBF-SVM, gradient boosting) on a four-moment summary of
MFCC+deltas, chroma, tonnetz, spectral contrast, spectral shape, ZCR, onset
envelope, tempo, and dynamics. Optional fault-filtered split and segment-level
training with track-level aggregation are available behind CLI flags.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install it
([instructions](https://docs.astral.sh/uv/getting-started/installation/)),
then:

```bash
uv sync
```

That resolves the lockfile and creates `.venv/`. Prefix commands with `uv run`
(shown below) or `source .venv/bin/activate` once and drop the prefix.

## Datasets

The dataset is selected with `--dataset` (default `gtzan`) on every CLI. Each
dataset has its own genre vocabulary, on-disk layout under `data/raw/`, and
split files under `data/splits/<dataset>/`.

| Dataset     | Genres | Clips  | Notes                                                        |
| ----------- | -----: | -----: | ------------------------------------------------------------ |
| `gtzan`     |     10 |  1,000 | 30s mono WAV @ 22050 Hz. Tiny, fast to iterate on.           |
| `fma_small` |      8 |  8,000 | 30s MP3, Creative-Commons, ships real artist ids. 8× GTZAN.  |

Fetch a dataset (idempotent; re-running is a no-op once the audio is on disk,
`--force` redownloads):

```bash
uv run python -m src.data --dataset gtzan --download       # ~1.2 GB genres.tar.gz from the marsyas/gtzan HF mirror
uv run python -m src.data --dataset fma_small --download   # ~7.5 GB: fma_small.zip + fma_metadata.zip from os.unil.cloud.switch.ch
```

GTZAN lands at `data/raw/genres_original/<genre>/<genre>.NNNNN.wav`; FMA at
`data/raw/fma_small/fma_small/<prefix>/<id>.mp3` with metadata in
`data/raw/fma_small/fma_metadata/tracks.csv`. MP3 decoding goes through
`librosa` — if your installed `libsndfile` predates MP3 support you may also
need `ffmpeg` on `PATH` (the `audioread` fallback uses it).

## Run

```bash
uv run python -m src.train --model logreg                            # classical baseline (GTZAN)
uv run python -m src.train --model svm_rbf --augment default         # SVM + train-only augmentation
uv run python -m src.train --model gbt --segments                    # segment-level training, track-aggregated metrics
uv run python -m src.train --model logreg --split filtered           # honest, leakage-controlled split
uv run python -m src.train --model logreg --dataset fma_small        # train on FMA small instead of GTZAN
```

Available models: `random`, `logreg`, `svm_rbf`, `gbt`. Add `--dataset
fma_small` to any command to run on FMA instead of GTZAN. Artifacts land in
`results/<run-name>/` (run name defaults to `<dataset>-<model>-<timestamp>`):

- `confusion_val.png` - count + row-normalized confusion matrices
- `per_class_val.png` - precision / recall / F1 per genre
- `pred_vs_actual_val.png` - jittered scatter, one point per track
- `predictions_val.csv` - per-track predictions + class probabilities
- `predictions_segments_val.csv` - per-segment predictions, only when `--segments` is on
- `classification_report_val.json`, `summary_val.json`

The held-out test split is only evaluated when you pass `--eval-test`, to
avoid test-set leakage during model iteration.

### Fault-filtered split

Duplicates and artist/album leakage inflate accuracy. Build the
leakage-controlled split once, then opt into it via `--split filtered`:

```bash
uv run python -m src.data --build-filtered-split                      # GTZAN, one-time
uv run python -m src.data --dataset fma_small --build-filtered-split  # FMA, one-time
uv run python -m src.train --model logreg --split filtered
```

The grouping strategy depends on the dataset, but either way whole groups are
assigned to train/val/test so no group spans folds:

- **GTZAN** has no artist metadata, so the builder drops near-duplicates and
  clusters tracks within each genre into *pseudo-artist* groups (cosine
  distance on MFCC means). Numbers will drop 5–15 pp; they're more honest.
- **FMA small** ships real artist ids in `tracks.csv`, so the builder groups
  by *true artist* directly — no dedup heuristic needed.

### Segments + augmentation

`--segments` slides a 3 s window with 1.5 s hop across each clip, trains
per-segment, and averages probabilities per `track_id` for the headline
metrics. Use it with `--augment default` (random gain, polarity inversion,
gaussian noise, train-only) for a stronger model.

## Visualizing the feature set

Before trusting the feature set in a model, look at it. `src/visualize.py`
renders a 3D PCA scatter, one point per track, color-coded by genre:

```bash
uv run python -m src.visualize                                       # pca3d, naive split, train fold
uv run python -m src.visualize --split filtered --fold all           # full dataset, filtered split
uv run python -m src.visualize --dataset fma_small --fold all        # FMA feature set
```

`--fold` takes `train` / `val` / `test` / `all` (`all` ignores `--split` and
uses the full dataset index). `--dataset` selects the dataset as elsewhere.
Artifacts land in `results/visualizations/`:

- `pca3d_<split>_<fold>.png` - static, fixed-angle render for run artifacts
- `pca3d_<split>_<fold>.html` - interactive, rotatable; this is the one to
  actually read cluster separation from

Features are z-scored before PCA (the set mixes scales like tempo ~120 against
near-zero MFCC moments). Tight per-genre clouds mean the features carry genre
signal; one uniform blob means they don't - classical and metal should sit far
apart, while rock/country/disco tend to overlap. Add another visualization by
writing a `plot_*` function and registering it in `PLOT_REGISTRY`.

## Results

Validation-set numbers from the runs in `results/`. The split is seed-fixed and reproducible, so re-running yields the same metrics
modulo `sklearn` non-determinism.

| Dataset     | Model     | Split    | Segments | Val acc | Macro F1 | n_samples |
| ----------- | --------- | -------- | :------: | ------: | -------: | --------: |
| `gtzan`     | random    | naive    |    -     |   0.100 |    0.101 |       100 |
| `gtzan`     | logreg    | naive    |    -     |   0.760 |    0.761 |       100 |
| `gtzan`     | svm_rbf   | naive    |    -     |   0.750 |    0.745 |       100 |
| `gtzan`     | gbt       | naive    |    -     |   0.740 |    0.739 |       100 |
| `gtzan`     | logreg    | filtered |    -     |   0.624 |    0.617 |        93 |
| `gtzan`     | svm_rbf   | filtered |    -     |   0.677 |    0.667 |        93 |
| `gtzan`     | gbt       | filtered |    -     |   0.656 |    0.635 |        93 |
| `gtzan`     | logreg    | filtered |    ✓     |   0.753 |    0.748 |        93 |
| `gtzan`     | svm_rbf   | filtered |    ✓     |   0.753 |    0.753 |        93 |
| `gtzan`     | gbt       | filtered |    ✓     |   0.699 |    0.691 |        93 |
| `fma_small` | random    | naive    |    -     |   0.122 |    0.122 |       800 |
| `fma_small` | logreg    | naive    |    -     |   0.513 |    0.511 |       800 |
| `fma_small` | svm_rbf   | naive    |    -     |   0.575 |    0.573 |       800 |
| `fma_small` | gbt       | naive    |    -     |   0.629 |    0.628 |       800 |
| `fma_small` | logreg    | filtered |    -     |   0.463 |    0.454 |       800 |
| `fma_small` | svm_rbf   | filtered |    -     |   0.487 |    0.487 |       800 |
| `fma_small` | gbt       | filtered |    -     |   0.524 |    0.518 |       800 |
| `fma_small` | logreg    | filtered |    ✓     |   0.500 |    0.487 |       800 |
| `fma_small` | svm_rbf   | filtered |    ✓     |   n/a † |    n/a † |       800 |
| `fma_small` | gbt       | filtered |    ✓     |   0.554 |    0.545 |       800 |

† `svm_rbf` + segments on FMA is impractical to fit: sklearn's libsvm is
between O(n²) and O(n³), and with `probability=True` the internal 5-fold CV
for Platt scaling roughly 6× that — projected 1–4 h for the ~120k segment
training set. Not run.

Reading across the two datasets:

1. **GTZAN naive is optimistic.** 0.760 looks strong, but the naive →
   filtered drop (–13.6 pp) is GTZAN's documented leakage (duplicates
   and artist/album overlap, Sturm 2013) being priced out. The filtered
   number is what the model actually generalizes to.
2. **FMA naive (0.513) is much lower than GTZAN naive (0.760)** — and
   that is the point. FMA is 8× larger, drawn from a far broader pool of
   artists, and Creative-Commons curated; 0.513 on 8 genres (vs. a 0.122
   random floor) is a *harder, more honest* number than GTZAN's leaky
   0.760. The gap is the dataset, not the model.
3. **FMA's naive → filtered drop is only ~5 pp** (0.513 → 0.463), vs.
   GTZAN's ~14 pp. Less leakage to subtract: FMA is curated and the
   filtered split groups by *real* artist ids, while GTZAN's filtered
   split has to approximate artist groups from MFCC clusters.
4. **Segments help GTZAN much more than FMA** (+12.9 pp vs. +3.7 pp on
   the filtered split). On GTZAN segments effectively replace the data
   the dedup step removed; FMA's filtered train set is already 6,400
   tracks, so the marginal data gain from 19 segments/track is much
   smaller. Soft-vote aggregation still flips wrong segment-majorities
   to right at the track level on both.
5. **Best model depends on the dataset.** `logreg` only wins on GTZAN
   naive — which is the row most distorted by leakage. On the
   leakage-controlled splits the picture flips: **`svm_rbf` wins GTZAN
   filtered** (0.677 vs. logreg's 0.624; with segments it ties `logreg`
   at 0.753) and **`gbt` wins FMA** at every configuration — peaking at
   **0.554 / 0.545 on filtered+segments**, the best FMA result in the
   table and +5 pp above logreg's segments row. The linear model's bias
   hurts more as the dataset gets larger and noisier; the tree
   ensemble's feature-interaction modelling pays off where the linear
   baseline plateaus.

## Project layout

```
src/
├── config.py             # Dataset descriptors (genres, layout), paths, seed
├── data.py               # GTZAN + FMA indexers, downloaders, naive and fault-filtered splits
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
├── visualize.py          # exploratory feature-set plots (PCA), own CLI
└── train.py              # CLI driver
```

## Adding a new model

1. Subclass `src.models.base.GenreClassifier`. The contract is sklearn-style:
   `fit(X, y)`, `predict(X)`, optional `predict_proba(X)`. Set
   `requires_features = False` only if your model can train without the
   audio feature pipeline (it still receives a zero-width X carrying labels
   and track ids).
2. Register it in `src/models/__init__.py`.
3. `uv run python -m src.train --model <name>`.

## Adding augmentation

`src/augment.py` defines a transform interface `(audio, sr) -> audio` and a
`Compose`. Build a chain and pass it to `features.extract(..., transform=...)`,
or wire it through `src/train.py`'s `--augment` flag (which already enforces
train-only). The `default_chain` is a conservative `RandomGain →
PolarityInversion → AddGaussianNoise` sequence; `TimeStretch` and
`PitchShift` are available but excluded by default - both can confuse genre
cues (tempo for rhythm-heavy genres, key/timbre for classical/jazz).
**Apply only to the training split** - val/test must stay clean.

## Next steps

Feature-set ideas not yet implemented, roughly in order of expected payoff
per unit of effort:

- **Beat-synchronous framing.** Aggregate the spectral/timbral features over
  beat-aligned frames (`librosa.util.sync` against `librosa.beat.beat_track`)
  instead of fixed-length frames. Makes the four-moment summaries
  tempo-invariant, which should help the genres where tempo varies within a
  class but timbre doesn't.
- **Percussive-component features.** `_features_from_audio` already runs HPSS
  to get the harmonic component for tonnetz - the percussive residual is
  computed and thrown away. Summarizing it (or its onset envelope) is nearly
  free and carries rhythm/attack signal the current set doesn't capture.
- **Compact rhythm descriptor.** We deliberately skip the full tempogram for
  feature-dim reasons (`features.py:137-139`). A low-dimensional middle
  ground - a coarse beat histogram, or tempogram-ratio features - would add
  rhythm structure without the dim explosion that hurts the linear models.
- **Log-mel spectrogram extractor.** A second extractor producing frame-level
  log-mel spectrograms (already foreshadowed in the `features.py` module
  docstring) would unlock a CNN model behind the existing
  `requires_features` contract.

## Dataset caveats

**GTZAN** is known to contain duplicate clips, mislabelings, and artist/album
leakage that inflate naive accuracies (Sturm 2013, Kereliuk 2015). One
known-corrupt file (`jazz.00054.wav`) is dropped at the indexer level; the
fault-filtered split (`--split filtered`) handles the duplicates and
pseudo-artist grouping. For headline numbers worth quoting, train and
evaluate on the filtered split.

**FMA small** is larger (8× the clips) and Creative-Commons licensed, with
real artist ids that make the filtered split exact rather than heuristic. A
handful of corrupt/truncated mp3s are dropped at the indexer level. It is the
better choice when scale matters; GTZAN remains the fast smoke-test default.
`fma_medium` (25k clips, 16 unbalanced genres) is a natural next step but is
not wired in — it would need unbalanced-aware splitting.
