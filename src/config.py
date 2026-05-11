"""Project-wide constants and paths.

Centralized so models, features, and visualization all agree on the same
genre order, sample rate, and on-disk layout.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # GTZAN audio lives here (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"  # cached feature arrays
SPLITS_DIR = DATA_DIR / "splits"        # serialized train/val/test indices
RESULTS_DIR = PROJECT_ROOT / "results"  # per-run artifacts (plots, CSVs, metrics)

# GTZAN: 10 balanced genres, 100 tracks each, 30s mono clips at 22050 Hz.
GENRES: tuple[str, ...] = (
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
)
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

SAMPLE_RATE = 22050
CLIP_DURATION_SEC = 30.0

# Single seed used everywhere we touch randomness, so runs are reproducible.
RANDOM_SEED = 42

# Default stratified split. Held-out test set is locked once and never touched
# during model selection - val is what we tune on.
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
