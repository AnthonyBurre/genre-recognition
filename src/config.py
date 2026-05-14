"""Project-wide constants, paths, and dataset descriptors.

Centralized so models, features, and visualization all agree on genre
ordering, sample rate, and on-disk layout.

The genre vocabulary is *dataset-specific* - GTZAN has 10 genres, FMA small
has 8 different ones - so it lives on a ``Dataset`` descriptor rather than a
single global. Code that needs the genre list (data indexing, evaluation,
visualization) takes a ``Dataset`` and reads ``dataset.genres``.
"""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # dataset audio lives here (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"  # cached feature arrays
SPLITS_DIR = DATA_DIR / "splits"        # serialized train/val/test indices
RESULTS_DIR = PROJECT_ROOT / "results"  # per-run artifacts (plots, CSVs, metrics)

# Single seed used everywhere we touch randomness, so runs are reproducible.
RANDOM_SEED = 42

# Default stratified split. Held-out test set is locked once and never touched
# during model selection - val is what we tune on.
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# Default analysis sample rate. Both shipped datasets are resampled to this on
# load; ``features.py`` uses it as its default and callers pass
# ``Dataset.sample_rate`` explicitly.
SAMPLE_RATE = 22050


@dataclass(frozen=True)
class Dataset:
    """One genre-recognition dataset: its genre vocabulary and on-disk layout.

    ``genres`` fixes the class ordering (label ``i`` is ``genres[i]``) the
    same way the old module-level ``GENRES`` tuple did - but per dataset, so
    GTZAN and FMA can coexist without colliding.
    """
    name: str
    genres: tuple[str, ...]
    raw_subdir: str          # audio root, relative to RAW_DIR
    splits_subdir: str       # split CSVs, relative to SPLITS_DIR
    sample_rate: int = SAMPLE_RATE
    clip_duration_sec: float = 30.0

    @property
    def genre_to_idx(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.genres)}

    @property
    def raw_dir(self) -> Path:
        return RAW_DIR / self.raw_subdir

    @property
    def splits_dir(self) -> Path:
        return SPLITS_DIR / self.splits_subdir


# GTZAN: 10 balanced genres, 100 tracks each, 30s mono clips at 22050 Hz.
GTZAN = Dataset(
    name="gtzan",
    genres=(
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    ),
    raw_subdir="genres_original",
    splits_subdir="gtzan",
)

# FMA small: 8 balanced genres, 1000 tracks each, 30s clips. Genre strings
# match the ``track.genre_top`` values in FMA's ``tracks.csv`` exactly, so
# the indexer can join on them without normalization.
FMA_SMALL = Dataset(
    name="fma_small",
    genres=(
        "Electronic", "Experimental", "Folk", "Hip-Hop",
        "Instrumental", "International", "Pop", "Rock",
    ),
    raw_subdir="fma_small",
    splits_subdir="fma_small",
)

DATASETS: dict[str, Dataset] = {d.name: d for d in (GTZAN, FMA_SMALL)}
DEFAULT_DATASET = "gtzan"


def get_dataset(name: str) -> Dataset:
    """Resolve a dataset name to its descriptor, or raise with the valid set."""
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{name}'. Known datasets: {sorted(DATASETS)}"
        ) from None
