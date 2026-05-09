"""GTZAN indexing and reproducible train/val/test splits.

GTZAN ships as ``data/raw/genres_original/<genre>/<genre>.NNNNN.wav``.
We never assume any particular ordering of files on disk: we glob, sort,
and then split with a fixed seed so the manifest is byte-identical across
machines.

Known issues (Sturm, "The GTZAN dataset: Its contents, its faults, their
effects on evaluation, and its future use", 2013): exact duplicates,
mislabelings, and artist/album leakage across the splits. Not corrected
here — flagged so the user can swap in a fault-filtered split file later.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    GENRES,
    GENRE_TO_IDX,
    RANDOM_SEED,
    RAW_DIR,
    SPLITS_DIR,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)


GTZAN_AUDIO_SUBDIR = "genres_original"


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        rows = []
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            counts = df["genre"].value_counts().reindex(GENRES, fill_value=0)
            rows.append(f"{name:>5}: {len(df):4d}  per-genre={counts.tolist()}")
        return "\n".join(rows)


def build_index(audio_root: Path | None = None) -> pd.DataFrame:
    """Discover GTZAN tracks under ``audio_root`` and return a sorted manifest.

    Columns: ``path`` (absolute), ``genre`` (str), ``label`` (int 0..9),
    ``track_id`` (str, e.g. ``blues.00042``).
    """
    audio_root = (audio_root or RAW_DIR / GTZAN_AUDIO_SUBDIR).resolve()
    if not audio_root.exists():
        raise FileNotFoundError(
            f"GTZAN audio not found at {audio_root}. "
            f"Place the dataset so that {audio_root}/<genre>/<genre>.NNNNN.wav exists."
        )

    rows = []
    for genre in GENRES:
        genre_dir = audio_root / genre
        if not genre_dir.is_dir():
            raise FileNotFoundError(f"Missing genre directory: {genre_dir}")
        wavs = sorted(genre_dir.glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files under {genre_dir}")
        for wav in wavs:
            rows.append({
                "path": str(wav),
                "genre": genre,
                "label": GENRE_TO_IDX[genre],
                "track_id": wav.stem,
            })

    df = pd.DataFrame(rows).sort_values("track_id", kind="stable").reset_index(drop=True)
    return df


def stratified_split(
    index: pd.DataFrame,
    seed: int = RANDOM_SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> Split:
    """Stratified split that preserves genre balance in every fold."""
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    trainval, test = train_test_split(
        index,
        test_size=test_frac,
        stratify=index["label"],
        random_state=seed,
    )
    val_relative = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        trainval,
        test_size=val_relative,
        stratify=trainval["label"],
        random_state=seed,
    )
    sort_key = "track_id"
    return Split(
        train=train.sort_values(sort_key).reset_index(drop=True),
        val=val.sort_values(sort_key).reset_index(drop=True),
        test=test.sort_values(sort_key).reset_index(drop=True),
    )


def save_split(split: Split, out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(out_dir / "train.csv", index=False)
    split.val.to_csv(out_dir / "val.csv", index=False)
    split.test.to_csv(out_dir / "test.csv", index=False)


def load_split(split_dir: Path = SPLITS_DIR) -> Split:
    return Split(
        train=pd.read_csv(split_dir / "train.csv"),
        val=pd.read_csv(split_dir / "val.csv"),
        test=pd.read_csv(split_dir / "test.csv"),
    )


def get_or_build_split(rebuild: bool = False) -> Split:
    """Idempotent: builds the split once, reuses it forever after."""
    expected = [SPLITS_DIR / f for f in ("train.csv", "val.csv", "test.csv")]
    if not rebuild and all(p.exists() for p in expected):
        return load_split()
    index = build_index()
    split = stratified_split(index)
    save_split(split)
    return split


# ---------------------------------------------------------------------------
# Dataset acquisition
# ---------------------------------------------------------------------------

EXPECTED_PER_GENRE = 100  # GTZAN: 100 tracks per genre, 10 genres.

# The HF `marsyas/gtzan` repo hosts the original Marsyas tarball under
# data/genres.tar.gz. We pull it directly: the script-based loader was
# removed in `datasets` 4.0, and the original opihi.cs.uvic.ca host is
# long dead. Tarball layout: genres/<genre>/<genre>.NNNNN.wav.
GTZAN_TARBALL_URL = (
    "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
)
GTZAN_TARBALL_BYTES = 1_226_192_050  # for the progress bar; advisory only.


def _is_already_downloaded() -> bool:
    """True iff the on-disk layout matches GTZAN's full 10x100 shape."""
    audio_root = RAW_DIR / GTZAN_AUDIO_SUBDIR
    if not audio_root.exists():
        return False
    for genre in GENRES:
        genre_dir = audio_root / genre
        if not genre_dir.is_dir():
            return False
        if sum(1 for _ in genre_dir.glob("*.wav")) < EXPECTED_PER_GENRE:
            return False
    return True


def download(force: bool = False) -> None:
    """Fetch GTZAN from the ``marsyas/gtzan`` HF mirror and lay it out for us.

    Idempotent: returns immediately if every genre directory already
    contains the expected number of wavs, unless ``force`` is set.

    Streams ``genres.tar.gz`` (~1.2 GB) into a temp file, then extracts
    each ``genres/<genre>/<genre>.NNNNN.wav`` into
    ``data/raw/genres_original/<genre>/`` so the rest of the pipeline sees
    the canonical layout.
    """
    if not force and _is_already_downloaded():
        print(f"[download] {RAW_DIR / GTZAN_AUDIO_SUBDIR} already populated. "
              "Pass force=True to redownload.")
        return

    import tarfile
    import tempfile
    import urllib.request
    from tqdm import tqdm

    audio_root = RAW_DIR / GTZAN_AUDIO_SUBDIR
    for genre in GENRES:
        (audio_root / genre).mkdir(parents=True, exist_ok=True)

    print(f"[download] fetching {GTZAN_TARBALL_URL} (~1.2 GB)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    genre_set = set(GENRES)

    with tempfile.NamedTemporaryFile(
        suffix=".tar.gz", dir=RAW_DIR, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        req = urllib.request.Request(
            GTZAN_TARBALL_URL,
            headers={"User-Agent": "genre-recognition/1.0"},
        )
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length") or GTZAN_TARBALL_BYTES)
            with open(tmp_path, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc="downloading"
            ) as bar:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    fh.write(chunk)
                    bar.update(len(chunk))

        print("[download] extracting wavs...")
        extracted = 0
        with tarfile.open(tmp_path, "r:gz") as tf:
            members = tf.getmembers()
            for member in tqdm(members, desc="extracting"):
                if not member.isfile():
                    continue
                name = Path(member.name)
                # Expect: genres/<genre>/<genre>.NNNNN.wav
                if len(name.parts) != 3 or name.suffix.lower() != ".wav":
                    continue
                # Skip macOS AppleDouble resource forks (`._foo.wav`) that
                # ride along when the tarball was packed on macOS.
                if name.name.startswith("._"):
                    continue
                genre = name.parts[1]
                if genre not in genre_set:
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                out_path = audio_root / genre / name.name
                with src, open(out_path, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                extracted += 1
        if extracted == 0:
            raise RuntimeError(
                f"No wavs extracted from {GTZAN_TARBALL_URL}; "
                "tarball layout may have changed."
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Verify the layout end-to-end through the same indexer training uses.
    index = build_index()
    counts = index["genre"].value_counts().reindex(GENRES, fill_value=0)
    if (counts != EXPECTED_PER_GENRE).any():
        raise RuntimeError(
            f"Post-download layout check failed. Expected {EXPECTED_PER_GENRE} "
            f"tracks per genre, got:\n{counts.to_string()}"
        )
    print(f"[download] OK — {len(index)} tracks across {len(GENRES)} genres at "
          f"{audio_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="GTZAN data utilities — download and split inspection."
    )
    parser.add_argument("--download", action="store_true",
                        help="Fetch GTZAN via Hugging Face into data/raw/.")
    parser.add_argument("--force", action="store_true",
                        help="Redownload even if data/raw/ already looks populated.")
    args = parser.parse_args(argv)

    if args.download:
        download(force=args.force)
        return 0

    split = get_or_build_split()
    print(split.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
