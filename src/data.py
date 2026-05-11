"""GTZAN indexing and reproducible train/val/test splits.

GTZAN ships as ``data/raw/genres_original/<genre>/<genre>.NNNNN.wav``.
We never assume any particular ordering of files on disk: we glob, sort,
and then split with a fixed seed so the manifest is byte-identical across
machines.

Known issues (Sturm, "The GTZAN dataset: Its contents, its faults, their
effects on evaluation, and its future use", 2013): exact duplicates,
mislabelings, and artist/album leakage across the splits. Not corrected
here - flagged so the user can swap in a fault-filtered split file later.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

# `jazz.00054.wav` ships with a corrupted RIFF header in every redistribution
# of GTZAN - soundfile can't open it. The standard remedy is to drop it; the
# alternative (ffmpeg re-encode) introduces a binary dependency for one file.
# Net effect: jazz has 99 tracks instead of 100. Acceptable given that the
# dataset is already documented as fault-ridden (see Sturm 2013).
_KNOWN_BAD_TRACK_IDS: frozenset[str] = frozenset({"jazz.00054"})


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
            if wav.stem in _KNOWN_BAD_TRACK_IDS:
                continue
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


def _split_dir_for(variant: str) -> Path:
    if variant == "naive":
        return SPLITS_DIR
    if variant == "filtered":
        return SPLITS_DIR / "filtered"
    raise ValueError(f"Unknown split variant '{variant}' - expected 'naive' or 'filtered'.")


def load_split(variant: str = "naive") -> Split:
    split_dir = _split_dir_for(variant)
    return Split(
        train=pd.read_csv(split_dir / "train.csv"),
        val=pd.read_csv(split_dir / "val.csv"),
        test=pd.read_csv(split_dir / "test.csv"),
    )


def get_or_build_split(rebuild: bool = False, variant: str = "naive") -> Split:
    """Idempotent: builds the requested split once, reuses it forever after."""
    split_dir = _split_dir_for(variant)
    expected = [split_dir / f for f in ("train.csv", "val.csv", "test.csv")]
    if not rebuild and all(p.exists() for p in expected):
        return load_split(variant=variant)
    index = build_index()
    if variant == "naive":
        split = stratified_split(index)
    else:
        split = build_filtered_split(index)
    save_split(split, out_dir=split_dir)
    return split


# ---------------------------------------------------------------------------
# Fault-filtered split - addresses GTZAN's documented duplicates and
# artist/album leakage (Sturm 2013, Kereliuk 2015).
# ---------------------------------------------------------------------------

# Cosine-distance threshold in z-scored 20-d MFCC-mean space. Anything below
# this is considered a near-duplicate; raw MFCCs would have mfcc0 (energy)
# dominate the angle, hence the z-score. Empirically captures roughly the
# bottom 1% of within-genre pairs on GTZAN.
DUP_THRESH = 0.05

# Target number of pseudo-artist groups per genre. Fixed-k clustering (rather
# than fixed-distance) keeps every genre well-populated regardless of how
# tight or sparse the within-genre MFCC manifold is - classical/jazz are
# both far more homogeneous than rock/disco.
N_GROUPS_PER_GENRE = 15


def _zscored_cosine_distances(X: np.ndarray) -> np.ndarray:
    """Cosine distances in the per-dimension z-scored space.

    With raw MFCC means the first coefficient (overall log-energy) has 25×
    the spread of the higher coefficients, so cosine angles are dominated
    by it. Z-scoring equalizes axis weights before the angle is taken.
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    return cosine_distances(Xs)


def build_filtered_split(
    index: Optional[pd.DataFrame] = None,
    seed: int = RANDOM_SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    dup_thresh: float = DUP_THRESH,
    n_groups_per_genre: int = N_GROUPS_PER_GENRE,
    verbose: bool = True,
) -> Split:
    """Build a fault-filtered split.

    Pipeline:
      1. Extract per-clip MFCC means (reusing the cached feature pipeline).
      2. Within each genre, drop near-duplicates (z-scored cosine distance
         < ``dup_thresh``). The lexicographically smallest ``track_id``
         survives each duplicate cluster.
      3. Within each genre, agglomerative-cluster the survivors with
         ``n_clusters=n_groups_per_genre`` and ``linkage='complete'`` to
         form pseudo-artist groups. Fixed-k keeps every genre
         well-populated; complete linkage avoids chain-merging.
      4. Per-genre, sort groups by size descending and greedy-assign each
         to the bucket with the largest size-weighted deficit. The largest
         groups get placed first so they can't all clump into train.
         Each non-empty bucket is guaranteed at least one group per genre
         when group count allows.

    Group ids are written into the resulting ``group_id`` column so callers
    can confirm the no-leak property at any time.
    """
    from sklearn.cluster import AgglomerativeClustering

    from .features import extract  # local import to avoid circular at module load

    if index is None:
        index = build_index()

    if verbose:
        print(f"[filter] extracting MFCC means for {len(index)} tracks "
              "(joblib-cached; fast on re-runs)")
    fm = extract(index, desc="filter:mfcc")
    mfcc_cols = [f"mfcc{i}_mean" for i in range(20)]
    col_idx = [fm.feature_names.index(c) for c in mfcc_cols]
    mfcc_means = fm.X[:, col_idx]  # (n_tracks, 20)

    # Distances are computed once in z-scored space and reused for both
    # dedup and grouping.
    dist_all = _zscored_cosine_distances(mfcc_means)

    # 1+2: per-genre dedup.
    keep_mask = np.ones(len(index), dtype=bool)
    track_ids = index["track_id"].to_numpy()
    labels = index["label"].to_numpy()
    for genre_idx in range(len(GENRES)):
        genre_indices = np.where(labels == genre_idx)[0]
        if len(genre_indices) < 2:
            continue
        for ii in range(len(genre_indices)):
            gi = genre_indices[ii]
            if not keep_mask[gi]:
                continue
            for jj in range(ii + 1, len(genre_indices)):
                gj = genre_indices[jj]
                if not keep_mask[gj]:
                    continue
                if dist_all[gi, gj] < dup_thresh:
                    # Keep the lexicographically-smaller track_id.
                    if track_ids[gi] <= track_ids[gj]:
                        keep_mask[gj] = False
                    else:
                        keep_mask[gi] = False
                        break
    n_dropped = int((~keep_mask).sum())
    if verbose:
        print(f"[filter] dropped {n_dropped} near-duplicates "
              f"(z-scored cosine threshold={dup_thresh})")

    filtered_index = index[keep_mask].copy().reset_index(drop=True)
    kept_orig_indices = np.where(keep_mask)[0]

    # 3: per-genre agglomerative grouping with fixed k.
    group_ids = np.empty(len(filtered_index), dtype=object)
    flabels = filtered_index["label"].to_numpy()
    for genre_idx, genre in enumerate(GENRES):
        local_idxs = np.where(flabels == genre_idx)[0]
        if len(local_idxs) <= 1:
            for k, i in enumerate(local_idxs):
                group_ids[i] = f"{genre}-{k}"
            continue
        # Use the precomputed z-scored cosine submatrix.
        orig_idxs = kept_orig_indices[local_idxs]
        sub_dist = dist_all[np.ix_(orig_idxs, orig_idxs)]
        n_clusters = min(n_groups_per_genre, len(local_idxs))
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="complete",
            metric="precomputed",
        )
        clusters = clusterer.fit_predict(sub_dist)
        for i, c in zip(local_idxs, clusters):
            group_ids[i] = f"{genre}-{int(c)}"
    filtered_index["group_id"] = group_ids

    # 4: per-genre size-aware greedy split with bucket-coverage guarantee.
    rng = np.random.default_rng(seed)
    target_fracs = {"train": train_frac, "val": val_frac, "test": test_frac}
    pieces: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    summary_rows = []
    for genre_idx, genre in enumerate(GENRES):
        sub = filtered_index[filtered_index["label"] == genre_idx]
        groups = [(gid, gdf) for gid, gdf in sub.groupby("group_id", sort=False)]
        # Within equal-size buckets, the random shuffle decides which group
        # goes where, so the assignment is reproducibly randomized.
        rng.shuffle(groups)
        groups.sort(key=lambda gx: -len(gx[1]))  # largest first; stable on ties
        n_total = len(sub)
        current = {"train": 0, "val": 0, "test": 0}

        # Reservation pass - guarantee each non-empty bucket gets at least one
        # group, picked from the smallest groups so the dominant cluster
        # doesn't end up isolated in val or test.
        reserved: dict[str, str] = {}
        small_first = sorted(groups, key=lambda gx: len(gx[1]))
        for bucket, frac in target_fracs.items():
            if frac <= 0:
                continue
            for gid, gdf in small_first:
                if gid in reserved.values():
                    continue
                reserved[bucket] = gid
                pieces[bucket].append(gdf)
                current[bucket] += len(gdf)
                break

        # Greedy fill for the remaining groups.
        for gid, gdf in groups:
            if gid in reserved.values():
                continue
            deficit = {
                bucket: target_fracs[bucket] * n_total - current[bucket]
                for bucket in target_fracs
            }
            best = max(deficit, key=deficit.get)
            pieces[best].append(gdf)
            current[best] += len(gdf)
        summary_rows.append((genre, len(groups), n_total, current))

    if verbose:
        print("[filter] per-genre groups (genre, n_groups, n_tracks, "
              "train/val/test counts):")
        for genre, n_groups, n_total, current in summary_rows:
            print(f"  {genre:>10}: {n_groups:3d} groups  "
                  f"{n_total:3d} tracks  "
                  f"train={current['train']:3d}  "
                  f"val={current['val']:3d}  "
                  f"test={current['test']:3d}")

    train_df = pd.concat(pieces["train"], ignore_index=True)
    val_df = pd.concat(pieces["val"], ignore_index=True)
    test_df = pd.concat(pieces["test"], ignore_index=True)
    sort_key = "track_id"
    return Split(
        train=train_df.sort_values(sort_key).reset_index(drop=True),
        val=val_df.sort_values(sort_key).reset_index(drop=True),
        test=test_df.sort_values(sort_key).reset_index(drop=True),
    )


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

    # Verify the layout matches GTZAN's full 10x100 shape on disk. We check
    # filesystem counts directly rather than going through ``build_index`` -
    # the indexer drops known-bad tracks (see ``_KNOWN_BAD_TRACK_IDS``), but
    # those files do still arrive in the tarball and should land on disk.
    on_disk = {g: sum(1 for _ in (audio_root / g).glob("*.wav")) for g in GENRES}
    bad = {g: c for g, c in on_disk.items() if c != EXPECTED_PER_GENRE}
    if bad:
        raise RuntimeError(
            f"Post-download layout check failed. Expected {EXPECTED_PER_GENRE} "
            f"tracks per genre, got mismatches: {bad}"
        )
    total = sum(on_disk.values())
    print(f"[download] OK - {total} tracks across {len(GENRES)} genres at "
          f"{audio_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="GTZAN data utilities - download and split inspection."
    )
    parser.add_argument("--download", action="store_true",
                        help="Fetch GTZAN via Hugging Face into data/raw/.")
    parser.add_argument("--force", action="store_true",
                        help="Redownload even if data/raw/ already looks populated.")
    parser.add_argument("--build-filtered-split", action="store_true",
                        help="Construct the fault-filtered split (dedup + "
                             "pseudo-artist grouping). Prints per-genre group "
                             "counts and writes data/splits/filtered/.")
    args = parser.parse_args(argv)

    if args.download:
        download(force=args.force)
        return 0

    if args.build_filtered_split:
        split = get_or_build_split(rebuild=True, variant="filtered")
        print("[filtered split]")
        print(split.describe())
        return 0

    split = get_or_build_split()
    print(split.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
