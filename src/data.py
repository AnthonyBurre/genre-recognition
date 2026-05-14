"""Dataset indexing and reproducible train/val/test splits.

Two datasets are supported, selected by a ``Dataset`` descriptor
(see ``config.py``):

* **GTZAN** - ``data/raw/genres_original/<genre>/<genre>.NNNNN.wav``.
  Known issues (Sturm, "The GTZAN dataset: Its contents, its faults, their
  effects on evaluation, and its future use", 2013): exact duplicates,
  mislabelings, and artist/album leakage across the splits.
* **FMA small** - ``data/raw/fma_small/fma_small/<prefix>/<id>.mp3`` with
  genre and artist metadata in ``data/raw/fma_small/fma_metadata/tracks.csv``.
  8 balanced genres, 8000 clips - 8x GTZAN, Creative-Commons licensed, and
  ships *real* artist ids so the fault-filtered split can group by true
  artist instead of GTZAN's MFCC-clustered pseudo-artists.

We never assume any particular ordering of files on disk: we glob, sort,
and then split with a fixed seed so the manifest is byte-identical across
machines.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DATASETS,
    DEFAULT_DATASET,
    RANDOM_SEED,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
    Dataset,
    get_dataset,
)


# Per-dataset known-bad track ids, dropped at indexing time.
#   GTZAN: ``jazz.00054.wav`` ships with a corrupted RIFF header in every
#   redistribution - soundfile can't open it. The standard remedy is to drop
#   it; the alternative (ffmpeg re-encode) introduces a binary dependency for
#   one file. Net effect: jazz has 99 tracks instead of 100.
#   FMA small: a handful of mp3s are corrupt, truncated, or near-silent; the
#   FMA authors document these as unusable.
_KNOWN_BAD_TRACK_IDS: dict[str, frozenset[str]] = {
    "gtzan": frozenset({"jazz.00054"}),
    "fma_small": frozenset({
        "098565", "098567", "098569", "099134", "108925", "133297",
    }),
}

# FMA's two zips extract to these directory names under ``dataset.raw_dir``.
FMA_AUDIO_DIRNAME = "fma_small"
FMA_METADATA_DIRNAME = "fma_metadata"


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self, genres: tuple[str, ...]) -> str:
        rows = []
        for name, df in [("train", self.train), ("val", self.val), ("test", self.test)]:
            counts = df["genre"].value_counts().reindex(genres, fill_value=0)
            rows.append(f"{name:>5}: {len(df):4d}  per-genre={counts.tolist()}")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(dataset: Dataset) -> pd.DataFrame:
    """Discover ``dataset``'s tracks and return a sorted manifest.

    Columns: ``path`` (absolute), ``genre`` (str), ``label`` (int), and
    ``track_id`` (str - ``blues.00042`` for GTZAN, the zero-padded numeric id
    for FMA). Sorted by ``track_id`` so the manifest is reproducible.
    """
    if dataset.name == "fma_small":
        return _build_index_fma(dataset)
    return _build_index_gtzan(dataset)


def _build_index_gtzan(dataset: Dataset) -> pd.DataFrame:
    audio_root = dataset.raw_dir.resolve()
    if not audio_root.exists():
        raise FileNotFoundError(
            f"GTZAN audio not found at {audio_root}. "
            f"Run `python -m src.data --dataset {dataset.name} --download` first."
        )

    bad = _KNOWN_BAD_TRACK_IDS[dataset.name]
    genre_to_idx = dataset.genre_to_idx
    rows = []
    for genre in dataset.genres:
        genre_dir = audio_root / genre
        if not genre_dir.is_dir():
            raise FileNotFoundError(f"Missing genre directory: {genre_dir}")
        wavs = sorted(genre_dir.glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files under {genre_dir}")
        for wav in wavs:
            if wav.stem in bad:
                continue
            rows.append({
                "path": str(wav),
                "genre": genre,
                "label": genre_to_idx[genre],
                "track_id": wav.stem,
            })

    return pd.DataFrame(rows).sort_values("track_id", kind="stable").reset_index(drop=True)


def _fma_metadata_path(dataset: Dataset) -> Path:
    return dataset.raw_dir / FMA_METADATA_DIRNAME / "tracks.csv"


def _load_fma_tracks(dataset: Dataset) -> pd.DataFrame:
    """Load FMA's ``tracks.csv``. It ships with a two-level column header and
    the integer track id as the index - the canonical read recipe from the
    FMA repo's ``utils.py``."""
    meta_path = _fma_metadata_path(dataset)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"FMA metadata not found at {meta_path}. "
            f"Run `python -m src.data --dataset {dataset.name} --download` first."
        )
    return pd.read_csv(meta_path, index_col=0, header=[0, 1])


def _build_index_fma(dataset: Dataset) -> pd.DataFrame:
    audio_root = dataset.raw_dir / FMA_AUDIO_DIRNAME
    if not audio_root.exists():
        raise FileNotFoundError(
            f"FMA audio not found at {audio_root}. "
            f"Run `python -m src.data --dataset {dataset.name} --download` first."
        )

    tracks = _load_fma_tracks(dataset)
    genre_top = tracks[("track", "genre_top")]
    bad = _KNOWN_BAD_TRACK_IDS[dataset.name]
    genre_to_idx = dataset.genre_to_idx

    rows = []
    for mp3 in sorted(audio_root.glob("*/*.mp3")):
        track_id = mp3.stem  # zero-padded 6-digit id, e.g. "000002"
        if track_id in bad:
            continue
        genre = genre_top.get(int(track_id))
        # fma_small is pre-filtered to 8 genres, but guard anyway: skip tracks
        # with a missing or out-of-vocabulary genre rather than crashing.
        if genre is None or pd.isna(genre) or genre not in genre_to_idx:
            continue
        rows.append({
            "path": str(mp3.resolve()),
            "genre": genre,
            "label": genre_to_idx[genre],
            "track_id": track_id,
        })

    if not rows:
        raise FileNotFoundError(
            f"No usable FMA tracks found under {audio_root}; "
            "the download may be incomplete."
        )
    return pd.DataFrame(rows).sort_values("track_id", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Naive stratified split
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Split persistence
# ---------------------------------------------------------------------------

def _split_dir_for(dataset: Dataset, variant: str) -> Path:
    if variant == "naive":
        return dataset.splits_dir
    if variant == "filtered":
        return dataset.splits_dir / "filtered"
    raise ValueError(f"Unknown split variant '{variant}' - expected 'naive' or 'filtered'.")


def save_split(split: Split, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(out_dir / "train.csv", index=False)
    split.val.to_csv(out_dir / "val.csv", index=False)
    split.test.to_csv(out_dir / "test.csv", index=False)


def load_split(dataset: Dataset, variant: str = "naive") -> Split:
    split_dir = _split_dir_for(dataset, variant)
    return Split(
        train=pd.read_csv(split_dir / "train.csv"),
        val=pd.read_csv(split_dir / "val.csv"),
        test=pd.read_csv(split_dir / "test.csv"),
    )


def get_or_build_split(
    dataset: Dataset, rebuild: bool = False, variant: str = "naive",
) -> Split:
    """Idempotent: builds the requested split once, reuses it forever after.

    Splits are namespaced per dataset under ``data/splits/<dataset>/`` so
    GTZAN and FMA never collide.
    """
    split_dir = _split_dir_for(dataset, variant)
    expected = [split_dir / f for f in ("train.csv", "val.csv", "test.csv")]
    if not rebuild and all(p.exists() for p in expected):
        return load_split(dataset, variant=variant)
    index = build_index(dataset)
    if variant == "naive":
        split = stratified_split(index)
    else:
        split = build_filtered_split(dataset, index)
    save_split(split, out_dir=split_dir)
    return split


# ---------------------------------------------------------------------------
# Fault-filtered split - addresses dataset duplicates and artist/album leakage.
#
# GTZAN has no artist metadata, so we approximate artist groups by clustering
# MFCC means (Sturm 2013, Kereliuk 2015). FMA ships real artist ids, so for
# FMA we just group by the true artist - cleaner and exact.
# ---------------------------------------------------------------------------

# Cosine-distance threshold in z-scored 20-d MFCC-mean space. Anything below
# this is considered a near-duplicate; raw MFCCs would have mfcc0 (energy)
# dominate the angle, hence the z-score. Empirically captures roughly the
# bottom 1% of within-genre pairs on GTZAN.
DUP_THRESH = 0.05

# Target number of pseudo-artist groups per genre for GTZAN. Fixed-k
# clustering (rather than fixed-distance) keeps every genre well-populated
# regardless of how tight or sparse the within-genre MFCC manifold is.
N_GROUPS_PER_GENRE = 15


def build_filtered_split(
    dataset: Dataset,
    index: Optional[pd.DataFrame] = None,
    seed: int = RANDOM_SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    dup_thresh: float = DUP_THRESH,
    n_groups_per_genre: int = N_GROUPS_PER_GENRE,
    verbose: bool = True,
) -> Split:
    """Build a fault-filtered split with no group leakage across folds.

    The grouping strategy depends on the dataset:

    * **FMA small** - group by the real artist id from ``tracks.csv``. No
      dedup pass: FMA is curated and the artist ids are authoritative.
    * **GTZAN** - no artist metadata, so: (1) drop near-duplicates by
      z-scored cosine distance on MFCC means, (2) agglomerative-cluster the
      survivors within each genre into ``n_groups_per_genre`` pseudo-artist
      groups.

    Either way, the result carries a ``group_id`` column, and whole groups
    are greedily assigned to train/val/test so no group spans folds.
    """
    if index is None:
        index = build_index(dataset)

    if dataset.name == "fma_small":
        filtered_index = index.copy().reset_index(drop=True)
        filtered_index["group_id"] = _fma_artist_groups(dataset, filtered_index)
        if verbose:
            n_groups = filtered_index["group_id"].nunique()
            print(f"[filter] fma_small: {len(filtered_index)} tracks grouped "
                  f"into {n_groups} real-artist groups (no dedup pass)")
    else:
        filtered_index = _gtzan_filtered_index(
            dataset, index, dup_thresh, n_groups_per_genre, verbose,
        )

    return _greedy_group_split(
        filtered_index, dataset.genres, seed,
        train_frac, val_frac, test_frac, verbose,
    )


def _fma_artist_groups(dataset: Dataset, index: pd.DataFrame) -> list[str]:
    """Map each FMA track to a ``artist-<id>`` group. Tracks missing an artist
    id fall back to a singleton ``track-<id>`` group."""
    tracks = _load_fma_tracks(dataset)
    artist_id = tracks[("artist", "id")]
    groups = []
    for track_id in index["track_id"]:
        aid = artist_id.get(int(track_id))
        if aid is None or pd.isna(aid):
            groups.append(f"track-{track_id}")
        else:
            groups.append(f"artist-{int(aid)}")
    return groups


def _zscored_cosine_distances(X: np.ndarray) -> np.ndarray:
    """Cosine distances in the per-dimension z-scored space.

    With raw MFCC means the first coefficient (overall log-energy) has 25x
    the spread of the higher coefficients, so cosine angles are dominated
    by it. Z-scoring equalizes axis weights before the angle is taken.
    """
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    return cosine_distances(Xs)


def _gtzan_filtered_index(
    dataset: Dataset,
    index: pd.DataFrame,
    dup_thresh: float,
    n_groups_per_genre: int,
    verbose: bool,
) -> pd.DataFrame:
    """GTZAN-specific dedup + pseudo-artist grouping.

    Returns ``index`` with near-duplicates dropped and a ``group_id`` column
    added (agglomerative MFCC clusters, one set of ids per genre).
    """
    from sklearn.cluster import AgglomerativeClustering

    from .features import extract  # local import to avoid circular at module load

    n_genres = len(dataset.genres)

    if verbose:
        print(f"[filter] extracting MFCC means for {len(index)} tracks "
              "(joblib-cached; fast on re-runs)")
    fm = extract(index, sr=dataset.sample_rate, desc="filter:mfcc")
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
    for genre_idx in range(n_genres):
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
    for genre_idx, genre in enumerate(dataset.genres):
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
    return filtered_index


def _greedy_group_split(
    filtered_index: pd.DataFrame,
    genres: tuple[str, ...],
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    verbose: bool,
) -> Split:
    """Per-genre, size-aware greedy assignment of whole ``group_id`` groups to
    train/val/test, with a bucket-coverage guarantee.

    The largest groups are placed first so they can't all clump into train;
    each non-empty bucket is reserved at least one (small) group per genre so
    the dominant cluster never ends up isolated in val or test.
    """
    rng = np.random.default_rng(seed)
    target_fracs = {"train": train_frac, "val": val_frac, "test": test_frac}
    pieces: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    summary_rows = []
    for genre_idx, genre in enumerate(genres):
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
            print(f"  {genre:>13}: {n_groups:4d} groups  "
                  f"{n_total:4d} tracks  "
                  f"train={current['train']:4d}  "
                  f"val={current['val']:4d}  "
                  f"test={current['test']:4d}")

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

# FMA is distributed as two zips from the project's host at UNIL. The audio
# zip extracts to ``fma_small/<prefix>/<id>.mp3``; the metadata zip extracts
# to ``fma_metadata/tracks.csv`` (plus other csvs we don't use).
FMA_SMALL_URL = "https://os.unil.ch/fma/fma_small.zip"
FMA_METADATA_URL = "https://os.unil.ch/fma/fma_metadata.zip"
FMA_SMALL_BYTES = 7_761_454_080      # ~7.2 GiB; advisory only.
FMA_METADATA_BYTES = 358_412_237     # ~342 MiB; advisory only.
FMA_EXPECTED_TRACKS = 8000


def _is_already_downloaded(dataset: Dataset) -> bool:
    """True iff ``dataset``'s expected on-disk layout is already present."""
    if dataset.name == "fma_small":
        audio_root = dataset.raw_dir / FMA_AUDIO_DIRNAME
        if not audio_root.exists() or not _fma_metadata_path(dataset).exists():
            return False
        n_mp3 = sum(1 for _ in audio_root.glob("*/*.mp3"))
        return n_mp3 >= FMA_EXPECTED_TRACKS - 50

    audio_root = dataset.raw_dir
    if not audio_root.exists():
        return False
    for genre in dataset.genres:
        genre_dir = audio_root / genre
        if not genre_dir.is_dir():
            return False
        if sum(1 for _ in genre_dir.glob("*.wav")) < EXPECTED_PER_GENRE:
            return False
    return True


def _stream_to_tempfile(
    url: str, expected_bytes: int, dest_dir: Path, desc: str,
) -> Path:
    """Stream ``url`` to a temp file under ``dest_dir`` with a progress bar.

    Returns the temp file path; the caller is responsible for unlinking it.
    """
    import tempfile
    import urllib.request

    from tqdm import tqdm

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    req = urllib.request.Request(url, headers={"User-Agent": "genre-recognition/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or expected_bytes)
        with open(tmp_path, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc
        ) as bar:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))
    return tmp_path


def download(dataset: Dataset, force: bool = False) -> None:
    """Fetch ``dataset``'s audio and lay it out under ``data/raw/``.

    Idempotent: returns immediately if the expected layout is already on
    disk, unless ``force`` is set.
    """
    if not force and _is_already_downloaded(dataset):
        print(f"[download] {dataset.raw_dir} already populated. "
              "Pass --force to redownload.")
        return
    if dataset.name == "fma_small":
        _download_fma(dataset)
    else:
        _download_gtzan(dataset)


def _download_gtzan(dataset: Dataset) -> None:
    """Stream ``genres.tar.gz`` (~1.2 GB) and extract each
    ``genres/<genre>/<genre>.NNNNN.wav`` into ``dataset.raw_dir``."""
    import tarfile

    audio_root = dataset.raw_dir
    for genre in dataset.genres:
        (audio_root / genre).mkdir(parents=True, exist_ok=True)

    print(f"[download] fetching {GTZAN_TARBALL_URL} (~1.2 GB)...")
    genre_set = set(dataset.genres)
    tmp_path = _stream_to_tempfile(
        GTZAN_TARBALL_URL, GTZAN_TARBALL_BYTES, audio_root.parent, "downloading gtzan",
    )
    try:
        from tqdm import tqdm

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
    # the indexer drops known-bad tracks, but those files do still arrive in
    # the tarball and should land on disk.
    on_disk = {g: sum(1 for _ in (audio_root / g).glob("*.wav")) for g in dataset.genres}
    bad = {g: c for g, c in on_disk.items() if c != EXPECTED_PER_GENRE}
    if bad:
        raise RuntimeError(
            f"Post-download layout check failed. Expected {EXPECTED_PER_GENRE} "
            f"tracks per genre, got mismatches: {bad}"
        )
    total = sum(on_disk.values())
    print(f"[download] OK - {total} tracks across {len(dataset.genres)} genres at "
          f"{audio_root}")


def _download_fma(dataset: Dataset) -> None:
    """Stream FMA's metadata and audio zips and extract them under
    ``dataset.raw_dir`` (-> ``fma_metadata/`` and ``fma_small/``)."""
    import zipfile

    raw_dir = dataset.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Metadata first - it's small, and a failure there is cheap to retry
    # before committing to the 7 GB audio download.
    for url, expected_bytes, label in [
        (FMA_METADATA_URL, FMA_METADATA_BYTES, "fma metadata"),
        (FMA_SMALL_URL, FMA_SMALL_BYTES, "fma audio"),
    ]:
        print(f"[download] fetching {url}...")
        tmp_path = _stream_to_tempfile(url, expected_bytes, raw_dir, f"downloading {label}")
        try:
            print(f"[download] extracting {label}...")
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(raw_dir)
        finally:
            tmp_path.unlink(missing_ok=True)

    audio_root = raw_dir / FMA_AUDIO_DIRNAME
    if not _fma_metadata_path(dataset).exists():
        raise RuntimeError(
            f"Post-download check failed: {_fma_metadata_path(dataset)} missing."
        )
    n_mp3 = sum(1 for _ in audio_root.glob("*/*.mp3"))
    if n_mp3 < FMA_EXPECTED_TRACKS - 50:
        raise RuntimeError(
            f"Post-download check failed: expected ~{FMA_EXPECTED_TRACKS} mp3s "
            f"under {audio_root}, found {n_mp3}."
        )
    print(f"[download] OK - {n_mp3} tracks across {len(dataset.genres)} genres at "
          f"{audio_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Dataset utilities - download and split inspection."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASETS),
                        help="Which dataset to operate on.")
    parser.add_argument("--download", action="store_true",
                        help="Fetch the dataset's audio into data/raw/.")
    parser.add_argument("--force", action="store_true",
                        help="Redownload even if data/raw/ already looks populated.")
    parser.add_argument("--build-filtered-split", action="store_true",
                        help="Construct the fault-filtered split (per-artist or "
                             "pseudo-artist grouping). Prints per-genre group "
                             "counts and writes data/splits/<dataset>/filtered/.")
    args = parser.parse_args(argv)

    dataset = get_dataset(args.dataset)

    if args.download:
        download(dataset, force=args.force)
        return 0

    if args.build_filtered_split:
        split = get_or_build_split(dataset, rebuild=True, variant="filtered")
        print("[filtered split]")
        print(split.describe(dataset.genres))
        return 0

    split = get_or_build_split(dataset)
    print(split.describe(dataset.genres))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
