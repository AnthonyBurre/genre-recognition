"""Audio feature extraction with on-disk caching.

Produces a fixed-length descriptor per clip - four-moment (mean, std, skew,
kurtosis) summaries over time of MFCC + delta + delta-delta, chroma, tonnetz,
spectral contrast, spectral centroid/bandwidth/rolloff, ZCR, and the onset
envelope; plus scalar tempo and RMS statistics. Sklearn-style classifiers
consume the resulting matrix unchanged.

This module is deliberately separate from the model layer so:
  - swapping classifiers is trivial (kNN, SVM, RF, etc. all consume the same X)
  - heavy frame-level / spectrogram features (for CNNs) can be added as a
    second extractor without disturbing existing code
  - models that don't need features (e.g. the random baseline) skip this
    step entirely (see ``model.requires_features``)

Augmentation hook: ``extract`` accepts an optional ``transform`` callable
applied to the raw waveform before featurization. Pipe training-set audio
through your augmentation chain and call ``extract`` with ``transform=...``;
augmented runs bypass the cache automatically since each call produces
different audio.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from joblib import Memory
from scipy.stats import kurtosis, skew
from tqdm import tqdm

from .config import PROCESSED_DIR, SAMPLE_RATE


_memory = Memory(location=str(PROCESSED_DIR / "joblib"), verbose=0)


AudioTransform = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class FeatureMatrix:
    X: np.ndarray            # (n_tracks, n_features)
    y: np.ndarray            # (n_tracks,)
    track_ids: np.ndarray    # (n_tracks,)
    feature_names: list[str]

    @classmethod
    def empty(cls, manifest: pd.DataFrame) -> "FeatureMatrix":
        """Zero-width matrix carrying labels and track ids only.

        Used by ``train.py`` when a model declares ``requires_features=False``
        - keeps the call site uniform (``model.fit(X, y)`` for everyone)
        without paying the ~30-min cold-cache cost of real feature extraction.
        """
        n = len(manifest)
        return cls(
            X=np.empty((n, 0), dtype=np.float32),
            y=manifest["label"].to_numpy(dtype=np.int64),
            track_ids=manifest["track_id"].to_numpy(),
            feature_names=[],
        )


def _summarize(name: str, values: np.ndarray) -> dict[str, float]:
    """Four-moment summary along the time axis.

    For 2D inputs we summarize each feature bin independently (so MFCC-20
    becomes 80 features, not 4). Skew and kurtosis are zeroed out where
    the variance is degenerate (constant slice) - scipy returns NaN there.
    """
    out: dict[str, float] = {}
    if values.ndim == 1:
        out[f"{name}_mean"] = float(values.mean())
        out[f"{name}_std"] = float(values.std())
        out[f"{name}_skew"] = float(np.nan_to_num(skew(values)))
        out[f"{name}_kurt"] = float(np.nan_to_num(kurtosis(values)))
        return out
    means = values.mean(axis=1)
    stds = values.std(axis=1)
    skews = np.nan_to_num(skew(values, axis=1))
    kurts = np.nan_to_num(kurtosis(values, axis=1))
    for i, (m, s, sk, ku) in enumerate(zip(means, stds, skews, kurts)):
        out[f"{name}{i}_mean"] = float(m)
        out[f"{name}{i}_std"] = float(s)
        out[f"{name}{i}_skew"] = float(sk)
        out[f"{name}{i}_kurt"] = float(ku)
    return out


@_memory.cache
def _extract_one_v2(path: str, sr: int) -> dict[str, float]:
    """Cached single-clip extraction. The ``_v2`` suffix is the cache namespace
    - bump it whenever the feature schema below changes so old caches are
    cleanly orphaned rather than silently mixed with new ones."""
    import librosa  # imported lazily so the random baseline doesn't pay for it

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return _features_from_audio(audio, sr)


def _features_from_audio(
    audio: np.ndarray,
    sr: int,
    harmonic: Optional[np.ndarray] = None,
) -> dict[str, float]:
    import librosa

    feats: dict[str, float] = {}

    # Timbre - MFCCs and their first/second time derivatives.
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    feats.update(_summarize("mfcc", mfcc))
    feats.update(_summarize("mfcc_d1", librosa.feature.delta(mfcc, order=1)))
    feats.update(_summarize("mfcc_d2", librosa.feature.delta(mfcc, order=2)))

    # Tonality - chroma and tonnetz. Tonnetz is derived from the harmonic
    # component since percussion smears the chroma it relies on. Callers
    # can pass a precomputed slice to skip the per-segment HPSS - used by
    # the segment extractor, which computes HPSS once for the whole clip.
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats.update(_summarize("chroma", chroma))
    if harmonic is None:
        harmonic = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    feats.update(_summarize("tonnetz", tonnetz))

    # Spectral shape - contrast captures peak/valley energy across bands and
    # is one of the strongest single descriptors for polyphonic vs.
    # percussive material on GTZAN.
    feats.update(_summarize("contrast", librosa.feature.spectral_contrast(y=audio, sr=sr)))
    feats.update(_summarize("centroid", librosa.feature.spectral_centroid(y=audio, sr=sr)[0]))
    feats.update(_summarize("bandwidth", librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]))
    feats.update(_summarize("rolloff", librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]))
    feats.update(_summarize("zcr", librosa.feature.zero_crossing_rate(audio)[0]))

    # Rhythm - tempo plus onset-envelope shape. We deliberately avoid the
    # full tempogram (hundreds of lag bins) because feature-dim explosion
    # hurts the linear classifiers on a 1000-clip dataset.
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    feats.update(_summarize("onset", onset_env))
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    feats["tempo"] = float(np.atleast_1d(tempo)[0])

    # Dynamics - RMS scalars complement the time-resolved spectral features.
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    feats["rms_mean"] = float(rms)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    feats["crest_factor"] = float(peak / rms)
    return feats


def extract(
    manifest: pd.DataFrame,
    sr: int = SAMPLE_RATE,
    transform: Optional[AudioTransform] = None,
    desc: str = "features",
) -> FeatureMatrix:
    """Featurize every row in ``manifest`` (must have ``path`` and ``label``).

    With ``transform`` set, audio is loaded uncached, transformed, and
    featurized fresh every call - appropriate for stochastic train-time
    augmentation where caching would defeat the purpose.
    """
    rows: list[dict[str, float]] = []
    for path in tqdm(manifest["path"].tolist(), desc=desc):
        if transform is None:
            rows.append(_extract_one_v2(path, sr))
        else:
            import librosa
            audio, _ = librosa.load(path, sr=sr, mono=True)
            audio = transform(audio, sr)
            rows.append(_features_from_audio(audio, sr))

    df = pd.DataFrame(rows)
    feature_names = list(df.columns)
    X = df.to_numpy(dtype=np.float32)
    y = manifest["label"].to_numpy(dtype=np.int64)
    track_ids = manifest["track_id"].to_numpy()
    return FeatureMatrix(X=X, y=y, track_ids=track_ids, feature_names=feature_names)


# ---------------------------------------------------------------------------
# Segment-level extraction
# ---------------------------------------------------------------------------

def _segment_features_from_audio(
    audio: np.ndarray,
    sr: int,
    segment_sec: float,
    hop_sec: float,
) -> list[tuple[int, dict[str, float]]]:
    """Slide a window across ``audio`` and featurize each segment.

    HPSS is computed once on the full clip and the harmonic component is
    sliced per segment - saves ~40% of segment-featurization wall time on
    cold cache versus running HPSS independently per segment, with edge
    effects negligible relative to the 3 s segment length.

    If the clip is shorter than one segment, falls back to a single
    whole-clip segment with index 0 - keeps the function total over any
    valid input.
    """
    import librosa
    seg_len = int(round(segment_sec * sr))
    hop_len = int(round(hop_sec * sr))
    n = len(audio)
    if n < seg_len:
        return [(0, _features_from_audio(audio, sr))]
    harmonic_full = librosa.effects.harmonic(audio)
    out: list[tuple[int, dict[str, float]]] = []
    start = 0
    idx = 0
    while start + seg_len <= n:
        seg = audio[start:start + seg_len]
        seg_harm = harmonic_full[start:start + seg_len]
        out.append((idx, _features_from_audio(seg, sr, harmonic=seg_harm)))
        start += hop_len
        idx += 1
    return out


@_memory.cache
def _extract_segments_v2(
    path: str, sr: int, segment_sec: float, hop_sec: float,
) -> list[tuple[int, dict[str, float]]]:
    """Cached per-clip segment extraction. Cache key is (path, sr, seg, hop)."""
    import librosa
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return _segment_features_from_audio(audio, sr, segment_sec, hop_sec)


def extract_segments(
    manifest: pd.DataFrame,
    sr: int = SAMPLE_RATE,
    segment_sec: float = 3.0,
    hop_sec: float = 1.5,
    transform: Optional[AudioTransform] = None,
    desc: str = "segments",
) -> tuple[FeatureMatrix, pd.DataFrame]:
    """Slide a window across each clip and featurize every window.

    Returns:
      - A ``FeatureMatrix`` with one row per segment (X, y, track_ids
        repeated per segment, feature_names).
      - An expanded segment-level manifest with the same number of rows.
        Each row has every column from ``manifest`` plus a ``segment_idx``
        column carrying its position within the parent clip.

    With ``transform`` set, audio is loaded uncached and transformed before
    segmentation - augmentation must be stochastic per call to be useful.
    """
    feat_rows: list[dict[str, float]] = []
    meta_rows: list[dict] = []
    base_cols = [c for c in manifest.columns]
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=desc):
        path = row["path"]
        if transform is None:
            segs = _extract_segments_v2(path, sr, segment_sec, hop_sec)
        else:
            import librosa
            audio, _ = librosa.load(path, sr=sr, mono=True)
            audio = transform(audio, sr)
            segs = _segment_features_from_audio(audio, sr, segment_sec, hop_sec)
        for seg_idx, feats in segs:
            feat_rows.append(feats)
            meta = {c: row[c] for c in base_cols}
            meta["segment_idx"] = seg_idx
            meta_rows.append(meta)

    feat_df = pd.DataFrame(feat_rows)
    feature_names = list(feat_df.columns)
    X = feat_df.to_numpy(dtype=np.float32)
    seg_manifest = pd.DataFrame(meta_rows).reset_index(drop=True)
    y = seg_manifest["label"].to_numpy(dtype=np.int64)
    track_ids = seg_manifest["track_id"].to_numpy()
    fm = FeatureMatrix(X=X, y=y, track_ids=track_ids, feature_names=feature_names)
    return fm, seg_manifest


def clear_cache() -> None:
    _memory.clear(warn=False)
