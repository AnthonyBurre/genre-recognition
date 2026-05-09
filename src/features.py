"""Audio feature extraction with on-disk caching.

Default feature set is a per-clip summary vector — mean and std-dev over
time of MFCCs, chroma, spectral centroid/bandwidth/rolloff, and zero-crossing
rate. That gives a fixed-length descriptor per track which any sklearn-style
classifier can consume.

This is deliberately separate from the model layer so that:
  - swapping models is trivial (kNN, SVM, RF, etc. all consume the same X)
  - heavy frame-level / spectrogram features (for CNNs) can be added as a
    second extractor without disturbing existing code
  - the random baseline can skip this step entirely (see ``model.requires_features``)

Augmentation hook: ``extract`` accepts an optional ``transform`` callable
applied to the raw waveform before featurization. Pipe training-set audio
through your augmentation chain and call ``extract`` with ``cache=False``.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from joblib import Memory
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


def _summarize(name: str, values: np.ndarray) -> dict[str, float]:
    """Mean + std along the time axis, flattened across feature bins."""
    out: dict[str, float] = {}
    if values.ndim == 1:
        out[f"{name}_mean"] = float(values.mean())
        out[f"{name}_std"] = float(values.std())
        return out
    means = values.mean(axis=1)
    stds = values.std(axis=1)
    for i, (m, s) in enumerate(zip(means, stds)):
        out[f"{name}{i}_mean"] = float(m)
        out[f"{name}{i}_std"] = float(s)
    return out


@_memory.cache
def _extract_one(path: str, sr: int) -> dict[str, float]:
    """Cached single-clip extraction. Cache key is (path, sr, code version)."""
    import librosa  # imported lazily so the random baseline doesn't pay for it

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return _features_from_audio(audio, sr)


def _features_from_audio(audio: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    feats: dict[str, float] = {}
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    feats.update(_summarize("mfcc", mfcc))

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats.update(_summarize("chroma", chroma))

    feats.update(_summarize("centroid", librosa.feature.spectral_centroid(y=audio, sr=sr)[0]))
    feats.update(_summarize("bandwidth", librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]))
    feats.update(_summarize("rolloff", librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]))
    feats.update(_summarize("zcr", librosa.feature.zero_crossing_rate(audio)[0]))
    feats["rms_mean"] = float(np.sqrt(np.mean(audio ** 2)))
    return feats


def extract(
    manifest: pd.DataFrame,
    sr: int = SAMPLE_RATE,
    transform: Optional[AudioTransform] = None,
    desc: str = "features",
) -> FeatureMatrix:
    """Featurize every row in ``manifest`` (must have ``path`` and ``label``).

    With ``transform`` set, audio is loaded uncached, transformed, and
    featurized fresh every call — appropriate for stochastic train-time
    augmentation where caching would defeat the purpose.
    """
    rows: list[dict[str, float]] = []
    for path in tqdm(manifest["path"].tolist(), desc=desc):
        if transform is None:
            rows.append(_extract_one(path, sr))
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


def clear_cache() -> None:
    _memory.clear(warn=False)
