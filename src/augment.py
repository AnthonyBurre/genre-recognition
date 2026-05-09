"""Audio augmentation primitives — applied to the raw waveform.

The interface is intentionally minimal: any callable ``(audio, sr) -> audio``
is a valid transform, and ``Compose`` chains them. This composes naturally
with ``features.extract(..., transform=...)``.

Only apply augmentations to the *training* split. Validation and test must
see clean audio so reported metrics reflect real-world generalization.
"""
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .config import RANDOM_SEED


AudioTransform = Callable[[np.ndarray, int], np.ndarray]


class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: Sequence[AudioTransform]):
        self.transforms = list(transforms)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        for t in self.transforms:
            audio = t(audio, sr)
        return audio


class Identity:
    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return audio


@dataclass
class AddGaussianNoise:
    """Add white noise at a randomly sampled SNR in dB.

    Lower SNR ⇒ louder noise. Range chosen so the signal is still clearly
    audible; tighten if the model starts overfitting to the noise itself.
    """
    snr_db_range: tuple[float, float] = (10.0, 30.0)
    rng: np.random.Generator = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(RANDOM_SEED)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        snr_db = self.rng.uniform(*self.snr_db_range)
        signal_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        noise_rms = signal_rms / (10 ** (snr_db / 20))
        noise = self.rng.normal(0.0, noise_rms, size=audio.shape).astype(audio.dtype)
        return audio + noise


# Stubs — wire up when we move past the random baseline. librosa.effects
# already implements both transforms; we just haven't committed to ranges
# or to the additional dependency cost yet.
@dataclass
class TimeStretch:
    rate_range: tuple[float, float] = (0.9, 1.1)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        rate = float(np.random.default_rng().uniform(*self.rate_range))
        return librosa.effects.time_stretch(audio, rate=rate)


@dataclass
class PitchShift:
    semitones_range: tuple[float, float] = (-2.0, 2.0)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        n_steps = float(np.random.default_rng().uniform(*self.semitones_range))
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
