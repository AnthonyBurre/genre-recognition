"""Audio augmentation primitives — applied to the raw waveform.

The interface is intentionally minimal: any callable ``(audio, sr) -> audio``
is a valid transform, and ``Compose`` chains them. This composes naturally
with ``features.extract(..., transform=...)``.

Only apply augmentations to the *training* split. Validation and test must
see clean audio so reported metrics reflect real-world generalization.

Each transform stores its own ``np.random.Generator`` so seeding is honored
end-to-end. Pass the same RNG to multiple transforms (or use ``default_chain``)
to get a single deterministic randomness stream across the whole chain.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

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
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(RANDOM_SEED)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        snr_db = self.rng.uniform(*self.snr_db_range)
        signal_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        noise_rms = signal_rms / (10 ** (snr_db / 20))
        noise = self.rng.normal(0.0, noise_rms, size=audio.shape).astype(audio.dtype)
        return audio + noise


@dataclass
class RandomGain:
    """Scale the waveform by a random gain in dB.

    Cheap, label-preserving, and the simplest way to teach the model that
    absolute level is not informative for genre.
    """
    db_range: tuple[float, float] = (-6.0, 6.0)
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(RANDOM_SEED)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        gain_db = float(self.rng.uniform(*self.db_range))
        return (audio * (10 ** (gain_db / 20))).astype(audio.dtype)


@dataclass
class PolarityInversion:
    """Flip the waveform's sign with probability ``p``.

    Audibly identical to the original but doubles the variety the encoder
    sees in the time domain. Spectral magnitudes are unchanged, so the
    effect is subtle — it mostly regularizes the time-domain features (zcr,
    crest factor) against fragile sign conventions.
    """
    p: float = 0.5
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(RANDOM_SEED)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.rng.random() < self.p:
            return -audio
        return audio


@dataclass
class TimeStretch:
    """Resample in time without altering pitch."""
    rate_range: tuple[float, float] = (0.9, 1.1)
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(RANDOM_SEED)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        rate = float(self.rng.uniform(*self.rate_range))
        return librosa.effects.time_stretch(audio, rate=rate)


@dataclass
class PitchShift:
    """Shift pitch in semitones without altering duration."""
    semitones_range: tuple[float, float] = (-2.0, 2.0)
    rng: Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(RANDOM_SEED)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        n_steps = float(self.rng.uniform(*self.semitones_range))
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def default_chain(seed: int = RANDOM_SEED) -> Compose:
    """Conservative augmentation chain that's safe to apply unconditionally.

    Excludes time-stretch / pitch-shift by default — both are slow and have
    the strongest potential to confuse genre cues (tempo for rhythm-heavy
    genres, key/timbre for classical/jazz). Add them deliberately when
    iterating, not as a baseline.
    """
    rng = np.random.default_rng(seed)
    return Compose([
        RandomGain(rng=rng),
        PolarityInversion(rng=rng),
        AddGaussianNoise(rng=rng),
    ])
