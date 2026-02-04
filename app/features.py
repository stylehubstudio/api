import numpy as np
import librosa
from scipy.stats import entropy, pearsonr


# ================= UTILITIES =================

def safe_entropy(values, bins=30):
    hist, _ = np.histogram(values, bins=bins, density=True)
    return entropy(hist + 1e-8)


def windowed_pitch_energy_coupling(y, sr, window_sec=0.25):
    """
    Computes pitch-energy coupling over short windows.
    Returns mean and variability of coupling.
    """
    hop_length = int(window_sec * sr)
    pitches = librosa.yin(
        y,
        fmin=75,
        fmax=500,
        sr=sr,
        hop_length=hop_length
    )

    rms = librosa.feature.rms(
        y=y,
        hop_length=hop_length
    )[0]

    min_len = min(len(pitches), len(rms))
    pitches = pitches[:min_len]
    rms = rms[:min_len]

    couplings = []

    for i in range(min_len):
        if np.isfinite(pitches[i]) and rms[i] > 1e-6:
            couplings.append(pitches[i] * rms[i])

    if len(couplings) < 5:
        return 0.0, 0.0

    couplings = np.array(couplings)
    return float(np.mean(couplings)), float(np.std(couplings))


def pitch_energy_correlation(y, sr):
    """
    Correlation between pitch and energy over voiced regions.
    """
    pitches = librosa.yin(y, fmin=75, fmax=500, sr=sr)
    rms = librosa.feature.rms(y=y)[0]

    min_len = min(len(pitches), len(rms))
    pitches = pitches[:min_len]
    rms = rms[:min_len]

    mask = np.isfinite(pitches) & (rms > 1e-6)

    if np.sum(mask) < 5:
        return 0.0

    if np.std(pitches[mask]) < 1e-6 or np.std(rms[mask]) < 1e-6:
        return 0.0

    corr, _ = pearsonr(pitches[mask], rms[mask])
    return abs(float(corr))


def temporal_drift_irregularity(y, sr):
    """
    Measures non-smooth timing irregularity in speech.
    AI tends to be smoother than human speech.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    if len(onset_env) < 3:
        return 0.0

    diff = np.diff(onset_env)
    return float(np.std(diff))


# ================= FAST FEATURE SET =================

def extract_features_fast(y, sr):
    """
    FAST inference feature set (used in API)
    - No Praat
    - No disk I/O
    - Biologically motivated
    """

    # Energy envelope
    rms = librosa.feature.rms(y=y)[0]

    # 1️⃣ Breath / energy entropy
    breath_entropy = safe_entropy(rms)

    # 2️⃣ Pitch–energy coupling (mean & variability)
    coupling_mean, coupling_var = windowed_pitch_energy_coupling(y, sr)

    # 3️⃣ Temporal drift irregularity
    drift = temporal_drift_irregularity(y, sr)

    return [
        breath_entropy,
        coupling_mean,
        coupling_var,
        drift
    ]


