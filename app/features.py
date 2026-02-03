import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.stats import entropy, pearsonr

SILENCE_THRESHOLD = 0.02
WINDOW_SEC = 1.0


def get_rms(y):
    return librosa.feature.rms(y=y)[0]


def breath_entropy(rms):
    hist, _ = np.histogram(rms, bins=30, density=True)
    return entropy(hist + 1e-6)


def jitter_value(path):
    snd = parselmouth.Sound(path)
    pitch = call(snd, "To Pitch", 0.0, 75, 500)
    pp = call([snd, pitch], "To PointProcess (cc)")

    if call(pp, "Get number of points") < 10:
        return 0.0

    return call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)


def windowed_coupling(y, sr):
    hop = int(sr * WINDOW_SEC)
    couplings = []

    for i in range(0, len(y) - hop, hop):
        chunk = y[i:i + hop]

        pitch = librosa.yin(chunk, fmin=75, fmax=500, sr=sr)
        rms = librosa.feature.rms(y=chunk)[0]

        m = min(len(pitch), len(rms))
        pitch, rms = pitch[:m], rms[:m]

        mask = (~np.isnan(pitch)) & (rms > SILENCE_THRESHOLD)
        if np.sum(mask) > 5:
            corr, _ = pearsonr(pitch[mask], rms[mask])
            couplings.append(abs(corr))

    if len(couplings) < 3:
        return 0.0, 0.0, 0.0

    hist, _ = np.histogram(couplings, bins=10, density=True)
    return np.mean(couplings), np.std(couplings), entropy(hist + 1e-6)


def drift_irregularity(y, sr):
    pitch = librosa.yin(y, fmin=75, fmax=500, sr=sr)
    pitch = pitch[~np.isnan(pitch)]

    if len(pitch) < 30:
        return 0.0

    curvature = np.diff(np.diff(pitch))
    return np.std(curvature)


def extract_features(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    rms = get_rms(y)

    cm, cs, ce = windowed_coupling(y, sr)

    return [
        jitter_value(path),
        breath_entropy(rms),
        cm,
        cs,
        ce,
        drift_irregularity(y, sr)
    ]
