import librosa
import numpy as np

def extract(path, duration=60):
    y, sr = librosa.load(path, duration=duration)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    row = {
        "tempo": float(np.atleast_1d(tempo)[0]),
        "energy": float(librosa.feature.rms(y=y).mean()),
        "spectral_centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y).mean()),
    }
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    for i, v in enumerate(mfcc):
        row[f"mfcc_{i+1}"] = float(v)
    return row