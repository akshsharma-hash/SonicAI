import librosa
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"

files = []
for ext in ("*.mp3", "*.wav", "*.flac"):
    files += list(DATA.glob(ext))

rows = []
for f in sorted(files):
    print("Analyzing:", f.name)
    y, sr = librosa.load(f, duration=60)  # first 60 seconds only

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    row = {
        "song": f.name,
        "tempo": float(np.atleast_1d(tempo)[0]),
        "energy": float(librosa.feature.rms(y=y).mean()),
        "spectral_centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y).mean()),
    }
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    for i, v in enumerate(mfcc):
        row[f"mfcc_{i+1}"] = float(v)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(BASE / "features.csv", index=False)
print(df[["song", "tempo", "energy", "spectral_centroid"]])
print("Saved features.csv")