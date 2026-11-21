import sys
import joblib
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
bundle = joblib.load(BASE / "models" / "mood_lite.joblib")
model = bundle["model"]

if len(sys.argv) < 2:
    print('Usage: python predict_song.py "part of song name"')
    sys.exit()

query = sys.argv[1].lower()
matches = [
    f for ext in ("*.mp3", "*.wav", "*.flac")
    for f in DATA.glob(ext) if query in f.name.lower()
]
if not matches:
    print("Song not found in data folder.")
    sys.exit()

f = matches[0]
print("Analyzing:", f.name)

y, sr = librosa.load(f, duration=60)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
import pyloudnorm as pyln
loudness = pyln.Meter(sr).integrated_loudness(y)
duration = librosa.get_duration(path=f)
if bundle["duration_in_ms"]:
    duration *= 1000

row = pd.DataFrame(
    [[loudness, float(np.atleast_1d(tempo)[0]), duration]],
    columns=bundle["features"],
)
probs = model.predict_proba(row)[0]
print(f"Values -> loudness: {loudness:.1f} dB, tempo: {float(np.atleast_1d(tempo)[0]):.0f}, duration: {duration:.0f}")
print(f"\nPredicted mood: {model.classes_[probs.argmax()]}")
for cls, p in sorted(zip(model.classes_, probs), key=lambda x: -x[1]):
    print(f"  {cls}: {p:.0%}")