import sys
import joblib
import pandas as pd
from pathlib import Path
from audio_features import extract

BASE = Path(__file__).parent
bundle = joblib.load(BASE / "models" / "mood_audio.joblib")
model = bundle["model"]

if len(sys.argv) < 2:
    print('Usage: python predict_audio.py "part of song name"')
    sys.exit()

query = sys.argv[1].lower()
matches = [f for ext in ("*.mp3", "*.wav", "*.flac")
           for f in (BASE / "data").glob(ext) if query in f.name.lower()]
if not matches:
    print("Song not found in data folder.")
    sys.exit()

f = matches[0]
print("Analyzing:", f.name)
row = pd.DataFrame([extract(f)])[bundle["features"]]
probs = model.predict_proba(row)[0]

print(f"\nPredicted mood: {model.classes_[probs.argmax()]}")
for cls, p in sorted(zip(model.classes_, probs), key=lambda x: -x[1]):
    print(f"  {cls}: {p:.0%}")