import os
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Use the ML code from the ml/ folder
ML = Path(__file__).resolve().parent.parent / "ml"
sys.path.append(str(ML))
from audio_features import extract  # noqa: E402

app = FastAPI(title="SonicAI API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

bundle = joblib.load(ML / "models" / "mood_audio.joblib")
model = bundle["model"]
feature_cols = bundle["features"]

catalog = pd.read_csv(ML / "features.csv")
scaler = StandardScaler().fit(catalog[feature_cols])
catalog_X = scaler.transform(catalog[feature_cols])


@app.get("/")
def health():
    return {"status": "SonicAI API is running"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".mp3", ".wav", ".flac"):
        raise HTTPException(400, "Please upload an mp3, wav or flac file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        path = tmp.name
    try:
        feats = extract(path)
    finally:
        os.remove(path)

    row = pd.DataFrame([feats])[feature_cols]

    # Mood prediction
    probs = model.predict_proba(row)[0]
    mood = model.classes_[probs.argmax()]
    mood_probs = {c: round(float(p), 3) for c, p in zip(model.classes_, probs)}

    # Similar songs from your library (skip the uploaded song itself)
    sims = cosine_similarity(scaler.transform(row), catalog_X)[0]
    order = [i for i in sims.argsort()[::-1] if catalog.loc[i, "song"] != file.filename]
    top = order[:3]
    similar = [
        {"song": catalog.loc[i, "song"], "similarity": round(float(sims[i]), 2)}
        for i in top
    ]

    return {
        "filename": file.filename,
        "tempo_bpm": round(feats["tempo"]),
        "mood": mood,
        "mood_probabilities": mood_probs,
        "similar_songs": similar,
    }