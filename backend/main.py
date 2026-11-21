import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import joblib
import librosa
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Use the ML code from the ml/ folder
ML = Path(__file__).resolve().parent.parent / "ml"
DATA = ML / "data"
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


# ---------- Library (mood + duration of every song) ----------
def load_durations():
    cache = DATA / "durations.csv"
    known = {}
    if cache.exists():
        d = pd.read_csv(cache)
        known = dict(zip(d["song"], d["seconds"]))
    changed = False
    for s in catalog["song"]:
        if s not in known and (DATA / s).exists():
            try:
                known[s] = float(librosa.get_duration(path=str(DATA / s)))
                changed = True
            except Exception:
                pass
    if changed:
        pd.DataFrame({"song": list(known), "seconds": list(known.values())}).to_csv(
            cache, index=False
        )
    return known


print("Loading library (first start can take a minute)...")
durations = load_durations()
classes = list(model.classes_)
cat_probs = model.predict_proba(catalog[feature_cols])

library = []
for i, s in enumerate(catalog["song"]):
    if s not in durations:
        continue
    library.append(
        {
            "song": s,
            "mood": classes[int(cat_probs[i].argmax())],
            "probs": dict(zip(classes, map(float, cat_probs[i]))),
            "tempo": float(catalog.loc[i, "tempo"]),
            "seconds": durations[s],
        }
    )
library_names = {s["song"] for s in library}
print("Library ready:", len(library), "songs")


# ---------- Simple NLP: understand playlist requests ----------
# (keywords, moods, (min tempo, max tempo))
RULES = [
    (["workout", "gym", "exercise", "run", "running", "training", "pump"],
     ["energetic", "intense"], (110, None)),
    (["party", "dance", "dancing", "club"], ["energetic"], (100, None)),
    (["energetic", "upbeat", "hype", "happy"], ["energetic"], (None, None)),
    (["intense", "aggressive", "rock", "angry"], ["intense"], (None, None)),
    (["study", "studying", "focus", "reading", "relax", "relaxing", "chill",
      "calm", "peaceful", "lofi"], ["calm"], (None, 120)),
    (["sleep", "sleeping", "bedtime", "meditation"], ["calm"], (None, 100)),
    (["sad", "heartbreak", "breakup", "emotional", "lonely", "cry"],
     ["sad"], (None, None)),
]


def parse_request(q: str):
    t = q.lower()
    moods, tmin, tmax = [], None, None
    for words, ms, (lo, hi) in RULES:
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            for m in ms:
                if m not in moods:
                    moods.append(m)
            if lo is not None:
                tmin = lo if tmin is None else max(tmin, lo)
            if hi is not None:
                tmax = hi if tmax is None else min(tmax, hi)

    total = 0.0
    h = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(?:hours?|hrs?|h)\b", t)
    m = re.search(r"(\d+)\s*-?\s*(?:minutes?|mins?|m)\b", t)
    if h:
        total += float(h.group(1)) * 60
    if m:
        total += float(m.group(1))
    return moods, tmin, tmax, (total or None)


@app.get("/")
def health():
    return {"status": "SonicAI API is running"}


@app.get("/audio/{name}")
def audio(name: str):
    if name not in library_names:
        raise HTTPException(404, "Song not found")
    return FileResponse(DATA / name)


@app.get("/playlist")
def playlist(q: str):
    moods, tmin, tmax, minutes = parse_request(q)
    if not moods:
        return {
            "query": q, "detected_moods": [], "songs": [],
            "note": "I couldn't find a mood or activity. Try: 'energetic workout', "
                    "'calm songs for studying', 'sad songs 30 minutes'.",
        }

    target = (minutes or 30) * 60
    pool = [s for s in library if s["mood"] in moods]
    pool.sort(key=lambda s: sum(s["probs"][m] for m in moods), reverse=True)

    def tempo_ok(s):
        return (tmin is None or s["tempo"] >= tmin) and (tmax is None or s["tempo"] <= tmax)

    # songs that fit the tempo range come first, the rest only if needed
    pool = [s for s in pool if tempo_ok(s)] + [s for s in pool if not tempo_ok(s)]

    chosen, total = [], 0.0
    for s in pool:
        if total >= target:
            break
        chosen.append(s)
        total += s["seconds"]

    note = ""
    if not chosen:
        note = "No songs with this mood in your library yet. Add more songs."
    elif total < target:
        note = f"Only {round(total / 60)} min of matching songs found. Add more songs for a longer playlist."

    return {
        "query": q,
        "detected_moods": moods,
        "target_minutes": round(target / 60),
        "total_minutes": round(total / 60, 1),
        "songs": [
            {
                "song": s["song"], "mood": s["mood"], "tempo": round(s["tempo"]),
                "minutes": round(s["seconds"] / 60, 1),
                "url": "/audio/" + quote(s["song"], safe=""),
            }
            for s in chosen
        ],
        "note": note,
    }


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