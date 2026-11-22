import os
import re
import sqlite3
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import joblib
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
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
mood_by_song = {s["song"]: s["mood"] for s in library}
print("Library ready:", len(library), "songs")


# ---------- Sound DNA ----------
DNA_KEYS = ["energy", "brightness", "bass", "vocal", "rhythm", "tempo"]
DNA_CACHE = DATA / "dna.csv"


def dna_raw(path):
    y, sr = librosa.load(str(path), duration=45)
    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    power = S ** 2
    total = power.sum() + 1e-9
    bass = power[(freqs >= 20) & (freqs < 250)].sum() / total
    vocal = power[(freqs >= 300) & (freqs < 3400)].sum() / total
    energy = float(librosa.feature.rms(S=S).mean())
    brightness = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    times = librosa.frames_to_time(beats, sr=sr)
    if len(times) > 4:
        gaps = np.diff(times)
        rhythm = float(max(0.0, 1.0 - gaps.std() / (gaps.mean() + 1e-9)))
    else:
        rhythm = 0.0

    return {
        "energy": energy,
        "brightness": brightness,
        "bass": float(bass),
        "vocal": float(vocal),
        "rhythm": rhythm,
        "tempo": float(np.atleast_1d(tempo)[0]),
    }


def load_dna():
    known = {}
    if DNA_CACHE.exists():
        d = pd.read_csv(DNA_CACHE)
        for _, r in d.iterrows():
            known[r["song"]] = {k: float(r[k]) for k in DNA_KEYS}
    missing = [s["song"] for s in library if s["song"] not in known]
    changed = False
    for n, s in enumerate(missing, 1):
        print(f"  Sound DNA {n}/{len(missing)}: {s[:50]}")
        try:
            known[s] = dna_raw(DATA / s)
            changed = True
        except Exception as e:
            print("   skipped:", e)
    if changed:
        rows = [{"song": s, **v} for s, v in known.items()]
        pd.DataFrame(rows).to_csv(DNA_CACHE, index=False)
    return known


print("Computing Sound DNA (only new songs, first time takes a few minutes)...")
dna_lib = load_dna()
dna_df = pd.DataFrame(dna_lib).T if dna_lib else pd.DataFrame(columns=DNA_KEYS)
print("Sound DNA ready:", len(dna_lib), "songs")


def dna_scores(raw):
    """Turn raw values into 0-100 scores relative to your library."""
    out = {}
    for k in DNA_KEYS:
        if k == "tempo":
            lo, hi = 60.0, 180.0
        elif len(dna_df):
            lo, hi = float(dna_df[k].min()), float(dna_df[k].max())
        else:
            lo, hi = 0.0, 1.0
        if hi - lo < 1e-9:
            out[k] = 50
        else:
            out[k] = int(round(100 * min(1.0, max(0.0, (raw[k] - lo) / (hi - lo)))))
    return out


# ---------- Listening history (SQLite) ----------
DB_PATH = DATA / "listening.db"
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db_lock = threading.Lock()
db.execute(
    "CREATE TABLE IF NOT EXISTS plays ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, song TEXT, mood TEXT, "
    "seconds REAL, played_at TEXT)"
)
db.commit()


class LogItem(BaseModel):
    song: str
    seconds: float


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


# ---------- Endpoints ----------
@app.get("/")
def health():
    return {"status": "SonicAI API is running"}


@app.get("/audio/{name}")
def audio(name: str):
    if name not in library_names:
        raise HTTPException(404, "Song not found")
    return FileResponse(DATA / name)


@app.get("/dna")
def dna(song: str):
    if song not in dna_lib:
        raise HTTPException(404, "Song not in library")
    raw = dna_lib[song]
    return {"song": song, "bpm": round(raw["tempo"]), "scores": dna_scores(raw)}


@app.post("/log")
def log_play(item: LogItem):
    if item.seconds < 5:
        return {"saved": False}
    with db_lock:
        db.execute(
            "INSERT INTO plays (song, mood, seconds, played_at) VALUES (?, ?, ?, ?)",
            (
                item.song,
                mood_by_song.get(item.song, "unknown"),
                round(min(item.seconds, 3600.0), 1),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        db.commit()
    return {"saved": True}


@app.get("/stats")
def stats():
    with db_lock:
        total_s, sessions = db.execute(
            "SELECT COALESCE(SUM(seconds), 0), COUNT(*) FROM plays"
        ).fetchone()
        moods = db.execute(
            "SELECT mood, SUM(seconds) AS s FROM plays GROUP BY mood ORDER BY s DESC"
        ).fetchall()
        songs = db.execute(
            "SELECT song, SUM(seconds) AS s FROM plays GROUP BY song ORDER BY s DESC LIMIT 5"
        ).fetchall()
        hrs = db.execute(
            "SELECT strftime('%H', played_at) AS h, SUM(seconds) FROM plays GROUP BY h"
        ).fetchall()
    hours = [0.0] * 24
    for h, s in hrs:
        hours[int(h)] = round(s / 60, 1)
    return {
        "total_minutes": round(total_s / 60, 1),
        "sessions": sessions,
        "moods": [{"mood": m, "minutes": round(s / 60, 1)} for m, s in moods],
        "top_songs": [{"song": n, "minutes": round(s / 60, 1)} for n, s in songs],
        "hours": hours,
    }


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
        raw = dna_raw(path)
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
        "dna": {"bpm": round(raw["tempo"]), "scores": dna_scores(raw)},
    }


@app.get("/app")
def web_app():
    return FileResponse(Path(__file__).resolve().parent.parent / "app" / "index.html")